#!/usr/bin/env julia

# GPU Sparse Linear Solve Experiment
# Goal: Compare CPU vs GPU sparse direct solvers for ADMM W matrix

import FOMPrototypes
using CUDA, CUDA.CUSPARSE, CUDSS
using SparseArrays, LinearAlgebra, Random, Printf

println("="^80)
println("GPU Sparse Direct Solver Experiment (CPU vs cuDSS)")
println("="^80)

# ============================================================================
# Setup
# ============================================================================

# Check GPU availability
CUDA.functional() || error("CUDA GPU required for this experiment")
println("\n[✓] GPU: ", CUDA.name(CUDA.device()))

# Load test problem
problem = FOMPrototypes.fetch_data("sslsq", "NYPA_Maragal_3_huber")
P, A, m, n = problem.P, problem.A, problem.m, problem.n
println("[✓] Problem loaded: n=$n, m=$m, nnz(P)=$(nnz(P)), nnz(A)=$(nnz(A))")

# Setup ADMM system: W = P + ρ * A' * A
ρ = 0.1
δ = 1e-10  # Regularization shift
Random.seed!(12345)
b_test = randn(n)

# ============================================================================
# CPU Baseline (Sparse Cholesky)
# ============================================================================

println("\n" * "="^80)
println("CPU Sparse Cholesky")
println("="^80)

# Setup: form W matrix
cpu_setup_time = @elapsed begin
    W_cpu = P + ρ * (A' * A)
end
println("W: $(size(W_cpu)), nnz=$(nnz(W_cpu))")

# Decomposition: Cholesky factorization
cpu_decomp_time = @elapsed begin
    F_cpu = cholesky(W_cpu; shift=δ)
end

# Solve: forward + backward substitution (combined in Julia's ldiv)
cpu_solve_time = @elapsed begin
    x_cpu = F_cpu \ b_test
end

residual_cpu = norm((W_cpu + δ*I) * x_cpu - b_test) / norm(b_test)
println(@sprintf("Setup: %.4f ms | Decomp: %.4f ms | Solve: %.4f ms | Residual: %.2e",
    cpu_setup_time * 1000, cpu_decomp_time * 1000, cpu_solve_time * 1000, residual_cpu))

# ============================================================================
# GPU cuDSS (Sparse Direct)
# ============================================================================

println("\n" * "="^80)
println("GPU cuDSS Sparse Direct Solver")
println("="^80)

try
    # Setup: transfer to GPU and form W (CSR format required by cuDSS)
    gpu_setup_time = @elapsed begin
        P_gpu = CuSparseMatrixCSR(P)
        A_gpu = CuSparseMatrixCSR(A)
        W_gpu_raw = P_gpu + ρ * (A_gpu' * A_gpu)
        # Add regularization shift (cuDSS doesn't support shift parameter)
        W_gpu = W_gpu_raw + δ * I
        b_gpu = CuArray(b_test)
        x_gpu = similar(b_gpu)
    end

    # Decomposition: cuDSS analysis + factorization
    gpu_decomp_time = @elapsed begin
        solver = CudssSolver(W_gpu, "SPD", 'F')  # Symmetric Positive Definite, Full matrix
        cudss("analysis", solver, x_gpu, b_gpu)
        cudss("factorization", solver, x_gpu, b_gpu)
    end

    # Forward substitution (L y = b)
    # Note: cuDSS combines forward/backward into single "solve" phase
    # We time them together as Julia's CPU cholesky does
    gpu_fwd_time = @elapsed begin
        cudss("solve", solver, x_gpu, b_gpu)
    end
    gpu_bwd_time = 0.0  # Combined with forward in cuDSS

    # Transfer result back
    gpu_back_time = @elapsed begin
        x_gpu_cpu = Array(x_gpu)
    end

    # Validation
    sol_diff = norm(x_cpu - x_gpu_cpu) / norm(x_cpu)
    residual_gpu = norm(W_cpu * x_gpu_cpu - b_test) / norm(b_test)

    println(@sprintf("Setup: %.4f ms | Decomp: %.4f ms | Solve: %.4f ms | Back: %.4f ms | Residual: %.2e",
        gpu_setup_time * 1000, gpu_decomp_time * 1000, gpu_fwd_time * 1000, gpu_back_time * 1000, residual_gpu))

    # Phase-by-phase comparison
    println("\n" * "="^80)
    println("Phase-by-Phase Comparison (CPU vs GPU)")
    println("="^80)

    println(@sprintf("Setup:        CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_setup_time * 1000, gpu_setup_time * 1000, cpu_setup_time / gpu_setup_time))

    println(@sprintf("Decomposition: CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_decomp_time * 1000, gpu_decomp_time * 1000, cpu_decomp_time / gpu_decomp_time))

    println(@sprintf("Solve:        CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_solve_time * 1000, gpu_fwd_time * 1000, cpu_solve_time / gpu_fwd_time))

    cpu_total = cpu_setup_time + cpu_decomp_time + cpu_solve_time
    gpu_total = gpu_setup_time + gpu_decomp_time + gpu_fwd_time + gpu_back_time

    println(@sprintf("\nTotal:        CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx %s",
        cpu_total * 1000, gpu_total * 1000, cpu_total / gpu_total,
        cpu_total / gpu_total > 1.0 ? "✓" : "⚠"))

    println(@sprintf("\nSolution diff: %.2e %s", sol_diff, sol_diff < 1e-6 ? "✓" : "⚠"))

    # Realistic iteration scenario: decompose once, solve many times
    println("\n" * "="^80)
    println("Iteration Scenario Analysis")
    println("="^80)
    println("Simulating: 1 decomposition + N solves (typical ADMM pattern)")
    println()

    for n_iters in [10, 100, 1000]
        # CPU cost: setup once + decompose once + solve N times
        cpu_iter_time = cpu_setup_time + cpu_decomp_time + n_iters * cpu_solve_time

        # GPU cost: setup once (includes transfer) + decompose once + solve N times + transfer back once
        # Note: In real usage, you'd keep data on GPU between solves, so setup is one-time
        gpu_iter_time = gpu_setup_time + gpu_decomp_time + n_iters * gpu_fwd_time + gpu_back_time

        speedup_iter = cpu_iter_time / gpu_iter_time

        println(@sprintf("%4d iters:  CPU %8.2f ms  |  GPU %8.2f ms  |  Speedup: %.2fx %s",
            n_iters, cpu_iter_time * 1000, gpu_iter_time * 1000, speedup_iter,
            speedup_iter > 1.0 ? "✓" : "⚠"))
    end

    println("\nNote: GPU setup cost is amortized across iterations.")
    println("      Data stays on GPU between solves in real algorithms.")

catch e
    println("✗ cuDSS failed: $e")
    println("\nPossible reasons:")
    println("  • CUDSS.jl not installed: add it with `] add CUDSS`")
    println("  • Matrix not positive definite (try larger δ)")
    println("  • GPU memory insufficient")
end

println("\n" * "="^80)
