#!/usr/bin/env julia

# GPU Sparse Linear Solve Experiment
# Goal: Compare CPU vs GPU sparse direct solvers for ADMM W matrix

import FOMPrototypes
using CUDA, CUDA.CUSPARSE, CUDSS
using SparseArrays, LinearAlgebra, Random, Printf, BenchmarkTools

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
problem = FOMPrototypes.fetch_data("mpc", "nonlinearChain_2")
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

# Pre-allocate output buffer for allocation-free solves
x_cpu = zeros(n)

# Solve: forward + backward substitution
# Warmup solve for validation
ldiv!(x_cpu, F_cpu, b_test)
residual_cpu = norm((W_cpu + δ*I) * x_cpu - b_test) / norm(b_test)

# Benchmark allocating solve (x = F \ b) - shows cost of allocation
println("  Benchmarking CPU solve (allocating)...")
cpu_solve_alloc_bench = @benchmark $F_cpu \ $b_test
cpu_solve_alloc_time = median(cpu_solve_alloc_bench.times) / 1e9
cpu_solve_alloc_allocs = cpu_solve_alloc_bench.allocs
cpu_solve_alloc_memory = cpu_solve_alloc_bench.memory

# Benchmark in-place solve (ldiv!) - realistic for ADMM iterations
println("  Benchmarking CPU solve (in-place)...")
cpu_solve_inplace_bench = @benchmark ldiv!($x_cpu, $F_cpu, $b_test)
cpu_solve_inplace_time = median(cpu_solve_inplace_bench.times) / 1e9
cpu_solve_inplace_allocs = cpu_solve_inplace_bench.allocs
cpu_solve_inplace_memory = cpu_solve_inplace_bench.memory

# Use in-place timing for iteration scenarios (realistic for ADMM)
cpu_solve_time = cpu_solve_inplace_time

println(@sprintf("Setup: %.4f ms | Decomp: %.4f ms | Solve (alloc): %.4f ms (%d allocs, %.2f KB) | Solve (in-place): %.4f ms (%d allocs, %.2f KB) | Residual: %.2e",
    cpu_setup_time * 1000, cpu_decomp_time * 1000,
    cpu_solve_alloc_time * 1000, cpu_solve_alloc_allocs, cpu_solve_alloc_memory / 1024.0,
    cpu_solve_inplace_time * 1000, cpu_solve_inplace_allocs, cpu_solve_inplace_memory / 1024.0,
    residual_cpu))

# ============================================================================
# GPU cuDSS (Sparse Direct)
# ============================================================================

println("\n" * "="^80)
println("GPU cuDSS Sparse Direct Solver")
println("="^80)

try
    # Setup: transfer to GPU and form W (CSR format required by cuDSS)
    gpu_setup_time = CUDA.@elapsed begin
        P_gpu = CuSparseMatrixCSR(P)
        A_gpu = CuSparseMatrixCSR(A)
        W_gpu_raw = P_gpu + ρ * (A_gpu' * A_gpu)
        # Add regularization shift (cuDSS doesn't support shift parameter)
        W_gpu = W_gpu_raw + δ * I
        b_gpu = CuArray(b_test)
        x_gpu = similar(b_gpu)
    end

    # Decomposition: cuDSS analysis + factorization
    gpu_decomp_time = CUDA.@elapsed begin
        solver = CudssSolver(W_gpu, "SPD", 'F')  # Symmetric Positive Definite, Full matrix
        cudss("analysis", solver, x_gpu, b_gpu)
        cudss("factorization", solver, x_gpu, b_gpu)
    end

    # Solve: forward + backward substitution (combined in cuDSS)
    # Warmup solve to ensure CUDA kernels are compiled
    cudss("solve", solver, x_gpu, b_gpu)
    CUDA.synchronize()

    # Benchmark the solve using CUDA.@sync for proper synchronization
    # Note: BenchmarkTools reports CPU allocations only, not GPU allocations
    println("  Benchmarking GPU solve (in-place)...")
    gpu_solve_bench = @benchmark CUDA.@sync cudss("solve", $solver, $x_gpu, $b_gpu)
    gpu_fwd_time = median(gpu_solve_bench.times) / 1e9  # Convert ns to seconds
    gpu_solve_allocs = gpu_solve_bench.allocs  # CPU allocations
    gpu_solve_memory = gpu_solve_bench.memory  # CPU memory
    gpu_bwd_time = 0.0  # Combined with forward in cuDSS

    # Transfer result back
    gpu_back_time = CUDA.@elapsed begin
        x_gpu_cpu = Array(x_gpu)
    end

    # Validation
    sol_diff = norm(x_cpu - x_gpu_cpu) / norm(x_cpu)
    residual_gpu = norm(W_cpu * x_gpu_cpu - b_test) / norm(b_test)

    println(@sprintf("Setup: %.4f ms | Decomp: %.4f ms | Solve: %.4f ms (%d allocs, %.2f KB) | Back: %.4f ms | Residual: %.2e",
        gpu_setup_time * 1000, gpu_decomp_time * 1000, gpu_fwd_time * 1000,
        gpu_solve_allocs, gpu_solve_memory / 1024.0, gpu_back_time * 1000, residual_gpu))

    # Phase-by-phase comparison
    println("\n" * "="^80)
    println("Phase-by-Phase Comparison (Median Times from BenchmarkTools)")
    println("="^80)

    println(@sprintf("Setup:         CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_setup_time * 1000, gpu_setup_time * 1000, cpu_setup_time / gpu_setup_time))

    println(@sprintf("Decomposition: CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_decomp_time * 1000, gpu_decomp_time * 1000, cpu_decomp_time / gpu_decomp_time))

    println(@sprintf("Solve (alloc): CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_solve_alloc_time * 1000, gpu_fwd_time * 1000, cpu_solve_alloc_time / gpu_fwd_time))

    println(@sprintf("Solve (reuse): CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx",
        cpu_solve_time * 1000, gpu_fwd_time * 1000, cpu_solve_time / gpu_fwd_time))

    cpu_total = cpu_setup_time + cpu_decomp_time + cpu_solve_time
    gpu_total = gpu_setup_time + gpu_decomp_time + gpu_fwd_time + gpu_back_time

    println(@sprintf("\nTotal (reuse): CPU %8.4f ms  |  GPU %8.4f ms  |  Speedup: %.2fx %s",
        cpu_total * 1000, gpu_total * 1000, cpu_total / gpu_total,
        cpu_total / gpu_total > 1.0 ? "✓" : "⚠"))

    println(@sprintf("\nSolution diff: %.2e %s", sol_diff, sol_diff < 1e-6 ? "✓" : "⚠"))

    # Allocation Analysis
    println("\n" * "="^80)
    println("Allocation Analysis")
    println("="^80)
    println(@sprintf("  CPU solve (allocating):  %d bytes/solve, %d allocs (%.2f KB)",
        cpu_solve_alloc_memory, cpu_solve_alloc_allocs, cpu_solve_alloc_memory / 1024.0))
    println(@sprintf("  CPU solve (in-place):    %d bytes/solve, %d allocs (%.2f KB)",
        cpu_solve_inplace_memory, cpu_solve_inplace_allocs, cpu_solve_inplace_memory / 1024.0))
    println(@sprintf("  GPU solve (in-place):    %d bytes/solve, %d allocs (%.2f KB)",
        gpu_solve_memory, gpu_solve_allocs, gpu_solve_memory / 1024.0))
    println()
    println("Note: Both CPU ldiv! and GPU cuDSS use pre-allocated buffers (realistic for ADMM)")
    println("      Allocation overhead avoided by reusing output vector across iterations")

    # Realistic iteration scenario: decompose once, solve many times
    println("\n" * "="^80)
    println("Iteration Scenario Analysis")
    println("="^80)
    println("Simulating: 1 decomposition + N solves (typical ADMM pattern)")
    println()

    for n_iters in [10, 100, 1000]
        # CPU cost: setup once + decompose once + solve N times (in-place)
        cpu_iter_time_reuse = cpu_setup_time + cpu_decomp_time + n_iters * cpu_solve_time
        cpu_iter_alloc_mb_reuse = (n_iters * cpu_solve_inplace_memory) / (1024.0^2)

        # CPU cost if allocating each solve (for comparison)
        cpu_iter_time_alloc = cpu_setup_time + cpu_decomp_time + n_iters * cpu_solve_alloc_time
        cpu_iter_alloc_mb_alloc = (n_iters * cpu_solve_alloc_memory) / (1024.0^2)

        # GPU cost: setup once (includes transfer) + decompose once + solve N times + transfer back once
        # Note: In real usage, you'd keep data on GPU between solves, so setup is one-time
        gpu_iter_time = gpu_setup_time + gpu_decomp_time + n_iters * gpu_fwd_time + gpu_back_time
        gpu_iter_alloc_mb = (n_iters * gpu_solve_memory) / (1024.0^2)

        speedup_iter_reuse = cpu_iter_time_reuse / gpu_iter_time
        speedup_iter_alloc = cpu_iter_time_alloc / gpu_iter_time

        println(@sprintf("%4d iters (reuse):  CPU %8.2f ms (%.2f MB)  |  GPU %8.2f ms (%.2f MB)  |  Speedup: %.2fx %s",
            n_iters, cpu_iter_time_reuse * 1000, cpu_iter_alloc_mb_reuse,
            gpu_iter_time * 1000, gpu_iter_alloc_mb, speedup_iter_reuse,
            speedup_iter_reuse > 1.0 ? "✓" : "⚠"))

        println(@sprintf("%4d iters (alloc):  CPU %8.2f ms (%.2f MB)  |  GPU %8.2f ms (%.2f MB)  |  Speedup: %.2fx %s",
            n_iters, cpu_iter_time_alloc * 1000, cpu_iter_alloc_mb_alloc,
            gpu_iter_time * 1000, gpu_iter_alloc_mb, speedup_iter_alloc,
            speedup_iter_alloc > 1.0 ? "✓" : "⚠"))
        println()
    end

    println("Notes:")
    println("  - 'reuse' = CPU uses ldiv! with pre-allocated buffer (Julia 1.12+)")
    println("  - 'alloc' = CPU uses \\ operator, allocating new vector each solve")
    println("  - GPU setup cost amortized across iterations")
    println("  - Both reuse patterns reflect realistic ADMM implementations")

catch e
    println("✗ cuDSS failed: $e")
    println("\nPossible reasons:")
    println("  • CUDSS.jl not installed: add it with `] add CUDSS`")
    println("  • Matrix not positive definite (try larger δ)")
    println("  • GPU memory insufficient")
end

println("\n" * "="^80)
