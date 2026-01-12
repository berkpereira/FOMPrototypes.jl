#!/usr/bin/env julia

# GPU Parallel ADMM Experiment
# Goal: Run multiple ADMM instances in parallel on GPU with different initial conditions
# Tier 1 experiment - proof of concept for parallel solver execution

import FOMPrototypes
using CUDA, CUDA.CUSPARSE, CUDSS
using SparseArrays, LinearAlgebra, Random, Printf

println("="^80)
println("GPU Parallel ADMM Experiment")
println("="^80)

# ============================================================================
# Configuration
# ============================================================================
N_INSTANCES = 8          # Number of parallel solver instances
N_ITERATIONS = 100       # ADMM iterations per instance
ρ = 0.1                  # ADMM penalty parameter
δ = 1e-10                # Regularization for Cholesky

# ============================================================================
# Check GPU availability
# ============================================================================
CUDA.functional() || error("CUDA GPU required for this experiment")
println("\n[✓] GPU: ", CUDA.name(CUDA.device()))

# ============================================================================
# Problem Setup (CPU)
# ============================================================================
println("\n" * "="^80)
println("Problem Setup")
println("="^80)

problem = FOMPrototypes.fetch_data("sslsq", "NYPA_Maragal_3_lasso")
P, A, b, c = problem.P, problem.A, problem.b, problem.c
m, n = problem.m, problem.n

println("[✓] Problem loaded: n=$n, m=$m")
println("    nnz(P)=$(nnz(P)), nnz(A)=$(nnz(A))")

# Build W = P + ρ*A'*A on CPU first
println("\n  Building W matrix (CPU)...")
setup_time = @elapsed begin
    W_cpu = P + ρ * (A' * A) + δ * I  # Add regularization directly
end
println("    W: $(size(W_cpu)), nnz=$(nnz(W_cpu)) (built in $(round(setup_time*1000, digits=2)) ms)")

# Transfer problem matrices to GPU and setup CUDSS solver
println("\n  Transferring matrices to GPU...")
gpu_transfer_time = @elapsed begin
    P_gpu = CuSparseMatrixCSR(P)
    A_gpu = CuSparseMatrixCSR(A)
    b_gpu = CuArray(b)
    c_gpu = CuArray(c)
    W_gpu = CuSparseMatrixCSR(W_cpu)
    CUDA.synchronize()
end
println("    [✓] Matrices on GPU ($(round(gpu_transfer_time*1000, digits=2)) ms)")

# Setup CUDSS solver for W (factorize once, solve many times)
println("\n  Setting up CUDSS solver...")
cudss_setup_time = @elapsed begin
    # Create solver for SPD matrix (we'll use the full matrix)
    F_gpu = cholesky(W_gpu, view='F')
    CUDA.synchronize()
end
println("    [✓] CUDSS factorization complete ($(round(cudss_setup_time*1000, digits=2)) ms)")

# ============================================================================
# Initialize Solver Instances (Batched for GPU parallelism)
# ============================================================================
println("\n" * "="^80)
println("Initializing $N_INSTANCES Solver Instances (Batched)")
println("="^80)

Random.seed!(42)

init_time = @elapsed begin
    # State variables as matrices (n × N_INSTANCES) for batched operations
    # Each column is one instance
    X = CuMatrix(randn(n, N_INSTANCES))  # Primal variables
    Y = CuMatrix(randn(m, N_INSTANCES))  # Dual variables

    # Scratch buffers (batched)
    temp_m = CuMatrix{Float64}(undef, m, N_INSTANCES)  # For A*x - b + y/ρ
    temp_n = CuMatrix{Float64}(undef, n, N_INSTANCES)  # For gradient computation
    Y_bar = CuMatrix{Float64}(undef, m, N_INSTANCES)   # Extragradient term
    AtY = CuMatrix{Float64}(undef, n, N_INSTANCES)     # For A'*y_bar (n-dimensional!)
    RHS = CuMatrix{Float64}(undef, n, N_INSTANCES)     # RHS for linear solve
    delta_X = CuMatrix{Float64}(undef, n, N_INSTANCES) # Solution from CUDSS

    CUDA.synchronize()
end

println("[✓] $N_INSTANCES instances initialized ($(round(init_time*1000, digits=2)) ms)")
println("    Memory per instance: $(round(2*(n+m)*8/1024^2, digits=2)) MB (x, y)")
println("    Total GPU memory: $(round(N_INSTANCES*2*(n+m)*8/1024^2, digits=2)) MB")

# ============================================================================
# Core ADMM Iteration Function (Batched GPU version)
# ============================================================================

"""
Batched ADMM iteration for all solver instances simultaneously.
All operations happen on GPU, including the linear solve via CUDSS.

Implements the iteration from src/solver.jl (onecol_method_operator):
  1. y-update: project(ρ*(A*X - b) + Y) to dual cone
  2. Extragradient: Y_bar = 2*Y_new - Y_old
  3. Gradient: RHS = P*X + A'*Y_bar + c
  4. x-update: X_new = X - W^{-1}*RHS (batched solve)

Arguments:
  X, Y           - State matrices (n × N_INSTANCES) and (m × N_INSTANCES)
  P_gpu, A_gpu   - Problem matrices on GPU
  b_gpu, c_gpu   - Problem vectors on GPU
  F_gpu          - CUDSS Cholesky factorization of W
  temp_m, Y_bar  - Scratch buffers (m × N_INSTANCES)
  temp_n, AtY    - Scratch buffers (n × N_INSTANCES)
  RHS, delta_X   - Scratch buffers for solve (n × N_INSTANCES)
  ρ              - ADMM penalty parameter
"""
function admm_step_batched!(
    X, Y,                              # State variables (in/out)
    P_gpu, A_gpu, b_gpu, c_gpu,       # Problem data on GPU
    F_gpu,                             # CUDSS Cholesky factorization
    temp_m, Y_bar, temp_n, AtY, RHS, delta_X,  # Scratch buffers
    ρ                                  # Penalty parameter
)
    # ========================================================================
    # Step 1: y-update (batched)
    # ========================================================================
    # Compute: temp_m = ρ*(A*X - b) + Y for all instances
    mul!(temp_m, A_gpu, X)              # temp_m = A*X (m × N_INSTANCES)
    temp_m .-= b_gpu                    # temp_m = A*X - b (broadcasts b)
    temp_m .*= ρ                        # temp_m = ρ*(A*X - b)
    temp_m .+= Y                        # temp_m = ρ*(A*X - b) + Y

    # Project to dual cone (nonnegative for lasso)
    Y_new = max.(temp_m, 0.0)          # Element-wise max

    # ========================================================================
    # Step 2: Extragradient (batched)
    # ========================================================================
    # Y_bar = 2*Y_new - Y_old (for θ=1.0)
    Y_bar .= 2.0 .* Y_new .- Y

    # ========================================================================
    # Step 3: Compute gradient terms (batched)
    # ========================================================================
    # RHS = P*X + A'*Y_bar + c
    mul!(temp_n, P_gpu, X)              # temp_n = P*X (n × N_INSTANCES)
    mul!(AtY, A_gpu', Y_bar)            # AtY = A'*Y_bar (n × N_INSTANCES)
    RHS .= temp_n .+ AtY .+ c_gpu       # RHS = P*X + A'*Y_bar + c

    # ========================================================================
    # Step 4: Batched linear solve on GPU via CUDSS
    # ========================================================================
    # Solve W * delta_X = RHS for all N_INSTANCES right-hand sides at once
    ldiv!(delta_X, F_gpu, RHS)

    # ========================================================================
    # Step 5: Update X and Y (batched)
    # ========================================================================
    X .-= delta_X                       # X_new = X - delta_X
    Y .= Y_new                          # Y_new = projected Y

    CUDA.synchronize()
end

# ============================================================================
# Run Batched ADMM Iterations (All instances solved together via CUDSS)
# ============================================================================

println("\n" * "="^80)
println("Running Batched ADMM Iterations (GPU-native)")
println("="^80)
println("Configuration:")
println("  N_INSTANCES  = $N_INSTANCES")
println("  N_ITERATIONS = $N_ITERATIONS")
println("  ρ (penalty)  = $ρ")
println("  Solver: CUDSS (batched RHS)")
println()

# Warmup: run one iteration to compile kernels
println("Warming up GPU kernels...")
warmup_time = @elapsed begin
    admm_step_batched!(
        X, Y,
        P_gpu, A_gpu, b_gpu, c_gpu, F_gpu,
        temp_m, Y_bar, temp_n, AtY, RHS, delta_X,
        ρ
    )
end
println("  [✓] Warmup complete ($(round(warmup_time*1000, digits=2)) ms)\n")

# Timed run: batched execution (all instances in one call)
println("Starting timed execution...")
println()

total_time = @elapsed begin
    for iter in 1:N_ITERATIONS
        # Single call processes all N_INSTANCES simultaneously
        admm_step_batched!(
            X, Y,
            P_gpu, A_gpu, b_gpu, c_gpu, F_gpu,
            temp_m, Y_bar, temp_n, AtY, RHS, delta_X,
            ρ
        )

        if iter % 10 == 0
            println("  Iteration $iter / $N_ITERATIONS")
        end
    end
end

per_iteration = total_time / N_ITERATIONS
per_instance_iter = total_time / (N_ITERATIONS * N_INSTANCES)

println()
println("="^80)
println("Timing Results (Batched GPU)")
println("="^80)
println(@sprintf("Total time:                    %.4f s", total_time))
println(@sprintf("Time per iteration (all %d):   %.4f ms", N_INSTANCES, per_iteration * 1000))
println(@sprintf("Time per instance-iteration:   %.4f ms", per_instance_iter * 1000))
println(@sprintf("Effective throughput:          %.2f instance-iters/sec",
                 N_ITERATIONS * N_INSTANCES / total_time))

# ============================================================================
# Validation: Check residuals
# ============================================================================
println("\n" * "="^80)
println("Validation (Final Residuals)")
println("="^80)

X_cpu = Array(X)
Y_cpu = Array(Y)

for i in 1:N_INSTANCES
    x_i = X_cpu[:, i]
    y_i = Y_cpu[:, i]

    # Primal residual: ||A*x - b||
    r_primal = norm(A * x_i - b)

    # Dual residual: ||P*x + A'*y + c||
    r_dual = norm(P * x_i + A' * y_i + c)

    println(@sprintf("  Instance %d: r_primal=%.2e, r_dual=%.2e",
                     i, r_primal, r_dual))
end

# ============================================================================
# Benchmark: Single-instance baseline (for throughput comparison)
# ============================================================================
println("\n" * "="^80)
println("Single-Instance Baseline (for throughput comparison)")
println("="^80)

# Create single-instance buffers for baseline measurement
println("Setting up single-instance baseline...")
Random.seed!(42)
x_single = CuArray(randn(n))
y_single = CuArray(randn(m))
temp_m_single = CuArray{Float64}(undef, m)
temp_n_single = CuArray{Float64}(undef, n)
Y_bar_single = CuArray{Float64}(undef, m)
AtY_single = CuArray{Float64}(undef, n)
RHS_single = CuArray{Float64}(undef, n)
delta_X_single = CuArray{Float64}(undef, n)
CUDA.synchronize()

# Single-instance ADMM step function (for baseline)
function admm_step_single!(
    x, y,
    P_gpu, A_gpu, b_gpu, c_gpu,
    F_gpu,
    temp_m, Y_bar, temp_n, AtY, RHS, delta_x,
    ρ
)
    # y-update
    mul!(temp_m, A_gpu, x)
    temp_m .-= b_gpu
    temp_m .*= ρ
    temp_m .+= y
    y_new = max.(temp_m, 0.0)

    # Extragradient
    Y_bar .= 2.0 .* y_new .- y

    # Gradient
    mul!(temp_n, P_gpu, x)
    mul!(AtY, A_gpu', Y_bar)
    RHS .= temp_n .+ AtY .+ c_gpu

    # Solve (single RHS)
    ldiv!(delta_x, F_gpu, RHS)

    # Update
    x .-= delta_x
    y .= y_new
    CUDA.synchronize()
end

# Warmup single-instance
admm_step_single!(
    x_single, y_single,
    P_gpu, A_gpu, b_gpu, c_gpu, F_gpu,
    temp_m_single, Y_bar_single, temp_n_single, AtY_single, RHS_single, delta_X_single,
    ρ
)

println("Running $N_ITERATIONS iterations × $N_INSTANCES instances (sequential single-instance)...")

sequential_time = @elapsed begin
    for i in 1:N_INSTANCES
        # Reset for each "instance"
        Random.seed!(42 + i)
        copyto!(x_single, randn(n))
        copyto!(y_single, randn(m))
        CUDA.synchronize()

        for iter in 1:N_ITERATIONS
            admm_step_single!(
                x_single, y_single,
                P_gpu, A_gpu, b_gpu, c_gpu, F_gpu,
                temp_m_single, Y_bar_single, temp_n_single, AtY_single, RHS_single, delta_X_single,
                ρ
            )
        end
    end
end

seq_per_instance_iter = sequential_time / (N_ITERATIONS * N_INSTANCES)

println()
println(@sprintf("Sequential time:               %.4f s", sequential_time))
println(@sprintf("Sequential per instance-iter:  %.4f ms", seq_per_instance_iter * 1000))

# ============================================================================
# Speedup Analysis
# ============================================================================
println("\n" * "="^80)
println("Speedup Analysis")
println("="^80)

speedup = sequential_time / total_time
efficiency = speedup / N_INSTANCES * 100

println(@sprintf("Batched time (all %d):  %.4f s", N_INSTANCES, total_time))
println(@sprintf("Sequential time:        %.4f s", sequential_time))
println(@sprintf("Speedup:                %.2fx", speedup))
println(@sprintf("Parallel efficiency:    %.1f%% (ideal: 100%% for %d instances)",
                 efficiency, N_INSTANCES))

if speedup >= 3.0
    println("\n✓ SUCCESS: Speedup ≥ 3x achieved! GPU batched solving working well.")
    println("  → CUDSS batched RHS is effective for parallel instances")
    println("  → Consider increasing N_INSTANCES to explore scaling limits")
elseif speedup >= 1.5
    println("\n⚠ MODERATE: Speedup between 1.5-3x.")
    println("  → Batching provides some benefit but not linear scaling")
    println("  → Consider: CUDSS overhead per solve may dominate for small problems")
else
    println("\n✗ LOW: Speedup < 1.5x.")
    println("  → Batching overhead may exceed parallelism benefit")
    println("  → Consider: Problem may be too small for GPU to show benefit")
end

println("\n" * "="^80)
println("Notes:")
println("  • All operations now run entirely on GPU (no CPU transfers)")
println("  • CUDSS ldiv! with matrix RHS solves all instances in one call")
println("  • Speedup depends on problem size vs GPU kernel launch overhead")
println("  • For small problems, batching many RHS amortizes launch costs")
println("  • For large problems, single-instance may already saturate GPU")
println("="^80)
