"""
This script is meant to try playing with linearisation matrices,
saved into JLD2 files in the saved_matrices directory.

These are Jacobian matrices of the operator T using some method
(eg ADMM) on the problem at hand. It depends on the method (ADMM vs PDHG vs ...)
as well as step size parameters, along with problem data P, A, ...
"""

using JLD2
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Plots
using Random

# Directory containing saved matrices
const MATRICES_DIR = joinpath(@__DIR__, "saved_matrices")

# TODO store also a fixed point residual vector so that
# I can construct a random subspace augmented with that
# direction, for comparison of that too

# Select which file to load by specifying its components
const PROBLEM_SET = "sslsq"
const PROBLEM_NAME = "NYPA_Maragal_1_lasso"
const VARIANT = :ADMM  # e.g., :ADMM, :PDHG
const RHO = 0.1        # e.g., 0.1, 1.0
const TAG = "optimal"  # "optimal" or "non-optimal"

# Construct the filename from components
rho_str = replace(string(RHO), "." => "p")
filename = "$(PROBLEM_SET)_$(PROBLEM_NAME)_$(VARIANT)_rho$(rho_str)_$(TAG).jld2"
filepath = joinpath(MATRICES_DIR, filename)

if !isfile(filepath)
    error("File not found: $filepath")
end

@info "Loading matrix data from: $filename"

# Load all data from the JLD2 file
data = load(filepath)

# Extract the matrices and metadata
tilde_A = data["tilde_A"]
tilde_b = data["tilde_b"]
W_inv_mat = data["W_inv_mat"]
problem_set = data["problem_set"]
problem_name = data["problem_name"]
tag = data["tag"]
variant = data["variant"]
rho = data["rho"]

# Display information about the loaded data
@info "Loaded data summary:"
@info "  Problem: $problem_set / $problem_name"
@info "  Variant: $variant, ρ = $rho"
@info "  Tag: $tag"
@info "  tilde_A size: $(size(tilde_A))"
@info "  tilde_b size: $(size(tilde_b))"
@info "  W_inv_mat size: $(size(W_inv_mat))"
@info "  tilde_A type: $(typeof(tilde_A))"

# Compute some basic properties
@info "Matrix properties:"
@info "  tilde_A sparsity: $(nnz(tilde_A)) / $(prod(size(tilde_A))) = $(nnz(tilde_A) / prod(size(tilde_A)))"
@info "  tilde_A norm: $(norm(tilde_A))"

# =============================================================================
# Linear System Setup
# =============================================================================
# We want to solve: (tilde_A - I) x = -tilde_b
# In the optimal case, a solution exists even if (tilde_A - I) is singular.

n = size(tilde_A, 1)
A_system = tilde_A - I  # The system matrix
b_system = -tilde_b     # The RHS

@info "Linear system setup:"
@info "  System size: $n × $n"
@info "  ||A_system||: $(norm(A_system))"
@info "  ||b_system||: $(norm(b_system))"

# =============================================================================
# Iterative Method Configuration
# =============================================================================
const MAX_ITERS = 500
const RELTOL = 1e-10

# Randomized subspace method parameters
const RAND_SUBSPACE_DIM = 20       # Default subspace dimension (s)
const RAND_REGULARIZATION = 1e-8   # Tikhonov regularization for Gram matrix
const RAND_RESTART_EVERY = 50      # Regenerate subspace every N LS solves

# =============================================================================
# GMRES with convergence logging
# =============================================================================
"""
Run GMRES and return the convergence history (residual norms per iteration).
"""
function run_gmres(A, b; maxiter=MAX_ITERS, reltol=RELTOL, restart=nothing)
    # Use full GMRES (no restart) by default, or specify restart
    restart_val = isnothing(restart) ? min(maxiter, size(A, 1)) : restart

    # Track residual history
    residual_history = Float64[]

    # Initial residual
    x0 = zeros(length(b))
    r0_norm = norm(b - A * x0)
    push!(residual_history, r0_norm)

    # Custom logging callback
    function log_residual(resnorm)
        push!(residual_history, resnorm)
    end

    # Run GMRES
    x, history = gmres(A, b;
        maxiter=maxiter,
        restart=restart_val,
        reltol=reltol,
        log=true,
        initially_zero=true
    )

    # Extract residual history from the ConvergenceHistory object
    residual_history = history.data[:resnorm]

    return x, residual_history
end

# =============================================================================
# Randomized Subspace Method
# =============================================================================
"""
Run randomized subspace method for solving Ax = b.

The method iteratively:
1. Generates a random subspace Ω (optionally augmented with current residual)
2. Computes V = A * Ω (s matrix-vector products)
3. Forms Gram matrix G = V'V + λI
4. Solves least-squares problems min ||Vz - r|| to update x

Returns (solution, residual_history, matvec_counts) where:
- residual_history[i] is the residual norm after the i-th LS solve
- matvec_counts[i] is the cumulative matvec count at that point

Parameters:
- subspace_dim (s): Dimension of random subspace
- regularization (λ): Tikhonov regularization for numerical stability
- restart_every: Regenerate subspace after this many LS solves
- augment_with_residual: If true, first column of Ω is the normalized residual
"""
function run_randomized_subspace(A, b;
    maxiter=MAX_ITERS,
    reltol=RELTOL,
    subspace_dim=RAND_SUBSPACE_DIM,
    regularization=RAND_REGULARIZATION,
    restart_every=RAND_RESTART_EVERY,
    augment_with_residual=false)

    n = size(A, 1)
    s = subspace_dim

    # Pre-allocate matrices
    Ω = zeros(n, s)
    V = zeros(n, s)
    G = zeros(s, s)

    # Initialize solution
    x = zeros(n)

    # Track convergence (residual norm after each LS solve, indexed by matvec count)
    residual_history = Float64[]
    matvec_counts = Int[]

    # Initial residual
    r = b - A * x  # Since x=0, this is just b
    r_norm = norm(r)
    push!(residual_history, r_norm)
    push!(matvec_counts, 0)

    total_matvecs = 0
    ls_solves_since_restart = 0
    subspace_regenerations = 0

    while total_matvecs < maxiter && r_norm > reltol
        # Generate random subspace
        randn!(Ω)

        # Augment with residual direction if requested
        if augment_with_residual && r_norm > 1e-14
            Ω[:, 1] .= r ./ r_norm
        end

        # Apply operator: V = A * Ω (s matvecs)
        mul!(V, A, Ω)
        total_matvecs += s
        subspace_regenerations += 1

        # Form Gram matrix G = V'V + λI
        mul!(G, V', V)
        for i in 1:s
            G[i, i] += regularization
        end

        # Factor G for repeated solves
        G_chol = try
            cholesky(Hermitian(G))
        catch e
            @warn "Cholesky factorization failed at matvec $total_matvecs" exception=e
            break
        end

        ls_solves_since_restart = 0

        # Inner loop: solve LS problems with current subspace
        while ls_solves_since_restart < restart_every && total_matvecs < maxiter && r_norm > reltol
            # Compute current residual
            r .= b
            mul!(r, A, x, -1.0, 1.0)  # r = b - A*x
            r_norm = norm(r)

            # Check convergence before LS solve
            if r_norm <= reltol
                push!(residual_history, r_norm)
                push!(matvec_counts, total_matvecs)
                break
            end

            # Project residual onto subspace: rhs = V' * r
            rhs = V' * r

            # Solve G * z = rhs
            z = G_chol \ rhs

            # Update solution: x = x + Ω * z (move in subspace direction)
            # Note: We use Ω*z not V*z because we want to move in the original
            # subspace, and V = A*Ω is the image under A
            mul!(x, Ω, z, 1.0, 1.0)  # x += Ω * z

            ls_solves_since_restart += 1

            # Compute new residual and log
            r .= b
            mul!(r, A, x, -1.0, 1.0)  # r = b - A*x
            r_norm = norm(r)

            push!(residual_history, r_norm)
            push!(matvec_counts, total_matvecs)
        end
    end

    return x, residual_history, matvec_counts, subspace_regenerations
end

# =============================================================================
# Run Methods and Collect Results
# =============================================================================
# Store results for comparison: Dict{method_name => (solution, residual_history)}
results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

# --- GMRES (full, no restart) ---
@info "Running GMRES (full)..."
x_gmres, hist_gmres = run_gmres(A_system, b_system)
results["GMRES"] = (x_gmres, hist_gmres)
@info "  Final residual: $(hist_gmres[end])"
@info "  Iterations: $(length(hist_gmres) - 1)"

# --- GMRES with restart (for comparison) ---
@info "Running GMRES(50) (restarted)..."
x_gmres50, hist_gmres50 = run_gmres(A_system, b_system; restart=50)
results["GMRES(50)"] = (x_gmres50, hist_gmres50)
@info "  Final residual: $(hist_gmres50[end])"
@info "  Iterations: $(length(hist_gmres50) - 1)"

# Store randomized method metadata separately (for enhanced summary)
rand_metadata = Dict{String, NamedTuple{(:matvec_counts, :regenerations), Tuple{Vector{Int}, Int}}}()

# --- Randomized (s=20, pure random) ---
@info "Running Randomized(s=20)..."
x_rand20, hist_rand20, mvcs_rand20, regens_rand20 = run_randomized_subspace(
    A_system, b_system; subspace_dim=20, augment_with_residual=false)
results["Rand(s=20)"] = (x_rand20, hist_rand20)
rand_metadata["Rand(s=20)"] = (matvec_counts=mvcs_rand20, regenerations=regens_rand20)
@info "  Final residual: $(hist_rand20[end])"
@info "  LS solves: $(length(hist_rand20) - 1), Subspace regenerations: $regens_rand20"

# --- Randomized (s=20, augmented with residual) ---
@info "Running Randomized(s=20,aug)..."
x_rand20_aug, hist_rand20_aug, mvcs_rand20_aug, regens_rand20_aug = run_randomized_subspace(
    A_system, b_system; subspace_dim=20, augment_with_residual=true)
results["Rand(s=20,aug)"] = (x_rand20_aug, hist_rand20_aug)
rand_metadata["Rand(s=20,aug)"] = (matvec_counts=mvcs_rand20_aug, regenerations=regens_rand20_aug)
@info "  Final residual: $(hist_rand20_aug[end])"
@info "  LS solves: $(length(hist_rand20_aug) - 1), Subspace regenerations: $regens_rand20_aug"

# --- Randomized (s=50, pure random) ---
@info "Running Randomized(s=50)..."
x_rand50, hist_rand50, mvcs_rand50, regens_rand50 = run_randomized_subspace(
    A_system, b_system; subspace_dim=50, augment_with_residual=false)
results["Rand(s=50)"] = (x_rand50, hist_rand50)
rand_metadata["Rand(s=50)"] = (matvec_counts=mvcs_rand50, regenerations=regens_rand50)
@info "  Final residual: $(hist_rand50[end])"
@info "  LS solves: $(length(hist_rand50) - 1), Subspace regenerations: $regens_rand50"

# --- Randomized (s=10, for comparison) ---
@info "Running Randomized(s=10)..."
x_rand10, hist_rand10, mvcs_rand10, regens_rand10 = run_randomized_subspace(
    A_system, b_system; subspace_dim=10, augment_with_residual=false)
results["Rand(s=10)"] = (x_rand10, hist_rand10)
rand_metadata["Rand(s=10)"] = (matvec_counts=mvcs_rand10, regenerations=regens_rand10)
@info "  Final residual: $(hist_rand10[end])"
@info "  LS solves: $(length(hist_rand10) - 1), Subspace regenerations: $regens_rand10"

# =============================================================================
# Plot Convergence
# =============================================================================
function plot_convergence(results; title_suffix="")
    plt = plot(
        title="Iterative Method Convergence" * title_suffix,
        xlabel="Iteration",
        ylabel="Residual Norm",
        yscale=:log10,
        legend=:topright,
        linewidth=2,
        size=(800, 500)
    )

    for (name, (_, hist)) in results
        plot!(plt, 0:length(hist)-1, hist, label=name)
    end

    # Add tolerance line
    hline!(plt, [RELTOL], linestyle=:dash, color=:gray, label="Tolerance")

    return plt
end

plt = plot_convergence(results; title_suffix=" ($PROBLEM_NAME, $VARIANT, ρ=$RHO)")
display(plt)

# =============================================================================
# Summary Table
# =============================================================================
@info "=== Summary ==="
for (name, (x, hist)) in results
    final_res = hist[end]
    iters = length(hist) - 1
    converged = final_res < RELTOL

    if haskey(rand_metadata, name)
        # Randomized method: show both LS solves and subspace regenerations
        meta = rand_metadata[name]
        total_matvecs = isempty(meta.matvec_counts) ? 0 : meta.matvec_counts[end]
        @info "  $name: $(iters) LS solves, $(meta.regenerations) regenerations, " *
              "$(total_matvecs) matvecs, residual=$(final_res), converged=$converged"
    else
        # GMRES: iterations = matvecs
        @info "  $name: $(iters) iters/matvecs, residual=$(final_res), converged=$converged"
    end
end
