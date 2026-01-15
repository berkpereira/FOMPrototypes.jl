"""
Linear Solver Comparison for Linearization Matrices

This script compares iterative linear system solvers on the system:
    (tilde_A - I) x = -tilde_b

where tilde_A is the Jacobian matrix of the operator T using some method
(e.g., ADMM) on the problem at hand.

Solvers compared:
- GMRES (full, no restart)
- GMRES with restart (e.g., GMRES(10))
- Randomized subspace methods (various configurations)

This script is fully standalone and does NOT require fp_residuals_history.
"""

using JLD2
using LinearAlgebra
using SparseArrays
using Statistics
using IterativeSolvers
using Plots
using Random

# Include shared utilities
include("utils.jl")

# Set up plotting
setup_plotting()

# =============================================================================
# Configuration
# =============================================================================
# Import defaults from utils.jl (can override locally if needed)
const MAX_ITERS = DEFAULT_MAX_ITERS
const ABSTOL = DEFAULT_ABSTOL
const RAND_SUBSPACE_DIM = DEFAULT_RAND_SUBSPACE_DIM
const RAND_REGULARIZATION = DEFAULT_RAND_REGULARIZATION
const RAND_REGEN_EVERY = DEFAULT_RAND_REGEN_EVERY

# =============================================================================
# File Selection
# =============================================================================
# Select which file to load by specifying its components
const MATRICES_DIR = @__DIR__
const PROBLEM_SET = "mpc"
const PROBLEM_NAME = "pendulum_1"
const VARIANT = :ADMM  # e.g., :ADMM, :PDHG
const RHO = 100.0       # e.g., 0.1, 1.0
const TAG = "non-optimal"  # "optimal" or "non-optimal"

# =============================================================================
# Load Data
# =============================================================================
filename, filepath = construct_filepath(MATRICES_DIR, PROBLEM_SET, PROBLEM_NAME, VARIANT, RHO, TAG)
@info "Loading matrix data from: $filename"

data = load_matrix_data(filepath)
display_matrix_info(data)

# Extract the matrices
tilde_A = data["tilde_A"]
tilde_b = data["tilde_b"]

display_matrix_properties(tilde_A)

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
# GMRES with convergence logging
# =============================================================================
"""
Run GMRES and return the convergence history (residual norms per iteration).
Note: Uses absolute tolerance for consistency with randomized methods.
"""
function run_gmres(A, b; maxiter=MAX_ITERS, abstol=ABSTOL, restart=nothing)
    # Use full GMRES (no restart) by default, or specify restart
    restart_val = isnothing(restart) ? min(maxiter, size(A, 1)) : restart

    # Track residual history
    residual_history = Float64[]

    # Initial residual
    # NB we're not even initialising this in the better
    # way which would be to derive a better guess from a
    # current optimisation iterate.
    x0 = zeros(length(b))
    r0_norm = norm(b - A * x0)
    push!(residual_history, r0_norm)

    # Custom logging callback
    function log_residual(resnorm)
        push!(residual_history, resnorm)
    end

    # Run GMRES with absolute tolerance
    x, history = gmres(A, b;
        maxiter=maxiter,
        restart=restart_val,
        abstol=abstol,
        reltol=0.0,  # Disable relative tolerance, use only absolute
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

Returns (solution, residual_history, matvec_counts, subspace_regenerations) where:
- residual_history[i] is the residual norm after the i-th LS solve
- matvec_counts[i] is the cumulative matvec count at that point
- subspace_regenerations is the total number of times the subspace was regenerated
- NOTE: iterations are counted as LS solves, NOT matvecs

Parameters:
- maxiter: Maximum number of LS solves (NOT matvecs)
- abstol: Absolute residual tolerance for convergence
- subspace_dim (s): Dimension of random subspace
- regularization (λ): Tikhonov regularization for numerical stability
- restart_every: Regenerate subspace after this many LS solves
- augment_with_residual: If true, first column of Ω is the normalized residual
"""
function run_randomized_subspace(A, b;
    maxiter=MAX_ITERS,
    abstol=ABSTOL,
    subspace_dim=RAND_SUBSPACE_DIM,
    regularization=RAND_REGULARIZATION,
    restart_every=RAND_REGEN_EVERY,
    augment_with_residual=false)

    n = size(A, 1)
    s = subspace_dim

    # Pre-allocate matrices
    Ω = zeros(n, s)
    V = zeros(n, s)
    G = zeros(s, s)

    # Initialize solution
    x = zeros(n)

    # Track convergence (residual norm after each LS solve)
    residual_history = Float64[]
    matvec_counts = Int[]

    # Initial residual
    r = b - A * x  # Since x=0, this is just b
    r_norm = norm(r)
    push!(residual_history, r_norm)
    push!(matvec_counts, 0)

    total_matvecs = 0
    total_ls_solves = 0
    ls_solves_since_restart = 0
    subspace_regenerations = 0

    while total_ls_solves < maxiter && r_norm > abstol
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
        while ls_solves_since_restart < restart_every && total_ls_solves < maxiter && r_norm > abstol
            # Compute current residual
            r .= b
            mul!(r, A, x, -1.0, 1.0)  # r = b - A*x
            r_norm = norm(r)

            # Check convergence before LS solve
            if r_norm <= abstol
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
            total_ls_solves += 1

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
@info "Running GMRES(10) (restarted)..."
x_gmres10, hist_gmres10 = run_gmres(A_system, b_system; restart=10)
results["GMRES(10)"] = (x_gmres10, hist_gmres10)
@info "  Final residual: $(hist_gmres10[end])"
@info "  Iterations: $(length(hist_gmres10) - 1)"

# Store randomized method metadata separately (for enhanced summary)
rand_metadata = Dict{String, NamedTuple{(:matvec_counts, :regenerations), Tuple{Vector{Int}, Int}}}()

# --- Randomized (s=20, pure random) ---
@info "Running Randomized(s=20)..."
x_rand20, hist_rand20, mvcs_rand20, regens_rand20 = run_randomized_subspace(
    A_system, b_system; subspace_dim=20, augment_with_residual=false, restart_every=1)
results["Rand(s=20)"] = (x_rand20, hist_rand20)
rand_metadata["Rand(s=20)"] = (matvec_counts=mvcs_rand20, regenerations=regens_rand20)
@info "  Final residual: $(hist_rand20[end])"
@info "  Iterations (= LS solves): $(length(hist_rand20) - 1), Subspace regenerations: $regens_rand20"

# --- Randomized (s=20, augmented with residual) ---
@info "Running Randomized(s=20,aug)..."
x_rand20_aug, hist_rand20_aug, mvcs_rand20_aug, regens_rand20_aug = run_randomized_subspace(
    A_system, b_system; subspace_dim=20, augment_with_residual=true, restart_every=1)
results["Rand(s=20,aug)"] = (x_rand20_aug, hist_rand20_aug)
rand_metadata["Rand(s=20,aug)"] = (matvec_counts=mvcs_rand20_aug, regenerations=regens_rand20_aug)
@info "  Final residual: $(hist_rand20_aug[end])"
@info "  Iterations (= LS solves): $(length(hist_rand20_aug) - 1), Subspace regenerations: $regens_rand20_aug"

# --- Randomized (s=50, pure random) ---
@info "Running Randomized(s=50)..."
x_rand50, hist_rand50, mvcs_rand50, regens_rand50 = run_randomized_subspace(
    A_system, b_system; subspace_dim=50, augment_with_residual=false, restart_every=1)
results["Rand(s=50)"] = (x_rand50, hist_rand50)
rand_metadata["Rand(s=50)"] = (matvec_counts=mvcs_rand50, regenerations=regens_rand50)
@info "  Final residual: $(hist_rand50[end])"
@info "  Iterations (= LS solves): $(length(hist_rand50) - 1), Subspace regenerations: $regens_rand50"

# --- Randomized (s=100, for comparison) ---
@info "Running Randomized(s=100)..."
x_rand100, hist_rand100, mvcs_rand100, regens_rand100 = run_randomized_subspace(
    A_system, b_system; subspace_dim=100, augment_with_residual=false, restart_every=1)
results["Rand(s=100)"] = (x_rand100, hist_rand100)
rand_metadata["Rand(s=100)"] = (matvec_counts=mvcs_rand100, regenerations=regens_rand100)
@info "  Final residual: $(hist_rand100[end])"
@info "  Iterations (= LS solves): $(length(hist_rand100) - 1), Subspace regenerations: $regens_rand100"

# =============================================================================
# Plot Convergence
# =============================================================================
function plot_convergence(results; title_suffix="")
    plt = plot(
        title="Iterative Method Convergence" * title_suffix,
        xlabel="Iteration (GMRES) / LS Solves (Rand)",
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
    hline!(plt, [ABSTOL], linestyle=:dash, color=:gray, label="Tolerance")

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
    initial_res = hist[1]
    iters = length(hist) - 1
    # Check absolute convergence (consistent across all methods)
    converged = final_res < ABSTOL

    if haskey(rand_metadata, name)
        # Randomized method: iterations = LS solves
        meta = rand_metadata[name]
        total_matvecs = isempty(meta.matvec_counts) ? 0 : meta.matvec_counts[end]
        @info "  $name: $(iters) iterations (= LS solves), $(meta.regenerations) regenerations, " *
              "$(total_matvecs) matvecs, residual=$(final_res), converged=$converged"
    else
        # GMRES: iterations = matvecs
        @info "  $name: $(iters) iters/matvecs, residual=$(final_res), converged=$converged"
    end
end
