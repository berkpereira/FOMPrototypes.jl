"""
Shared utilities for linearization analysis scripts.

This file provides common functions for:
- File I/O and data loading
- Configuration constants
- Plotting setup
- Matrix property display

Both `linear_solver_comparison.jl` and `spectral_residual_analysis.jl` include this file.
"""

using JLD2
using LinearAlgebra
using Plots

# =============================================================================
# Default Configuration Constants
# =============================================================================
# These can be overridden in individual scripts if needed

# Iterative solver defaults
const DEFAULT_MAX_ITERS = 300           # Maximum iterations for solvers
const DEFAULT_ABSTOL = 1e-9             # Absolute residual tolerance

# Randomized subspace method defaults
const DEFAULT_RAND_SUBSPACE_DIM = 20    # Default subspace dimension (s)
const DEFAULT_RAND_REGULARIZATION = 1e-8 # Tikhonov regularization
const DEFAULT_RAND_REGEN_EVERY = 1      # Regenerate subspace frequency

# Spectral analysis defaults
const DEFAULT_UNIT_TOL = 0.01           # Tolerance for "near-unit" eigenvalues
const DEFAULT_COMPLEX_TOL = 1e-6        # Tolerance for complex eigenvalues
const DEFAULT_EIGEN_MAX_SIZE = 5000     # Max matrix size for full eigen
const DEFAULT_EIGEN_WARN_THRESHOLD = 2000 # Warn before expensive eigen

# Plotting defaults
const DEFAULT_PLOT_DPI = 800            # High DPI for publication quality

# =============================================================================
# File I/O Helpers
# =============================================================================

"""
    construct_filepath(matrices_dir, problem_set, problem_name, variant, rho, tag)

Construct the filename and full filepath from problem specification components.

Returns a tuple `(filename, filepath)`.

# Arguments
- `matrices_dir`: Directory containing the .jld2 files
- `problem_set`: Problem set name (e.g., "mpc", "sslsq")
- `problem_name`: Problem name (e.g., "pendulum_1", "NYPA_Maragal_1_lasso")
- `variant`: Solver variant symbol (e.g., :ADMM, :PDHG)
- `rho`: Rho parameter value (e.g., 0.1, 100.0)
- `tag`: Tag string ("optimal" or "non-optimal")
"""
function construct_filepath(matrices_dir, problem_set, problem_name, variant, rho, tag)
    rho_str = replace(string(rho), "." => "p")
    filename = "$(problem_set)_$(problem_name)_$(variant)_rho$(rho_str)_$(tag).jld2"
    filepath = joinpath(matrices_dir, filename)
    return filename, filepath
end

"""
    load_matrix_data(filepath)

Load and validate matrix data from a JLD2 file.

Returns a dictionary with all loaded data.

# Required fields in file
- `tilde_A`: Linearization/Jacobian matrix
- `tilde_b`: Corresponding RHS vector
- `W_inv_mat`: Inverse weight matrix
- `problem_set`, `problem_name`, `tag`, `variant`, `rho`: Metadata

# Optional fields
- `fp_residuals_history`: Fixed-point residual history (may not be present in older files)
"""
function load_matrix_data(filepath)
    if !isfile(filepath)
        error("File not found: $filepath")
    end

    data = load(filepath)

    # Validate required fields
    required_fields = ["tilde_A", "tilde_b", "W_inv_mat", "problem_set",
                       "problem_name", "tag", "variant", "rho"]
    for field in required_fields
        if !haskey(data, field)
            error("Required field '$field' missing from file: $filepath")
        end
    end

    return data
end

# =============================================================================
# Display Helpers
# =============================================================================

"""
    display_matrix_info(data)

Display summary information about loaded matrix data.
"""
function display_matrix_info(data)
    @info "Loaded data summary:"
    @info "  Problem: $(data["problem_set"]) / $(data["problem_name"])"
    @info "  Variant: $(data["variant"]), ρ = $(data["rho"])"
    @info "  Tag: $(data["tag"])"
    @info "  tilde_A size: $(size(data["tilde_A"]))"
    @info "  tilde_b size: $(size(data["tilde_b"]))"
    @info "  W_inv_mat size: $(size(data["W_inv_mat"]))"

    fp_residuals_history = get(data, "fp_residuals_history", nothing)
    if !isnothing(fp_residuals_history)
        @info "  fp_residuals_history size: $(size(fp_residuals_history))"
    end

    @info "  tilde_A type: $(typeof(data["tilde_A"]))"
end

"""
    display_matrix_properties(tilde_A; sparsity_tol=1e-12)

Compute and display basic matrix properties (sparsity, norm).
"""
function display_matrix_properties(tilde_A; sparsity_tol=1e-12)
    num_nonzeros = count(x -> abs(x) > sparsity_tol, tilde_A)
    total_entries = prod(size(tilde_A))

    @info "Matrix properties:"
    @info "  tilde_A sparsity (|entry| > $sparsity_tol): $num_nonzeros / $total_entries = $(num_nonzeros / total_entries)"
    @info "  tilde_A norm: $(norm(tilde_A))"
end

# =============================================================================
# Plotting Setup
# =============================================================================

"""
    setup_plotting(; dpi=DEFAULT_PLOT_DPI)

Initialize plotting backend with high DPI for publication-quality plots.
"""
function setup_plotting(; dpi=DEFAULT_PLOT_DPI)
    gr(dpi=dpi)
end
