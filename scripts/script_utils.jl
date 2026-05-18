# Shared utilities for REPL and terminal scripts
using JLD2

"""
    save_diagnostic_matrix(ws_diag, problem_set, problem_name, tag, variant, rho, config)

Save diagnostic matrices (tilde_A, tilde_b, W_inv_mat) and FP residual history to JLD2 file.
Only works when full_diagnostics was enabled and acceleration is :none.

Files are saved to `scripts/experimental/saved_matrices/` with naming convention:
`{problem_set}_{problem_name}_{variant}_rho{rho}_{tag}.jld2`
"""
function save_diagnostic_matrix(ws_diag, problem_set, problem_name, tag, variant, rho, config)
    if ws_diag === nothing
        @warn "Cannot save matrix: ws_diag is nothing (full_diagnostics was false)"
        return
    end

    # Validate that acceleration is disabled (FP residuals only tracked for vanilla method)
    if config.acceleration != :none
        error("Cannot save FP residual history with acceleration enabled. " *
              "FP residuals are only tracked for vanilla (:none) acceleration. " *
              "Current acceleration: $(config.acceleration)")
    end

    # Create output directory
    output_dir = joinpath(@__DIR__, "experimental", "saved_matrices")
    mkpath(output_dir)

    # Format rho for filename (e.g., 0.1 -> rho0p1, 1.0 -> rho1p0)
    rho_str = replace(string(rho), "." => "p")

    # Create filename: problemset_problemname_variant_rhoXpY_tag.jld2
    filename = "$(problem_set)_$(problem_name)_$(variant)_rho$(rho_str)_$(tag).jld2"
    filepath = joinpath(output_dir, filename)

    # Extract FP residuals history in correct chronological order
    # Handle circular buffer wraparound
    col_idx = ws_diag.fp_history_col[]
    history_size = size(ws_diag.fp_residuals_history, 2)

    # Determine how many columns were actually filled
    # col_idx points to the NEXT column to write, so col_idx-1 columns have been written
    # (unless we've wrapped around)
    if col_idx == 1
        # Either no iterations or exactly wrapped back to 1
        # Check if the last column has non-zero data to distinguish
        if all(ws_diag.fp_residuals_history[:, history_size] .== 0.0)
            # No data at all
            fp_residuals_history = zeros(eltype(ws_diag.fp_residuals_history), size(ws_diag.fp_residuals_history, 1), 0)
        else
            # Fully wrapped, take all columns in order [1:end]
            fp_residuals_history = ws_diag.fp_residuals_history
        end
    elseif col_idx <= history_size
        # Check if we've wrapped by looking at the column after current write position
        if col_idx < history_size && all(ws_diag.fp_residuals_history[:, col_idx] .== 0.0)
            # Haven't wrapped yet, take first (col_idx-1) columns
            fp_residuals_history = ws_diag.fp_residuals_history[:, 1:col_idx-1]
        else
            # Wrapped around: oldest data is at col_idx, newest at col_idx-1
            # Order: [col_idx:end, 1:col_idx-1]
            fp_residuals_history = hcat(
                ws_diag.fp_residuals_history[:, col_idx:end],
                ws_diag.fp_residuals_history[:, 1:col_idx-1]
            )
        end
    end

    # Save matrices, FP history, and metadata
    K_F    = ws_diag.K_F
    F_mask = ws_diag.F_mask
    @save filepath tilde_A=ws_diag.tilde_A tilde_b=ws_diag.tilde_b W_inv_mat=ws_diag.W_inv_mat fp_residuals_history K_F F_mask problem_set problem_name tag variant rho

    @info "Matrix saved to: $filepath"
    @info "  tilde_A size: $(size(ws_diag.tilde_A))"
    @info "  tilde_b size: $(size(ws_diag.tilde_b))"
    @info "  W_inv_mat size: $(size(ws_diag.W_inv_mat))"
    @info "  fp_residuals_history size: $(size(fp_residuals_history))"
    if K_F !== nothing
        @info "  K_F size: $(size(K_F)), active constraints (m_F): $(count(F_mask))"
    end
end
