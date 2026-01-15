using LinearAlgebra
using SparseArrays
using Clarabel

function vanilla_step!(
    ws::VanillaWorkspace,
    record::AbstractRecord,
    config::SolverConfig,
    ws_diag::Union{DiagnosticsWorkspace, Nothing} = nothing,
    )
    # copy older iterate before iterating
    ws.vars.state_prev .= ws.vars.state

    onecol_method_operator!(ws, Val{ws.method.variant}(), ws.vars.state, ws.scratch.extra.swap_vec, true, true)

    # Compute fixed-point metric BEFORE swap (when !run_fast)
    # At this point:
    # - ws.vars.state contains old state
    # - ws.scratch.extra.swap_vec contains FOM(old state)
    if !config.run_fast && ws.k[] > 0
        # Compute fp_residual = FOM(state) - state
        # Use temp_mn_vec1 as storage for fp_residual (it gets overwritten by custom_swap! anyway)
        ws.scratch.base.temp_mn_vec1 .= ws.scratch.extra.swap_vec .- ws.vars.state

        # Save FP residual to diagnostics history if enabled
        if !isnothing(ws_diag)
            col_idx = ws_diag.fp_history_col[]
            ws_diag.fp_residuals_history[:, col_idx] .= ws.scratch.base.temp_mn_vec1
            ws_diag.fp_history_col[] = mod1(col_idx + 1, FP_HISTORY_SIZE)
        end

        # Compute metric
        fp_metric = compute_fp_metric!(ws, ws.scratch.base.temp_mn_vec1)

        # Print ratio (skip first iteration since prev_fp_metric = Inf)
        if ws.prev_fp_metric[] < Inf
            metric_ratio = fp_metric / ws.prev_fp_metric[]
            println("Vanilla iter $(ws.k[]): fp metric ratio: $(metric_ratio), metric: $(fp_metric)")
        end

        # Update for next iteration
        ws.prev_fp_metric[] = fp_metric
    end

    # swap contents of ws.vars.state and ws.scratch.extra.swap_vec
    custom_swap!(ws.vars.state, ws.scratch.extra.swap_vec, ws.scratch.base.temp_mn_vec1)
    # now ws.vars.state contains newer iterate, while
    # ws.scratch.extra.swap_vec contains older one

    push_update_to_record!(ws, record)

    ws.res.residual_check_count[] += 1
    ws.method.rho_update_count[] += 1
end