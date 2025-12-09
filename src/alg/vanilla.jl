using LinearAlgebra
using SparseArrays
using Clarabel

function vanilla_step!(
    ws::VanillaWorkspace,
    config::SolverConfig,
    record::AbstractRecord,
    )
    # copy older iterate before iterating
    ws.vars.state_prev .= ws.vars.state

    onecol_method_operator!(ws, Val{ws.method.variant}(), ws.vars.state, ws.scratch.extra.swap_vec, true, true)
    # swap contents of ws.vars.state and ws.scratch.extra.swap_vec
    custom_swap!(ws.vars.state, ws.scratch.extra.swap_vec, ws.scratch.base.temp_mn_vec1)
    # now ws.vars.state contains newer iterate, while
    # ws.scratch.extra.swap_vec contains older one

    # Attempt linesearch (modifies ws.vars.state in-place if successful)
    linesearch_success = adaptive_linesearch!(
        ws, config, record,
        ws.vars.state_prev,
        ws.vars.state,
        ws.scratch.base.temp_mn_vec2
    )
    if linesearch_success && record isa IterationRecord
        push!(record.linesearch_iters, ws.k[])
    end

    push_update_to_record!(ws, record)

    ws.res.residual_check_count[] += 1
    ws.method.rho_update_count[] += 1
end