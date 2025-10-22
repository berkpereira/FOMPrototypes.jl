using LinearAlgebra
using SparseArrays
using Clarabel

function vanilla_step!(
    ws::VanillaWorkspace,
    record::AbstractRecord,
    )
    # copy older iterate before iterating
    ws.vars.state_prev .= ws.vars.state

    onecol_method_operator!(ws, ws.vars.state, ws.scratch.swap_vec, true)
    # swap contents of ws.vars.state and ws.scratch.swap_vec
    custom_swap!(ws.vars.state, ws.scratch.swap_vec, ws.scratch.temp_mn_vec1)
    # now ws.vars.state contains newer iterate,  while
    # ws.scratch.swap_vec contains older one

    push_update_to_record!(ws, record)
end