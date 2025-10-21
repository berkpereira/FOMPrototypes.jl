using LinearAlgebra
using SparseArrays
using Clarabel

function vanilla_step!(
    ws::VanillaWorkspace,
    args::Dict{String, Any},
    record::Union{NamedTuple, Nothing},
    )
    # copy older iterate before iterating
    ws.vars.xy_prev .= ws.vars.xy

    onecol_method_operator!(ws, ws.vars.xy, ws.scratch.swap_vec, true)
    # swap contents of ws.vars.xy and ws.scratch.swap_vec
    custom_swap!(ws.vars.xy, ws.scratch.swap_vec, ws.scratch.temp_mn_vec1)
    # now ws.vars.xy contains newer iterate,  while
    # ws.scratch.swap_vec contains older one

    if !args["run-fast"]
        curr_xy_update .= ws.vars.xy - ws.scratch.swap_vec

        # record iteration data here
        push!(record.xy_step_norms, norm(curr_xy_update))
        push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
    end
end