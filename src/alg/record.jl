abstract type AbstractRecord end

struct NullRecord <: AbstractRecord
end

using Base: @kwdef

@kwdef struct IterationRecord <: AbstractRecord
    primal_obj_vals::Vector{Float64} = Float64[]
    pri_res_norms::Vector{Float64} = Float64[]
    dual_obj_vals::Vector{Float64} = Float64[]
    dual_res_norms::Vector{Float64} = Float64[]
    record_proj_flags::Vector{Vector{Bool}} = Vector{Vector{Bool}}[]
    record_soc_states::Vector{Vector{SOCAction}} = Vector{Vector{SOCAction}}[]
    update_mat_ranks::Vector{Float64} = Float64[]
    update_mat_singval_ratios::Vector{Float64} = Float64[]
    update_mat_iters::Vector{Int} = Int[]
    acc_step_iters::Vector{Int} = Int[]
    acc_attempt_iters::Vector{Int} = Int[]
    linesearch_iters::Vector{Int} = Int[]
    state_step_norms::Vector{Float64} = Float64[]
    state_step_char_norms::Vector{Float64} = Float64[]
    state_update_cosines::Vector{Float64} = Float64[]
    x_dist_to_sol::Vector{Float64} = Float64[]
    y_dist_to_sol::Vector{Float64} = Float64[]
    state_chardist::Vector{Float64} = Float64[]
    current_update_mat_col::Ref{Int} = Ref(1)
    fp_metric_ratios::Vector{Float64} = Float64[]

    # SOC normal direction angle tracking
    soc_normal_angles::Vector{Vector{Float64}} = Vector{Vector{Float64}}[]

    updates_matrix::Matrix{Float64} = Matrix{Float64}(undef, 0, 0)
    curr_state_update::Vector{Float64} = Vector{Float64}(undef, 0)
    prev_state_update::Vector{Float64} = Vector{Float64}(undef, 0)

    char_norm_func::Function
end

function IterationRecord(ws::AbstractWorkspace)
    
    char_norm_mat = [(ws.method.W - ws.p.P) ws.p.A'; ws.p.A I(ws.p.m) / ws.method.ρ]
    function char_norm(vector::AbstractArray{Float64})
        return sqrt(dot(vector, char_norm_mat * vector))
    end
    
    # fix some number of past updates to monitor here, 20 in this case
    return IterationRecord(
        updates_matrix = zeros(Float64, ws.p.n + ws.p.m, 20),
        curr_state_update = zeros(Float64, ws.p.n + ws.p.m),
        prev_state_update = zeros(Float64, ws.p.n + ws.p.m),
        char_norm_func = char_norm,
    )
end

# functions to record data during solver run
function push_update_to_record!(
    ws::VanillaWorkspace,
    record::IterationRecord,
    )
    @views record.curr_state_update .= ws.vars.state - ws.vars.state_prev

    push!(record.state_step_norms, norm(record.curr_state_update))
    push!(record.state_step_char_norms, record.char_norm_func(record.curr_state_update))
    insert_update_into_matrix!(record.updates_matrix, record.curr_state_update, record.current_update_mat_col)
end

function push_update_to_record!(
    ws::AndersonWorkspace,
    record::IterationRecord,
    vanilla_step::Bool,
    )
    if vanilla_step
        record.curr_state_update .= ws.vars.state - ws.vars.state_prev

        push!(record.state_step_norms, norm(record.curr_state_update))
        push!(record.state_step_char_norms, record.char_norm_func(record.curr_state_update))
        insert_update_into_matrix!(record.updates_matrix, record.curr_state_update, record.current_update_mat_col)
        return
    end

    # ws.accelerator.success only indicates there was no
    # numerical blowup in the COSMOAccelerators.accelerate!
    # internals
    if ws.accelerator.success && ws.control_flags.accepted_accel
        record.curr_state_update .= ws.vars.state - ws.scratch.extra.state_pre_overwrite
        push!(record.acc_step_iters, ws.k[])
        record.updates_matrix .= 0.0
        record.current_update_mat_col[] = 1

        push!(record.state_step_norms, norm(record.curr_state_update))
        push!(record.state_step_char_norms, record.char_norm_func(record.curr_state_update))
    else # prevent recording nonsense
        push!(record.state_step_norms, NaN)
        push!(record.state_step_char_norms, NaN)
    end
end

function push_update_to_record!(
    ws::KrylovWorkspace,
    record::IterationRecord,
    vanilla_step::Bool,
    )
    if vanilla_step
        @views record.curr_state_update .= ws.vars.state_q[:, 1] - ws.vars.state_prev

        push!(record.state_step_norms, norm(record.curr_state_update))
        push!(record.state_step_char_norms, record.char_norm_func(record.curr_state_update))
        insert_update_into_matrix!(record.updates_matrix, record.curr_state_update, record.current_update_mat_col)
        
        return
    end

    if ws.control_flags.accepted_accel
        @views record.curr_state_update .= ws.scratch.extra.accelerated_point - ws.vars.state_q[:, 1]
        push!(record.acc_step_iters, ws.k[])
        record.updates_matrix .= 0.0
        record.current_update_mat_col[] = 1

        push!(record.state_step_norms, norm(record.curr_state_update))
        push!(record.state_step_char_norms, record.char_norm_func(record.curr_state_update))
    else
        @views record.curr_state_update .= 0.0
        push!(record.state_step_norms, NaN)
        push!(record.state_step_char_norms, NaN)
    end
end

function push_update_to_record!(
    ws::AbstractWorkspace,
    record::NullRecord,
    vanilla_step::Bool=false,
    )
    return
end

function push_res_to_record!(
    ws::AbstractWorkspace,
    record::IterationRecord,
    )
    if ws.k[] > 0
        # NB: effective rank counts number of normalised singular values
        # larger than a specified small tolerance (eg 1e-8).
        # curr_effective_rank, curr_singval_ratio = effective_rank(record.updates_matrix, 1e-8)
        # push!(record.update_mat_ranks, curr_effective_rank)
        # push!(record.update_mat_singval_ratios, curr_singval_ratio)
        # push!(record.update_mat_iters, ws.k[])

        push!(record.primal_obj_vals, ws.res.obj_primal)
        push!(record.dual_obj_vals, ws.res.obj_dual)
        push!(record.pri_res_norms, ws.res.rp_abs)
        push!(record.dual_res_norms, ws.res.rd_abs)
    end
end

function push_res_to_record!(
    ws::AbstractWorkspace,
    record::NullRecord,
    x_sol=nothing,
    )
    return
end

function push_ref_dist_to_record!(
    ws::AbstractWorkspace,
    record::IterationRecord,
    view_state::AbstractArray{Float64},
    state_ref::Union{Nothing, Vector{Float64}},
    )
    if !isnothing(state_ref)
        @views ws.scratch.base.temp_mn_vec1[1:ws.p.n] .= view_state[1:ws.p.n] - state_ref[1:ws.p.n]
        @views ws.scratch.base.temp_mn_vec1[ws.p.n+1:end] .= view_state[ws.p.n+1:end] - (-state_ref[ws.p.n+1:end]) # negation due to often opposite sign out of reference solver
        curr_state_chardist = record.char_norm_func(ws.scratch.base.temp_mn_vec1)

        @views curr_x_dist = norm(ws.scratch.base.temp_mn_vec1[1:ws.p.n])
        @views curr_y_dist = norm(ws.scratch.base.temp_mn_vec1[ws.p.n+1:end])

        push!(record.x_dist_to_sol, curr_x_dist)
        push!(record.y_dist_to_sol, curr_y_dist)
        push!(record.state_chardist, curr_state_chardist)
    end
end

function push_ref_dist_to_record!(
    ws::AbstractWorkspace,
    record::NullRecord,
    view_state::AbstractArray{Float64},
    state_ref::Union{Nothing, Vector{Float64}}
    )
    return
end

function push_cosines_projs!(
    ws::AbstractWorkspace,
    record::IterationRecord,
    )
    
    # store cosine between last two iterate updates
    if ws.k[] >= 1
        state_prev_updates_cos = abs(dot(record.curr_state_update, record.prev_state_update) / (norm(record.curr_state_update) * norm(record.prev_state_update)))
        push!(record.state_update_cosines, state_prev_updates_cos)
    end
    record.prev_state_update .= record.curr_state_update

    # store active set flags
    # TODO note whether this aligns with new methods' notions
    # of flags used for core dynamics in hot loops versus interpretation
    # in terms of active set (might be the NEGATION of PrePPM's interpretations)
    push!(record.record_proj_flags, copy(ws.proj_state.nn_mask))
    push!(record.record_soc_states, copy(ws.proj_state.soc_states))
end

function push_cosines_projs!(
    ws::AbstractWorkspace,
    record::NullRecord
    )
    return
end

"""
Records angular changes in SOC projection normal directions.

For each SOC that was in `soc_interesting` state for 2+ consecutive iterations,
computes the angular distance (in radians) between the previous and current
normal vectors. Stores NaN when angle cannot be computed (state transition,
first occurrence, or no normals available).

Only called when full_diagnostics=true and workspace is KrylovWorkspace.
"""
function push_soc_normal_angles!(
    ws::KrylovWorkspace,
    record::IterationRecord,
    ws_diag::DiagnosticsWorkspace,
    )

    # Bail early if normal tracking not initialized
    if isnothing(ws_diag.soc_normals_curr) || isnothing(ws_diag.soc_normals_prev)
        return
    end

    num_socs = length(ws.proj_state.soc_states)
    angles = Vector{Float64}(undef, num_socs)
    fill!(angles, NaN)  # Default to NaN

    # Only compute angles for SOCs that are currently interesting
    for soc_idx in 1:num_socs
        # Check if this SOC is in interesting state
        if ws.proj_state.soc_states[soc_idx] != soc_interesting
            # State transition or not interesting -> store NaN
            continue
        end

        # Check if we have a previous normal (was interesting last iteration)
        # This happens when iteration > 0 and SOC was interesting before
        if ws.k[] == 0
            # First iteration -> no previous normal
            continue
        end

        # Both current and previous normals exist and SOC is interesting
        # Compute angular distance
        v_prev = ws_diag.soc_normals_prev[soc_idx]
        v_curr = ws_diag.soc_normals_curr[soc_idx]

        # Use dot product to compute angle: θ = acos(v_prev · v_curr)
        # Clamp to handle numerical errors
        cos_angle = dot(v_prev, v_curr)
        cos_angle_clamped = clamp(cos_angle, -1.0, 1.0)
        angles[soc_idx] = acos(cos_angle_clamped)
    end

    push!(record.soc_normal_angles, angles)
end

# Overload for non-Krylov workspaces (no-op)
function push_soc_normal_angles!(
    ws::AbstractWorkspace,
    record::IterationRecord,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    )
    # Do nothing for VanillaWorkspace and AndersonWorkspace
    return
end

# Overload for NullRecord (no-op)
function push_soc_normal_angles!(
    ws::AbstractWorkspace,
    record::NullRecord,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    )
    return
end

"""
Rotates SOC normal vector storage: current → previous.

Should be called AFTER recording angles and BEFORE computing next iteration's normals.
This ensures proper sequencing of normal vector history.
"""
function rotate_soc_normals!(ws_diag::Union{DiagnosticsWorkspace, Nothing})
    if isnothing(ws_diag) || isnothing(ws_diag.soc_normals_curr)
        return
    end

    # Swap references (efficient, no copying)
    ws_diag.soc_normals_prev, ws_diag.soc_normals_curr =
        ws_diag.soc_normals_curr, ws_diag.soc_normals_prev
end
