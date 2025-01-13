using LinearAlgebra
using Plots
using Random
using Printf
include("types.jl")
include("utils.jl")
include("residuals.jl")
include("acceleration.jl")
include("printing.jl")

# Compute v iterate which consolidates the y and s iterates.
function compute_v(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
    x::AbstractVector{Float64}, y_prev::AbstractVector{Float64}, ρ::Float64)
    # With the vanilla method, $v_k = b - A x_k - y_{k-1} / ρ$
    v = b - A * x - y_prev / ρ
    return v
end

# This function recovers the multiplier variable y from the "consolidated" v
# iterate.
function recover_y(v::AbstractVector{Float64}, ρ::Float64,
    K::Vector{Clarabel.SupportedCone})
    y = ρ * (project_to_K(v, K) - v)
    return y
end

# This function recovers the slack variable s from the "consolidated" v iterate.
function recover_s(v::AbstractVector{Float64}, K::Vector{Clarabel.SupportedCone})
    s = project_to_K(v, K)
    return s
end

function iter_x!(x::AbstractVector{Float64},
    pre_matrix::Diagonal{Float64},
    P::AbstractMatrix{Float64},
    c::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    y_prev::AbstractVector{Float64})

    # NOTE: pre_matrix is stored inverted, hence just multiplication here.
    step = - pre_matrix * (P * x + c + A' * (2 * y - y_prev))
    
    x .+= step

    return step
end

function iter_s!(s::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    b::AbstractVector{Float64},
    y::AbstractVector{Float64},
    K::Vector{Clarabel.SupportedCone},
    ρ::Float64)

    s_old = copy(s)

    # Take step.
    s .= project_to_K(b - A * x - y / ρ, K)
    
    return s - s_old
end

function iter_y!(y::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    s::AbstractVector{Float64},
    b::AbstractVector{Float64},
    ρ::Float64)
    step = y_update(A, x, s, b, ρ)
    y .+= step
    return step
end

# NOTE: non-mutating iteration functions mostly (exclusively?) for debugging.
function x_update(x::AbstractVector{Float64},
    pre_matrix::Diagonal{Float64},
    P::AbstractMatrix{Float64},
    c::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    y_prev::AbstractVector{Float64})

    # NOTE: pre_matrix is stored inverted, hence just multiplication here.
    step = - pre_matrix * (P * x + c + A' * (2 * y - y_prev))

    return step
end

function iter_v(A::AbstractMatrix{Float64},
    b::AbstractVector{Float64},
    v::AbstractVector{Float64},
    x::AbstractVector{Float64},
    K::Vector{Clarabel.SupportedCone})
    v_new = b - A * x + v - project_to_K(v, K)

    return v_new
end

function y_update(A::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    s::AbstractVector{Float64},
    b::AbstractVector{Float64},
    ρ::Float64)
    # The step is y_{k+1} = y_k + ρ (A x_{k+1} + s_{k+1} - b)
    step = ρ * (A * x + s - b)
    return step
end

# Crude check for acceptance of an acceleration candidate.
function accept_acc_candidate(ws::Workspace,
    acc_pri_res::AbstractVector{Float64},
    acc_dual_res::AbstractVector{Float64},
    curr_pri_res::AbstractVector{Float64},
    curr_dual_res::AbstractVector{Float64},
    residual_norm::Real)
    # Compute residual norms.
    acc_pri_res_norm = norm(acc_pri_res, residual_norm)
    acc_dual_res_norm = norm(acc_dual_res, residual_norm)
    curr_pri_res_norm = norm(curr_pri_res, residual_norm)
    curr_dual_res_norm = norm(curr_dual_res, residual_norm)

    # Accept candidate if reduced residuals.
    return acc_pri_res_norm < curr_pri_res_norm && acc_dual_res_norm < curr_dual_res_norm
end

"""
When solver is run with (prototype) adaptive restart mechanism, this function 
determines whether it is time for a restart or not, based on the cumulative
angle data for each iterate sequence.
"""
function restart_trigger(restart_period::Union{Real, Symbol}, k::Integer,
    cumsum_angles::Float64...)
    if restart_period == Inf
        return false
    elseif restart_period == :adaptive
        return all(v >= 2π for v in cumsum_angles)
        # return any(v >= 2π for v in cumsum_angles)
    else
        return k > 0 && k % restart_period == 0
    end
end

"""
Run the optimiser for the initial inputs and solver options given.

    restart_period: Integer or Symbol
        If an integer, the number of iterations between restarts. The Only
        valid symbolic input is :adaptive, with obvious meaning.
"""
function optimise!(ws::Workspace, max_iter::Integer, print_modulo::Integer,
    restart_period::Union{Real, Symbol} = Inf, residual_norm::Real = Inf,
    acceleration::Bool = false,
    acceleration_memory::Integer = 20,
    x_sol::AbstractVector{Float64} = nothing,
    s_sol::AbstractVector{Float64} = nothing,
    y_sol::AbstractVector{Float64} = nothing,
    explicit_affine_operator::Bool = false,
    spectrum_plot_period::Int = 17,)

    # We maintain a matrix whose columns are formed by the past
    # acceleration_memory iterate updates, normalised to unit l2 norm.
    # Init this matrix.
    no_stored_updates_cond = 20
    updates_matrix = zeros(ws.p.n + ws.p.m, no_stored_updates_cond)
    current_update_mat_col = Ref(1) # This index is updated in circular fashion.
    # This variable determines how often we compute and store this rank.
    # updates_cond_period = Int(floor((acceleration_memory + 1) / 4))
    updates_cond_period = 1

    # Compute the (fixed) matrix that premultiplies the x step.
    # NOTE: I assume for now that A_gram is already in the cache.
    # It is suboptimal to be storing all these matrices explicitly.
    # It would be much better to specify how the operator computes products
    # against vectors instead. However, for now, we will leave it like this.
    ws.cache[:W_inv] = pre_x_step_matrix(ws.variant, ws.p.P, ws.cache[:A_gram], ws.τ, ws.ρ, ws.p.n)

    # Recall that W = M_1 + P + ρ A^T A.
    # TODO: remove this once no longer needed.
    # This is just for theoretically minded checks, when we have access to the
    # true problem solution.
    ws.cache[:seminorm_mat] = [Diagonal(1.0 ./ ws.cache[:W_inv].diag) - ws.p.P ws.p.A'; ws.p.A I(ws.p.m) / ws.ρ]
    
    # Compute fixed bits of tilde_{A} operator.
    # The affine operator is only constructed explicitly when the user so
    # requests, since it is computationally expensive and wasteful.
    if explicit_affine_operator
        ws.cache[:tlhs] = I(ws.p.n) - ws.cache[:W_inv] * (ws.p.P + ws.ρ * ws.cache[:A_gram])
        ws.cache[:blhs] = -A * ws.cache[:tlhs]
        ws.cache[:trhs_pre] = ws.ρ * ws.cache[:W_inv] * A'
    end

    # If acceleration, initialise memory for storing Krylov basis vectors, as
    # well as Hessenberg matrix H arising during the Gram-Schmidt process.
    if acceleration
        ws.cache[:krylov_basis] = zeros(Float64, ws.p.n + ws.p.m, acceleration_memory)
        ws.cache[:H] = init_upper_hessenberg(acceleration_memory)
    end

    # Initialise "artificial" y_{-1} to make first x update well-defined
    # and correct.
    ws.vars.y_prev .= ws.vars.y - y_update(ws.p.A, ws.vars.x, ws.vars.s, ws.p.b, ws.ρ)

    # Initialise running sums of step angles.
    # x_angle_sum, s_angle_sum, y_angle_sum = 0.0, 0.0, 0.0
    
    # for restarts, keep average iterates
    if restart_period != Inf
        x_avg = copy(ws.vars.x)
        s_avg = copy(ws.vars.s)
        y_avg = copy(ws.vars.y)
    end

    pri_res = zeros(Float64, ws.p.m)
    dual_res = zeros(Float64, ws.p.n)
    dual_res_temp = zeros(Float64, ws.p.n)
    acc_pri_res = zeros(Float64, ws.p.m)
    acc_dual_res = zeros(Float64, ws.p.n)
    acc_dual_res_temp = zeros(Float64, ws.p.n)
    j_restart = 0

    # Data containers for metrics (if return_run_data == true).
    # x_step_angles = Float64[]
    # s_step_angles = Float64[]
    # y_step_angles = Float64[]
    # concat_step_angles = Float64[]
    # normalised_concat_step_angles = Float64[]
    primal_obj_vals = Float64[]
    dual_obj_vals = Float64[]
    pri_res_norms = Float64[]
    dual_res_norms = Float64[]
    enforced_set_flags = Vector{Vector{Bool}}()
    update_mat_ranks = Float64[]
    update_mat_singval_ratios = Float64[]
    update_mat_iters = Int[]
    acc_step_iters = Int[]
    xv_step_norms = Float64[]
    # If real solution provided.
    if !isnothing(x_sol)
        x_dist_to_sol = Float64[]
        s_dist_to_sol = Float64[]
        y_dist_to_sol = Float64[]
        v_dist_to_sol = Float64[]
        xy_semidist = Float64[]
    end


    # Initialise other variables to be used during algorithm.
    # x_step = zeros(Float64, ws.p.n)
    # s_step = zeros(Float64, ws.p.m)
    # y_step = zeros(Float64, ws.p.m)
    # concat_step = zeros(Float64, ws.p.n + 2 * ws.p.m)
    # normalised_concat_step = zeros(Float64, ws.p.n + 2 * ws.p.m)
    # x_step_prev = zeros(Float64, ws.p.n)
    # s_step_prev = zeros(Float64, ws.p.m)
    # y_step_prev = zeros(Float64, ws.p.m)
    # concat_step_prev = zeros(Float64, ws.p.n + 2 * ws.p.m)
    # normalised_concat_step_prev = zeros(Float64, ws.p.n + 2 * ws.p.m)

    # This flag helps us keep track of whether we should forgo the notion of a
    # "previous" step, including in the very first iteration.
    just_restarted = true
    just_accelerated = true

    for k in 0:max_iter
        # Update average iterates
        if restart_period != Inf
            x_avg .= (j_restart * x_avg + ws.vars.x) / (j_restart + 1)
            s_avg .= (j_restart * s_avg + ws.vars.s) / (j_restart + 1)
            y_avg .= (j_restart * y_avg + ws.vars.y) / (j_restart + 1)
        end

        # Compute residuals, their norms, and objective values.
        primal_residual!(pri_res, ws.p.A, ws.vars.x, ws.vars.s, ws.p.b)
        dual_residual!(dual_res, dual_res_temp, ws.p.P, ws.p.A, ws.vars.x, ws.vars.y, ws.p.c)
        curr_pri_res_norm = norm(pri_res, residual_norm)
        curr_dual_res_norm = norm(dual_res, residual_norm)
        primal_obj = primal_obj_val(ws.p.P, ws.p.c, ws.vars.x)
        dual_obj = dual_obj_val(ws.p.P, ws.p.b, ws.vars.x, ws.vars.y)
        gap = duality_gap(primal_obj, dual_obj)

        if !isnothing(x_sol)
            curr_xy_semidist = !isnothing(x_sol) ? sqrt(dot([ws.vars.x - x_sol; ws.vars.y + y_sol], ws.cache[:seminorm_mat] * [ws.vars.x - x_sol; ws.vars.y + y_sol])) : nothing
            curr_x_dist = !isnothing(x_sol) ? norm(ws.vars.x - x_sol) : nothing
            curr_s_dist = !isnothing(s_sol) ? norm(ws.vars.s - s_sol) : nothing
            curr_y_dist = !isnothing(y_sol) ? norm(ws.vars.y - (-y_sol)) : nothing
            # NB: obtain expression for v distance by expanding norm  and
            # using v_k = s_k - y_k / ρ.
            curr_v_dist = !isnothing(x_sol) ? sqrt( norm(ws.vars.s - s_sol)^2 + norm((ws.vars.y - (-y_sol)) / ws.ρ)^2 + 2 * dot(ws.vars.s - s_sol, (-y_sol) - ws.vars.y) / ws.ρ) : nothing
        end

        # Print iteration info.
        if k % print_modulo == 0
            curr_xv_dist = sqrt.(curr_x_dist .^ 2 .+ curr_v_dist .^ 2)
            print_results(k, primal_obj, curr_pri_res_norm, curr_dual_res_norm, abs(gap), curr_xv_dist=curr_xv_dist)
        end

        # Note period of computation of the rank of updates.
        if k % updates_cond_period == 0 && k > 0
            # NB: effective rank counts number of normalised singular values
            # larger than a specified small tolerance (eg 1e-8).
            curr_effective_rank, curr_singval_ratio = effective_rank(updates_matrix, 1e-8)
            push!(update_mat_ranks, curr_effective_rank)
            push!(update_mat_singval_ratios, curr_singval_ratio)
            push!(update_mat_iters, k)
        end


        # Store metrics.
        push!(primal_obj_vals, primal_obj)
        push!(dual_obj_vals, dual_obj)
        push!(pri_res_norms, curr_pri_res_norm)
        push!(dual_res_norms, curr_dual_res_norm)
        if !isnothing(x_sol)
            push!(x_dist_to_sol, curr_x_dist)
            push!(s_dist_to_sol, curr_s_dist)
            push!(y_dist_to_sol, curr_y_dist)
            push!(v_dist_to_sol, curr_v_dist)
            push!(xy_semidist, curr_xy_semidist)
        end

        # Update linearised (affine) operator.
        # Doing this explicitly is (very) expensive, so it is only done when
        # the user requests it (eg for displaying spectra).
        if explicit_affine_operator
            update_affine_dynamics!(ws)

            # Plot spectrum of tilde_A operator.
            if k % spectrum_plot_period == 0
                # We keep the eigendecomposition for further analysis once
                # we have computed the iterate update from this operator. 
                eig_decomp = plot_spectrum(ws.tilde_A, k)
            end
        end

        ### ITERATE ###

        # NB, for mat-mat-free implementation, we replace the above heavy
        # function update_affine_dynamics! with updates of just the enforced
        # constraint set and tilde_b (avoiding matrix-matrix products).
        update_enforced_constraints!(ws)
        update_tilde_b!(ws)
        
        # ACCELERATED ITERATION UPDATE.
        if acceleration && k % (acceleration_memory + 1) == 0 && k > 0
            accelerated_point = custom_acceleration_candidate(ws)
            acc_x, acc_v = accelerated_point[1:ws.p.n], accelerated_point[ws.p.n+1:end]
            acc_y, acc_s = recover_y(acc_v, ws.ρ, ws.p.K), recover_s(acc_v, ws.p.K)
            
            primal_residual!(acc_pri_res, ws.p.A, acc_x, acc_s, ws.p.b)
            dual_residual!(acc_dual_res, acc_dual_res_temp, ws.p.P, ws.p.A, acc_x, acc_y, ws.p.c)

            if accept_acc_candidate(ws, acc_pri_res, acc_dual_res, pri_res, dual_res, residual_norm)
                println("Accepted acceleration candidate at iteration $k.")
                push!(acc_step_iters, k)
                
                ws.vars.x .= acc_x
                ws.vars.v .= acc_v
                ws.vars.s .= acc_s
                ws.vars.y .= acc_y

                curr_xv_update = [ws.vars.x; ws.vars.v] - ws.vars.x_v_q[:, 1]
                
                # If wishing to exclude accelerate steps from the updates matrix,
                # we have the line just below this commented out.
                # insert_update_into_matrix!(updates_matrix, curr_xv_update, current_update_mat_col)
                
                # If wishing to reset the stored matrix of iterate updates when
                # a successful accelerated step is taken:
                updates_matrix .= 0.0
                current_update_mat_col = Ref(1)
                
                # push!(xv_step_norms, norm(curr_xv_update))
                push!(xv_step_norms, NaN) # If wishing to exclude accelerated steps from monitoring matrix.

                # Update "for real":
                ws.vars.x_v_q[:, 1] .= [ws.vars.x; ws.vars.v]
                
                # It does not make sense to have ws.vars.y_prev as usual here.
                # This is more like a restart situation.
                ws.vars.y_prev .= ws.vars.y - y_update(ws.p.A, ws.vars.x, ws.vars.s, ws.p.b, ws.ρ)
            else
                # NOTE: if we have an unsuccessful acceleration candidate, iteration does NOT update at all!
                # Accordingly, I also do not write anything into the matrix
                # storing past iterate updates (for monitoring purposes only).
                # However, I must account for the step norm monitoring.
                push!(xv_step_norms, NaN)
                nothing
            end

            # NB: We reset the Krylov basis memory each time
            # we ATTEMPT an acceleration step.
            ws.cache[:krylov_basis] .= 0.0
            ws.cache[:H] .= 0.0

            # TODO: consider how to handle acceleration steps in the context of
            # running averages for restarts!?
            # RESET AVERAGES PERHAPS? ALONG WITH SETTING j_restart = 0 ?
            just_accelerated = true
            j_restart = 0
        # RESTARTED ITERATION UPDATE.
        elseif k > 0 && k % restart_period == 0
            # NOTE: suspended adaptive trigger in the line below.
            # if restart_trigger(restart_period, k, x_angle_sum, s_angle_sum, y_angle_sum)
            # Restart. Use averages of (x, s, y) sequence, then recover v.
            ws.vars.x .= x_avg
            ws.vars.s .= s_avg
            ws.vars.y .= y_avg
            ws.vars.y_prev .= ws.vars.y - y_update(ws.p.A, ws.vars.x, ws.vars.s, ws.p.b, ws.ρ)
            ws.vars.v .= compute_v(ws.p.A, ws.p.b, ws.vars.x, ws.vars.y_prev, ws.ρ)
            ws.vars.x_v_q[1:end, 1] .= [ws.vars.x; ws.vars.v]
            ws.vars.x_v_q[1:end, 2] .= ones(ws.p.n + ws.p.m) # TODO: ADAPT THIS TO SOMETHING SENSIBLE.

            # Reset iterate averages.
            x_avg .= ws.vars.x
            s_avg .= ws.vars.s
            y_avg .= ws.vars.y

            # Reset inner loop counters/sums.
            just_restarted = true # NOTE: this may not work as expected when reintroducing step angle stuff for adaptive restarts.
            j_restart = 0
            
            # x_angle_sum, s_angle_sum, y_angle_sum = 0.0, 0.0, 0.0
        else # STANDARD ITERATION.
            # Standard iterate update through x^+ = tilde_A * x + tilde_b.
            # Krylov iterate (2nd column) is instead updated through
            # q^+ = (tilde_A - I) * q + tilde_b.
            
            # NB: I replaced this with a mat-vec only imlpementation of tilde_A
            # vector products, just below.
            # ws.vars.x_v_q .= ws.tilde_A * ws.vars.x_v_q + [ws.tilde_b (-ws.vars.x_v_q[:, 2])]
            
            # NB: mat-vec-only implementation here.
            # NB: my first implementation of Arnoldi puts the sequence through
            # the operator B := A - I.
            prev_xv = ws.vars.x_v_q[:, 1]
            ws.vars.x_v_q .= tilde_A_prod(ws, ws.vars.x_v_q) + [ws.tilde_b (-ws.vars.x_v_q[:, 2])]
            curr_xv_update = ws.vars.x_v_q[:, 1] - prev_xv
            insert_update_into_matrix!(updates_matrix, curr_xv_update, current_update_mat_col)

            if explicit_affine_operator && k % spectrum_plot_period == 0
                plot_eigenvec_alignment_vs_phase(curr_xv_update, eig_decomp.values, eig_decomp.vectors, k)
            end
            
            push!(xv_step_norms, norm(curr_xv_update))


            # It is sensible to have the first vector in the Krylov basis be
            # the residual achieved by our initial guess.
            if acceleration && just_accelerated
                # When we are beginning to build a basis, the first step is
                # therefore just to assign the initial residual.
                ws.vars.x_v_q[:, 2] .= ws.vars.x_v_q[:, 1] - [ws.vars.x; ws.vars.v]
                
                ws.vars.x_v_q[:, 2] ./= norm(ws.vars.x_v_q[:, 2]) # Normalise.
                ws.cache[:krylov_basis][:, 1] .= ws.vars.x_v_q[:, 2]
            elseif acceleration
                # In these circumstances, what we have to do is orthogonalise
                # the new Krylov vector, resultant from applying (A - I) +
                # + tilde_b, against the previous vectors.
                # This is the modified Gram-Schmidt process.
                ws.vars.x_v_q[:, 2] .= arnoldi_step!(ws.cache[:krylov_basis], ws.vars.x_v_q[:, 2], ws.cache[:H])
            end
            
            # Extract variables.
            ws.vars.x .= ws.vars.x_v_q[1:ws.p.n, 1]
            ws.vars.v .= ws.vars.x_v_q[ws.p.n+1:end, 1]
            ws.vars.s .= recover_s(ws.vars.v, ws.p.K)
            ws.vars.y .= recover_y(ws.vars.v, ws.ρ, ws.p.K)
            ws.vars.q .= ws.vars.x_v_q[:, 2]

            just_accelerated = false
            j_restart += 1
        end

        push!(enforced_set_flags, ws.enforced_constraints)



        # concat_step[1:ws.p.n] .= x_step
        # normalised_concat_step[1:ws.p.n] .= dim_adjusted_vec_normalise(x_step)
        # concat_step[ws.p.n+1:ws.p.n+ws.p.m] .= s_step
        # normalised_concat_step[ws.p.n+1:ws.p.n+ws.p.m] .= dim_adjusted_vec_normalise(s_step)
        # concat_step[ws.p.n+ws.p.m+1:end] .= y_step
        # normalised_concat_step[ws.p.n+ws.p.m+1:end] .= dim_adjusted_vec_normalise(y_step)

        # NOTE: we skip the iterations just after a restart (or the first one).
        # if !just_restarted
        #     x_step_angle = acos(dot(x_step, x_step_prev) / (norm(x_step) * norm(x_step_prev)))
        #     x_angle_sum += x_step_angle
        #     s_step_angle = acos(dot(s_step, s_step_prev) / (norm(s_step) * norm(s_step_prev)))
        #     # Avoid nans for zero steps: put in zeros instead.
        #     # NOTE: may want to replicate this for x and y steps too.
        #     s_step_angle = isnan(s_step_angle) ? 0.0 : s_step_angle
        #     s_angle_sum += s_step_angle
        #     y_step_angle = acos(dot(y_step, y_step_prev) / (norm(y_step) * norm(y_step_prev)))
        #     y_angle_sum += y_step_angle
            
        #     # NOTE: this data has indexing "gaps" corresponding to when a
        #     # restart occurs, since it doesn't make sense to include those as
        #     # angles between consecutive steps in the usual sense.
        #     if return_run_data
        #         concat_step_angle = acos(dot(concat_step, concat_step_prev) / (norm(concat_step) * norm(concat_step_prev)))
        #         normalised_concat_step_angle = acos(dot(normalised_concat_step, normalised_concat_step_prev) / (norm(normalised_concat_step) * norm(normalised_concat_step_prev)))
        #         push!(x_step_angles, x_step_angle)
        #         push!(s_step_angles, s_step_angle)
        #         push!(y_step_angles, y_step_angle)
        #         push!(concat_step_angles, concat_step_angle)
        #         push!(normalised_concat_step_angles, normalised_concat_step_angle)
        #     end
        # end

        # x_step_prev = copy(x_step)
        # s_step_prev = copy(s_step)
        # y_step_prev = copy(y_step)
        # concat_step_prev = copy(concat_step)
        # normalised_concat_step_prev = copy(normalised_concat_step)

        # Notion of a previous iterate step makes sense (again).
        just_restarted = false
    end


    # END: Compute residuals, their norms, and objective values.
    # TODO: make this stuff modular as opposed to copied from the main loop.
    primal_residual!(pri_res, ws.p.A, ws.vars.x, ws.vars.s, ws.p.b)
    dual_residual!(dual_res, dual_res_temp, ws.p.P, ws.p.A, ws.vars.x, ws.vars.y, ws.p.c)
    curr_pri_res_norm = norm(pri_res, residual_norm)
    curr_dual_res_norm = norm(dual_res, residual_norm)
    primal_obj = primal_obj_val(ws.p.P, ws.p.c, ws.vars.x)
    dual_obj = dual_obj_val(ws.p.P, ws.p.b, ws.vars.x, ws.vars.y)
    gap = duality_gap(primal_obj, dual_obj)
    
    curr_xy_semidist = !isnothing(x_sol) ? sqrt(dot([ws.vars.x - x_sol; ws.vars.y + y_sol], ws.cache[:seminorm_mat] * [ws.vars.x - x_sol; ws.vars.y + y_sol])) : nothing
    curr_x_dist = !isnothing(x_sol) ? norm(ws.vars.x - x_sol) : nothing
    curr_s_dist = !isnothing(s_sol) ? norm(ws.vars.s - s_sol) : nothing
    curr_y_dist = !isnothing(y_sol) ? norm(ws.vars.y - (-y_sol)) : nothing
    
    # NB: obtain expression for v distance by expanding norm  and
    # using v_k = s_k - y_k / ρ.
    curr_v_dist = !isnothing(x_sol) ? sqrt( norm(ws.vars.s - s_sol)^2 + norm((ws.vars.y - (-y_sol)) / ws.ρ)^2 + 2 * dot(ws.vars.s - s_sol, (-y_sol) - ws.vars.y) / ws.ρ) : nothing

    # END: Store metrics if requested.
    # TODO: make this stuff modular as opposed to copied from the main loop.
    push!(primal_obj_vals, primal_obj)
    push!(dual_obj_vals, dual_obj)
    push!(pri_res_norms, curr_pri_res_norm)
    push!(dual_res_norms, curr_dual_res_norm)
    if !isnothing(x_sol)
        push!(x_dist_to_sol, curr_x_dist)
        push!(s_dist_to_sol, curr_s_dist)
        push!(y_dist_to_sol, curr_y_dist)
        push!(v_dist_to_sol, curr_v_dist)
        push!(xy_semidist, curr_xy_semidist)
    end

    # END: print iteration info.
    curr_xv_dist = sqrt.(curr_x_dist .^ 2 .+ curr_v_dist .^ 2)
    print_results(max_iter, primal_obj, curr_pri_res_norm, curr_dual_res_norm, abs(gap), curr_xv_dist = curr_xv_dist, terminated = true)

    return Results(primal_obj_vals, dual_obj_vals, pri_res_norms, dual_res_norms, enforced_set_flags, x_dist_to_sol, s_dist_to_sol, y_dist_to_sol, v_dist_to_sol, xy_semidist, update_mat_iters, update_mat_ranks, update_mat_singval_ratios, acc_step_iters, xv_step_norms)
end