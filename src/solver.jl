using LinearAlgebra
using Plots
include("utils.jl")
include("residuals.jl")
include("acceleration.jl")
include("printing.jl")
# include("QPProblem.jl")

# Compute v iterate which consolidates the y and s iterates.
function compute_v(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
    x::AbstractVector{Float64}, y_prev::AbstractVector{Float64}, ρ::Float64)
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
    step = ρ * (A * x + s - b)
    return step
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
    acceleration::Bool = false)
    # I am disabling the restart feature for now, as I experiment with
    # how best to handle the Krylov acceleration methods.
    if restart_period != Inf
        throw(ArgumentError("Restart feature suspended."))
    end
    ####################################################################

    # Compute the (fixed) matrix that premultiplies the x step.
    # NOTE: I assume for now that A_gram is already in the cache.
    # It is suboptimal to be storing all these matrices explicitly.
    # It would be much better to specify how the operator computes products
    # against vectors instead. However, for now, we will leave it like this.
    ws.cache[:W_inv] = pre_x_step_matrix(ws.variant, ws.p.P, ws.cache[:A_gram], ws.τ, ws.ρ, ws.p.n)
    # Compute fixed bits of \tilde_{A} for acceleration.
    ws.cache[:tlhs] = I(ws.p.n) - ws.cache[:W_inv] * (ws.p.P + ws.ρ * ws.cache[:A_gram])
    ws.cache[:blhs] = -A * ws.cache[:tlhs]
    
    ws.cache[:trhs_pre] = ws.ρ * ws.cache[:W_inv] * A'
    ws.cache[:brhs_pre] = -A * ws.cache[:trhs_pre]



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

    primal_res = zeros(Float64, ws.p.m)
    dual_res = zeros(Float64, ws.p.n)
    dual_res_temp = zeros(Float64, ws.p.n)
    j = 0

    # Data containers for metrics (if return_run_data == true).
    # x_step_angles = Float64[]
    # s_step_angles = Float64[]
    # y_step_angles = Float64[]
    # concat_step_angles = Float64[]
    # normalised_concat_step_angles = Float64[]
    primal_obj_vals = Float64[]
    dual_obj_vals = Float64[]
    primal_res_norms = Float64[]
    dual_res_norms = Float64[]
    enforced_set_flags = Vector{Vector{Bool}}()


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

    for k in 0:max_iter
        # Update average iterates
        if restart_period != Inf
            x_avg .= (j * x_avg + ws.vars.x) / (j + 1)
            s_avg .= (j * s_avg + ws.vars.s) / (j + 1)
            y_avg .= (j * y_avg + ws.vars.y) / (j + 1)
            if restart_trigger(restart_period, k, x_angle_sum, s_angle_sum, y_angle_sum)
                # Restart.
                ws.vars.x .= x_avg
                ws.vars.s .= s_avg
                ws.vars.y .= y_avg
                ws.vars.y_prev .= ws.vars.y - y_update(ws.p.A, ws.vars.x, ws.vars.s, ws.p.b, ws.ρ)

                # Reset iterate averages.
                x_avg .= ws.vars.x
                s_avg .= ws.vars.s
                y_avg .= ws.vars.y

                # Reset inner loop counters/sums.
                j = 0
                x_angle_sum, s_angle_sum, y_angle_sum = 0.0, 0.0, 0.0

                just_restarted = true
            end
        end

        # Compute residuals, their norms, and objective values.
        primal_residual!(primal_res, ws.p.A, ws.vars.x, ws.vars.s, ws.p.b)
        dual_residual!(dual_res, dual_res_temp, ws.p.P, ws.p.A, ws.vars.x, ws.vars.y, ws.p.c)
        curr_primal_res_norm = norm(primal_res, residual_norm)
        curr_dual_res_norm = norm(dual_res, residual_norm)
        primal_obj = primal_obj_val(ws.p.P, ws.p.c, ws.vars.x)
        dual_obj = dual_obj_val(ws.p.P, ws.p.b, ws.vars.x, ws.vars.y)
        gap = duality_gap(primal_obj, dual_obj)

        # Print iteration info.
        if k % print_modulo == 0
            print_results(k, primal_obj, curr_primal_res_norm, curr_dual_res_norm, abs(gap))
        end

        # Store metrics if requested.
        push!(primal_obj_vals, primal_obj)
        push!(dual_obj_vals, dual_obj)
        push!(primal_res_norms, curr_primal_res_norm)
        push!(dual_res_norms, curr_dual_res_norm)


        ### ITERATE and record step vectors.
        # ACCELERATED STEP.
        # v = compute_v(ws.p.A, ws.p.b, ws.vars.x, ws.vars.y_prev, ws.ρ)
        
        # Compute linearised (affine) operator.
        tilde_A, tilde_b, D_k_π_diag = local_affine_dynamics(ws)
        
        # Wherever projections "have had" to act, flags are set to true (1.0).
        # In other blocks, they are false (0.0).

        constraint_flags = D_k_π_diag .!= 0.0


        # We now plot the spectrum of tilde_A on the complex plane.
        # plot_spectrum(tilde_A)
        # Compute the eigenvalues (spectrum) of the matrix
        if k % 50 == 0
            spectrum = eigvals(Matrix(tilde_A))
            println("Condition number of tilde_A: ", cond(Matrix(tilde_A)))
            println("Norm of tilde_b: ", norm(tilde_b))
            println()

            # Extract real and imaginary parts of the eigenvalues
            real_parts = real(spectrum)
            imag_parts = imag(spectrum)

            # Plot the spectrum in the complex plane
            display(scatter(real_parts, imag_parts,
                xlabel="Re", ylabel="Im",
                title="Spectrum of tilde_A",
                legend=false, aspect_ratio=:equal, marker=:circle))
        end

        # Apply operator.
        # ACCELERATED ITERATION UPDATE.
        if acceleration && k >= 100 && k % 20 == 0
            accelerated_point = acceleration_candidate(tilde_A, tilde_b, ws.vars.x, ws.vars.v, ws.p.n, ws.p.m)
            acc_x, acc_v = accelerated_point[1:ws.p.n], accelerated_point[ws.p.n+1:end]
            acc_y, acc_s = recover_y(acc_v, ws.ρ, ws.p.K), recover_s(acc_v, ws.p.K)

            ws.vars.x .= acc_x
            ws.vars.v .= acc_v
            ws.vars.s .= acc_s
            ws.vars.y .= acc_y
            ws.vars.x_v_q[:, 1] .= [acc_x; acc_v]
            ws.vars.x_v_q[:, 2] .= ones(n + m) # TODO: ADAPT THIS TO SOMETHING SENSIBLE.
            
            # It does not make sense to have ws.vars.y_prev as usual here.
            # This is more like a restart situation.
            ws.vars.y_prev .= ws.vars.y - y_update(ws.p.A, ws.vars.x, ws.vars.s, ws.p.b, ws.ρ)
        else # STANDARD ITERATION.
            ws.vars.x_v_q .= tilde_A * ws.vars.x_v_q + (tilde_b * ones(2)')

            # TODO: ORTHOGONALISE THE q VECTOR FOR BASIS BUILDING
            # TODO: THIS IMPLIES HAVING A REGISTER OF SOME MAX MEMORY STORED
            # IN THE WORKSPACE AS WE ITERATE.
            
            # Extract variables.
            ws.vars.x .= ws.vars.x_v_q[1:ws.p.n, 1]
            ws.vars.v .= ws.vars.x_v_q[ws.p.n+1:end, 1]
            ws.vars.s .= recover_s(ws.vars.v, ws.p.K)
            ws.vars.y .= recover_y(ws.vars.v, ws.ρ, ws.p.K)
            ws.vars.q .= ws.vars.x_v_q[:, 2]
            push!(enforced_set_flags, constraint_flags)
        end



            # NOTE: check for whether the local affinisation and the actual
            # method operator give the same result when evaluated at the
            # current iterate (they should!).
            # x_actual = ws.vars.x + x_update(ws.vars.x, ws.cache[:W_inv], ws.p.P, ws.p.c, ws.p.A, ws.vars.y, ws.vars.y_prev)
            # v_actual = iter_v(ws.p.A, ws.p.b, v, ws.vars.x, ws.p.K)

            # println(norm((tilde_A * [ws.vars.x; v] + tilde_b) - [x_actual; v_actual]), 2)
            
            # x_step = acc_x - ws.vars.x
            # s_step = acc_s - ws.vars.s
            # y_step = acc_y - ws.vars.y
            


        # else # STANDARD ITERATION (no acceleration).
        #     x_step = iter_x!(ws.vars.x, ws.cache[:W_inv], ws.p.P, ws.p.c, ws.p.A, ws.vars.y, ws.vars.y_prev)
        #     s_step, constraint_flags = iter_s!(ws.vars.s, ws.p.A, ws.vars.x, ws.p.b, ws.vars.y, ws.p.K, ws.ρ)
        #     ws.vars.y_prev .= ws.vars.y
        #     y_step = iter_y!(ws.vars.y, ws.p.A, ws.vars.x, ws.vars.s, ws.p.b, ws.ρ)

        #     if return_run_data
        #         push!(enforced_set_flags, constraint_flags)
        #     end
        # end

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
        
        j += 1
    end


    # END: Compute residuals, their norms, and objective values.
    # TODO: make this stuff modular as opposed to copied from the main loop.
    primal_residual!(primal_res, ws.p.A, ws.vars.x, ws.vars.s, ws.p.b)
    dual_residual!(dual_res, dual_res_temp, ws.p.P, ws.p.A, ws.vars.x, ws.vars.y, ws.p.c)
    curr_primal_res_norm = norm(primal_res, residual_norm)
    curr_dual_res_norm = norm(dual_res, residual_norm)
    primal_obj = primal_obj_val(ws.p.P, ws.p.c, ws.vars.x)
    dual_obj = dual_obj_val(ws.p.P, ws.p.b, ws.vars.x, ws.vars.y)
    gap = duality_gap(primal_obj, dual_obj)

    # END: Store metrics if requested.
    # TODO: make this stuff modular as opposed to copied from the main loop.
    push!(primal_obj_vals, primal_obj)
    push!(dual_obj_vals, dual_obj)
    push!(primal_res_norms, curr_primal_res_norm)
    push!(dual_res_norms, curr_dual_res_norm)

    # END: print iteration info.
    print_results(max_iter, primal_obj, curr_primal_res_norm, curr_dual_res_norm, abs(gap), terminated = true)

    return primal_obj_vals, dual_obj_vals, primal_res_norms, dual_res_norms, enforced_set_flags #, x_step_angles, s_step_angles, y_step_angles, concat_step_angles, normalised_concat_step_angles, 
end