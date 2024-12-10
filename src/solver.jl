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

    # Take "preliminary" step.
    s .= b - A * x - y / ρ

    # Project to cones in K.
    projected_s = project_to_K(s, K)

    # NOTE: proj_flags is a vector of flags. Entries where the projection passed
    # through the projection get zeros in these flags. Others get ones.
    proj_flags = projected_s .!= s
    
    s .= projected_s
    
    # NOTE: line below is how we will simply want to do this in the end.
    # project_to_K!(s, K)
    
    return s - s_old, proj_flags
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
function optimise!(problem::ProblemData, variant::Integer, x::Vector{Float64},
    s::Vector{Float64}, y::Vector{Float64}, τ::Float64, ρ::Float64,
    A_gram::AbstractMatrix, max_iter::Integer, print_modulo::Integer,
    restart_period::Union{Real, Symbol} = Inf, residual_norm::Real = Inf,
    return_run_data::Bool = false, acceleration::Bool = false)

    # Unpack problem data for easier reference
    P, c, A, b, K = problem.P, problem.c, problem.A, problem.b, problem.K
    m, n = size(A)

    # Compute the (fixed) matrix that premultiplies the x step
    pre_matrix = pre_x_step_matrix(variant, P, A_gram, τ, ρ, n)

    # Initialise "artificial" y_{-1} to make first x update well-defined
    # and correct.
    y_prev = y - y_update(A, x, s, b, ρ)

    # Initialise running sums of step angles.
    x_angle_sum, s_angle_sum, y_angle_sum = 0.0, 0.0, 0.0
    
    # for restarts, keep average iterates
    if restart_period != Inf
        x_avg = copy(x)
        s_avg = copy(s)
        y_avg = copy(y)
    end

    primal_res = zeros(Float64, m)
    dual_res = zeros(Float64, n)
    dual_res_temp = zeros(Float64, n)
    j = 0

    # Data containers for metrics (if return_run_data == true).
    if return_run_data
        x_step_angles = Float64[]
        s_step_angles = Float64[]
        y_step_angles = Float64[]
        concat_step_angles = Float64[]
        normalised_concat_step_angles = Float64[]
        primal_obj_vals = Float64[]
        dual_obj_vals = Float64[]
        primal_res_norms = Float64[]
        dual_res_norms = Float64[]
        v_proj_flags = Vector{Vector{Bool}}()
    end

    # Initialise other variables to be used during algorithm.
    x_step = zeros(Float64, n)
    s_step = zeros(Float64, m)
    y_step = zeros(Float64, m)
    concat_step = zeros(Float64, n + 2 * m)
    normalised_concat_step = zeros(Float64, n + 2 * m)
    x_step_prev = zeros(Float64, n)
    s_step_prev = zeros(Float64, m)
    y_step_prev = zeros(Float64, m)
    concat_step_prev = zeros(Float64, n + 2 * m)
    normalised_concat_step_prev = zeros(Float64, n + 2 * m)

    # This flag helps us keep track of whether we should forgo the notion of a
    # "previous" step, including in the very first iteration.
    just_restarted = true

    for k in 0:max_iter
        # Update average iterates
        if restart_period != Inf
            x_avg .= (j * x_avg + x) / (j + 1)
            s_avg .= (j * s_avg + s) / (j + 1)
            y_avg .= (j * y_avg + y) / (j + 1)
            if restart_trigger(restart_period, k, x_angle_sum, s_angle_sum, y_angle_sum)
                # Restart.
                x .= x_avg
                s .= s_avg
                y .= y_avg
                y_prev .= y - y_update(A, x, s, b, ρ)

                # Reset iterate averages.
                x_avg .= x
                s_avg .= s
                y_avg .= y

                # Reset inner loop counters/sums.
                j = 0
                x_angle_sum, s_angle_sum, y_angle_sum = 0.0, 0.0, 0.0

                just_restarted = true
            end
        end

        # Compute residuals, their norms, and objective values.
        primal_residual!(primal_res, A, x, s, b)
        dual_residual!(dual_res, dual_res_temp, P, A, x, y, c)
        curr_primal_res_norm = norm(primal_res, residual_norm)
        curr_dual_res_norm = norm(dual_res, residual_norm)
        primal_obj = primal_obj_val(P, c, x)
        dual_obj = dual_obj_val(P, b, x, y)
        gap = duality_gap(primal_obj, dual_obj)

        # Print iteration info.
        if k % print_modulo == 0
            print_results(k, primal_obj, curr_primal_res_norm, curr_dual_res_norm, abs(gap))
        end

        # Store metrics if requested.
        if return_run_data
            push!(primal_obj_vals, primal_obj)
            push!(dual_obj_vals, dual_obj)
            push!(primal_res_norms, curr_primal_res_norm)
            push!(dual_res_norms, curr_dual_res_norm)
        end

        ### Iterate and record step vectors.
        if k >= 100 && k % 20 == 0 && acceleration
            v = compute_v(A, b, x, y_prev, ρ)
            tilde_A, tilde_b = local_affine_dynamics(P, A, A_gram, b, c, pre_matrix,
            s, v, ρ, n, m, K)

            # We now plot the spectrum of tilde_A on the complex plane.
            # This is to check that the matrix is indeed PSD.
            # plot_spectrum(tilde_A)
            # Compute the eigenvalues (spectrum) of the matrix
            spectrum = eigvals(Matrix(tilde_A))

            # Extract real and imaginary parts of the eigenvalues
            real_parts = real(spectrum)
            imag_parts = imag(spectrum)

            # Plot the spectrum in the complex plane
            display(scatter(real_parts, imag_parts,
                xlabel="Re", ylabel="Im",
                title="Spectrum of tilde_A",
                legend=false, aspect_ratio=:equal, marker=:circle))



            # NOTE: check for whether the local affinisation and the actual
            # method operator give the same result when evaluated at the
            # current iterate (they should!).
            x_actual = x + x_update(x, pre_matrix, P, c, A, y, y_prev)
            v_actual = iter_v(A, b, v, x, K)

            println(norm((tilde_A * [x; v] + tilde_b) - [x_actual; v_actual]), 2)

            accelerated_point = acceleration_candidate(tilde_A, tilde_b, x, v, n, m)

            println()

            acc_x, acc_v = accelerated_point[1:n], accelerated_point[n+1:end]
            acc_y, acc_s = recover_y(acc_v, ρ, K), recover_s(acc_v, K)
            
            x_step = acc_x - x
            s_step = acc_s - s
            y_step = acc_y - y
            
            x .= acc_x
            s .= acc_s
            y .= acc_y
            # It does not make sense to have y_prev as usual here.
            # This is more like a restart situation.
            y_prev .= y - y_update(A, x, s, b, ρ)
        else
            x_step = iter_x!(x, pre_matrix, P, c, A, y, y_prev)
            s_step, proj_flags = iter_s!(s, A, x, b, y, K, ρ)
            y_prev .= y
            y_step = iter_y!(y, A, x, s, b, ρ)

            if return_run_data
                push!(v_proj_flags, proj_flags)
            end
        end

        concat_step[1:n] .= x_step
        normalised_concat_step[1:n] .= dim_adjusted_vec_normalise(x_step)
        concat_step[n+1:n+m] .= s_step
        normalised_concat_step[n+1:n+m] .= dim_adjusted_vec_normalise(s_step)
        concat_step[n+m+1:end] .= y_step
        normalised_concat_step[n+m+1:end] .= dim_adjusted_vec_normalise(y_step)

        # NOTE: we skip the iterations just after a restart (or the first one).
        if !just_restarted
            x_step_angle = acos(dot(x_step, x_step_prev) / (norm(x_step) * norm(x_step_prev)))
            x_angle_sum += x_step_angle
            s_step_angle = acos(dot(s_step, s_step_prev) / (norm(s_step) * norm(s_step_prev)))
            # Avoid nans for zero steps: put in zeros instead.
            # NOTE: may want to replicate this for x and y steps too.
            s_step_angle = isnan(s_step_angle) ? 0.0 : s_step_angle
            s_angle_sum += s_step_angle
            y_step_angle = acos(dot(y_step, y_step_prev) / (norm(y_step) * norm(y_step_prev)))
            y_angle_sum += y_step_angle
            
            # NOTE: this data has indexing "gaps" corresponding to when a
            # restart occurs, since it doesn't make sense to include those as
            # angles between consecutive steps in the usual sense.
            if return_run_data
                concat_step_angle = acos(dot(concat_step, concat_step_prev) / (norm(concat_step) * norm(concat_step_prev)))
                normalised_concat_step_angle = acos(dot(normalised_concat_step, normalised_concat_step_prev) / (norm(normalised_concat_step) * norm(normalised_concat_step_prev)))
                push!(x_step_angles, x_step_angle)
                push!(s_step_angles, s_step_angle)
                push!(y_step_angles, y_step_angle)
                push!(concat_step_angles, concat_step_angle)
                push!(normalised_concat_step_angles, normalised_concat_step_angle)
            end
        end

        x_step_prev = copy(x_step)
        s_step_prev = copy(s_step)
        y_step_prev = copy(y_step)
        concat_step_prev = copy(concat_step)
        normalised_concat_step_prev = copy(normalised_concat_step)

        # Notion of a previous iterate step makes sense (again).
        just_restarted = false

        # Keep "previous" dual variable, used in the x update.
        # y_prev = copy(y)
        
        j += 1
    end


    # TODO: make this stuff modular as opposed to copied from the main loop.
    # END: Compute residuals, their norms, and objective values.
    primal_residual!(primal_res, A, x, s, b)
    dual_residual!(dual_res, dual_res_temp, P, A, x, y, c)
    curr_primal_res_norm = norm(primal_res, residual_norm)
    curr_dual_res_norm = norm(dual_res, residual_norm)
    primal_obj = primal_obj_val(P, c, x)
    dual_obj = dual_obj_val(P, b, x, y)
    gap = duality_gap(primal_obj, dual_obj)

    # TODO: make this stuff modular as opposed to copied from the main loop.
    # END: Store metrics if requested.
    if return_run_data
        push!(primal_obj_vals, primal_obj)
        push!(dual_obj_vals, dual_obj)
        push!(primal_res_norms, curr_primal_res_norm)
        push!(dual_res_norms, curr_dual_res_norm)
    end

    # END: print iteration info.
    print_results(max_iter, primal_obj, curr_primal_res_norm, curr_dual_res_norm, abs(gap), terminated = true)

    if return_run_data
        return primal_obj_vals, dual_obj_vals, primal_res_norms, dual_res_norms, x_step_angles, s_step_angles, y_step_angles, concat_step_angles, normalised_concat_step_angles, v_proj_flags
    else
        return primal_obj_val(P, c, x), dual_obj_val(P, b, x, y)
    end
end