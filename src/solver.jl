module PrototypeMethod

using LinearAlgebra
include("utils.jl")
include("residuals.jl")
include("printing.jl")
include("QPProblem.jl")

export QPProblem, optimise!

function iter_x!(x::AbstractVector{Float64},
    pre_matrix::Diagonal{Float64},
    P::AbstractMatrix{Float64},
    c::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    y_prev::AbstractVector{Float64})
    # NOTE: pre_matrix should already have been inverted!
    step = - pre_matrix * (P * x + c + A' * (2 * y - y_prev))
    x .+= step
end

function iter_s!(s::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    b::AbstractVector{Float64},
    y::AbstractVector{Float64},
    K::Vector{Clarabel.SupportedCone},
    ρ::Float64)

    s .= b - A * x - y / ρ

    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1

        # Project portion of s depending on the cone type
        if cone isa Clarabel.NonnegativeConeT
            @views s[start_idx:end_idx] .= max.(s[start_idx:end_idx], 0)
        elseif cone isa Clarabel.ZeroConeT
            @views s[start_idx:end_idx] .= zeros(Float64, cone.dim)
        else
            error("Unsupported cone type: $typeof(cone)")
        end
        
        start_idx = end_idx + 1
    end
end

function iter_y!(y::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    s::AbstractVector{Float64},
    b::AbstractVector{Float64},
    ρ::Float64)
    y .+= ρ * (A * x + s - b)
end


function optimise!(problem::QPProblem, variant::Integer, x::Vector{Float64},
    s::Vector{Float64}, y::Vector{Float64}, τ::Float64, ρ::Float64,
    A_gram::AbstractMatrix, max_iter::Integer, print_modulo::Integer,
    restart_period::Real = Inf)

    # Unpack problem data for easier reference
    P, c, A, b, K = problem.P, problem.c, problem.A, problem.b, problem.K
    n, m = size(A)

    # Compute the (fixed) matrix that premultiplies the x step
    pre_matrix = pre_x_step_matrix(variant, P, A_gram, τ, ρ, n)

    # keep old dual variable, convenient for x update.
    y_prev = copy(y)
    
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
    for k in 0:max_iter
        # Update average iterates
        if restart_period != Inf
            x_avg .= (j * x_avg + x) / (j + 1)
            s_avg .= (j * s_avg + s) / (j + 1)
            y_avg .= (j * y_avg + y) / (j + 1)
            if k > 0 && k % restart_period == 0
                # restart
                x .= x_avg
                s .= s_avg
                y .= y_avg

                # reset averages
                x_avg .= x
                s_avg .= s
                y_avg .= y

                j = 0
            end
        end

        # Compute residuals and objective values.
        primal_residual!(primal_res, A, x, s, b)
        dual_residual!(dual_res, dual_res_temp, P, A, x, y, c)
        primal_obj = primal_obj_val(P, c, x)
        dual_obj = dual_obj_val(P, b, x, y)
        gap = duality_gap(primal_obj, dual_obj)

        if k % print_modulo == 0
            # Note that the l-infinity norm is customary for residuals.
            print_results(k, primal_obj, norm(primal_res, Inf), norm(dual_res, Inf), abs(gap))
        end
        
        # Iterate
        iter_x!(x, pre_matrix, P, c, A, y, y_prev)
        iter_s!(s, A, x, b, y, K, ρ)
        iter_y!(y, A, x, s, b, ρ)

        # Keep "previous" dual variable
        y_prev = y

        j += 1
    end

    # Return final iterates and objective values
    return x, s, y, primal_obj_val(P, c, x), dual_obj_val(P, b, x, y)
end

end # module PrototypeMethod