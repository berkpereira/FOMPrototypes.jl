# Compute primal residual (in place)
# r_prim = A * x + s - b
function primal_residual!(r::AbstractVector{Float64}, A::AbstractMatrix{Float64}, x::AbstractVector{Float64}, s::AbstractVector{Float64}, b::AbstractVector{Float64}) 
    mul!(r, A, x)
    @. r += s
    @. r -= b
    
    return nothing
end

function primal_residual(A::AbstractMatrix{Float64}, x::AbstractVector{Float64}, s::AbstractVector{Float64}, b::AbstractVector{Float64})
    return A * x + s - b
end

# Compute dual residual (in place)
# r_dual = P * x + A' * y + c
function dual_residual!(r::AbstractVector{Float64}, r_temp::AbstractVector{Float64}, P::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, x::AbstractVector{Float64}, y::AbstractVector{Float64}, c::AbstractVector{Float64})
    mul!(r, A', y)
    mul!(r_temp, P, x)
    @. r += r_temp
    @. r += c

    return nothing
end

function dual_residual(P::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, x::AbstractVector{Float64}, y::AbstractVector{Float64}, c::AbstractVector{Float64})
    return P * x + A' * y + c
end

function is_primal_feasible(r_primal::AbstractVector{Float64}, A::AbstractMatrix{Float64}, x::AbstractVector{Float64}, s::AbstractVector{Float64}, b::AbstractVector{Float64}, ϵ_abs::Float64, ϵ_rel::Float64)
    r_norm = norm(r_primal, Inf)
    return r_norm <= ϵ_abs + ϵ_rel * max(norm(A * x, Inf), norm(s, Inf), norm(b, Inf))
end

function is_dual_feasible(r_dual::AbstractVector{Float64}, P::AbstractMatrix{Float64}, x::AbstractVector{Float64}, c::AbstractVector{Float64}, A::AbstractMatrix{Float64}, y::AbstractVector{Float64}, ϵ_abs::Float64, ϵ_rel::Float64)
    r_norm = norm(r_dual, Inf)
    return r_norm <= ϵ_abs + ϵ_rel * max(norm(P * x, Inf), norm(A' * y, Inf), norm(c, Inf))
end

function is_feasible(r_primal::AbstractVector{Float64}, r_dual::AbstractVector{Float64}, A::AbstractMatrix{Float64}, P::AbstractMatrix{Float64}, x::AbstractVector{Float64}, s::AbstractVector{Float64}, y::AbstractVector{Float64}, b::AbstractVector{Float64}, c::AbstractVector{Float64}, ϵ_abs::Float64, ϵ_rel::Float64)
    return is_primal_feasible(r_primal, A, x, s, b, ϵ_abs, ϵ_rel) && is_dual_feasible(r_dual, P, x, c, A, y, ϵ_abs, ϵ_rel)
end
