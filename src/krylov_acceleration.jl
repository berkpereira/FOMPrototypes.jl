using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Clarabel

"""
For LINEAR CONSTRAINTS DO ENTRY BY ENTRY, not cone by cone. Check for
entry-wise equality/inequality.
"""
function update_proj_flags!(proj_flags::BitVector,
    preproj_y::AbstractVector{Float64},
    postproj_y::AbstractVector{Float64})

    proj_flags .= (preproj_y .== postproj_y)
end

# helper to form and store the first Krylov basis vector
function init_krylov_basis!(ws::KrylovWorkspace)
    # form fixed‐point residual in slot 2
    @views ws.vars.xy_q[:, 2] .= ws.vars.xy_q[:, 1] .- ws.vars.xy_prev
    # normalise
    @views ws.vars.xy_q[:, 2] ./= norm(ws.vars.xy_q[:, 2])
    # store in krylov_basis[:,1]
    @views ws.krylov_basis[:, 1] .= ws.vars.xy_q[:, 2]
end

"""
This function implements a mat-mat free specification of the application of
tilde_A to a vector, at some iteration.

NOTE: one must pass in a bit vector of enforced constraints as well as the
AbstractWorkspace struct ws, even though AbstractWorkspace already has a field storing a
bit vector of this kind.
The issue is that the AbstractWorkspace field corresponds to the actual current
iterate of the method, whereas we may want to evaluate the method's output
from some arbitrary, different iterate!
"""
function tilde_A_prod(ws::AbstractWorkspace,
    enforced_constraints::BitVector,
    q::AbstractArray{Float64})

    @views top_left = q[1:ws.p.n, :] - (ws.W_inv * (ws.p.P * q[1:ws.p.n, :] + ws.ρ * ws.A_gram * q[1:ws.p.n, :]))
    bot_left = - ws.p.A * top_left
    @views top_right = ws.ρ * ws.W_inv * (ws.p.A' * ((enforced_constraints - .!enforced_constraints) .* q[ws.p.n+1:end, :]))
    @views bot_right = - ws.p.A * top_right + enforced_constraints .* q[ws.p.n+1:end, :]

    # NB: matrix, NOT vector, is returned.
    # This is to allow for multiplication by a matrix of 2 columns (maintain
    # two sequences of vector iterates simultaneously).
    return [top_left + top_right; bot_left + bot_right]
end

function acceleration_candidate(tilde_A::AbstractMatrix{Float64},
    tilde_b::AbstractVector{Float64}, x::AbstractVector{Float64},
    v::AbstractVector{Float64}, n::Integer, m::Integer,
    use_package_krylov::Bool = true,
    krylov_basis::AbstractMatrix{Float64} = zeros(Float64, n + m, 0))
    if use_package_krylov
        krylov_iterate = copy([x; v])
        
        # NOTE: may want to set maxiter argument in call to gmres! below.
        gmres!(krylov_iterate, I(n + m) - tilde_A, tilde_b, maxiter = 20)

        return tilde_A * krylov_iterate + tilde_b
    else
        # Throw exception because this is not yet implemented.
        throw(ArgumentError("Non-package acceleration not yet implemented."))
    end
end

"""
This implementation of the accelerated point computation relies on our custom
implementations of the Arnoldi and Krylov procedures.
"""
function custom_acceleration_candidate!(ws::KrylovWorkspace,
    result_vec::Vector{Float64},
    temp_n_vec1::Vector{Float64},
    temp_n_vec2::Vector{Float64},
    temp_m_vec::Vector{Float64})

    # TODO pre-allocate working vectors in this function when acceleration
    # is used

    # compute iterate from xy_q[:, 1], and store it in result_vec
    @views onecol_method_operator!(ws, ws.vars.xy_q[:, 1], result_vec, temp_n_vec1, temp_n_vec2, temp_m_vec)
    rhs_res_custom = ws.krylov_basis' * (result_vec - ws.vars.xy_q[:, 1])
    
    # compute Hessenber LLS solution, reduced dimension y
    if ws.krylov_operator == :tilde_A
        shifted_hessenberg = ws.H - [I(ws.mem - 1); zeros(1, ws.mem - 1)]
        
        y_krylov_sol = krylov_least_squares!(shifted_hessenberg, rhs_res_custom)
    else # ie use B := tilde_A - I as the Arnoldi/Krylov operator.
        y_krylov_sol = krylov_least_squares!(ws.H, rhs_res_custom)
    end

    # compute full-dimension LLS solution
    @views gmres_sol = ws.vars.xy_q[:, 1] + ws.krylov_basis[:, 1:end-1] * y_krylov_sol

    # obtain actual acceleration candidate, write it to result_vec
    onecol_method_operator!(ws, gmres_sol, result_vec, temp_n_vec1, temp_n_vec2, temp_m_vec)

    return nothing
end