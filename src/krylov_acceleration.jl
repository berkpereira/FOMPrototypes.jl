include("utils.jl")
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

"""
This function implements a mat-mat free specification of the application of
tilde_A to a vector, at some iteration.

NOTE: one must pass in a bit vector of enforced constraints as well as the
Workspace struct ws, even though Workspace already has a field storing a
bit vector of this kind.
The issue is that the Workspace field corresponds to the actual current
iterate of the method, whereas we may want to evaluate the method's output
from some arbitrary, different iterate!
"""
function tilde_A_prod(ws::Workspace,
    enforced_constraints::BitVector,
    q::AbstractArray{Float64})

    @views top_left = q[1:ws.p.n, :] - (ws.cache[:W_inv] * (ws.p.P * q[1:ws.p.n, :] + ws.ρ * ws.cache[:A_gram] * q[1:ws.p.n, :]))
    bot_left = - ws.p.A * top_left
    @views top_right = ws.ρ * ws.cache[:W_inv] * (ws.p.A' * ((enforced_constraints - .!enforced_constraints) .* q[ws.p.n+1:end, :]))
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

        # NOTE: the {k+1}th Anderson acceleration iterate is given by the affine 
        # operator of the kth Krylov iterate (see Theorem 2.2 from Walker and Ni,
        # Anderson Acceleration for Fixed-Point Iterations, 2011).

        # println("Rank of (tilde_A - I): ", rank(tilde_A - I(n + m)))
        # println("Condition number of (tilde_A - I): ", cond(Matrix(tilde_A - I(n + m))))
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
function custom_acceleration_candidate!(ws::Workspace,
    krylov_operator_tilde_A::Bool,
    acceleration_memory::Integer,
    result_vec::AbstractVector{Float64},
    temp_n_vec1::AbstractVector{Float64},
    temp_n_vec2::AbstractVector{Float64},
    temp_m_vec::AbstractVector{Float64})
    # note, Workspace should contain:
    # ws.cache[:H], the Arnoldi upper Hessenberg matrix.
    # ws.cache[:krylov_basis], the Arnoldi-Krylov orthonormal basis matrix.

    # TODO pre-allocate working vectors in this function when acceleration
    # is used

    # compute iterate from xy_q[:, 1], and store it in result_vec
    @views onecol_method_operator!(ws, ws.vars.xy_q[:, 1], result_vec, temp_n_vec1, temp_n_vec2, temp_m_vec)
    rhs_res_custom = ws.cache[:krylov_basis]' * (result_vec - ws.vars.xy_q[:, 1])
    
    # compute Hessenber LLS solution, reduced dimension y
    if krylov_operator_tilde_A
        shifted_hessenberg = ws.cache[:H] - [I(acceleration_memory - 1); zeros(1, acceleration_memory - 1)]
        
        # TODO pre-allocate vector/memory for y_krylov_sol ?
        y_krylov_sol = krylov_least_squares!(shifted_hessenberg, rhs_res_custom)
    else # ie use B := tilde_A - I as the Arnoldi/Krylov operator.
        # TODO pre-allocate vector/memory for y_krylov_sol ?
        y_krylov_sol = krylov_least_squares!(ws.cache[:H], rhs_res_custom)
    end

    # compute full-dimension LLS solution
    gmres_sol = ws.vars.xy_q[:, 1] + ws.cache[:krylov_basis][:, 1:end - 1] * y_krylov_sol

    # obtain actual acceleration candidate, write it to result_vec
    onecol_method_operator!(ws, gmres_sol, result_vec, temp_n_vec1, temp_n_vec2, temp_m_vec)

    return nothing
end