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
Compute the Krylov accelerant candidate, write to result_vec.

This function performs an accelerated update of the candidate solution vector by:
    1. Evaluating the operator on an initial column vector from the workspace.
    2. Computing the residual in the Krylov subspace from the difference between the evaluated result and the reference vector.
    3. Solving a least squares problem based on a (possibly shifted) Hessenberg matrix derived from the workspace.
    4. Constructing a full-dimensional approximate solution using the Krylov basis, and re-applying the operator it to obtain the acceleration candidate.

# Parameters besides working vectors
- ws::KrylovWorkspace: 
        The workspace containing the Krylov subspace information, including the Hessenberg matrix, 
        Krylov basis, and other variables required for the computation.
- result_vec::Vector{Float64}: 
        Vector where the computed accelerated candidate solution will be stored.

# Returns
- Nothing. The function updates `result_vec` with the computed acceleration candidate.
"""
function compute_krylov_accelerant!(ws::KrylovWorkspace,
    result_vec::Vector{Float64},
    temp_n_vec1::Vector{Float64},
    temp_n_vec2::Vector{Float64},
    temp_m_vec::Vector{Float64})

    # TODO pre-allocate working vectors in this function when acceleration
    # is used

    # TODO slice of H to use is H[1:ws.givens_count[], 1:ws.givens_count[]]


    # compute FOM(xy_q[:, 1]) and store it in result_vec
    @views onecol_method_operator!(ws, ws.vars.xy_q[:, 1], result_vec, temp_n_vec1, temp_n_vec2, temp_m_vec)
    @views rhs_res_custom = (ws.krylov_basis[:, 1:ws.givens_count[] + 1])' * (result_vec - ws.vars.xy_q[:, 1])
    
    # ws.H is passed as triangular, so we now need to apply the
    # Givens rotations to Q_{krylov + 1}^T * residual
    # and solve the appropriate triangular system to obtain the LLS solution
    
    # compute Hessenber LLS solution, reduced dimension y
    # the difference between ws.krylov_operator == tilde_A
    # and ws.krylov_operator == :B is accounted for in the ws.H matrix
    # itself, which in the former case is shifted by -I at every step of the
    # way leading up to here (see Obsidian 2024/wee51 notes)
    solve_current_least_squares!(
        ws.H,
        ws.givens_rotations,
        ws.givens_count,
        rhs_res_custom # NB solution overwrites rhs_res_custom
        )

    # compute full-dimension LLS solution
    @views gmres_sol = ws.vars.xy_q[:, 1] + ws.krylov_basis[:, 1:ws.givens_count[]] * rhs_res_custom[1:ws.givens_count[]]

    # obtain actual acceleration candidate by applying FOM to
    # this, and write candidate to result_vec
    onecol_method_operator!(ws, gmres_sol, result_vec, temp_n_vec1, temp_n_vec2, temp_m_vec)

    return nothing
end