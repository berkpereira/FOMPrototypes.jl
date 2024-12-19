include("utils.jl")
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Clarabel


function flag_arrays(v::AbstractVector{Float64}, s::AbstractVector{Float64},
    m::Integer, K::Vector{Clarabel.SupportedCone})
    # NOTE: commented out code below is more along the lines of the generalised
    # notion of parameterising these projection operations at a particular
    # iterate (x_k, v_k). HOWEVER, we must go finer than this: in particular,
    # when using nonnegative orthants and zero cones, we must actually do this
    # process entry-by-entry! NOT cone by cone, since only this can offer us
    # the "same facet projection" interpretation of papers such as Boley 2013.
    
    # !!! MAY REQUIRE ADJUSTMENTS, ADAPT FROM ENTRY-WISE CODE FURTHER BELOW!!!

    # # Start partition indexing
    # start_idx = 1
    # for cone in K
    #     end_idx = start_idx + cone.dim - 1

    #     # Extract partitions of s and v for the current block
    #     s_block = s[start_idx:end_idx]
    #     v_block = v[start_idx:end_idx]

    #     # Check whether the blocks of s and v are equal
    #     if s_block == v_block
    #         # Fill diagonal entries for the case where s == v
    #         D_k_x_diag[start_idx:end_idx] .= 1.0
    #         D_k_x_b_diag[start_idx:end_idx] .= 0.0
    #     else
    #         # Fill diagonal entries for the case where s != v
    #         D_k_x_diag[start_idx:end_idx] .= -1.0
    #         D_k_x_b_diag[start_idx:end_idx] .= 2.0
    #     end

    #     # Update the start index for the next block
    #     start_idx = end_idx + 1
    # end

    # For LINEAR CONSTRAINTS DO as in the above but ENTRY BY ENTRY, not cone
    # by cone. Check for entry-wise equality/inequality.
    return Diagonal(s .== v)
end

# Recall that pre_x_matrix is often denoted by $W^{-1}$ in my/Paul's notes.
# TODO: consider making operations in this whole "affinisation" process more
# efficient. At the moment it is done in quite a naive fashion for proof of 
# concept.
function update_affine_dynamics!(ws::Workspace)

    # Compute the diagonal flag matrices required.
    D_k_π = flag_arrays(ws.vars.v, ws.vars.s, ws.p.m, ws.p.K)
    new_enforced_constraints = .!D_k_π.diag # Bitwise negation.

    # Dynamics have not changed (and we are past the first iteration).
    if new_enforced_constraints == ws.enforced_constraints && ws.tilde_A != spzeros(eltype(ws.tilde_A), ws.p.n + ws.p.m, ws.p.n + ws.p.m)
        return
    # Enforced constraints (and therefore "affinised" dynamics) have changed.
    else
        println("Dynamics have changed.")
        # Update enforced constraints.
        ws.enforced_constraints .= new_enforced_constraints
        
        # Assemble updated tilde_A.
        D_k_π_neg = Diagonal(ws.enforced_constraints)
        trhs = ws.cache[:trhs_pre] * (D_k_π_neg - D_k_π)
        brhs = -A * trhs + D_k_π_neg
        ws.tilde_A .= [ws.cache[:tlhs] trhs; ws.cache[:blhs] brhs]

        # NOTE: b_k_π IS ALWAYS ZERO WHEN WE ARE CONSIDERING LINEAR CONSTRAINTS
        # IE NONNEGATIVE ORTHANT AND ZERO CONES.
        # b_k_π = D_k_π_neg * ws.vars.s # GENERAL case
        b_k_π = zeros(Float64, ws.p.m)  # LINEAR CONSTRAINTS case

        # tilde_b IN THE GENERAL CASE
        tilde_b_top = -ws.cache[:trhs_pre] * (2 * b_k_π - ws.p.b) - ws.cache[:W_inv] * ws.p.c
        tilde_b_bot = -ws.p.A * tilde_b_top + ws.p.b - b_k_π
        # NOTICE THAT tilde_b IS INVARIANT IN THE LINEAR CONSTRAINTS CASE

        # Assemble tilde_b.
        ws.tilde_b .= [tilde_b_top; tilde_b_bot]

        return ws.tilde_A, ws.tilde_b
    end
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
TODO: make sure that this thing works.
"""
function custom_acceleration_candidate(ws::Workspace)
    # The steps at a high-level are the following.
    # The workspace contains:
    # ws.cache[:H], the Arnoldi upper Hessenberg matrix.
    # ws.cache[:krylov_basis], the Arnoldi-Krylov orthonormal basis matrix.
    # ws.tilde_A, the affine operator matrix.
    # ws.tilde_b, the affine operator vector.
    
    
    # rhs_res = ws.tilde_A * ws.vars.x_v_q[:, 1] + ws.tilde_b - ws.vars.x_v_q[:, 1]

    # NOTE: need to pre-multiply by the transpose of "Q_{k+1}" here.
    # This is assumed by the function krylov_least_squares!
    rhs_res_custom = ws.cache[:krylov_basis]' * (ws.tilde_A * ws.vars.x_v_q[:, 1] + ws.tilde_b - ws.vars.x_v_q[:, 1])
    
    y_krylov_sol = krylov_least_squares!(ws.cache[:H], rhs_res_custom)

    # coeff_mat = ws.cache[:krylov_basis] * ws.cache[:H]
    # y_krylov_sol = (coeff_mat' * coeff_mat) \ (coeff_mat' * (-rhs_res))

    gmres_sol = ws.vars.x_v_q[:, 1] + ws.cache[:krylov_basis][:, 1:end - 1] * y_krylov_sol
    
    # NOTE: only left to convert from GMRES problem solution to Anderson
    # acceleration problem solution (see Walker and Ni, 2011 for details).
    acceleration_point = ws.tilde_A * gmres_sol + ws.tilde_b
    
    return acceleration_point
end