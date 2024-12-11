using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Clarabel


function flag_arrays(v::AbstractVector{Float64}, s::AbstractVector{Float64},
    m::Integer, K::Vector{Clarabel.SupportedCone})
    # Initialize empty diagonal vectors for the diagonal matrices
    D_k_π_diag = zeros(Float64, m)

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
    D_k_π_diag = s .== v

    # Construct and return diagonal matrices from the diagonal vectors
    D_k_π = Diagonal(D_k_π_diag)

    return D_k_π
end

# Recall that pre_x_matrix is often denoted by $W^{-1}$ in my/Paul's notes.
# TODO: consider making operations in this whole "affinisation" process more
# efficient. At the moment it is done in quite a naive fashion for proof of 
# concept.
function local_affine_dynamics!(ws::Workspace)

    # Compute the diagonal flag matrices required.
    D_k_π = flag_arrays(ws.vars.v, ws.vars.s, ws.p.m, ws.p.K)
    D_k_π_neg = Diagonal(1.0 .- D_k_π.diag) # Negation of flag matrix.

    # Define affine dynamics: g(x, v) = tilde_A * (x, v) + tilde_b.
    
    # tlhs = I(n) - pre_x_matrix * (P + ρ * A_gram)
    # blhs = -A * (I(n) - pre_x_matrix * (P + ρ * A_gram))
    
    trhs = ws.cache[:trhs_pre] * (D_k_π_neg - D_k_π)
    # brhs = ρ * A * pre_x_matrix * A' * (2 * D_k_π - I(m)) + I(m) - D_k_π
    brhs = -A * trhs + D_k_π_neg
    
    # Assemble tilde_A.
    ws.tilde_A .= [ws.cache[:tlhs] trhs; ws.cache[:blhs] brhs]

    b_k_π = D_k_π_neg * ws.vars.s
    tilde_b_top = -ws.cache[:trhs_pre] * (2 * b_k_π - ws.p.b) - ws.cache[:W_inv] * ws.p.c
    tilde_b_bot = -ws.p.A * tilde_b_top + ws.p.b - b_k_π

    # println("1-norm of D_k_π_neg is ", norm(D_k_π_neg, 1))
    # println("D_k_π_neg: ", D_k_π_neg)

    # Assemble tilde_b.
    ws.tilde_b .= [tilde_b_top; tilde_b_bot]

    # tilde_b = [-pre_x_matrix * (c + ρ * A' * (D_k_x_b * s - b)); A * pre_x_matrix * (ρ * A' * (2 * b_k_π - b) + c) + b - b_k_π]

    # NOTE: we'd like to compare this local affinisation of the operator with
    # the method's actual operator. Evaluated at the current iterate, the
    # result should be the exact same!

    # println("Rank of tilde_A: ", rank(tilde_A))
    # println("Condition number of tilde_A: ", cond(Matrix(tilde_A)))
    
    return D_k_π.diag
end

function acceleration_candidate(tilde_A::AbstractMatrix{Float64},
    tilde_b::AbstractVector{Float64}, x::AbstractVector{Float64},
    v::AbstractVector{Float64}, n::Integer, m::Integer)
    krylov_iterate = copy([x; v])
    
    # NOTE: may want to set maxiter argument in call to gmres! below.
    gmres!(krylov_iterate, I(n + m) - tilde_A, tilde_b, maxiter = 30)

    # NOTE: the {k+1}th Anderson acceleration iterate is given by the affine 
    # operator of the kth Krylov iterate (see Theorem 2.2 from Walker and Ni,
    # Anderson Acceleration for Fixed-Point Iterations, 2011).

    # println("Rank of (tilde_A - I): ", rank(tilde_A - I(n + m)))
    # println("Condition number of (tilde_A - I): ", cond(Matrix(tilde_A - I(n + m))))
    return tilde_A * krylov_iterate + tilde_b
end