using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Clarabel

function diag_flag_matrices(v::AbstractVector{Float64}, s::AbstractVector{Float64},
    m::Integer, K::Vector{Clarabel.SupportedCone})
    # Initialize empty diagonal vectors for the diagonal matrices
    D_k_v_diag = zeros(Float64, m)
    D_k_x_diag = zeros(Float64, m)
    D_k_x_b_diag = zeros(Float64, m)

    # Start partition indexing
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1

        # Extract partitions of s and v for the current block
        s_block = s[start_idx:end_idx]
        v_block = v[start_idx:end_idx]

        # Check whether the blocks of s and v are equal
        if s_block == v_block
            # Fill diagonal entries for the case where s == v
            D_k_v_diag[start_idx:end_idx] .= 0.0
            D_k_x_diag[start_idx:end_idx] .= 1.0
            D_k_x_b_diag[start_idx:end_idx] .= 0.0
        else
            # Fill diagonal entries for the case where s != v
            D_k_v_diag[start_idx:end_idx] .= 1.0
            D_k_x_diag[start_idx:end_idx] .= -1.0
            D_k_x_b_diag[start_idx:end_idx] .= 2.0
        end

        # Update the start index for the next block
        start_idx = end_idx + 1
    end

    # Construct and return diagonal matrices from the diagonal vectors
    D_k_v = Diagonal(D_k_v_diag)
    D_k_x = Diagonal(D_k_x_diag)
    D_k_x_b = Diagonal(D_k_x_b_diag)

    return D_k_v, D_k_x, D_k_x_b
    
end

# Recall that pre_x_matrix is often denoted by $W^{-1}$ in my/Paul's notes.
# TODO: consider making operations in this whole "affinisation" process more
# efficient. At the moment it is done in quite a naive fashion for proof of 
# concept. 
function local_affine_dynamics(P::AbstractMatrix{Float64},
    A::AbstractMatrix{Float64}, A_gram::AbstractMatrix{Float64},
    b::AbstractVector{Float64}, pre_x_matrix::AbstractMatrix{Float64},
    s::AbstractVector{Float64}, v::AbstractVector{Float64}, ρ::Float64,
    n::Integer, m::Integer, K::Vector{Clarabel.SupportedCone})

    # Compute the diagonal flag matrices required.
    D_k_v, D_k_x, D_k_x_b = diag_flag_matrices(v, s, m, K)

    # Define affine dynamics: g(x, v) = tilde_A * (x, v) + tilde_b.
    tilde_A = [(I(n) - pre_x_matrix * (P + ρ * A_gram)) -ρ * pre_x_matrix * A' * D_k_x; -A D_k_v]
    tilde_b = [-ρ * pre_x_matrix * A' * (D_k_x_b * s); b - D_k_v * s]

    return tilde_A, tilde_b, v
end

function acceleration_candidate(tilde_A::AbstractMatrix{Float64},
    tilde_b::AbstractVector{Float64}, x::AbstractVector{Float64},
    v::AbstractVector{Float64}, n::Integer, m::Integer)
    krylov_iterate = copy([x; v])
    
    # NOTE: may want to set maxiter argument in call to minres! below.
    gmres!(krylov_iterate, I(n + m) - tilde_A, tilde_b)

    # NOTE: the {k+1}th iterate is given by the affine operator of the kth
    # Krylov iterate (see Theorem 2.2 from Walker and Ni, Anderson Acceleration 
    # for Fixed-Point Iterations, 2011).
    return tilde_A * krylov_iterate + tilde_b
end