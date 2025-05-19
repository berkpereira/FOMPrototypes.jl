using LinearAlgebra, SparseArrays
import LinearAlgebra:givensAlgorithm

# Forward substitution for L*x = b.
# Writes result to x in place.
# Assumes L is a SparseMatrixCSC{Float64,Int64} that is LOWER triangular.
function forward_solve!(L::SparseMatrixCSC{Float64,Int64}, x::Union{Vector{Float64}, Vector{Complex{Float64}}})
    n = size(L, 1)
    @inbounds for j in 1:n
        # only thing left to do to x[j] is to divide by L[j, j],
        # which is the first entry in column j:
        @inbounds x[j] /= L.nzval[L.colptr[j]]

        # inner loop: subtract contributions from the just-computed x[j]
        # from other rows
        @inbounds xj = x[j] # accumulator to reduce array access
        @inbounds for i in L.colptr[j]+1:L.colptr[j+1]-1
            @inbounds x[L.rowval[i]] -= L.nzval[i] * xj
            # @inbounds x[L.rowval[i]] -= L.nzval[i] * x[j] # without using accumulator
        end
    end

    return nothing
end

# Backward substitution for L'*x = b.
# Writes result to x in place.
# Assumes L is a SparseMatrixCSC{Float64,Int64} that is LOWER triangular.
function backward_solve!(L::SparseMatrixCSC{Float64,Int64}, x::Union{Vector{Float64}, Vector{Complex{Float64}}})
    n = size(L, 1)
    
    # We read the vectors associated with CSC matrix L in reverse.
    # This corresponds to iterating on the rows of L', then on its columns.
    # Outer loop moves up, inner loop moves left. This is why we divide
    # by the diagonal entry AFTER the end of the inner loop, as opposed to
    # what we do in forward_solve!
    @inbounds for j in n:-1:1

        # inner loop, subtract contributions
        a = 0. # accumulator to reduce array access
        @inbounds for i in L.colptr[j+1]-1:-1:L.colptr[j]+1
            @inbounds a += L.nzval[i] * x[L.rowval[i]]
            # x[j] -= L.nzval[i] * x[L.rowval[i]] # without using accumulator
        end
        @inbounds x[j] -= a
        @inbounds x[j] /= L.nzval[L.colptr[j]]
    end

    return nothing
end


"""

This puts custom forward and backward solve routines together to solve a system
A x = b, with A symmetric PD, using a pivoted sparse Cholesky factorisation.

x should initially be set to right-hand side b. It is overwritten in-place
with the solution to the linear system.

Lmat and perm arguments will usually have been obtained from sparse(F.L)
and F.p respectively, where F is
a SparseArrays.CHOLMOD.Factor{Float64,Int64} object.

Outline of the maths is as follows. An object of type
SparseArrays.CHOLMOD.Factor{Float64,Int64} uses row-column pivoting to reduce
fill-in of the factors. Refer to P as a permutation matrix.

The factors computed are P A P^T = L L^T, with L lower-triangular.

We can define z = P x, y = L^T z, and b_perm = P b.

Then we solve the system A x = b with the following steps:\\
A x = b \\
`P A P^T z = b_perm`  -> permute b in-place \\
`L y = b_perm`        -> forward solve for y in-place \\
`L^T z = y`           -> backward solve for z in-place \\
`x = P^T z`           -> inverse permute z in place
"""
function sparse_cholmod_solve!(Lsp::SparseMatrixCSC{Float64, Int64}, perm::Vector{Int64}, inv_perm::Vector{Int64}, x::Union{Vector{Float64}, Vector{Complex{Float64}}})
    permute!(x, perm)
    forward_solve!(Lsp, x)
    backward_solve!(Lsp, x)
    permute!(x, inv_perm)
    
    return nothing
end

function sparse_cholmod_solve!(Lsp::SparseMatrixCSC{Float64, Int64}, perm::Vector{Int64}, inv_perm::Vector{Int64}, x::Matrix{Float64}, temp_n_vec::Vector{Complex{Float64}})
    @assert size(x, 2) == 2 "Custom matrix method only defined for TWO simultaneous right-hand sides."

    @views temp_n_vec .= complex.(x[:, 1], x[:, 2])
    
    permute!(temp_n_vec, perm)
    forward_solve!(Lsp, temp_n_vec)
    backward_solve!(Lsp, temp_n_vec)
    permute!(temp_n_vec, inv_perm)

    @views x[:, 1] .= real.(temp_n_vec)
    @views x[:, 2] .= imag.(temp_n_vec)
    
    return nothing
end

"""
    arnoldi_step!(V, v_new, H)

Perform one Modified Gram-Schmidt orthogonalisation step for Krylov methods.
Orthogonalises the new vector IN-PLACE.

Inputs:
- V::Matrix{T}: Existing tall matrix of orthonormal vectors (pre-allocated, may contain zero columns).
- v_new::Vector{T}: New vector to orthonormalise in-place.
- H::Matrix{T}: Upper Hessenberg matrix (to be updated in-place).

Assumptions:
- V is pre-allocated (with zeros); the leftmost zero column determines where to place the new orthonormal vector.
- H has been pre-allocated with the correct size (e.g. mem+1 by mem).
- Orthogonalisation uses modified Gram-Schmidt (numerically stable).
"""
function arnoldi_step!(V::AbstractMatrix{T},
    v_new::AbstractVector{T},
    H::AbstractMatrix{T}) where T <: AbstractFloat
    # Determine the current column of interest.
    k = findfirst(col -> all(iszero, view(V, :, col)), 1:size(V, 2))
    if isnothing(k)
        k = size(V, 2) - 1
    else
        k -= 1
    end

    # Loop through previous basis vectors to orthogonalise v_new    
    for j in 1:k
        @views Hjk = dot(V[:, j], v_new)
        H[j, k] = Hjk
        
        # Use BLAS.axpy! for efficient in-place subtraction:
        # this does v_new = v_new - Hjk * V[:, j]
        BLAS.axpy!(-Hjk, view(V, :, j), v_new)

        # BAD way apparently:
        # Vj = view(V, :, j)
        # Hjk = dot(Vj, v_new)
        # H[j, k] = Hjk
        
        # @. v_new = v_new - Hjk * Vj
    end

    # Compute the norm of the orthogonalised vector.
    H[k + 1, k] = norm(v_new)

    # Check for breakdown
    if H[k + 1, k] == 0.0
        error("(Happy?) breakdown: orthogonalized vector has zero norm.")
    end

    # Normalize v_new to make it unit length
    v_new ./= H[k + 1, k]

    # Place the orthonormalized vector into the basis matrix V
    V[:, k + 1] .= v_new

    return nothing
end

"""
    krylov_least_squares!(H, rhs_res)

Solve the least-squares problem `min_y ||H * y + rhs_res||^2`
arising in Krylov subspace methods.

Inputs:
- `H`: Upper Hessenberg matrix of size (k+1) × k (modified in-place).
- `rhs_res`: Vector of size (k+1) (modified in-place).

Outputs:
- `y`: Solution vector of size k.
"""
function krylov_least_squares!(H::AbstractMatrix{T}, rhs_res::AbstractVector{T}) where T
    k = size(H, 2)  # Number of columns in H

    # Givens rotation application to reduce H to upper triangular form
    @inbounds for i in 1:k
        # Generate Givens rotation for element (i, i) and (i+1, i)
        # givensAlgorithm generates a plane rotation so that
        # [  c  s  ]  .  [ f ]  =  [ r ]
        # [ -s  c  ]     [ g ]     [ 0 ]
        # (note that this interprets positive rotations as clockwise!)
        c, s, r = givensAlgorithm(H[i, i], H[i+1, i])

        # Apply Givens rotation to the jth and (i+1)th rows of H
        @inbounds for j in i:k
            temp = c * H[i, j] + s * H[i+1, j]
            H[i+1, j] = -s * H[i, j] + c * H[i+1, j]
            H[i, j] = temp
        end

        # Apply Givens rotation to the RHS vector
        temp = c * rhs_res[i] + s * rhs_res[i+1]
        rhs_res[i+1] = -s * rhs_res[i] + c * rhs_res[i+1]
        rhs_res[i] = temp
    end

    # The Givens rotations reduce the problem to a k by k linear system.
    # Now solve the upper triangular system H[1:k, 1:k] * y = -rhs_res[1:k]
    return H[1:k, 1:k] \ (- rhs_res[1:k])
end

"""
Initialises an UpperHessenberg view of a n by (n - 1) matrix filled with zeros.
"""
function init_upper_hessenberg(n::Int)
    # H = spzeros(n + 1, n)
    H = zeros(n, n - 1)
    return UpperHessenberg(H)
end