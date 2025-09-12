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
function arnoldi_step!(
    V::AbstractMatrix{T},
    v_new::AbstractVector{T},
    H::AbstractMatrix{T},
    rot_count::Base.Ref{Int}) where T <: AbstractFloat
    # Determine the current column of interest.
    k = rot_count[] + 1

    mn = size(V, 1) # number of rows in V

    # Loop through previous basis vectors to orthogonalise v_new    
    # init_idx = max(1, k-4)
    init_idx = 1 # this is the canonical choice, of course
    
    @inbounds for j in init_idx:k
        @views Hjk = dot(V[:, j], v_new)
        H[j, k] = Hjk
        
        # Use BLAS.axpy! for efficient in-place subtraction:
        # this does v_new = v_new - Hjk * V[:, j]
        # THIS IS QUICK:
        BLAS.axpy!(-Hjk, view(V, :, j), v_new)

        # THIS IS SLOW
        # @views v_new .= v_new - Hjk * V[:, j]

        # BAD way apparently:
        # Vj = view(V, :, j)
        # Hjk = dot(Vj, v_new)
        # H[j, k] = Hjk
        
        # @. v_new = v_new - Hjk * Vj
    end
    # Compute the norm of the orthogonalised vector.
    @inbounds H[k + 1, k] = norm(v_new)

    # Check for breakdown
    @inbounds if H[k + 1, k] <= 1e-12
        @info "(Happy?) Arnoldi breakdown: orthogonalized vector in Arnoldi is approx zero."

        @inbounds H[k + 1, k] = 0.0 # set to clean zero

        # return true to indicate breakdown
        return true
    end

    # Normalize v_new to make it unit length
    # s = H[k + 1, k]
    # @inbounds for i in eachindex(v_new)
    #     v_new[i] /= s
    # end
    
    # normalise v_new
    # @inbounds v_new /= H[k + 1, k] # or just this as opposed to @inbounds loop?

    # normalise v_new
    @inbounds s = 1 / H[k + 1, k]
    BLAS.scal!(s, v_new) # this does v_new = s * v_new

    # Place the orthonormalized vector into the basis matrix V
    @inbounds V[:, k + 1] .= v_new # simple way to do it
    
    # lower-level call to place orthonormalized vector into V
    # TODO if it works well, put it into a helper function fast_store_column! ?
    # @inbounds begin
    #     ptr_src = pointer(v_new)
    #     ptr_dest = pointer(V, k * mn + 1)   # pointer to the start of column (k+1)
    #     BLAS.blascopy!(mn, ptr_src, 1, ptr_dest, 1)
    # end

    # no breakdown, hence return false
    return false
end

"""
    apply_existing_rotations_new_col!(H, Gs, rot_count, col)

Apply the first `rot_count` Givens rotations in `Gs` to column `col` of `H`.
"""
function apply_existing_rotations_new_col!(H::AbstractMatrix{T}, Gs::Vector{GivensRotation{T}}, rot_count::Base.Ref{Int}) where T
    # TODO consider iterator here to avoid pointer arithmetic at every
    # iteration?
    col = rot_count[] + 1
    @inbounds for i in 1:col-1
        G = Gs[i]
        # apply rotation to rows i, i+1 of column rot_count[]
        tmp =  G.c * H[i, col] + G.s * H[i+1, col]
        H[i+1, col] = -G.s * H[i, col] + G.c * H[i+1, col]
        H[i, col] = tmp
    end
end

"""
    generate_store_apply_rotation!(H, Gs, rot_count, col)

Compute the Givens rotation that zeroes H[col+1, col], store it in Gs[rot_count+1],
and apply it to rows (col, col+1) of H[:, col:end].
Returns new rot_count = old rot_count + 1.
"""
function generate_store_apply_rotation!(H::AbstractMatrix{T}, Gs::Vector{GivensRotation{T}}, rot_count::Base.Ref{Int}) where T
    # compute rotation from H[col,col] and H[col+1,col]
    col = rot_count[] + 1
    c, s, r = givensAlgorithm(H[col, col], H[col+1, col])
    Gs[col] = GivensRotation{T}(c, s) # store

    # apply newest rotation to the NEW column only --- other rows/columns
    # would lead to us operating on zeros
    # NOTE that this requires us to carry out this rotation before we have
    # populated with any new non-zero columns in the pre-allocated H matrix,
    # ie do this EVERY ITERATION
    @inbounds begin
        tmp = c * H[col, col] + s * H[col+1, col]
        H[col+1, col] = -s * H[col, col] + c * H[col+1, col]
        H[col,   col] = tmp
    end

    # increment rotation counter after we have generated the new rotation
    rot_count[] += 1
    return nothing
end

"""
    apply_rotations_to_krylov_rhs!(rhs_res, Gs, rot_count)

Re-apply the first `rot_count` Givens rotations to the vector `rhs_res`
in place.

rhs_res should be a vector of size rot_count[] + 1, to which the first
rot_count[] rotations in Gs are applied.
"""
function apply_rotations_to_krylov_rhs!(
    rhs_res::AbstractVector{T},
    Gs::Vector{GivensRotation{T}},
    rot_count::Int) where T
    @inbounds for i in 1:rot_count
        G = Gs[i]
        tmp =  G.c * rhs_res[i] + G.s * rhs_res[i+1]
        rhs_res[i+1] = -G.s * rhs_res[i] + G.c * rhs_res[i+1]
        rhs_res[i]   = tmp
    end
end

"""
    solve_current_least_squares!(H, Gs, rot_count, rhs_res)

Assumes H has been upper-triangularised in its first `rot_count` cols/rows.
Solves the kxk system
    H[1:rot_count[], 1:rot_count[]] * y = -rhs_res[1:rot_count[]]
in place --- solution is written to rhs_res[1:rot_count[]].
"""
function solve_current_least_squares!(H::AbstractMatrix{T}, Gs::Vector{GivensRotation{T}}, rot_count::Base.Ref{Int}, rhs_res::AbstractVector{T}, arnoldi_breakdown::Bool) where T
    # apply the stored Givens rotations to the rhs_res vector
    if !arnoldi_breakdown
        apply_rotations_to_krylov_rhs!(rhs_res, Gs, rot_count[])
    else
        # if we had Arnoldi breakdown, really we ignore what would usually
        # have been the final Givens rotation here, since the breakdown
        # gave us an upper triangular subblock of H even without the use
        # of a final such rotation
        apply_rotations_to_krylov_rhs!(rhs_res, Gs, rot_count[] - 1)
    end
    
    # use UpperTriangular wrapper to exploit triangular structure
    # in ldiv!
    R = UpperTriangular(view(H, 1:rot_count[], 1:rot_count[]))

    @inbounds rhs_res[1:rot_count[]] .*= -1.0 # negate the rhs_res vector

    # this ought to be triggered after an Arnoldi breakdown
    # where we have found a 1-eigenvector of tilde_A (ie null vector of B = tilde_A - I)
    @inbounds if abs(R[rot_count[], rot_count[]]) < 10 * eps(T)
        return :B_nullvec
    end
    
    @views ldiv!(R, rhs_res[1:rot_count[]])
    
    # naive solution code:
    # @views lls_sol = H[1:rot_count[], 1:rot_count[]] \ (-rhs_res[1:rot_count[]])
    # return lls_sol
    
    return :success
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