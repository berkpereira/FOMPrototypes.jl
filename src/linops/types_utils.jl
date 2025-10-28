# A function that prepares an inverse operator based on the type of W
function prepare_inv(W::Diagonal{T},
    to::Union{TimerOutput, Nothing}=nothing) where T <: AbstractFloat
    # For a Diagonal matrix, simply compute the reciprocal of the diagonal entries.
    if to !== nothing
        @timeit to "Diagonal inverse" begin 
            inv_diag = 1 ./ Vector(diag(W))
        end
    else
        inv_diag = 1 ./ Vector(diag(W))
    end
    
    return DiagInvOp(inv_diag)
end

function prepare_inv(W::SparseMatrixCSC{T, I},
    to::Union{TimerOutput, Nothing}=nothing; δ::Float64=1e-10) where {T <: AbstractFloat, I <: Integer}
    # For a symmetric positive definite matrix, compute its Cholesky factorization.
    shifts = (nothing, δ, δ * 10)
    F = nothing
    last_error = nothing
    shift_used = zero(T)

    for (idx, shift) in pairs(shifts)
        try
            if to !== nothing
                F = @timeit to "Cholesky factorisation" begin
                    shift === nothing ? SparseArrays.cholesky(W) : SparseArrays.cholesky(W; shift=shift)
                end
            else
                F = shift === nothing ? SparseArrays.cholesky(W) : SparseArrays.cholesky(W; shift=shift)
            end
            break
        catch e
            last_error = e
            if idx == 1
                @warn "Cholesky failed; retrying with δ = $δ" exception=e
                shift_used = δ
            elseif idx == 2
                @warn "Cholesky failed even with shift δ=$δ. Retrying with shift $(δ * 10)" exception=e
                shift_used = δ * 10
            else # give up with error
                rethrow(e)
            end
        end
    end

    if F === nothing
        rethrow(last_error)
    end
    
    if to !== nothing
        @timeit to "other Cholesky prep" begin
            Lmat = sparse(F.L)
            # perm describes permutation matrix P used for pivoting and reduction
            # of fill-in in the sparse Cholesky factors
            # we apply this permutation to a vector v using v[perm]
            perm = F.p

            # need inverse permutation to compute solution to pivoted Cholesky
            # linear system. 
            inv_perm = invperm(perm)
        end
    else
        Lmat = sparse(F.L)
        perm = F.p
        inv_perm = invperm(perm)
    end

    println("Used shift δ = $shift_used for Cholesky factorisation.")

    return CholeskyInvOp(F, Lmat, perm, inv_perm, shift_used)
end

# we also define in-place operators of these preconditioners
function apply_inv!(op::DiagInvOp, x::AbstractArray)
    # Elementwise in-place multiplication: x becomes op.inv_diag .* x.
    # NB matrices get scaled column by column, as expected
    x .*= op.inv_diag
    return nothing
end

# would like in-place version of non-diagonal operator, but this is
# not trivial to do currently.
# standard option is to use op.F \ x, but I want an in-place alternative.
# this has been implemented for a sparse Cholesky factorisation recently,
# (https://github.com/JuliaSparse/SparseArrays.jl/pull/547)
# and will be available in Julia v1.12.
# for the moment we use a standard \ solve which unfortunately allocates
# memory. this is the same as is done
# in COSMO.jl/src/linear_solver/kkt_solver.jl
function apply_inv!(op::CholeskyInvOp, x::Vector{T}, scratch::Vector{T}) where T <: AbstractFloat
    # implementation using custom routines
    sparse_cholmod_solve!(op.Lsp, op.perm, op.inv_perm, x, scratch)

    # cf naive code:
    # x .= op.F \ x

    return nothing
end

"""
In addition to the method apply_inv!(op::CholeskyInvOp, x::Vector{T}) where T <: AbstractFloat,
intended for when x is a Vector, we also have a method for when x is a Matrix
with two columns.
"""
function apply_inv!(op::CholeskyInvOp, x::Matrix{T}, temp_n_vec::Vector{Complex{T}}, perm_scratch::Vector{Complex{T}}) where T <: AbstractFloat
    # implementation using custom routines
    # note that x in this method should have exactly two columns
    sparse_cholmod_solve!(op.Lsp, op.perm, op.inv_perm, x, temp_n_vec, perm_scratch)
    
    return nothing
end
    
