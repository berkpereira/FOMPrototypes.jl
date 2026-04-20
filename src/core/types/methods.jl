# method structs

abstract type AbstractMethod{T <: AbstractFloat, I <: Integer} end

# TODO restate existing code in terms of this new struct in the workspace
mutable struct PrePPM{T <: AbstractFloat, I <: Integer} <: AbstractMethod{T, I}
    variant::Symbol # in {:ADMM, :PDHG, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    ρ::Vector{T}   # per-constraint penalty vector (length m); uniform when scalar rho is used
    τ::Union{T, Nothing}
    θ::T
    W::AbstractMatrix{T} # all initialised when constructing workspace
    W_inv::AbstractInvOp

    dP::Vector{T} # diagonal of P
    dA::Vector{T} # diagonal of A' * A
    Asq::SparseMatrixCSC{T, I} # element-wise square of A; used for diag(A'*Diagonal(ρ)*A) = Asq'*ρ

    rho_update_count::Ref{Int}

    function PrePPM{T, I}(
        variant::Symbol,
        ρ::Vector{T},
        τ::Union{T, Nothing},
        θ::T,
        W::AbstractMatrix{T},
        W_inv::AbstractInvOp,
        dP::Vector{T},
        dA::Vector{T},
        Asq::SparseMatrixCSC{T, I},
        ) where {T <: AbstractFloat, I <: Integer}
        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        if !all(ρ .> 0.0)
            throw(ArgumentError("all entries of ρ must be positive"))
        end

        new{T, I}(variant, ρ, τ, θ, W, W_inv, dP, dA, Asq, Ref(0))
    end
end

# We now define some types to make the inversion of preconditioner + Hessian
# matrices, required for the x update, abstract. Thus we can use diagonal ones
# (as we intend in production) or non-diagonal symmetric ones for comparing
# with other methods (eg ADMM or vanilla PDHG).

# Concrete type for a diagonal inverse operator
struct DiagInvOp{T} <: AbstractInvOp
    inv_diag::AbstractVector{T}
end

# Concrete type for a symmetric matrix's Cholesky-based inverse operator
struct CholeskyInvOp{T, I} <: AbstractInvOp
    F::SparseArrays.CHOLMOD.Factor{T, I}  # Store the Cholesky factorization
    Lsp::SparseMatrixCSC{T, I} # Store the lower triangular factor
    perm::Vector{I}
    inv_perm::Vector{I}
    shift::T # δ used in Cholesky factorisation (regularisation δI term)
end
