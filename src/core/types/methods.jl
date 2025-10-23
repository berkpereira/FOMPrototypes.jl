# method structs

abstract type AbstractMethod{T <: AbstractFloat, I <: Integer} end

# TODO restate existing code in terms of this new struct in the workspace
struct PrePPM{T <: AbstractFloat, I <: Integer} <: AbstractMethod{T, I}
    variant::Symbol # in {:ADMM, :PDHG, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    ρ::T
    τ::Union{T, Nothing}
    θ::T
    W::AbstractMatrix{T} # all initialised when constructing workspace
    W_inv::AbstractInvOp

    dP::Vector{T} # diagonal of P
    dA::Vector{T} # diagonal of A' * A

    function PrePPM{T, I}(
        variant::Symbol,
        ρ::T,
        τ::Union{T, Nothing},
        θ::T,
        W::AbstractMatrix{T},
        W_inv::AbstractInvOp,
        dP::Vector{T},
        dA::Vector{T},
        ) where {T <: AbstractFloat, I <: Integer}
        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        if ρ <= 0.0
            throw(ArgumentError("ρ must be positive"))
        end

        new{T, I}(variant, ρ, τ, θ, W, W_inv, dP, dA)
    end
end

# TODO incorporate this into code
struct ADMM{T <: AbstractFloat, I <: Integer} <: AbstractMethod{T, I}
    # method-specific fields go here
    rho::T
    # ... any more needed?

    function ADMM{T, I}(rho::T) where {T <: AbstractFloat, I <: Integer}
        new{T, I}(rho)
    end
end