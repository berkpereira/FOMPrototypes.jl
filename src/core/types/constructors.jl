# --------------------------------------------------------------------
# Default constants
const DEFAULT_RESIDUAL_PERIOD::Int = 25

# Helper functions to build components via dispatch ---
# --------------------------------------------------------------------

# 1. Helpers for METHOD-SPECIFIC VECTORS (dispatches on Method type)
function _build_method_vecs(p::ProblemData{T}, ::Type{<:PrePPM}) where {T}
    return PrePPMScratchVecs{T}(p)
end

function _build_method_vecs(p::ProblemData{T}, ::Type{<:ADMM}) where {T}
    return ADMMScratchVecs{T}(p)
end

# 2. Helpers for METHOD-SPECIFIC MATRICES (dispatches on Method type)
function _build_method_mats(p::ProblemData{T}, ::Type{<:PrePPM}) where {T}
    return PrePPMScratchMats{T}(p)
end

function _build_method_mats(p::ProblemData{T}, ::Type{<:ADMM}) where {T}
    return ADMMScratchMats{T}(p)
end

# -----------------------------------------------------------------------
# Main "Assembler" functions (dispatches on Workspace type) ---
# -----------------------------------------------------------------------

"""
    build_scratch(p::ProblemData, ::Type{W}, ::Type{M})

Builds the complete scratch workspace struct by dispatching
on the workspace type `W` and method type `M`.
"""
function build_scratch(
    p::ProblemData{T},
    ::Type{<:VanillaWorkspace}, # Dispatch on Vanilla
    ::Type{M}
) where {T, M <: AbstractMethod}
    
    # 1. Build common components
    base = BaseScratch{T}(p)
    
    # 2. Build method-specific component by calling helper
    method_vecs = _build_method_vecs(p, M)
    
    # 3. Build workspace-specific component
    extra = VanillaScratchExtra{T}(p)

    # 4. Assemble and return
    return VanillaScratch(base, method_vecs, extra)
end


function build_scratch(
    p::ProblemData{T},
    ::Type{<:AndersonWorkspace}, # Dispatch on Anderson
    ::Type{M}
) where {T, M <: AbstractMethod}

    base = BaseScratch{T}(p)
    method_vecs = _build_method_vecs(p, M)
    extra = AndersonScratchExtra{T}(p)

    return AndersonScratch(base, method_vecs, extra)
end


function build_scratch(
    p::ProblemData{T},
    ::Type{<:KrylovWorkspace}, # Dispatch on Krylov
    ::Type{M}
) where {T, M <: AbstractMethod}

    base = BaseScratch{T}(p)
    method_vecs = _build_method_vecs(p, M)
    extra = KrylovScratchExtra{T}(p)

    # --- Krylov-specific step ---
    # Also build the method matrices
    method_mats = _build_method_mats(p, M)
    
    # Note the different field names in the final struct
    return KrylovScratch(base, method_vecs, method_mats, extra)
end


# Outer constructors of workspace objects
# dispatch on method type to construct method alongside
# just before workspace
function VanillaWorkspace{T, I}(
    p::ProblemData{T},
    ::Type{PrePPM}, # note dispatch on PrePPM type
    variant::Symbol, # in {:ADMM, :PDHG, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    τ::Union{T, Nothing},
    ρ::T,
    θ::T;
    residual_period::I = DEFAULT_RESIDUAL_PERIOD,
    vars::Union{VanillaVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing
    ) where {T <: AbstractFloat, I <: Integer}

    # TODO simplify syntax of call to W_operator using method struct
    @timeit to "W op prep" begin
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
    end

    W_inv = prepare_inv(W, to)

    @timeit to "dP prep" begin
        dP = Vector(diag(p.P))
    end
    @timeit to "dA prep" begin
        dA = vec(sum(abs2, p.A; dims=1))
    end

    # construct method object
    method = PrePPM{T, I}(variant, ρ, τ, θ, W, W_inv, dP, dA)

    scratch = build_scratch(p, VanillaWorkspace, PrePPM)

    # delegate to the inner constructor
    return VanillaWorkspace{T, I, PrePPM{T, I}}(p, method, residual_period, scratch, vars, A_gram)
end

VanillaWorkspace(args...; kwargs...) = VanillaWorkspace{DefaultFloat, DefaultInt}(args...; kwargs...)

# dispatch on method type to construct method alongside
# just before workspace
function KrylovWorkspace{T, I}(
    p::ProblemData{T},
    ::Type{PrePPM}, # note dispatch on PrePPM type
    variant::Symbol, # in {:ADMM, :PDHG, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    τ::Union{T, Nothing},
    ρ::T,
    θ::T,
    mem::Int,
    tries_per_mem::Int,
    safeguard_norm::Symbol,
    krylov_operator::Symbol;
    residual_period::I = DEFAULT_RESIDUAL_PERIOD,
    vars::Union{KrylovVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing
    ) where {T <: AbstractFloat, I <: Integer}

    # TODO simplify syntax of call to W_operator using method struct
    @timeit to "W op prep" begin
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
    end

    W_inv = prepare_inv(W, to)

    @timeit to "dP prep" begin
        dP = Vector(diag(p.P))
    end

    @timeit to "dA prep" begin
        dA = vec(sum(abs2, p.A; dims=1))
    end

    method = PrePPM{T, I}(variant, ρ, τ, θ, W, W_inv, dP, dA)

    scratch = build_scratch(p, KrylovWorkspace, PrePPM)
    
    # delegate to the inner constructor
    return KrylovWorkspace{T, I, PrePPM{T, I}}(
        p,
        method,
        residual_period,
        scratch,
        vars,
        A_gram,
        mem,
        tries_per_mem,
        safeguard_norm,
        krylov_operator
    )
end
KrylovWorkspace(args...; kwargs...) = KrylovWorkspace{DefaultFloat, DefaultInt}(args...; kwargs...)

# dispatch on method type to construct method alongside
# just before workspace
function AndersonWorkspace{T, I}(
    p::ProblemData{T},
    ::Type{PrePPM}, # note dispatch on PrePPM type
    variant::Symbol,
    τ::Union{T, Nothing},
    ρ::T,
    θ::T,
    mem::Int,
    anderson_interval::Int,
    safeguard_norm::Symbol;
    residual_period::I = DEFAULT_RESIDUAL_PERIOD,
    vars::Union{AndersonVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    broyden_type::Symbol = :normal2,
    memory_type::Symbol = :rolling,
    regulariser_type::Symbol = :none,
    anderson_log::Bool = false,
    to::Union{TimerOutput, Nothing} = nothing
    ) where {T <: AbstractFloat, I <: Integer}

    if broyden_type == :normal2
        broyden_type = Type2{NormalEquations}
    elseif broyden_type == :QR2
        broyden_type = Type2{QRDecomp}
    elseif broyden_type == Symbol(1)
        broyden_type = Type1
    else
        throw(ArgumentError("Unknown Broyden type: $broyden_type."))
    end

    if memory_type == :rolling
        memory_type = RollingMemory
    elseif memory_type == :restarted
        memory_type = RestartedMemory
    else
        throw(ArgumentError("Unknown memory type: $memory_type."))
    end

    if regulariser_type == :none
        regulariser_type = NoRegularizer
    elseif regulariser_type == :tikonov
        regulariser_type = TikonovRegularizer
    elseif regulariser_type == :frobenius
        regulariser_type = FrobeniusNormRegularizer
    else
        throw(ArgumentError("Unknown regulariser type: $regulariser_type."))
    end

    # TODO simplify syntax of call to W_operator using method struct
    @timeit to "W op prep" begin
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
    end

    W_inv = prepare_inv(W, to)

    @timeit to "dP prep" begin
        dP = Vector(diag(p.P))
    end

    @timeit to "dA prep" begin
        dA = vec(sum(abs2, p.A; dims=1))
    end

    method = PrePPM{T, I}(variant, ρ, τ, θ, W, W_inv, dP, dA)

    scratch = build_scratch(p, AndersonWorkspace, PrePPM)
    
    # delegate to the inner constructor
    return AndersonWorkspace{T, I, PrePPM{T, I}}(
        p,
        method,
        residual_period,
        scratch,
        vars,
        A_gram,
        mem,
        anderson_interval,
        safeguard_norm,
        broyden_type,
        memory_type,
        regulariser_type,
        anderson_log
        )
end

AndersonWorkspace(args...; kwargs...) = AndersonWorkspace{DefaultFloat, DefaultInt}(args...; kwargs...)