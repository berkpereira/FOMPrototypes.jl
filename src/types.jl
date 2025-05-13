using COSMOAccelerators
using Clarabel
using Parameters
using LinearAlgebra
using LinearAlgebra.LAPACK
import SparseArrays
using LinearMaps

const DefaultFloat = Float64
const DefaultInt = Int64

# Define an abstract type for an inverse linear operator
abstract type AbstractInvOp end

@with_kw struct ProblemData{T, I}
    problem_set::String
    problem_name::String
    P::Symmetric{T}
    c::Vector{T}
    A::SparseMatrixCSC{T, I}
    m::Int
    n::Int
    b::Vector{T}
    K::Vector{Clarabel.SupportedCone}

    # misc vector norms useful for relative KKT error
    b_norm_inf::T
    c_norm_inf::T

    function ProblemData{T, I}(problem_set::String, problem_name::String,
        P::Symmetric{T}, c::Vector{T}, A::SparseMatrixCSC{T, I}, b::Vector{T},
        K::Vector{Clarabel.SupportedCone}) where {T <: AbstractFloat, I <: Integer}
        m, n = size(A)
        
        b_norm_inf = norm(b, Inf)
        c_norm_inf = norm(c, Inf)

        new(problem_set, problem_name, P, c, A, m, n, b, K, b_norm_inf, c_norm_inf)
    end
end
ProblemData(args...) = ProblemData{DefaultFloat, DefaultInt}(args...)

abstract type AbstractVariables{T<:AbstractFloat} end

struct Variables{T <: AbstractFloat} <: AbstractVariables{T}
    # Consolidated (x, y) vector and q vector (for Krylov basis building).
    # (x, y) is exactly as it sounds. q is for building a basis with an
    # Arnoldi-like process simultaneously as we iterate on the (x, y) sequence.
    # Convenient for when we use the linearisation technique
    # for Anderson/Krylov acceleration.
    xy_q::Matrix{T}

    preproj_y::Vector{T} # thing fed to the projection to dual cone, useful to store
    y_qm_bar::Matrix{T} # Extrapolated primal variable

    # default (zeros) initialisation of variables
    function Variables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m, 2), zeros(m), zeros(m, 2))
    end
end
Variables(args...) = Variables{DefaultFloat}(args...)

struct OnecolVariables{T <: AbstractFloat} <: AbstractVariables{T}
    xy::Vector{T}
    preproj_y::Vector{T} # of interest just for recording active set
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function OnecolVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(m))
    end
end
OnecolVariables(args...) = OnecolVariables{DefaultFloat}(args...)

# Type for storing residuals in the workspace
mutable struct ProgressMetrics{T <: AbstractFloat}
    r_primal::Vector{T}
    r_dual::Vector{T}

    obj_primal::T # primal objective value
    obj_dual::T # dual objective value

    rp_abs::T # absolute primal residual metric
    rd_abs::T # absolute dual residual metric
    gap_abs::T # absolute duality gap metric

    rp_rel::T # relative primal residual metric
    rd_rel::T # relative dual residual metric
    gap_rel::T # relative duality gap metric

    # simple NaN constructor for all residual quantities
    function ProgressMetrics{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(fill(NaN, m), fill(NaN, n), NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end
end

abstract type AbstractWorkspace{T<:AbstractFloat, V <: AbstractVariables{T}} end

# workspace type for when :none acceleration is used
struct NoneWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, OnecolVariables{T}}
    k::Base.RefValue{Int} # iter counter
    p::ProblemData{T}
    vars::OnecolVariables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, Symmetric{T}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    function NoneWorkspace{T}(p::ProblemData{T},
        vars::Union{OnecolVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing}) where {T <: AbstractFloat}
        m, n = p.m, p.n
        res = ProgressMetrics{T}(m, n)
        if vars === nothing
            vars = OnecolVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end
        
        @timeit to "W operator" begin
            W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        end

        W_inv = prepare_inv(W, to)
        new{T}(Ref(0), p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m))
    end 
end

# thin wrapper to allow default constructor arguments
function NoneWorkspace(
    p::ProblemData{T},
    variant::Symbol,
    τ::Union{T, Nothing},
    ρ::T,
    θ::T;
    vars::Union{OnecolVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}
    
    # delegate to the inner constructor
    return NoneWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to)
end

NoneWorkspace(args...; kwargs...) = NoneWorkspace{DefaultFloat}(args...; kwargs...)

# workspace type for when Krylov acceleration is used
struct KrylovWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, Variables{T}}
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING unsuccessul Krylov acceleration attempts
    p::ProblemData{T}
    vars::Variables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, Symmetric{T}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    # additional Krylov-related fields
    mem::Int # memory for Krylov acceleration
    krylov_operator::Symbol # either :tilde_A or :B
    H::Matrix{T} # Arnoldi Hessenberg matrix, size (mem+1, mem)
    krylov_basis::Matrix{T} # Krylov basis matrix, size (m + n, mem)
    
    # NOTE that mem is (k+1) in the usual Arnoldi relation written as
    # $ A Q_k = Q_{k+1} \tilde{H}_k $

    # constructor where initial iterates are not passed (default set to zero)
    function KrylovWorkspace{T}(p::ProblemData{T},
        vars::Union{Variables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing},
        mem::Int,
        krylov_operator::Symbol) where {T <: AbstractFloat}
        m, n = p.m, p.n
        res = ProgressMetrics{T}(m, n)
        if vars === nothing
            vars = Variables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end

        @timeit to "W operator" begin
            W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        end

        W_inv = prepare_inv(W, to)
        new{T}(Ref(0), Ref(0), p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, krylov_operator, UpperHessenberg(zeros(mem, mem-1)), zeros(m + n, mem))
    end
end

# thin wrapper to allow default constructor arguments
function KrylovWorkspace(
    p::ProblemData{T},
    variant::Symbol,
    τ::Union{T, Nothing},
    ρ::T,
    θ::T,
    mem::Int,
    krylov_operator::Symbol;
    vars::Union{Variables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}
    
    # delegate to the inner constructor
    return KrylovWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to, mem, krylov_operator)
end

KrylovWorkspace(args...; kwargs...) = KrylovWorkspace{DefaultFloat}(args...; kwargs...)

# workspace type for when Anderson acceleration is used
struct AndersonWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, OnecolVariables{T}}
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING unsuccessul (Anderson) acceleration attempts
    k_vanilla::Base.RefValue{Int} # this counts just vanilla iterations throughout the run --- as expected by COSMOAccelerators functions
    p::ProblemData{T}
    vars::OnecolVariables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, Symmetric{T}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    # additional Anderson acceleration-related fields
    mem::Int # memory for Anderson accelerator
    attempt_period::Int # how often to attempt Anderson acceleration
    accelerator::AndersonAccelerator

    function AndersonWorkspace{T}(
        p::ProblemData{T},
        vars::Union{OnecolVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing},
        mem::Int,
        attempt_period::Int,
        broyden_type::Type{<:COSMOAccelerators.AbstractBroydenType},
        memory_type::Type{<:COSMOAccelerators.AbstractMemory},
        regulariser_type::Type{<:COSMOAccelerators.AbstractRegularizer}) where {T <: AbstractFloat}
        attempt_period >= 2 || throw(ArgumentError("Anderson acceleration attempt period must be at least 2."))
        m, n = p.m, p.n
        res = ProgressMetrics{T}(m, n)

        @timeit to "W operator" begin
            W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        end
        
        W_inv = prepare_inv(W, to)

        if vars === nothing
            vars = OnecolVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end

        # default constructor types:
        # AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}

        aa = AndersonAccelerator{Float64, broyden_type, memory_type, regulariser_type}(m + n, mem = mem)
        new{T}(Ref(0), Ref(0), Ref(0), p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, attempt_period, aa)
    end

end
AndersonWorkspace(args...; kwargs...) = AndersonWorkspace{DefaultFloat}(args...; kwargs...)

# thin wrapper to allow default constructor arguments
function AndersonWorkspace(
    p::ProblemData{T},
    variant::Symbol,
    τ::Union{T, Nothing},
    ρ::T,
    θ::T,
    mem::Int,
    attempt_period::Int;
    vars::Union{OnecolVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    broyden_type::Symbol = :normal2,
    memory_type::Symbol = :rolling,
    regulariser_type::Symbol = :none,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}

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
    
    # delegate to the inner constructor
    return AndersonWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to, mem, attempt_period, broyden_type, memory_type, regulariser_type)
end


mutable struct Results{T <: AbstractFloat}
    metrics_history::Dict{Symbol, Any}
    metrics_final::ProgressMetrics{T}
    exit_status::Symbol
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
end

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

function prepare_inv(W::Symmetric{T},
    to::Union{TimerOutput, Nothing}=nothing; δ::Float64=1e-10) where T <: AbstractFloat
    # For a symmetric positive definite matrix, compute its Cholesky factorization.
    
    if to !== nothing
        @timeit to "Cholesky factorisation" begin
            try
                F = SparseArrays.cholesky(W)
            catch e
                @warn "Cholesky failed; retrying with δ = $δ" exception=e
            end

            # try Cholesky with shift
            try
                F = SparseArrays.cholesky(W; shift=δ)
            catch e
                @warn "Cholesky failed even with shift δ=$δ. Retrying with shift 10δ"
            end

            δ *= 10.;
            F = SparseArrays.cholesky(W; shift=δ)
        end
    else
        try
            F = SparseArrays.cholesky(W)
        catch e
            @warn "Cholesky failed; retrying with δ = $δ" exception=e
        end

        # try Cholesky with shift
        try
            F = SparseArrays.cholesky(W; shift=δ)
        catch e
            @warn "Cholesky failed even with shift δ=$δ. Retrying with shift 10δ"
        end

        δ *= 10.;
        F = SparseArrays.cholesky(W; shift=δ)
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

    return CholeskyInvOp(F, Lmat, perm, inv_perm)
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
function apply_inv!(op::CholeskyInvOp, x::Vector{T}) where T <: AbstractFloat
    # implementation using custom routines
    sparse_cholmod_solve!(op.Lsp, op.perm, op.inv_perm, x)

    # cf naive code:
    # x .= op.F \ x

    return nothing
end

"""
In addition to the method apply_inv!(op::CholeskyInvOp, x::Vector{T}) where T <: AbstractFloat,
intended for when x is a Vector, we also have a method for when x is a Matrix
with two columns.
"""
function apply_inv!(op::CholeskyInvOp, x::Matrix{T}, temp_n_vec::Vector{Complex{T}}) where T <: AbstractFloat
    # implementation using custom routines
    # note that x in this method should have exactly two columns
    sparse_cholmod_solve!(op.Lsp, op.perm, op.inv_perm, x, temp_n_vec)
    
    return nothing
end
    