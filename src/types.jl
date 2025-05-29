using COSMOAccelerators
using Clarabel
using Parameters
using LinearAlgebra
import LinearAlgebra:givensAlgorithm
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

struct KrylovVariables{T <: AbstractFloat} <: AbstractVariables{T}
    # Consolidated (x, y) vector and q vector (for Krylov basis building).
    # (x, y) is exactly as it sounds. q is for building a basis with an
    # Arnoldi-like process simultaneously as we iterate on the (x, y) sequence.
    # Convenient for when we use the linearisation technique
    # for Anderson/Krylov acceleration.
    xy_q::Matrix{T}

    preproj_y::Vector{T} # thing fed to the projection to dual cone, useful to store
    y_qm_bar::Matrix{T} # Extrapolated primal variable

    # recycled iterate --- to recycle work done when computing fixed-point
    # residuals for acceleration acceptance criteria, then assigned
    # to the working optimisation variable xy_q[:, 1] in the next iteration
    xy_recycled::Vector{T}

    # to store "previous" working iterate for misc purposes
    xy_prev::Vector{T}

    # default (zeros) initialisation of variables
    function KrylovVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m, 2), zeros(m), zeros(m, 2), zeros(n + m), zeros(n + m))
    end
end
KrylovVariables(args...) = KrylovVariables{DefaultFloat}(args...)

struct AndersonVariables{T <: AbstractFloat} <: AbstractVariables{T}
    xy::Vector{T}
    xy_prev::Vector{T}
    xy_into_accelerator::Vector{T} # working iterate looking back up to anderson-interval iterations, to be passed into the COSMOAccelerators interface
    preproj_y::Vector{T} # of interest just for recording active set
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function AndersonVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(n + m), zeros(m))
    end
end
AndersonVariables(args...) = AndersonVariables{DefaultFloat}(args...)

struct NoneVariables{T <: AbstractFloat} <: AbstractVariables{T}
    xy::Vector{T}
    xy_prev::Vector{T}
    preproj_y::Vector{T} # of interest just for recording active set
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function NoneVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(m))
    end
end
NoneVariables(args...) = NoneVariables{DefaultFloat}(args...)

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

# this struct is a simpler version of ProgressMetrics, not
# storing the residual vectors --- this is for further
# processing purposes
mutable struct ReturnMetrics{T <: AbstractFloat}
    obj_primal::T # primal objective value
    obj_dual::T # dual objective value

    rp_abs::T # absolute primal residual metric
    rd_abs::T # absolute dual residual metric
    gap_abs::T # absolute duality gap metric

    rp_rel::T # relative primal residual metric
    rd_rel::T # relative dual residual metric
    gap_rel::T # relative duality gap metric
end

# outer constructor to create a ReturnMetrics object
# from an existing ProgressMetrics object
ReturnMetrics(pm::ProgressMetrics{T}) where {T<:AbstractFloat} = 
    ReturnMetrics(
        pm.obj_primal,
        pm.obj_dual,
        pm.rp_abs,
        pm.rd_abs,
        pm.gap_abs,
        pm.rp_rel,
        pm.rd_rel,
        pm.gap_rel,
       )

abstract type AbstractWorkspace{T<:AbstractFloat, V <: AbstractVariables{T}} end

# workspace type for when :none acceleration is used
struct NoneWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, NoneVariables{T}}
    k::Base.RefValue{Int} # iter counter
    p::ProblemData{T}
    vars::NoneVariables{T}
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
        vars::Union{NoneVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing}) where {T <: AbstractFloat}
        
        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        m, n = p.m, p.n
        res = ProgressMetrics{T}(m, n)
        if vars === nothing
            vars = NoneVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end
        
        @timeit to "W op prep" begin
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
    vars::Union{NoneVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}
    
    # delegate to the inner constructor
    return NoneWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to)
end

NoneWorkspace(args...; kwargs...) = NoneWorkspace{DefaultFloat}(args...; kwargs...)

# type used to store Givens rotation
# LinearAlgebra.givensAlgorithm is derived from LAPACK's dlartg
# (netlib.org/lapack/explore-html/da/dd3/group__lartg_ga86f8f877eaea0386cdc2c3c175d9ea88.html)
# givensAlgorithm generates a plane rotation so that
# [  c  s  ]  .  [ f ]  =  [ r ]
# [ -s  c  ]     [ g ]     [ 0 ]
# (note that this interprets positive rotations as clockwise!)
struct GivensRotation{T <: AbstractFloat}
    c::T
    s::T
end

# workspace type for when Krylov acceleration is used
struct KrylovWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, KrylovVariables{T}}
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING unsuccessul Krylov acceleration attempts
    p::ProblemData{T}
    vars::KrylovVariables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, Symmetric{T}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    # precomputed diagonals
    dP::Vector{T} # diagonal of P
    dA::Vector{T} # diagonal of A' * A

    # additional Krylov-related fields
    mem::Int # memory for Krylov acceleration
    tries_per_mem::Int # number of tries per Krylov memory fill-up
    trigger_givens_counts::Vector{Int} # trigger points for attempting Krylov acceleration
    krylov_operator::Symbol # either :tilde_A or :B
    H::Matrix{T} # Arnoldi Hessenberg matrix, size (mem+1, mem)
    krylov_basis::Matrix{T} # Krylov basis matrix, size (m + n, mem)
    givens_rotations::Vector{GivensRotation{T}}
    givens_count::Base.RefValue{Int}
    
    # NOTE that mem is (k+1) in the usual Arnoldi relation written
    # at the point of maximum memory usage as
    # A Q_k = Q_{k+1} \tilde{H}_k
    # ie plainly the number of columns allocated for krylov basis vectors

    # constructor where initial iterates are not passed (default set to zero)
    function KrylovWorkspace{T}(p::ProblemData{T},
        vars::Union{KrylovVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing},
        mem::Int,
        tries_per_mem::Int,
        krylov_operator::Symbol) where {T <: AbstractFloat}

        # make vector of trigger_givens_counts
        trigger_givens_counts = Vector{Int}(undef, tries_per_mem)
        for i in eachindex(trigger_givens_counts)
            trigger_givens_counts[i] = Int(floor(i * (mem - 1) / tries_per_mem))
        end

        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        m, n = p.m, p.n
        res = ProgressMetrics{T}(m, n)
        if vars === nothing
            vars = KrylovVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end

        @timeit to "dP prep" begin
            dP = Vector(diag(p.P))
        end
        @timeit to "dA prep" begin
            dA = vec(sum(abs2, p.A; dims=1))
        end
        @timeit to "W op prep" begin
            W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        end

        W_inv = prepare_inv(W, to)

        println(typeof(dP))

        new{T}(Ref(0), Ref(0), p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), dP, dA, mem, tries_per_mem, trigger_givens_counts, krylov_operator, UpperHessenberg(zeros(mem, mem-1)), zeros(m + n, mem), Vector{GivensRotation{Float64}}(undef, mem-1), Ref(0))
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
    tries_per_mem::Int,
    krylov_operator::Symbol;
    vars::Union{KrylovVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}
    
    # delegate to the inner constructor
    return KrylovWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to, mem, tries_per_mem, krylov_operator)
end

KrylovWorkspace(args...; kwargs...) = KrylovWorkspace{DefaultFloat}(args...; kwargs...)

# workspace type for when Anderson acceleration is used
struct AndersonWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, AndersonVariables{T}}
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING unsuccessul (Anderson) acceleration attempts
    k_vanilla::Base.RefValue{Int} # this counts just vanilla iterations throughout the run --- as expected by COSMOAccelerators functions (just for its logging purposes)
    composition_counter::Base.RefValue{Int} # counts the number of compositions of the operator, important for monitoring of data passed into COSMOAccelerators functions
    p::ProblemData{T}
    vars::AndersonVariables{T}
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
    anderson_interval::Int # how often to attempt Anderson acceleration
    accelerator::AndersonAccelerator

    function AndersonWorkspace{T}(
        p::ProblemData{T},
        vars::Union{AndersonVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing},
        mem::Int,
        anderson_interval::Int,
        broyden_type::Type{<:COSMOAccelerators.AbstractBroydenType},
        memory_type::Type{<:COSMOAccelerators.AbstractMemory},
        regulariser_type::Type{<:COSMOAccelerators.AbstractRegularizer},
        anderson_log::Bool) where {T <: AbstractFloat}
        anderson_interval >= 1 || throw(ArgumentError("Anderson acceleration interval must be at least 1."))

        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        m, n = p.m, p.n
        res = ProgressMetrics{T}(m, n)

        @timeit to "W op prep" begin
            W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        end
        
        W_inv = prepare_inv(W, to)

        if vars === nothing
            vars = AndersonVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end

        # default constructor types:
        # AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}

        aa = AndersonAccelerator{Float64, broyden_type, memory_type, regulariser_type}(m + n, mem = mem, activate_logging = anderson_log)
        
        new{T}(Ref(0), Ref(0), Ref(0), Ref(0), p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, anderson_interval, aa)
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
    anderson_interval::Int;
    vars::Union{AndersonVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    broyden_type::Symbol = :normal2,
    memory_type::Symbol = :rolling,
    regulariser_type::Symbol = :none,
    anderson_log::Bool = false,
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
    return AndersonWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to, mem, anderson_interval, broyden_type, memory_type, regulariser_type, anderson_log)
end


mutable struct Results{T <: AbstractFloat}
    metrics_history::Dict{Symbol, Any}
    metrics_final::ReturnMetrics{T}
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