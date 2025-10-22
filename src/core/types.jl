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
    P::SparseMatrixCSC{T, I}
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
        P::SparseMatrixCSC{T, I}, c::Vector{T}, A::SparseMatrixCSC{T, I}, b::Vector{T},
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
    # Consolidated state vector and q vector (for Krylov basis building).
    # state is exactly as it sounds. q is for building a basis with an
    # Arnoldi-like process simultaneously as we iterate on the state sequence.
    # Convenient for when we use the linearisation technique
    # for Anderson/Krylov acceleration.
    state_q::Matrix{T}
    # to store "previous" working iterate for misc purposes
    state_prev::Vector{T}

    preproj_y::Vector{T} # thing fed to the projection to dual cone, useful to store
    y_qm_bar::Matrix{T} # Extrapolated primal variable


    # default (zeros) initialisation of variables
    function KrylovVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m, 2), zeros(n + m), zeros(m), zeros(m, 2))
    end
end
KrylovVariables(args...) = KrylovVariables{DefaultFloat}(args...)

struct AndersonVariables{T <: AbstractFloat} <: AbstractVariables{T}
    state::Vector{T}
    state_prev::Vector{T}
    preproj_y::Vector{T} # of interest just for recording active set
    
    state_into_accelerator::Vector{T} # working iterate looking back up to anderson-interval iterations, to be passed into the COSMOAccelerators interface
    
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function AndersonVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(m), zeros(n + m))
    end
end
AndersonVariables(args...) = AndersonVariables{DefaultFloat}(args...)

struct VanillaVariables{T <: AbstractFloat} <: AbstractVariables{T}
    state::Vector{T}
    state_prev::Vector{T}
    preproj_y::Vector{T} # of interest just for recording active set
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function VanillaVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(m))
    end
end
VanillaVariables(args...) = VanillaVariables{DefaultFloat}(args...)

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


# types to hold flags aiding control flow in algorithms
abstract type AbstractControlFlags end

mutable struct KrylovControlFlags <: AbstractControlFlags
    recycle_next::Bool
    accepted_accel::Bool
    back_to_building_krylov_basis::Bool
    krylov_status::Symbol
end

# NOTE default init values
KrylovControlFlags() = KrylovControlFlags(false, false, true, :init)

mutable struct AndersonControlFlags <: AbstractControlFlags
    recycle_next::Bool
    accepted_accel::Bool
end

# NOTE default init values
AndersonControlFlags() = AndersonControlFlags(false, false)


# Types for scratch areas/working vectors to avoid heap allocations
abstract type AbstractWorkspaceScratch{T} end

struct VanillaScratch{T} <: AbstractWorkspaceScratch{T}
    temp_n_vec1::Vector{T}
    temp_n_vec2::Vector{T}
    temp_m_vec::Vector{T}
    temp_mn_vec1::Vector{T}
    temp_mn_vec2::Vector{T}
    
    # when using onecol_method_operator!
    swap_vec::Vector{T}
end

function VanillaScratch(p::ProblemData{T}) where {T <: AbstractFloat}
    m, n = p.m, p.n
    VanillaScratch{T}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
    )
end

VanillaScratch(p::ProblemData) = VanillaScratch{DefaultFloat}(p)

struct AndersonScratch{T} <: AbstractWorkspaceScratch{T}
    temp_n_vec1::Vector{T}
    temp_n_vec2::Vector{T}
    temp_m_vec::Vector{T}
    temp_mn_vec1::Vector{T}
    temp_mn_vec2::Vector{T}

    # when using onecol_method_operator!
    swap_vec::Vector{T}

    # acceleration and safeguarding
    accelerated_point::Vector{T}
    # recycled iterate --- to recycle work done when computing fixed-point
    state_recycled::Vector{T}

    state_lookahead::Vector{T}
    fp_res::Vector{T}

    # to check success after potentially being overwritten
    state_pre_overwrite::Vector{T}
end

function AndersonScratch(p::ProblemData{T}) where {T <: AbstractFloat}
    m, n = p.m, p.n
    AndersonScratch{T}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
    )
end

AndersonScratch(p::ProblemData) = AndersonScratch{DefaultFloat}(p)

struct KrylovScratch{T} <: AbstractWorkspaceScratch{T}
    temp_n_vec1::Vector{T}
    temp_n_vec2::Vector{T}
    temp_m_vec::Vector{T}
    temp_mn_vec1::Vector{T}
    temp_mn_vec2::Vector{T}

    temp_n_mat1::Matrix{T}
    temp_n_mat2::Matrix{T}
    temp_m_mat::Matrix{T}
    temp_n_vec_complex1::Vector{Complex{T}}
    temp_n_vec_complex2::Vector{Complex{T}}
    temp_m_vec_complex::Vector{Complex{T}}

    # briefly holds initial iterate
    initial_vec::Vector{T}
    
    accelerated_point::Vector{T}
    # recycled iterate --- to recycle work done when computing fixed-point
    # residuals for acceleration acceptance criteria, then assigned
    # to the working optimisation variable state_q[:, 1] in the next iteration
    state_recycled::Vector{T}
    state_lookahead::Vector{T}
    fp_res::Vector{T}
end

function KrylovScratch(p::ProblemData{T}) where {T <: AbstractFloat}
    m, n = p.m, p.n
    KrylovScratch{T}(
        zeros(T, n),
        zeros(T, n),
        zeros(T, m),
        zeros(T, m + n),
        zeros(T, m + n),

        zeros(T, n, 2),
        zeros(T, n, 2),
        zeros(T, m, 2),
        zeros(Complex{T}, n),
        zeros(Complex{T}, n),
        zeros(Complex{T}, m),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
        zeros(T, m + n),
    )
end

KrylovScratch(p::ProblemData) = KrylovScratch{DefaultFloat}(p)

abstract type AbstractWorkspace{T <: AbstractFloat, I <: Integer, V <: AbstractVariables{T}} end

# workspace type for when :none acceleration is used
struct VanillaWorkspace{T <: AbstractFloat, I <: Integer} <: AbstractWorkspace{T, I, VanillaVariables{T}}
    k::Base.RefValue{Int} # iter counter
    scratch::VanillaScratch{T}
    p::ProblemData{T}
    vars::VanillaVariables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, SparseMatrixCSC{T, I}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    function VanillaWorkspace{T, I}(
        p::ProblemData{T},
        vars::Union{VanillaVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing}) where {T <: AbstractFloat, I <: Integer}
        
        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        m, n = p.m, p.n
        scratch = VanillaScratch(p)
        res = ProgressMetrics{T}(m, n)
        if vars === nothing
            vars = VanillaVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end
        
        @timeit to "W op prep" begin
            W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        end

        W_inv = prepare_inv(W, to)
        new{T, I}(Ref(0), scratch, p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m))
    end 
end

# thin wrapper to allow default constructor arguments
function VanillaWorkspace(
    p::ProblemData{T},
    variant::Symbol,
    τ::Union{T, Nothing},
    ρ::T,
    θ::T;
    vars::Union{VanillaVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}
    
    # delegate to the inner constructor
    return VanillaWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to)
end

VanillaWorkspace(args...; kwargs...) = VanillaWorkspace{DefaultFloat, DefaultInt}(args...; kwargs...)

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
struct KrylovWorkspace{T <: AbstractFloat, I <: Integer} <: AbstractWorkspace{T, I, KrylovVariables{T}}
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING unsuccessul Krylov acceleration attempts
    k_operator::Base.RefValue{Int} # this counts number of operator applications, whether in vanilla iterations, for safeguarding, or acceleration. probably most useful notion of iter count
    scratch::KrylovScratch{T}
    p::ProblemData{T}
    vars::KrylovVariables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, SparseMatrixCSC{T, I}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    
    proj_flags::AbstractVector{Bool}
    control_flags::KrylovControlFlags


    # precomputed diagonals
    # TODO get rid of these when they are not needed
    # depends on safeguard norm used, as well as the variant, but
    # not always needed!
    # see src/utils.jl for more details
    dP::Vector{T} # diagonal of P
    dA::Vector{T} # diagonal of A' * A

    # additional Krylov-related fields
    mem::Int # memory for Krylov acceleration
    tries_per_mem::Int # number of tries per Krylov memory fill-up
    safeguard_norm::Symbol # in {:euclid, :char, :none}, norm used for Krylov acceleration safeguard. if :none, no safeguard is used
    trigger_givens_counts::Vector{Int} # trigger points for attempting Krylov acceleration
    krylov_operator::Symbol # either :tilde_A or :B
    H::Matrix{T} # Arnoldi Hessenberg matrix, size (mem+1, mem)
    krylov_basis::Matrix{T} # Krylov basis matrix, size (m + n, mem)
    givens_rotations::Vector{GivensRotation{T}}
    givens_count::Base.RefValue{Int}
    arnoldi_breakdown::Base.RefValue{Bool}
    fp_found::Base.RefValue{Bool} # used to trigger a termination condition when stuff goes wrong initialising Krylov basis (with NaNs due to a zero fixed-point residual having been found!)
    
    # NOTE that mem is (k+1) in the usual Arnoldi relation written
    # at the point of maximum memory usage as
    # A Q_k = Q_{k+1} \tilde{H}_k
    # ie plainly the number of columns allocated for krylov basis vectors

    # constructor where initial iterates are not passed (default set to zero)
    function KrylovWorkspace{T, I}(
        p::ProblemData{T},
        vars::Union{KrylovVariables{T}, Nothing},
        variant::Symbol,
        A_gram::Union{LinearMap{T}, Nothing},
        τ::Union{T, Nothing},
        ρ::T,
        θ::T,
        to::Union{TimerOutput, Nothing},
        mem::Int,
        tries_per_mem::Int,
        safeguard_norm::Symbol,
        krylov_operator::Symbol) where {T <: AbstractFloat, I <: Integer}

        if (mem - 1) >= p.m + p.n
            throw(ArgumentError("(mem - 1) must be less than m + n."))
        end

        if tries_per_mem > (mem - 1)
            throw(ArgumentError("tries_per_mem must be less than or equal to (mem - 1)."))
        end

        if !(krylov_operator in [:tilde_A, :B])
            throw(ArgumentError("krylov_operator must be one of :tilde_A, :B"))
        end

        if !(safeguard_norm in [:euclid, :char, :none])
            throw(ArgumentError("safeguard_norm must be one of :euclid, :char, :none"))
        end

        scratch = KrylovScratch(p)

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

        # TODO: get rid of duplication of 
        # code generating dP and dA, search in utils
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

        new{T, I}(Ref(0), Ref(0), Ref(0), scratch, p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), KrylovControlFlags(), dP, dA, mem, tries_per_mem, safeguard_norm, trigger_givens_counts, krylov_operator, UpperHessenberg(zeros(mem, mem-1)), zeros(m + n, mem), Vector{GivensRotation{Float64}}(undef, mem-1), Ref(0), Ref(false), Ref(false))
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
    safeguard_norm::Symbol,
    krylov_operator::Symbol;
    vars::Union{KrylovVariables{T}, Nothing} = nothing,
    A_gram::Union{LinearMap{T}, Nothing} = nothing,
    to::Union{TimerOutput, Nothing} = nothing) where {T <: AbstractFloat}
    
    # delegate to the inner constructor
    return KrylovWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to, mem, tries_per_mem, safeguard_norm, krylov_operator)
end
KrylovWorkspace(args...; kwargs...) = KrylovWorkspace{DefaultFloat, DefaultInt}(args...; kwargs...)

# workspace type for when Anderson acceleration is used
struct AndersonWorkspace{T <: AbstractFloat, I <: Integer} <: AbstractWorkspace{T, I, AndersonVariables{T}}
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING UNsuccessul (Anderson) acceleration attempts
    k_vanilla::Base.RefValue{Int} # this counts just vanilla iterations throughout the run --- as expected by COSMOAccelerators functions (just for its logging purposes)
    k_operator::Base.RefValue{Int} # this counts number of operator applications, whether in vanilla iterations, for safeguarding, or acceleration. probably most useful notion of iter count
    composition_counter::Base.RefValue{Int} # counts the number of compositions of the operator, important for monitoring of data passed into COSMOAccelerators functions
    scratch::AndersonScratch{T}
    p::ProblemData{T}
    vars::AndersonVariables{T}
    res::ProgressMetrics{T}
    variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
    W::Union{Diagonal{T}, SparseMatrixCSC{T, I}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T

    proj_flags::AbstractVector{Bool}
    control_flags::AndersonControlFlags

    # precomputed diagonals
    # TODO get rid of these when they are not needed
    # depends on safeguard norm used, as well as the variant, but
    # not always needed!
    # see src/utils.jl for more details
    dP::Vector{T} # diagonal of P
    dA::Vector{T} # diagonal of A' * A

    # additional Anderson acceleration-related fields
    mem::Int # memory for Anderson accelerator
    anderson_interval::Int # how often to attempt Anderson acceleration
    safeguard_norm::Symbol # in {:euclid, :char, :none}
    accelerator::AndersonAccelerator

    function AndersonWorkspace{T, I}(
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
        safeguard_norm::Symbol,
        broyden_type::Type{<:COSMOAccelerators.AbstractBroydenType},
        memory_type::Type{<:COSMOAccelerators.AbstractMemory},
        regulariser_type::Type{<:COSMOAccelerators.AbstractRegularizer},
        anderson_log::Bool) where {T <: AbstractFloat, I <: Integer}

        anderson_interval >= 1 || throw(ArgumentError("Anderson acceleration interval must be at least 1."))

        if !(safeguard_norm in [:euclid, :char, :none])
            throw(ArgumentError("safeguard_norm must be one of :euclid, :char, :none"))
        end

        if θ != 1.0
            throw(ArgumentError("θ ≠ 1.0 is not yet supported"))
        end

        m, n = p.m, p.n
        scratch = AndersonScratch(p)
        res = ProgressMetrics{T}(m, n)

        # TODO: get rid of duplication of 
        # code generating dP and dA, search in utils
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

        if vars === nothing
            vars = AndersonVariables(m, n)
        end
        if A_gram === nothing
            A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        end

        # default constructor types:
        # AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}

        aa = AndersonAccelerator{Float64, broyden_type, memory_type, regulariser_type}(m + n, mem = mem, activate_logging = anderson_log)
        
        new{T, I}(Ref(0), Ref(0), Ref(0), Ref(0), Ref(0), scratch, p, vars, res, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), AndersonControlFlags(), dP, dA, mem, anderson_interval, safeguard_norm, aa)
    end

end
AndersonWorkspace(args...; kwargs...) = AndersonWorkspace{DefaultFloat, DefaultInt}(args...; kwargs...)

# thin wrapper to allow default constructor arguments
function AndersonWorkspace(
    p::ProblemData{T},
    variant::Symbol,
    τ::Union{T, Nothing},
    ρ::T,
    θ::T,
    mem::Int,
    anderson_interval::Int,
    safeguard_norm::Symbol;
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
    return AndersonWorkspace(p, vars, variant, A_gram, τ, ρ, θ, to, mem, anderson_interval, safeguard_norm, broyden_type, memory_type, regulariser_type, anderson_log)
end


mutable struct Results{T <: AbstractFloat, I <: Integer}
    metrics_history::Dict{Symbol, Any}
    metrics_final::ReturnMetrics{T}
    exit_status::Symbol
    k_final::I
    k_operator_final::I
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

# Struct just for diagnostics data
struct DiagnosticsWorkspace{T <: AbstractFloat}
    tilde_A::AbstractMatrix{T}
    tilde_b::Vector{T}
    W_inv_mat::AbstractMatrix{T}

    # only relevant for Krylov variants
    H_unmod::Union{UpperHessenberg{T}, Nothing}
end

function DiagnosticsWorkspace(ws::AbstractWorkspace{T}) where T <: AbstractFloat
    m, n = ws.p.m, ws.p.n
    tilde_A = zeros(T, m + n, m + n)
    tilde_b = zeros(T, m + n)
    if ws isa KrylovWorkspace
        H_unmod = UpperHessenberg(zeros(ws.mem, ws.mem - 1))
    else
        H_unmod = nothing
    end
    
    # form dense identity
    dense_I = Matrix{Float64}(I, ws.p.n, ws.p.n)

    # form matrix inverse of W (= P + M_1)
    if ws.W_inv isa CholeskyInvOp
        W_inv_mat = ws.W_inv.F \ dense_I
        W_inv_mat = Symmetric(W_inv_mat)
    elseif ws.W_inv isa DiagInvOp
        W_inv_mat = Diagonal(ws.W_inv.inv_diag)
    end

    DiagnosticsWorkspace{T}(tilde_A, tilde_b, W_inv_mat, H_unmod)
end
DiagnosticsWorkspace(args...; kwargs...) = DiagnosticsWorkspace{DefaultFloat}(args...; kwargs...)