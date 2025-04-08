import COSMOAccelerators
using Clarabel
using Parameters
using LinearAlgebra
using LinearAlgebra.LAPACK
import SparseArrays
using LinearMaps
using Infiltrator

const DefaultFloat = Float64

# Define an abstract type for an inverse linear operator
abstract type AbstractInvOp end

@with_kw struct ProblemData{T}
    P::AbstractMatrix{T}
    c::AbstractVector{T}
    A::AbstractMatrix{T}
    m::Int
    n::Int
    b::AbstractVector{T}
    K::Vector{Clarabel.SupportedCone}
    K_star::Vector{Clarabel.SupportedCone}

    function ProblemData{T}(P::AbstractMatrix{T}, c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, K::Vector{Clarabel.SupportedCone}) where {T <: AbstractFloat}
        m, n = size(A)
        new(P, c, A, m, n, b, K)
    end
end
ProblemData(args...) = ProblemData{DefaultFloat}(args...)

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
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function OnecolVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m))
    end
end
OnecolVariables(args...) = OnecolVariables{DefaultFloat}(args...)

abstract type AbstractWorkspace{T<:AbstractFloat, V <: AbstractVariables{T}} end

# workspace type for when :none acceleration is used
struct NoneWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, OnecolVariables{T}}
    p::ProblemData{T}
    vars::OnecolVariables{T}
    variant::Union{Int, Symbol} # In {:PDHG, :ADMM, 1, 2, 3, 4}.
    W::Union{Diagonal{T}, Symmetric{T}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    # constructor where initial iterates are passed in
    function NoneWorkspace{T}(p::ProblemData{T}, vars::OnecolVariables{T}, variant::Union{Int, Symbol}, τ::Union{T, Nothing}, ρ::T, θ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)
        new{T}(p, vars, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m))
    end

    # constructor where initial iterates are not passed (default set to zero)
    function NoneWorkspace{T}(p::ProblemData{T}, variant::Union{Int, Symbol}, τ::Union{T, Nothing}, ρ::T, θ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)
        new{T}(p, OnecolVariables(m, n), variant, W, W_inv, A_gram, τ, ρ, θ, falses(m))
    end

    # constructor where initial iterates are not passed (default set to zero)
    # but A_gram is
    function NoneWorkspace{T}(p::ProblemData{T}, variant::Union{Int, Symbol}, A_gram::LinearMap{T}, τ::Union{T, Nothing}, ρ::T, θ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)
        new{T}(p, OnecolVariables(m, n), variant, W, W_inv, A_gram, τ, ρ, θ, falses(m))
    end
end
NoneWorkspace(args...) = NoneWorkspace{DefaultFloat}(args...)

# workspace type for when Krylov acceleration is used
struct KrylovWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, Variables{T}}
    p::ProblemData{T}
    vars::Variables{T}
    variant::Union{Int, Symbol} # In {:PDHG, :ADMM, 1, 2, 3, 4}.
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

    # constructor where initial iterates are passed in
    function KrylovWorkspace{T}(p::ProblemData{T}, vars::Variables{T}, variant::Union{Int, Symbol}, τ::Union{T, Nothing}, ρ::T, θ::T, mem::Int, krylov_operator::Symbol) where {T <: AbstractFloat}
        m, n = p.m, p.n
        A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)
        new{T}(p, vars, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, krylov_operator, UpperHessenberg(zeros(mem, mem-1)), zeros(m + n, mem))
    end

    # constructor where initial iterates are not passed (default set to zero)
    function KrylovWorkspace{T}(p::ProblemData{T}, variant::Union{Int, Symbol}, τ::Union{T, Nothing}, ρ::T, θ::T, mem::Int, krylov_operator::Symbol) where {T <: AbstractFloat}
        m, n = p.m, p.n
        A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)
        new{T}(p, Variables(m, n), variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, krylov_operator, UpperHessenberg(zeros(mem, mem-1)), zeros(m + n, mem))
    end

    # constructor where initial iterates are not passed (default set to zero)
    # but A_gram is
    function KrylovWorkspace{T}(p::ProblemData{T}, variant::Union{Int, Symbol}, A_gram::LinearMap{T}, τ::Union{T, Nothing}, ρ::T, θ::T, mem::Int, krylov_operator::Symbol) where {T <: AbstractFloat}
        m, n = p.m, p.n
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)
        new{T}(p, Variables(m, n), variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, krylov_operator, UpperHessenberg(zeros(mem, mem-1)), zeros(m + n, mem))
    end
end

# workspace type for when Anderson acceleration is used
struct AndersonWorkspace{T <: AbstractFloat} <: AbstractWorkspace{T, OnecolVariables{T}}
    p::ProblemData{T}
    vars::OnecolVariables{T}
    variant::Union{Int, Symbol} # In {:PDHG, :ADMM, 1, 2, 3, 4}.
    W::Union{Diagonal{T}, Symmetric{T}}
    W_inv::AbstractInvOp
    A_gram::LinearMap{T}
    τ::Union{T, Nothing}
    ρ::T
    θ::T
    proj_flags::AbstractVector{Bool}

    # additional Anderson-related fields
    mem::Int # memory for Anderson accelerator
    attempt_period::Int # how often to attempt Anderson acceleration
    accelerator::COSMOAccelerators.AndersonAccelerator

    # constructor where initial iterates are passed in
    function AndersonWorkspace{T}(p::ProblemData{T}, vars::OnecolVariables{T}, variant::Union{Int, Symbol}, τ::Union{T, Nothing}, ρ::T, θ::T, mem::Int, attempt_period::Int) where {T <: AbstractFloat}
        m, n = p.m, p.n
        A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)

        # default types:
        # COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.QRDecomp}, COSMOAccelerators.RestartedMemory, COSMOAccelerators.NoRegularizer}

        aa = COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.NormalEquations}, COSMOAccelerators.RollingMemory, COSMOAccelerators.NoRegularizer}(m + n, mem = mem)
        new{T}(p, vars, variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, attempt_period, aa)
    end

    # constructor where initial iterates are not passed (default set to zero)
    function AndersonWorkspace{T}(p::ProblemData{T}, variant::Union{Int, Symbol}, τ::Union{T, Nothing}, ρ::T, θ::T, mem::Int, attempt_period::Int) where {T <: AbstractFloat}
        m, n = p.m, p.n
        A_gram = LinearMap(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)

        # default types:
        # COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.QRDecomp}, COSMOAccelerators.RestartedMemory, COSMOAccelerators.NoRegularizer}

        aa = COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.NormalEquations}, COSMOAccelerators.RollingMemory, COSMOAccelerators.NoRegularizer}(m + n, mem = mem)
        new{T}(p, OnecolVariables(m, n), variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, attempt_period, aa)
    end

    # constructor where initial iterates are not passed (default set to zero)
    # but A_gram is
    function AndersonWorkspace{T}(p::ProblemData{T}, variant::Union{Int, Symbol}, A_gram::LinearMap{T}, τ::Union{T, Nothing}, ρ::T, θ::T, mem::Int, attempt_period::Int) where {T <: AbstractFloat}
        m, n = p.m, p.n
        W = W_operator(variant, p.P, p.A, A_gram, τ, ρ)
        W_inv = prepare_inv(W)

        # default types:
        # COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.QRDecomp}, COSMOAccelerators.RestartedMemory, COSMOAccelerators.NoRegularizer}

        aa = COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.NormalEquations}, COSMOAccelerators.RollingMemory, COSMOAccelerators.NoRegularizer}(m + n, mem = mem)
        new{T}(p, OnecolVariables(m, n), variant, W, W_inv, A_gram, τ, ρ, θ, falses(m), mem, attempt_period, aa)
    end

end
AndersonWorkspace(args...) = AndersonWorkspace{DefaultFloat}(args...)


@with_kw mutable struct Results{T <: AbstractFloat}
    data::Dict{Symbol, Any} = Dict{Symbol,Any}()
end

@with_kw mutable struct RunResults{T <: AbstractFloat}
    # Primal and dual objective values.
    primal_obj::AbstractVector{T} = T[]
    dual_obj::AbstractVector{T} = T[]

    # Duality gap.
    gaps::AbstractVector{T} = T[]

    # Primal and dual residuals.
    pri_res::AbstractVector{T} = T[]
    dual_res::AbstractVector{T} = T[]
end
RunResults(args...) = RunResults{DefaultFloat}(args...)

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
end

# A function that prepares an inverse operator based on the type of W
function prepare_inv(W::Diagonal{T}) where T <: Real
    # For a Diagonal matrix, simply compute the reciprocal of the diagonal entries.
    return DiagInvOp(1 ./ Vector(diag(W)))
end

function prepare_inv(W::Symmetric{T}) where T <: Real
    # For a symmetric positive definite matrix, compute its Cholesky factorization.
    F = SparseArrays.cholesky(W)

    # @infiltrate

    return CholeskyInvOp(F)
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
function apply_inv!(op::CholeskyInvOp, x::AbstractArray{T}) where T <: Real
    # TODO: use ldiv! for sparse Cholesky solve when using Julia v1.12.
    
    # println("inside apply_inv! just before: factorisation has dimension $(size(op.F))")
    # temp = (op.F \ x)
    # println("inside apply_inv! just after: factorisation has dimension $(size(op.F))")
    # x .= temp

    # should also just be able to do this:
    x .= op.F \ x

    return nothing
end