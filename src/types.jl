# Workspace to contain problem and algorithm data before/during/after solver
# runs.
# We enable the use of default values as well as keyword arguments.

using Clarabel
using Parameters
using LinearAlgebra
using LinearAlgebra.LAPACK

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

struct Variables{T}
    # Consolidated (x, y) vector and q vector (for Krylov basis building).
    # (x, y) is exactly as it sounds. q is for building a basis with an
    # Arnoldi-like process simultaneously as we iterate on the (x, y) sequence.
    # Convenient for when we use the linearisation technique
    # for Anderson/Krylov acceleration.
    xy_q::AbstractMatrix{T}

    preproj_y::AbstractVector{T} # thing fed to the projection to dual cone, useful to store
    y_qm_bar::AbstractMatrix{T} # Extrapolated primal variable

    # default (zeros) initialisation of variables
    function Variables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m, 2), zeros(m), zeros(m, 2))
    end
end
Variables(args...) = Variables{DefaultFloat}(args...)

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


@with_kw mutable struct Workspace{T <: AbstractFloat}
    # Problem data.
    p::ProblemData{T}
    vars::Variables{T}

    # select method variant (to do with the proximal penalty norm used).
    variant::Int # In {-1, 0, 1, 2, 3, 4}.

    # primal and dual step sizes
    τ::T
    ρ::T
    # extrapolation parameter
    θ::T

    # preconditioner operator for the x update
    W_inv::AbstractInvOp

    # indicator of projection by-pass (QP case: where entries "pass through")
    proj_flags::AbstractVector{Bool}

    # Cache for variety of useful quantities (e.g. fixed matrix-... products).
    cache::Dict{Symbol, Any}

    # Constructor where initial iterates are passed in.
    function Workspace{T}(p::ProblemData{T}, vars::Variables{T}, variant::Int, τ::T, ρ::T, θ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        new(p, vars, variant, τ, ρ, θ, falses(m), Dict{Symbol, Any}());
    end

    # Constructor where initial iterates are not passed (default set to zero).
    function Workspace{T}(p::ProblemData{T}, variant::Int, τ::T, ρ::T, θ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        new(p, Variables(p.m, p.n), variant, τ, ρ, θ, spzeros(T, n + m, n + m), spzeros(T, n + m), falses(m), Dict{Symbol, Any}())
    end
end
Workspace(args...) = Workspace{DefaultFloat}(args...)

@with_kw mutable struct Results{T <: AbstractFloat}
    data::Dict{Symbol, Any} = Dict{Symbol,Any}()
end

# We now define some types to make the inversion of preconditioner + Hessian
# matrices, required for the x update, abstract. Thus we can use diagonal ones
# (as we intend in production) or non-diagonal symmetric ones for comparing
# with other methods (eg ADMM or vanilla PDHG).

# Concrete type for a diagonal inverse operator
struct DiagInvOp{T} <: AbstractInvOp
    inv_diag::Vector{T}
end

# Concrete type for a symmetric matrix's Cholesky-based inverse operator
struct CholeskyInvOp{T} <: AbstractInvOp
    F::Cholesky{T,Matrix{T}}  # Store the Cholesky factorization
end

# A function that prepares an inverse operator based on the type of M
function prepare_inv(M::Diagonal{T}) where T <: Real
    # For a Diagonal matrix, simply compute the reciprocal of the diagonal entries.
    return DiagInvOp(1 ./ diag(M))
end

function prepare_inv(M::Symmetric{T}) where T <: Real
    # For a symmetric positive definite matrix, compute its Cholesky factorization.
    F = cholesky(M)
    return CholeskyInvOp(F)
end

# Define how to apply the inverse operator to a vector
function (op::DiagInvOp)(x::AbstractVector)
    # element-wise multiplication by the precomputed entry-wise reciprocal
    return op.inv_diag .* x
end

function (op::CholeskyInvOp)(x::AbstractVector)
    # Use the Cholesky factorization to compute the inverse-vector product.
    # This computes F \ x, which is equivalent to inv(M)*x.
    return op.F \ x
end

# we also define in-place operators of these preconditioners
function apply_inv!(op::DiagInvOp, x::AbstractMatrix)
    # Elementwise in-place multiplication: x becomes op.inv_diag .* x.
    # NB matrices get scaled column by column, as expected
    x .*= op.inv_diag
    return nothing
end

# we also define in-place operators of these preconditioners
function apply_inv!(op::CholeskyInvOp, x::AbstractMatrix{T}) where T <: Real
    # NB this does very well, and seems to be apply the inverse Cholesky 
    # factors in-place with next to no memory allocations.
    # IT CAN also do it for a two-column input in the same time as for a single
    # vector!! 
    ldiv!(op.F, x)
    return nothing
end

# mutable struct States
# 	IS_ASSEMBLED::Bool # The workspace has been assembled with problem data.
# 	IS_OPTIMISED::Bool # The optimisation function has been CALLED on the model.
# 	# IS_SCALED::Bool # The problem data has been scaled.
# 	States() = new(false, false, false, false, false)
# end