# Workspace to contain problem and algorithm data before/during/after solver
# runs.
# We enable the use of default values as well as keyword arguments.

using Clarabel
using Parameters

const DefaultFloat = Float64

@with_kw struct ProblemData{T}
    P::AbstractMatrix{T}
    c::AbstractVector{T}
    A::AbstractMatrix{T}
    m::Int
    n::Int
    b::AbstractVector{T}
    K::Vector{Clarabel.SupportedCone}

    function ProblemData{T}(P::AbstractMatrix{T}, c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector{T}, K::Vector{Clarabel.SupportedCone}) where {T <: AbstractFloat}
        m, n = size(A)
        new(P, c, A, m, n, b, K)
    end
end
ProblemData(args...) = ProblemData{DefaultFloat}(args...)

struct Variables{T}
    x::AbstractVector{T} # Primal variable.
    s::AbstractVector{T} # Slack variable.
    y::AbstractVector{T} # Dual variable.

    # Dual variable from "previous" iteration. N.B.: after (e.g.) restarts or
    # accelerated steps, this is actually an artificial "previous" iterate.
    y_prev::AbstractVector{T}
    
    # "Artificial" iterate consolidating s and y. Allows to reduce dimension of
    # the method's operator from (n + 2 * m) to just (n + m).
    v::AbstractVector{T}

    # Iterate for basis building with an Arnoldi-like process as we iterate.
    # This is for purposes of affine approximation-based acceleration.
    q::AbstractVector{T}

    # Consolidated (x, v) vector and q vector (for basis building) (two cols).
    # (x,v) is exactly as it sounds. q is for building a basis with an
    # Arnoldi-like process simultaneously as we iterate on the (x, v) sequence.
    # Convenient for when we use the linearisation technique
    # for acceleration.
    x_v_q::AbstractMatrix{T}

    function Variables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n), zeros(m), zeros(m), zeros(m), zeros(m), zeros(n + m), zeros(n + m, 2))
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

    # Select method variant (to do with the proximal penalty norm used).
    variant::Int # In {1, 2, 3, 4}.

    # Primal and dual step sizes.
    τ::T
    ρ::T

    # Affine dynamics descriptors.
    tilde_A::AbstractMatrix{T}
    tilde_b::AbstractVector{T}

    # Indicator of enforced constraints (are cone projections making a change
    # to these blocks?).
    enforced_constraints::AbstractVector{Bool}

    # Cache for variety of useful quantities (e.g. fixed matrix-... products).
    cache::Dict{Symbol, Any}

    # Constructor where initial iterates are passed in.
    function Workspace{T}(p::ProblemData{T}, vars::Variables{T}, variant::Int, τ::T, ρ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        new(p, vars, variant, τ, ρ, spzeros(T, n + m, n + m), spzeros(T, n + m), falses(m), Dict{Symbol, Any}());
    end

    # Constructor where initial iterates are not passed (default set to zero).
    function Workspace{T}(p::ProblemData{T}, variant::Int, τ::T, ρ::T) where {T <: AbstractFloat}
        m, n = p.m, p.n
        new(p, Variables(p.m, p.n), variant, τ, ρ, spzeros(T, n + m, n + m), spzeros(T, n + m), falses(m), Dict{Symbol, Any}())
    end
end
Workspace(args...) = Workspace{DefaultFloat}(args...)

@with_kw mutable struct Results{T <: AbstractFloat}
    primal_obj_vals::Vector{T}
    dual_obj_vals::Vector{T}
    pri_res_norms::Vector{T}
    dual_res_norms::Vector{T}
    enforced_set_flags::Vector{Vector{Bool}}
    x_dist_to_sol::Vector{T}
    s_dist_to_sol::Vector{T}
    y_dist_to_sol::Vector{T}
    v_dist_to_sol::Vector{T}
    xy_semidist::Vector{T}
    update_mat_iters::Vector{Int}
    update_mat_ranks::Vector{T}
    update_mat_singval_ratios::Vector{T}
    acc_step_iters::Vector{Int}
    linesearch_iters::Vector{Int}

    xy_step_norms::Vector{T}
    xy_step_char_norms::Vector{T}
    xv_step_norms::Vector{T}
    xy_update_cosines::Vector{T}
    xv_update_cosines::Vector{T}
end



# mutable struct States
# 	IS_ASSEMBLED::Bool # The workspace has been assembled with problem data.
# 	IS_OPTIMISED::Bool # The optimisation function has been CALLED on the model.
# 	# IS_SCALED::Bool # The problem data has been scaled.
# 	States() = new(false, false, false, false, false)
# end