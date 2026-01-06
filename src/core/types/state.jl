# problem data structure
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

# Variables structs

macro common_var_fields()
    return esc(quote
        state_prev::Vector{T}
        preproj_vec::Vector{T} # of interest just for recording active set
    end)
end

abstract type AbstractVariables{T<:AbstractFloat} end

struct KrylovVariables{T <: AbstractFloat} <: AbstractVariables{T}
    # Consolidated state vector and q vector (for Krylov basis building).
    # state is exactly as it sounds. q is for building a basis with an
    # Arnoldi-like process simultaneously as we iterate on the state sequence.
    # Convenient for when we use the linearisation technique
    # for Anderson/Krylov acceleration.
    state_q::Matrix{T}
    
    @common_var_fields()

    # default (zeros) initialisation of variables
    function KrylovVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m, 2), zeros(n + m), zeros(m))
    end
end
KrylovVariables(args...) = KrylovVariables{DefaultFloat}(args...)

struct AndersonVariables{T <: AbstractFloat} <: AbstractVariables{T}
    state::Vector{T}
    
    @common_var_fields()
    
    state_into_accelerator::Vector{T} # working iterate looking back up to anderson-interval iterations, to be passed into the COSMOAccelerators interface
    
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'
    
    function AndersonVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(m), zeros(n + m))
    end
end
AndersonVariables(args...) = AndersonVariables{DefaultFloat}(args...)

struct VanillaVariables{T <: AbstractFloat} <: AbstractVariables{T}
    state::Vector{T}

    @common_var_fields()
    # TODO perhaps add y_bar field for temporary storage, to be multiplied by A'

    function VanillaVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(m))
    end
end
VanillaVariables(args...) = VanillaVariables{DefaultFloat}(args...)

struct RandomizedVariables{T <: AbstractFloat} <: AbstractVariables{T}
    state::Vector{T}

    @common_var_fields()

    function RandomizedVariables{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(zeros(n + m), zeros(n + m), zeros(m))
    end
end
RandomizedVariables(args...) = RandomizedVariables{DefaultFloat}(args...)

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

    residual_check_count::Ref{Int}

    # simple NaN constructor for all residual quantities
    function ProgressMetrics{T}(m::Int, n::Int) where {T <: AbstractFloat}
        new(fill(NaN, m), fill(NaN, n), NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, Ref(0))
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