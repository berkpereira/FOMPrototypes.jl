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

struct BaseScratch{T <: AbstractFloat}
    temp_n_vec1::Vector{T}
    temp_n_vec2::Vector{T}
    temp_m_vec1::Vector{T}
    temp_m_vec2::Vector{T}
    temp_mn_vec1::Vector{T}
    temp_mn_vec2::Vector{T}
    bAx_proj_for_res::Vector{T} # to hold Pi_K(b - Ax), intermediate in primal residual computation
    prev_y::Vector{T} # to hold previous dual variable, for residual computation
    s_reconst::Vector{T}

    function BaseScratch{T}(p::ProblemData{T}) where {T <: AbstractFloat}
        new(
            zeros(T, p.n),
            zeros(T, p.n),
            zeros(T, p.m),
            zeros(T, p.m),
            zeros(T, p.m + p.n),
            zeros(T, p.m + p.n),
            zeros(T, p.m),
            zeros(T, p.m),
            zeros(T, p.m),
        )
    end
end

abstract type AbstractMethodScratchVecs{T <: AbstractFloat} end
abstract type AbstractMethodScratchMats{T <: AbstractFloat} end

struct PrePPMScratchVecs{T} <: AbstractMethodScratchVecs{T}
    y_bar::Vector{T} # Extrapolated dual variable
    Ax::Vector{T} # to hold "basic" A * x products
    ATy::Vector{T} # to hold "basic" A' * y products
    
    function PrePPMScratchVecs{T}(p::ProblemData{T}) where {T <: AbstractFloat}
        new(zeros(T, p.m), zeros(T, p.m), zeros(T, p.n))
    end
end

struct PrePPMScratchMats{T} <: AbstractMethodScratchMats{T}
    y_qm_bar::Matrix{T} # Extrapolated dual variable for Krylov basis building

    function PrePPMScratchMats{T}(p::ProblemData{T}) where {T <: AbstractFloat}
        new(zeros(T, p.m, 2))
    end
end

abstract type ScratchExtra{T <: AbstractFloat} end

struct VanillaScratchExtra{T} <: ScratchExtra{T}
    swap_vec::Vector{T}

    function VanillaScratchExtra{T}(p::ProblemData{T}) where {T <: AbstractFloat}
        new(zeros(T, p.m + p.n))
    end
end

struct AndersonScratchExtra{T} <: ScratchExtra{T}
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

    function AndersonScratchExtra{T}(p::ProblemData{T}) where {T <: AbstractFloat}
        len = p.m + p.n
        new{T}(
            zeros(T, len),
            zeros(T, len),
            zeros(T, len),
            zeros(T, len),
            zeros(T, len),
            zeros(T, len),
        )
    end
end

struct KrylovScratchExtra{T} <: ScratchExtra{T}
    temp_n_mat1::Matrix{T}
    temp_n_mat2::Matrix{T}
    temp_m_mat1::Matrix{T}
    temp_m_mat2::Matrix{T}
    Ax_mat::Matrix{T}
    temp_n_vec_complex1::Vector{Complex{T}}
    temp_n_vec_complex2::Vector{Complex{T}}
    temp_m_vec_complex::Vector{Complex{T}}
    
    # to hold normal vectors of SOC projection.
    # needs at the very most m entries. usually much fewer
    projection_normal::Vector{T} 

    # holds the rhs residual set up for GMRES solution
    # we use a view into it, since the relevant working
    # portion is of length ws.givens_count[] + 1
    rhs_res::Vector{T}
    gmres_increment::Vector{T}

    # briefly holds initial iterate
    initial_vec::Vector{T}
    
    accelerated_point::Vector{T}
    # recycled iterate --- to recycle work done when computing fixed-point
    # residuals for acceleration acceptance criteria, then assigned
    # to the working optimisation variable state_q[:, 1] in the next iteration
    state_recycled::Vector{T}
    state_lookahead::Vector{T}
    fp_res::Vector{T}

    # cache for next iterate
    # when we pause to compute the Krylov accelerated point,
    # we require FOM(current_state)
    # this cache gives us the chance to recycle that work again
    # within the subsequent safeguard call!
    step_when_computing_krylov::Vector{T}

    # TODO add rhs_res_custom to use in compute_krylov_accelerant!
    # will take a little finesse

    function KrylovScratchExtra{T}(p::ProblemData{T}) where {T <: AbstractFloat}
        m, n = p.m, p.n
        new{T}(
            zeros(T, n, 2),
            zeros(T, n, 2),
            zeros(T, m, 2),
            zeros(T, m, 2),
            zeros(T, m, 2),
            zeros(Complex{T}, n),
            zeros(Complex{T}, n),
            zeros(Complex{T}, m),
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
end


# Composite types which fit all scratch components together
abstract type AbstractWorkspaceScratch{T <: AbstractFloat} end

struct VanillaScratch{T} <: AbstractWorkspaceScratch{T}
    base::BaseScratch{T}
    method::AbstractMethodScratchVecs{T}
    extra::VanillaScratchExtra{T}
end

struct AndersonScratch{T} <: AbstractWorkspaceScratch{T}
    base::BaseScratch{T}
    method::AbstractMethodScratchVecs{T}
    extra::AndersonScratchExtra{T}
end

struct KrylovScratch{T} <: AbstractWorkspaceScratch{T}
    base::BaseScratch{T}
    method::AbstractMethodScratchVecs{T}
    method_mats::AbstractMethodScratchMats{T}
    extra::KrylovScratchExtra{T}
end