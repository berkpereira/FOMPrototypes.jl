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

    y_bar::Vector{T} # Extrapolated dual variable
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
        zeros(T, m),
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

    y_bar::Vector{T} # Extrapolated dual variable
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

        zeros(T, m),
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
   
    y_bar::Vector{T} # Extrapolated dual variable, for use in onecol operator
    y_qm_bar::Matrix{T} # Extrapolated dual variable
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

        zeros(T, m),
        zeros(T, m, 2)
    )
end

KrylovScratch(p::ProblemData) = KrylovScratch{DefaultFloat}(p)