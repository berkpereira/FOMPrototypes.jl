# structures for cone projection actions
@enum SOCAction::Int8 begin
    soc_zero        = 0
    soc_identity    = 1
    soc_interesting = 2
end

"""
State used to control per-cone projection behaviour.

`nn_mask` holds a flat mask for all nonnegative cones; `soc_states` stores the
action to apply to each SOC block.
"""
struct ProjectionState
    # 1. Nonnegative Cone Data
    # We flatten ALL nonnegative cones into a single boolean mask.
    # This allows you to perform one giant broadcasted SIMD operation 
    # across all NN variables at once, even when they are split across
    # multiple cone blocks.
    nn_mask::Vector{Bool} 

    # 2. SOC Data
    # One enum per SOC cone (not per variable).
    soc_states::Vector{SOCAction}

    # 3. Maps (optional)
    # If cones are interleaved in 'K', we might need to know that 
    # K[i] corresponds to soc_states[j].
    # If K is sorted (e.g. all NN then all SOC), we don't need this.
end

function ProjectionState(K::Vector{Clarabel.SupportedCone})
    # Calculate total
    soc_count = 0
    nn_dim = 0

    # TODO add check that cones are ordered as expected
    # namely: single nonnegative cone first, then SOCs, then single zero cone,
    # except in QPs, where it doesn't matter (?)
    for cone in K
        if cone isa Clarabel.NonnegativeConeT
            nn_dim += cone.dim
        elseif cone isa Clarabel.SecondOrderConeT
            soc_count += 1
        elseif cone isa Clarabel.ZeroConeT
            nothing
            # no need to populate anything
        else
            throw(ArgumentError("Unsupported cone type in K."))
        end
    end

    # Allocate
    nn_mask = Vector{Bool}(undef, nn_dim)
    soc_states = Vector{SOCAction}(undef, soc_count)
    
    # Initialise
    fill!(nn_mask, true)
    fill!(soc_states, soc_identity)

    return ProjectionState(nn_mask, soc_states)
end

# shared helpers for workspace constructors
function _resolve_vars(::Type{V}, p::ProblemData{T}, vars::Union{V, Nothing}) where {T <: AbstractFloat, V <: AbstractVariables{T}}
    return vars === nothing ? V(p.m, p.n) : vars
end

function _resolve_A_gram(p::ProblemData{T}, A_gram::Union{LinearMap{T}, Nothing}) where {T <: AbstractFloat}
    if A_gram === nothing
        return LinearMap{T}(x -> p.A' * (p.A * x), size(p.A, 2), size(p.A, 2); issymmetric = true)
    end
    return A_gram
end

function _init_common_workspace(::Type{V}, p::ProblemData{T}, vars::Union{V, Nothing}, A_gram::Union{LinearMap{T}, Nothing}) where {T <: AbstractFloat, V <: AbstractVariables{T}}
    resolved_vars = _resolve_vars(V, p, vars)
    gram = _resolve_A_gram(p, A_gram)
    res = ProgressMetrics{T}(p.m, p.n)
    proj_state = ProjectionState(p.K)
    return resolved_vars, gram, res, proj_state
end

function _validate_safeguard_norm(safeguard_norm::Symbol)
    safeguard_norm in (:euclid, :char, :none) || throw(ArgumentError("safeguard_norm must be one of :euclid, :char, :none"))
    return safeguard_norm
end

function _validate_krylov_mem(mem::Int, tries_per_mem::Int, limit::Int)
    (mem - 1) < limit || throw(ArgumentError("(mem - 1) must be less than m + n."))
    tries_per_mem <= (mem - 1) || throw(ArgumentError("tries_per_mem must be less than or equal to (mem - 1)."))
    return mem, tries_per_mem
end

function _validate_krylov_operator(krylov_operator::Symbol)
    krylov_operator in (:tilde_A, :B) || throw(ArgumentError("krylov_operator must be one of :tilde_A, :B"))
    return krylov_operator
end

function _validate_anderson_interval(anderson_interval::Int)
    anderson_interval >= 1 || throw(ArgumentError("Anderson acceleration interval must be at least 1."))
    return anderson_interval
end

# macro to inject common workspace fields (used inside struct bodies)
macro common_workspace_fields()
    return esc(quote
        p::ProblemData{T}
        method::AbstractMethod{T, I} # eg PrePPM
        vars::AbstractVariables{T}
        A_gram::LinearMap{T}
        res::ProgressMetrics{T}
        proj_state::ProjectionState
        
        residual_period::I

        
        # variant::Symbol # In {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}.
        # W::Union{Diagonal{T}, SparseMatrixCSC{T, I}}
        # W_inv::AbstractInvOp
        # τ::T
        # ρ::T
        # θ::T
    end)
end

abstract type AbstractWorkspace{T <: AbstractFloat, I <: Integer, V <: AbstractVariables{T}, M <: AbstractMethod{T, I}} end

# workspace type for when :none acceleration is used
"""
Workspace for vanilla iterations (no acceleration).
Maintains problem data, bookkeeping metrics, and scratch buffers.
"""
struct VanillaWorkspace{T <: AbstractFloat, I <: Integer, M <: AbstractMethod{T, I}} <: AbstractWorkspace{T, I, VanillaVariables{T}, M}
    @common_workspace_fields()
    
    k::Base.RefValue{Int} # iter counter
    scratch::VanillaScratch{T}

    function VanillaWorkspace{T, I, M}(
        p::ProblemData{T},
        method::M,
        residual_period::I,
        scratch::VanillaScratch{T},
        vars::Union{VanillaVariables{T}, Nothing},
        A_gram::Union{LinearMap{T}, Nothing},
        ) where {T <: AbstractFloat, I <: Integer, M <: AbstractMethod{T, I}}

        vars, A_gram, res, proj_state = _init_common_workspace(VanillaVariables{T}, p, vars, A_gram)

        new{T, I, M}(
            p,
            method,
            vars,
            A_gram,
            res,
            proj_state,
            residual_period,
            Ref(0),
            scratch
        )
    end 
end

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
"""
Workspace for Krylov acceleration.
Tracks Arnoldi state, control flags, and counters for accelerated iterations.
"""
struct KrylovWorkspace{T <: AbstractFloat, I <: Integer, M <: AbstractMethod{T, I}} <: AbstractWorkspace{T, I, KrylovVariables{T}, M}
    @common_workspace_fields()

    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING unsuccessul Krylov acceleration attempts
    k_operator::Base.RefValue{Int} # this counts number of operator applications, whether in vanilla iterations, for safeguarding, or acceleration. probably most useful notion of iter count
    scratch::KrylovScratch{T}

    control_flags::KrylovControlFlags

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
    fp_found::Base.RefValue{Bool} # used to trigger a termination condition when stuff goes wrong initialising Krylov basis (with NaNs due to a zero fixed-point residual having been found!) TODO move to AndersonControlFlags?
    
    # NOTE that mem is (k+1) in the usual Arnoldi relation written
    # at the point of maximum memory usage as
    # A Q_k = Q_{k+1} \tilde{H}_k
    # ie plainly the number of columns allocated for krylov basis vectors

    # constructor where initial iterates are not passed (default set to zero)
    function KrylovWorkspace{T, I, M}(
        p::ProblemData{T},
        method::M,
        residual_period::I,
        scratch::KrylovScratch{T},
        vars::Union{KrylovVariables{T}, Nothing},
        A_gram::Union{LinearMap{T}, Nothing},
        mem::Int,
        tries_per_mem::Int,
        safeguard_norm::Symbol,
        krylov_operator::Symbol) where {T <: AbstractFloat, I <: Integer, M <: AbstractMethod{T, I}}

        _validate_krylov_mem(mem, tries_per_mem, p.m + p.n)
        _validate_krylov_operator(krylov_operator)
        _validate_safeguard_norm(safeguard_norm)

        # make vector of trigger_givens_counts
        trigger_givens_counts = Vector{Int}(undef, tries_per_mem)
        for i in eachindex(trigger_givens_counts)
            trigger_givens_counts[i] = Int(floor(i * (mem - 1) / tries_per_mem))
        end

        m, n = p.m, p.n
        vars, A_gram, res, proj_state = _init_common_workspace(KrylovVariables{T}, p, vars, A_gram)

        new{T, I, M}(
            p,
            method,
            vars,
            A_gram,
            res,
            proj_state,
            residual_period,
            Ref(0),
            Ref(0),
            Ref(0),
            scratch,
            KrylovControlFlags(),
            mem,
            tries_per_mem,
            safeguard_norm,
            trigger_givens_counts,
            krylov_operator,
            UpperHessenberg(zeros(T, mem, mem-1)),
            zeros(T, m + n, mem),
            Vector{GivensRotation{T}}(undef, mem-1),
            Ref(0),
            Ref(false),
            Ref(false)
        )
    end
end

# workspace type for when Anderson acceleration is used
"""
Workspace for Anderson acceleration.
Stores accelerator state, control flags, counters, and scratch buffers.
"""
struct AndersonWorkspace{T <: AbstractFloat, I <: Integer, M <: AbstractMethod{T, I}} <: AbstractWorkspace{T, I, AndersonVariables{T}, M}
    @common_workspace_fields()
    
    k::Base.RefValue{Int} # iter counter
    k_eff::Base.RefValue{Int} # effective iter counter, ie EXCLUDING UNsuccessul (Anderson) acceleration attempts
    k_vanilla::Base.RefValue{Int} # this counts just vanilla iterations throughout the run --- as expected by COSMOAccelerators functions (just for its logging purposes)
    k_operator::Base.RefValue{Int} # this counts number of operator applications, whether in vanilla iterations, for safeguarding, or acceleration. probably most useful notion of iter count
    composition_counter::Base.RefValue{Int} # counts the number of compositions of the operator, important for monitoring of data passed into COSMOAccelerators functions
    scratch::AndersonScratch{T}

    control_flags::AndersonControlFlags

    # additional Anderson acceleration-related fields
    mem::Int # memory for Anderson accelerator
    anderson_interval::Int # how often to attempt Anderson acceleration
    safeguard_norm::Symbol # in {:euclid, :char, :none}
    accelerator::AndersonAccelerator

    function AndersonWorkspace{T, I, M}(
        p::ProblemData{T},
        method::M,
        residual_period::I,
        scratch::AndersonScratch{T},
        vars::Union{AndersonVariables{T}, Nothing},
        A_gram::Union{LinearMap{T}, Nothing},
        mem::Int,
        anderson_interval::Int,
        safeguard_norm::Symbol,
        broyden_type::Type{<:COSMOAccelerators.AbstractBroydenType},
        memory_type::Type{<:COSMOAccelerators.AbstractMemory},
        regulariser_type::Type{<:COSMOAccelerators.AbstractRegularizer},
        anderson_log::Bool
        ) where {T <: AbstractFloat, I <: Integer, M <: AbstractMethod{T, I}}

        _validate_anderson_interval(anderson_interval)
        _validate_safeguard_norm(safeguard_norm)

        m, n = p.m, p.n
        vars, A_gram, res, proj_state = _init_common_workspace(AndersonVariables{T}, p, vars, A_gram)

        # default constructor types:
        # AndersonAccelerator{Float64, Type2{QRDecomp}, RestartedMemory, NoRegularizer}

        aa = AndersonAccelerator{T, broyden_type, memory_type, regulariser_type}(m + n, mem = mem, activate_logging = anderson_log)
        
        new{T, I, M}(
            p,
            method,
            vars,
            A_gram,
            res,
            proj_state,
            residual_period,
            Ref(0),
            Ref(0),
            Ref(0),
            Ref(0),
            Ref(0),
            scratch,
            AndersonControlFlags(),
            mem,
            anderson_interval,
            safeguard_norm,
            aa
        )
    end
end
