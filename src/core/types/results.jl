# Results structs

mutable struct Results{T <: AbstractFloat, I <: Integer}
    metrics_history::Dict{Symbol, Any}
    metrics_final::ReturnMetrics{T}
    exit_status::Symbol
    k_final::I
    k_operator_final::I
end

# Struct just for diagnostics data
mutable struct DiagnosticsWorkspace{T <: AbstractFloat}
    tilde_A::AbstractMatrix{T}
    tilde_b::Vector{T}
    W_inv_mat::AbstractMatrix{T}

    # only relevant for Krylov variants
    H_unmod::Union{UpperHessenberg{T}, Nothing}

    # SOC normal direction tracking (only when full_diagnostics=true)
    soc_normals_prev::Union{Vector{Vector{T}}, Nothing}
    soc_normals_curr::Union{Vector{Vector{T}}, Nothing}
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
    if ws.method.W_inv isa CholeskyInvOp
        W_inv_mat = ws.method.W_inv.F \ dense_I
        W_inv_mat = Symmetric(W_inv_mat)
    elseif ws.method.W_inv isa DiagInvOp
        W_inv_mat = Diagonal(ws.method.W_inv.inv_diag)
    end

    # Initialize SOC normal storage for KrylovWorkspace only
    if ws isa KrylovWorkspace
        # Count SOC cones and get their dimensions
        num_socs = length(ws.proj_state.soc_states)
        soc_normals_prev = Vector{Vector{T}}(undef, num_socs)
        soc_normals_curr = Vector{Vector{T}}(undef, num_socs)

        # Allocate storage for each SOC based on cone dimensions
        soc_idx = 1
        for cone in ws.p.K
            if cone isa Clarabel.SecondOrderConeT
                soc_normals_prev[soc_idx] = Vector{T}(undef, cone.dim)
                soc_normals_curr[soc_idx] = Vector{T}(undef, cone.dim)
                soc_idx += 1
            end
        end
    else
        soc_normals_prev = nothing
        soc_normals_curr = nothing
    end

    DiagnosticsWorkspace{T}(tilde_A, tilde_b, W_inv_mat, H_unmod, soc_normals_prev, soc_normals_curr)
end
DiagnosticsWorkspace(args...; kwargs...) = DiagnosticsWorkspace{DefaultFloat}(args...; kwargs...)