# Results structs

mutable struct Results{T <: AbstractFloat, I <: Integer}
    metrics_history::Dict{Symbol, Any}
    metrics_final::ReturnMetrics{T}
    exit_status::Symbol
    k_final::I
    k_operator_final::I
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
    if ws.method.W_inv isa CholeskyInvOp
        W_inv_mat = ws.method.W_inv.F \ dense_I
        W_inv_mat = Symmetric(W_inv_mat)
    elseif ws.method.W_inv isa DiagInvOp
        W_inv_mat = Diagonal(ws.method.W_inv.inv_diag)
    end

    DiagnosticsWorkspace{T}(tilde_A, tilde_b, W_inv_mat, H_unmod)
end
DiagnosticsWorkspace(args...; kwargs...) = DiagnosticsWorkspace{DefaultFloat}(args...; kwargs...)