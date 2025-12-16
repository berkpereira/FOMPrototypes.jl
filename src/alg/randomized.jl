using LinearAlgebra
using SparseArrays
using Random
using Clarabel

"""
    linearised_proj_step_batch!(
        Y::AbstractMatrix{Float64},
        preproj_vec::AbstractVector{Float64},
        postproj_vec::AbstractVector{Float64},
        K::Vector{Clarabel.SupportedCone},
        proj_state::ProjectionState,
        temp_normal::AbstractVector{Float64}
    )

Apply linearised projection to each column of Y, using the current active set
defined by proj_state. This is the batch (s-column) version of linearised_proj_step!.

The linearisation is determined by preproj_vec and postproj_vec (from the state iterate),
which define the normal vector for SOC "interesting" cases.
"""
function linearised_proj_step_batch!(
    Y::AbstractMatrix{Float64},
    preproj_vec::AbstractVector{Float64},
    postproj_vec::AbstractVector{Float64},
    K::Vector{Clarabel.SupportedCone},
    proj_state::ProjectionState,
    temp_normal::AbstractVector{Float64},
    )
    start_idx = 1
    soc_idx = 1
    nn_idx = 1

    for cone in K
        end_idx = start_idx + cone.dim - 1
        if cone isa Clarabel.NonnegativeConeT
            nn_range = nn_idx:(nn_idx + cone.dim - 1)
            # For each column, multiply by the mask
            @views for col in axes(Y, 2)
                Y[start_idx:end_idx, col] .*= proj_state.nn_mask[nn_range]
            end
            nn_idx += cone.dim
        elseif cone isa Clarabel.SecondOrderConeT
            if proj_state.soc_states[soc_idx] == soc_identity
                nothing # identity operation
            elseif proj_state.soc_states[soc_idx] == soc_zero
                @views Y[start_idx:end_idx, :] .= 0.0
            else # interesting case: project onto hyperplane orthogonal to normal
                v_temp = view(temp_normal,  start_idx:end_idx)
                v_post = view(postproj_vec, start_idx:end_idx)
                v_pre  = view(preproj_vec,  start_idx:end_idx)

                # Compute unit normal vector (same for all columns)
                @. v_temp = v_post - v_pre
                inv_nrm = 1 / norm(v_temp)

                if inv_nrm > 1e4
                    println("🟠 Very small normal vector in batch linearised projection step for SOC at index $soc_idx !")
                end

                @. v_temp *= inv_nrm

                # Apply orthogonal projection to each column
                @views for col in axes(Y, 2)
                    v_y = Y[start_idx:end_idx, col]
                    scalar_proj = dot(v_y, v_temp)
                    @. v_y -= scalar_proj * v_temp
                end
            end
            soc_idx += 1
        end
        # NB zero cone needs no action to project to its dual (whole space)
        start_idx = end_idx + 1
    end
end


"""
    apply_linearized_operator_batch!(
        ws::RandomizedWorkspace{T, I, M},
        X_in::AbstractMatrix{T},
        X_out::AbstractMatrix{T},
        ::Val{P},
        use_shift::Bool = true
    )

Apply the linearized FOM operator L (or L-I if use_shift=true) to each column of X_in,
storing results in X_out. Uses the current active set from ws.proj_state.

This is the s-column batch version for randomized acceleration, analogous to
twocol_method_operator! but:
- Operates on s columns (not 2)
- Uses standard BLAS matrix ops (not complex vector trick)
- Applies linearised_proj_step_batch! using frozen active set

Arguments:
- ws: RandomizedWorkspace containing problem data and projection state
- X_in: Input matrix (m+n) × s
- X_out: Output matrix (m+n) × s (will be mutated)
- Val{P}: Method variant dispatch
- use_shift: If true, compute (L-I)X; if false, compute LX
"""
function apply_linearized_operator_batch!(
    ws::RandomizedWorkspace{T, I, M},
    X_in::AbstractMatrix{T},
    X_out::AbstractMatrix{T},
    ::Val{P},
    use_shift::Bool = true,
    ) where {T, I, M <: PrePPM, P}

    n = ws.p.n
    s = size(X_in, 2)

    # Extract views for primal (x) and dual (y) parts
    @views X_n_in = X_in[1:n, :]      # Primal part of input
    @views X_m_in = X_in[n+1:end, :]  # Dual part of input
    @views X_n_out = X_out[1:n, :]    # Primal part of output
    @views X_m_out = X_out[n+1:end, :] # Dual part of output

    # Scratch matrices from workspace
    temp_n_mat = ws.scratch.extra.temp_n_mat    # n × s
    temp_m_mat1 = ws.scratch.extra.temp_m_mat1  # m × s
    y_bar_mat = ws.scratch.extra.y_qm_bar_mat   # m × s

    # ========== DUAL UPDATE ==========
    # Compute A * X_n for all s columns
    mul!(temp_m_mat1, ws.p.A, X_n_in)

    # temp_m_mat1 now holds A * X_n
    # Scale by ρ: ρ * A * X_n
    temp_m_mat1 .*= ws.method.ρ

    # Add current dual part: ρ * A * X_n + X_m
    temp_m_mat1 .+= X_m_in

    # Now apply linearized projection to get new dual Y
    # X_m_out = linearized_proj(ρ * A * X_n + X_m)
    X_m_out .= temp_m_mat1

    linearised_proj_step_batch!(
        X_m_out,
        ws.vars.preproj_vec,           # preproj from state iterate
        ws.scratch.base.temp_m_vec1,   # postproj (we need to store this somewhere)
        ws.p.K,
        ws.proj_state,
        ws.scratch.extra.projection_normal,
    )

    # ========== EXTRAPOLATION: y_bar = 2*y_new - y_old ==========
    # y_bar_mat = 2 * X_m_out - X_m_in (θ = 1 specialization)
    @. y_bar_mat = 2.0 * X_m_out - X_m_in

    # If using B = L - I operator, subtract identity from dual part
    if use_shift
        @. X_m_out -= X_m_in
    end

    # ========== PRIMAL UPDATE ==========
    # Compute P * X_n into temp_n_mat
    mul!(temp_n_mat, ws.p.P, X_n_in)

    # Compute A' * y_bar into X_n_out (used as scratch, then overwritten)
    mul!(X_n_out, ws.p.A', y_bar_mat)

    # Combine: P * X_n + A' * y_bar
    temp_n_mat .+= X_n_out

    # Apply W^{-1} in-place to temp_n_mat
    # For batch operations, we apply column by column
    if ws.method.W_inv isa CholeskyInvOp
        # Cholesky solve needs concrete Vector, not view
        # Use scratch vectors and copy in/out
        work_vec = ws.scratch.base.temp_n_vec1
        scratch_vec = ws.scratch.base.temp_n_vec2
        for col in 1:s
            @views work_vec .= temp_n_mat[:, col]
            apply_inv!(ws.method.W_inv, work_vec, scratch_vec)
            @views temp_n_mat[:, col] .= work_vec
        end
    else
        # Diagonal case - DiagInvOp can handle matrix directly
        apply_inv!(ws.method.W_inv, temp_n_mat)
    end

    # Final primal update: X_n_out = X_n_in - W^{-1} * (P * X_n + A' * y_bar)
    if use_shift
        # For L - I: just store -W^{-1} * (...)
        @. X_n_out = -temp_n_mat
    else
        # For L: X_n - W^{-1} * (...)
        @. X_n_out = X_n_in - temp_n_mat
    end
end


"""
    generate_random_subspace!(ws::RandomizedWorkspace, rng::AbstractRNG = Random.GLOBAL_RNG)

Generate a new random Gaussian embedding Ω, compute its image V = (L-I)Ω under the
linearized operator, and form the regularized Gram matrix G = V'V + λI.

Returns :success or :cholesky_failure.
"""
function generate_random_subspace!(
    ws::RandomizedWorkspace,
    rng::AbstractRNG = Random.GLOBAL_RNG
    )

    s = ws.subspace_dim

    # Generate Gaussian random matrix Ω ~ N(0, 1)
    randn!(rng, ws.Omega)

    # Compute V = (L - I) * Omega
    # First, need to update the projection state from current iterate
    # This should have been done before calling this function

    use_shift = (ws.rand_operator == :tilde_A)
    apply_linearized_operator_batch!(ws, ws.Omega, ws.V, Val{ws.method.variant}(), use_shift)

    # Count operator applications (s applications)
    ws.k_operator[] += s

    # Form Gram matrix G = V'V + λI
    mul!(ws.G, ws.V', ws.V)

    # Add regularization
    @inbounds for i in 1:s
        ws.G[i, i] += ws.regularization
    end

    return :success
end


"""
    compute_randomized_accelerant!(ws::RandomizedWorkspace, result_vec::Vector{Float64})

Compute the randomized subspace acceleration candidate by solving:
    min_z ||V*z - r_k||^2
via the normal equations: G*z = V'*r_k, where G = V'V + λI.

The accelerated point is FOM(current + V*z), stored in result_vec.

Returns :success or :lls_failure.
"""
function compute_randomized_accelerant!(
    ws::RandomizedWorkspace,
    result_vec::Vector{Float64}
    )

    # 1. Compute FOM(current) and cache for safeguard
    @views onecol_method_operator!(
        ws,
        Val{ws.method.variant}(),
        ws.vars.state,
        result_vec,
        false,  # don't update proj action (already done)
        false   # don't update residuals
    )
    ws.scratch.extra.step_when_computing_randomized .= result_vec

    # 2. Compute fixed-point residual r_k = FOM(current) - current
    ws.scratch.base.temp_mn_vec1 .= result_vec
    ws.scratch.base.temp_mn_vec1 .-= ws.vars.state

    # 3. Compute V' * r_k
    mul!(ws.scratch.extra.lls_rhs, ws.V', ws.scratch.base.temp_mn_vec1)

    # 4. Solve G * z = V' * r_k via Cholesky
    ws.scratch.extra.lls_sol .= ws.scratch.extra.lls_rhs

    # Attempt Cholesky factorization and solve
    try
        G_chol = cholesky!(Hermitian(copy(ws.G)))
        ldiv!(G_chol, ws.scratch.extra.lls_sol)
    catch e
        @warn "Randomized least-squares solution failed" exception=e
        return :lls_failure
    end

    # 5. Form full-dimensional increment: V * z
    mul!(ws.scratch.extra.subspace_increment, ws.V, ws.scratch.extra.lls_sol)

    # 6. Compute accelerated point: current + V*z
    ws.scratch.base.temp_mn_vec1 .= ws.vars.state
    ws.scratch.base.temp_mn_vec1 .+= ws.scratch.extra.subspace_increment

    # 7. Apply operator to get candidate: FOM(current + V*z)
    @views onecol_method_operator!(
        ws,
        Val{ws.method.variant}(),
        ws.scratch.base.temp_mn_vec1,
        result_vec,
        false,
        false
    )

    return :success
end


"""
    randomized_step!(ws::RandomizedWorkspace, ws_diag, config, record, full_diagnostics, timer)

Main iteration step for randomized subspace acceleration.
- On first iteration or after regeneration flag: generate new random subspace
- Compute acceleration candidate
- Apply safeguard check
- Accept or reject, with regeneration on rejection
"""
function randomized_step!(
    ws::RandomizedWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    config::SolverConfig,
    record::AbstractRecord,
    full_diagnostics::Bool,
    timer::TimerOutput,
    )

    # Copy previous iterate for step norm computation
    ws.vars.state_prev .= ws.vars.state

    if ws.k[] == 0
        # Initial iteration: compute FOM(x_0) and generate first subspace
        @timeit timer "init FOM" begin
            @views onecol_method_operator!(
                ws,
                Val{ws.method.variant}(),
                ws.vars.state,
                ws.scratch.extra.state_recycled,
                true,   # update proj action
                true    # update residuals
            )
        end
        ws.k_operator[] += 1

        # Generate initial random subspace
        @timeit timer "gen subspace" begin
            generate_random_subspace!(ws)
        end

        # First iteration just takes the vanilla step
        ws.vars.state .= ws.scratch.extra.state_recycled
        ws.control_flags.need_regenerate = false

    else
        # Regenerate subspace if needed
        if ws.control_flags.need_regenerate
            # First take a vanilla step to update projection state
            @timeit timer "vanilla step" begin
                @views onecol_method_operator!(
                    ws,
                    Val{ws.method.variant}(),
                    ws.vars.state,
                    ws.scratch.extra.state_recycled,
                    true,   # update proj action
                    true    # update residuals
                )
            end
            ws.k_operator[] += 1

            # Update state with vanilla step
            ws.vars.state .= ws.scratch.extra.state_recycled

            @timeit timer "gen subspace" begin
                generate_random_subspace!(ws)
            end
            ws.control_flags.need_regenerate = false
        end

        # Compute acceleration candidate
        @timeit timer "randomized accel" begin
            ws.control_flags.randomized_status = compute_randomized_accelerant!(
                ws,
                ws.scratch.extra.accelerated_point
            )
        end
        ws.k_operator[] += 2  # Two operator applications in compute_randomized_accelerant!

        if ws.control_flags.randomized_status == :success
            # Safeguard check
            @timeit timer "safeguard" begin
                accepted = accel_fp_safeguard!(
                    ws,
                    ws_diag,
                    ws.vars.state,
                    ws.scratch.extra.accelerated_point,
                    config.safeguard_factor,
                    record,
                    full_diagnostics
                )
            end

            if accepted
                # Accept accelerated point
                ws.vars.state .= ws.scratch.extra.accelerated_point
                ws.k_eff[] += 1
                ws.control_flags.accepted_accel = true

                # Update residuals on the accepted point
                @views onecol_method_operator!(
                    ws,
                    Val{ws.method.variant}(),
                    ws.vars.state,
                    ws.scratch.extra.state_recycled,
                    false,  # don't update proj action
                    true    # DO update residuals
                )
                # Keep same subspace for next iteration
            else
                # Reject: take vanilla step and flag for regeneration
                ws.vars.state .= ws.scratch.extra.state_recycled
                ws.control_flags.accepted_accel = false
                ws.control_flags.need_regenerate = true
            end
        else
            # Acceleration failed: take vanilla step and regenerate
            @warn "Randomized acceleration failed: $(ws.control_flags.randomized_status)"

            @views onecol_method_operator!(
                ws,
                Val{ws.method.variant}(),
                ws.vars.state,
                ws.scratch.extra.state_recycled,
                true,
                false
            )
            ws.k_operator[] += 1

            ws.vars.state .= ws.scratch.extra.state_recycled
            ws.control_flags.accepted_accel = false
            ws.control_flags.need_regenerate = true
        end
    end

    # Update residual check counter
    ws.res.residual_check_count[] += 1
end
