# acceleration functions shared to both Anderson and Krylov interfaces

"""
Compute FOM(state) into `fom_out` and set `fp_out .= fom_out - state`.
Both `fom_out` and `fp_out` must be preallocated vectors of length m+n.
This centralises the small sequence of operations and the associated scratch usage.
"""
function compute_fom_and_fp!(
    ws::AbstractWorkspace,
    state_in::AbstractVector{Float64},
    fom_out::AbstractVector{Float64},
    fp_out::AbstractVector{Float64},
    )
    # compute FOM(state_in) into fom_out (in-place)
    onecol_method_operator!(ws, Val{ws.method.variant}(), state_in, fom_out)
    # fp_out := FOM(state_in) - state_in
    fp_out .= fom_out .- state_in
    return nothing
end

"""
Compute fixed-point metric for fp_res (length m+n) according to ws.safeguard_norm.

Inputs:
- ws: workspace (provides ws.p, ws.method.ρ, ws.method.variant, etc)
- fp_res: fixed-point residual vector (fom - state) of length m+n
- temp_n_vec1, temp_n_vec2: scratch vectors length n
- temp_m_vec: scratch vector length m

Returns:
- scalar fp metric (Float64)

Behavior:
- If ws.safeguard_norm == :euclid, returns norm(fp_res).
- If ws.safeguard_norm == :char, computes the quadratic form using M1/M2 and the cross term:
    metric^2 = <M1 * fp_x, fp_x> + ||fp_y||^2 / ρ + 2 <A * fp_x, fp_y>
  and returns sqrt(metric^2).
This mirrors your original logic and reuses the provided scratch buffers.
"""
function compute_fp_metric!(
    ws::AbstractWorkspace,
    fp_res::AbstractVector{Float64},
    )
    n = ws.p.n
    m = ws.p.m

    if ws.safeguard_norm == :euclid
        return norm(fp_res)
    elseif ws.safeguard_norm == :char
        # ws.scratch.base.temp_n_vec1 will hold M1 * fp_x
        @views M1_op!(
            fp_res[1:n],
            ws.scratch.base.temp_n_vec1,
            ws,
            ws.method.variant,
            ws.scratch.base.temp_n_vec2,
            ws.scratch.base.temp_m_vec2)

        # <M1 * fp_x, fp_x>
        @views metric_sq = dot(ws.scratch.base.temp_n_vec1, fp_res[1:n])

        # + <M2 * fp_y, fp_y> where M2 = (1/ρ) I  (so becomes ||fp_y||^2 / ρ)
        @views metric_sq += norm(fp_res[n+1:end])^2 / ws.method.ρ

        # + 2 <A * fp_x, fp_y>
        @views mul!(ws.scratch.base.temp_m_vec1, ws.p.A, fp_res[1:n])    # ws.scratch.base.temp_m_vec1 := A * fp_x

        @views metric_sq += 2 * dot(ws.scratch.base.temp_m_vec1, fp_res[n+1:end])

        return sqrt(metric_sq)
    else
        throw(ArgumentError("Unknown ws.safeguard_norm: $(ws.safeguard_norm)"))
    end
end

"""
Check for acceptance of a Krylov acceleration candidate.
Refactored to call helper routines that encapsulate FOM and fixed-point metric computations.
"""
function accel_fp_safeguard!(
    ws::Union{KrylovWorkspace, AndersonWorkspace},
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    current_state::AbstractVector{Float64},
    accelerated_state::AbstractVector{Float64},
    safeguard_factor::Float64,
    record::AbstractRecord,
    full_diagnostics::Bool = false,
    )
    # Unguarded (fast) path
    if ws.safeguard_norm == :none
        onecol_method_operator!(ws, Val{ws.method.variant}(), accelerated_state, ws.scratch.extra.state_recycled)
        return true
    end

    # compute FOM(current_state) in ws.scratch.extra.state_lookahead and fp_res_current in ws.scratch.extra.fp_res
    if ws isa AndersonWorkspace
        compute_fom_and_fp!(ws, current_state, ws.scratch.extra.state_lookahead, ws.scratch.extra.fp_res)
    else # Krylov case
        # here we can recycle from specialised cache container written to
        # in the GMRES setup inside compute_krylov_accelerant!
        ws.scratch.extra.state_lookahead .= ws.scratch.extra.step_when_computing_krylov
        ws.scratch.extra.fp_res .= ws.scratch.extra.state_lookahead - current_state
    end

    # preserve the vanilla FOM iterate into ws.scratch.extra.state_recycled for now
    ws.scratch.extra.state_recycled .= ws.scratch.extra.state_lookahead

    # compute fp metric for the vanilla iterate
    fp_metric_vanilla = compute_fp_metric!(ws, ws.scratch.extra.fp_res)

    # optional diagnostics based on tilde_A
    # Only available for QPs (not SOCPs) since explicit operator construction
    # is not yet supported for second-order cones
    if !isnothing(ws_diag) && !any(cone -> cone isa Clarabel.SecondOrderConeT, ws.p.K)
        try
            # pinv_sol = (tilde_A - I) \ (-tilde_b)
            pinv_sol = pinv(ws_diag.tilde_A - I) * (-ws_diag.tilde_b)
            pinv_residual = norm(ws_diag.tilde_A * pinv_sol + ws_diag.tilde_b - pinv_sol)
            if pinv_residual > 1e-9
                println("❌ no true fixed-point at this affine operator! pinv sol residual: ", pinv_residual)
            else
                println("✅ true fixed-point at this affine operator! pinv sol residual: ",  pinv_residual)
            end
            println("Norm of pinv solution: ", norm(pinv_sol))
            println("Norm of accelerated_state: ", norm(accelerated_state))

            error_to_est = pinv_sol - accelerated_state
            rel_err = norm(error_to_est) / norm(pinv_sol)
            if rel_err < 1e-2
                println("✅ good Krylov-approximation of pinv-fixed point: rel error at iter $(ws.k[]): ", rel_err)
            else
                println("❌ bad Krylov-approximation of pinv-fixed point: rel error at iter $(ws.k[]): ", rel_err)
            end

            char_mat = Matrix([(ws.method.W - ws.p.P) ws.p.A'; ws.p.A I(ws.p.m) / ws.method.ρ])
            true_lookahead = zeros(ws.p.m + ws.p.n)
            onecol_method_operator!(ws, Val{ws.method.variant}(), pinv_sol, true_lookahead)
            # take a step from the linear system ≈solution, as in our Krylov method
            onecol_method_operator!(ws, Val{ws.method.variant}(), true_lookahead, pinv_sol)

            # swap: after this, true_lookahead stores T(pinv_sol)
            custom_swap!(true_lookahead, pinv_sol, ws.scratch.base.temp_mn_vec1)
            
            true_lookahead_step = true_lookahead - pinv_sol
            if ws.safeguard_norm == :euclid
                fp_metric_true = norm(true_lookahead_step)
            elseif ws.safeguard_norm == :char
                fp_metric_true = dot(true_lookahead_step, char_mat * true_lookahead_step)
                fp_metric_true = sqrt(fp_metric_true)
            end

            true_metric_ratio = fp_metric_true / fp_metric_vanilla
            if fp_metric_true < 1e-6
                println("✅ T(pinv sol) achieves small fp metric: fp_metric_true / fp_metric_vanilla = ", true_metric_ratio)
            else
                println("❌ T(pinv sol) does NOT achieve small fp metric: fp_metric_true / fp_metric_vanilla = ", true_metric_ratio)
            end
        catch e
            @info "Failed something when considering pinv-≈true fixed point at iter $(ws.k[]). $e"
        end
    end

    # optional Krylov diagnostics
    if full_diagnostics && ws isa KrylovWorkspace
        krylov_basis_svdvals = svdvals(ws.krylov_basis)
        krylov_basis_svdvals = krylov_basis_svdvals[findall(krylov_basis_svdvals .>= 1e-10)]
        # max_sval = maximum(krylov_basis_svdvals)
        # min_sval = minimum(krylov_basis_svdvals)
        # cond_no = max_sval / min_sval
        # if cond_no > 1.1
        #     println("❌ condition number of Krylov basis at iter $(ws.k[]) is ", cond_no)
        #     println("largest singular value: ", max_sval)
        #     println("smallest singular value: ", min_sval)
        #     @show krylov_basis_svdvals
        # else
        #     println("✅ condition number of Krylov basis at iter $(ws.k[]) is ", cond_no)
        # end
    end

    # compute FOM(accelerated_state) into ws.scratch.extra.state_lookahead and fp_res_acc in ws.scratch.extra.fp_res
    compute_fom_and_fp!(ws, accelerated_state, ws.scratch.extra.state_lookahead, ws.scratch.extra.fp_res)

    # compute fp metric for accelerated iterate
    fp_metric_acc = compute_fp_metric!(ws, ws.scratch.extra.fp_res)

    # ratio and record keeping
    metric_ratio = fp_metric_acc / fp_metric_vanilla

    if !(record isa NullRecord)
        push!(record.acc_attempt_iters, ws.k[])
        push!(record.fp_metric_ratios, metric_ratio)
    end

    acceleration_success = fp_metric_acc <= safeguard_factor * fp_metric_vanilla
    
    # finalize ws.scratch.extra.state_recycled
    if acceleration_success
        if ws isa KrylovWorkspace
            println("Givens count is $(ws.givens_count[]),")
        end
        println("✅ accel success, safeguard ratio: $(fp_metric_acc / fp_metric_vanilla)")
        ws.scratch.extra.state_recycled .= ws.scratch.extra.state_lookahead    # ws.scratch.extra.state_lookahead == FOM(accelerated_state)
    end

    # note this flag serves to indicate that we should recycle
    # ws.scratch.extra.state_recycled at the next iteration
    # (NOT set to true when safeguard is off)
    ws.control_flags.recycle_next = true

    return acceleration_success
end