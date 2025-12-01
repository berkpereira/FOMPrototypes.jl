using LinearAlgebra
using Random
using Printf
using Infiltrator
using TimerOutputs
import COSMOAccelerators

# data to store history of when a run uses run_fast == false
const HISTORY_KEYS = [
    :primal_obj_vals,
    :dual_obj_vals,
    :pri_res_norms,
    :dual_res_norms,
    :record_proj_flags,
    :record_soc_states,
    :x_dist_to_sol,
    :y_dist_to_sol,
    :state_chardist,
    :update_mat_iters,
    :update_mat_ranks,
    :update_mat_singval_ratios,
    :acc_step_iters,
    :linesearch_iters,
    :state_step_norms,
    :state_step_char_norms,
    :state_update_cosines,

    :fp_metric_ratios,
    :acc_attempt_iters,
]

const RHO_BALANCE_THRESHOLD = 10.0
const RHO_SCALE_UP = 2.0
const RHO_SCALE_DOWN = 0.5

accel_ready_for_rho_update(::VanillaWorkspace) = true
accel_ready_for_rho_update(ws::KrylovWorkspace) = ws.givens_count[] == 0
# with Anderson from COSMOAccelerators, if this is 0 then the next iteration
# will fill in the first column of the Anderson memory matrices:
accel_ready_for_rho_update(ws::AndersonWorkspace) = (ws.accelerator.iter % ws.accelerator.mem) == 0

function propose_new_rho(ws::AbstractWorkspace)
    rp = ws.res.rp_abs # TODO rel or abs resiuals? other rules a la OSQP to see
    rd = ws.res.rd_abs
    ρ  = ws.method.ρ

    if rp > RHO_BALANCE_THRESHOLD * rd
        return ρ * RHO_SCALE_UP
    elseif rd > RHO_BALANCE_THRESHOLD * rp
        return ρ * RHO_SCALE_DOWN
    else
        return ρ
    end
end

function update_admm_rho!(ws::AbstractWorkspace, new_ρ::Float64, timer::TimerOutput)
    old_inv = ws.method.W_inv
    perm_hint = old_inv isa CholeskyInvOp ? old_inv.perm : nothing
    typed_ρ = convert(typeof(ws.method.ρ), new_ρ)

    @timeit timer "rho update" begin
        ws.method.ρ = typed_ρ
        ws.method.W = W_operator(ws.method.variant, ws.p.P, ws.p.A, ws.A_gram, ws.method.τ, ws.method.ρ)
        ws.method.W_inv = prepare_inv(ws.method.W, timer; perm_hint = perm_hint)
    end
end

function maybe_update_rho!(ws::AbstractWorkspace, config::SolverConfig, timer::TimerOutput)
    period = config.rho_update_period
    if !isfinite(period) || period <= 0
        return
    end
    if ws.method.variant != :ADMM
        return
    end
    
    # only update when residuals were just refreshed?
    # this is unlikely to happen unless we explicitly couple the two
    # if ws.res.residual_check_count[] != 0
    #     return
    # end

    if !accel_ready_for_rho_update(ws)
        return
    end

    period_int = max(1, Int(floor(period)))
    if ws.k[] % period_int != 0
        return
    end

    new_ρ = propose_new_rho(ws)
    if new_ρ != ws.method.ρ
        println("Updating rho now.")
        update_admm_rho!(ws, new_ρ, timer)
    end
end

"""
Efficiently computes FOM iteration at point state_in, and stores it in state_out.
Vector state_in is left unchanged.

Set update_proj_action to true IFF the method is being used for a bona fide
vanilla FOM iteration, where the active set flags and working residual metrics
should be updated by recycling the computational work (mostly mat-vec
products involving A, A', and P) which is carried out here regardless.

This function is somewhat analogous to twocol_method_operator!, but it is
concerned only with applying the iteration to some actual iterate, ie
it does not consider any sort of concurrent Arnoldi-like process.

This is convenient whenever we need the method's operator but without
updating an Arnoldi vector, eg when solving the upper Hessenber linear least
squares problem following from the Krylov acceleration subproblem.

Note that state_in is the "current" iterate as a single vector in R^{n+m}.
New iterate is written (in-place) into state_out input vector.

Notw that ws.proj_state.nn_mask may have different interpretations depending on the
method in question. Eg in PrePPM it
The common theme is that it takes the form which makes it easier

TODO make x update more efficient (no need for P product
just to update x, just rearrange)
"""
function onecol_method_operator!(
    ws::AbstractWorkspace{T, I, V, M},
    ::Val{P},
    state_in::AbstractVector{Float64},
    state_out::AbstractVector{Float64},
    update_proj_action::Bool = false,
    update_residuals::Bool = false,
    ) where {T, I, V, M <: PrePPM, P} # note dispatch on PrePPM

    confirm_residual_update = update_residuals && ws.res.residual_check_count[] >= ws.residual_period

    # compute A * x
    @views mul!(ws.scratch.base.temp_m_vec1, ws.p.A, state_in[1:ws.p.n])

    if confirm_residual_update
        @views Ax_norm = norm(ws.scratch.base.temp_m_vec1, Inf)
        
        # note assign -Ax to ws.scratch.base.s_reconst, so we use
        # it later in this function when computing the full primal residual
        ws.scratch.base.s_reconst .= ws.scratch.base.temp_m_vec1
        ws.scratch.base.s_reconst .*= -1.0
    end

    # subtract b
    ws.scratch.base.temp_m_vec1 .-= ws.p.b

    # multiply by ρ
    ws.scratch.base.temp_m_vec1 .*= ws.method.ρ
    # add to current y_k
    @views ws.scratch.base.temp_m_vec1 .+= state_in[ws.p.n+1:end]
    # this is what's fed through the projection to dual cone
    
    if update_proj_action
        # this is what's fed into dual cone projection
        @views ws.vars.preproj_vec .= ws.scratch.base.temp_m_vec1
    end
    
    # project to dual cone K, this stores y_{k+1}
    @views project_to_dual_K!(ws.scratch.base.temp_m_vec1, ws.p.K) # ws.scratch.base.temp_m_vec1 now stores y_{k+1}

    # update ADMM-like primal residual based on reconstruction
    # of slack variable s 
    if confirm_residual_update
        ws.res.r_primal .= ws.scratch.base.temp_m_vec1
        ws.res.r_primal .-= state_in[ws.p.n+1:end]
        ws.res.r_primal .*= (1 / ws.method.ρ)

        # NOTE that from above in this function,
        # right now ws.scratch.base.s_reconst holds -Ax
        ws.scratch.base.s_reconst .+= ws.res.r_primal
        ws.scratch.base.s_reconst .+= ws.p.b

        ws.res.rp_abs = norm(ws.res.r_primal, Inf)
        s_norm = norm(ws.scratch.base.s_reconst, Inf)
        ws.res.rp_rel = ws.res.rp_abs / (1 + max(Ax_norm, s_norm, ws.p.b_norm_inf))
    end
        

    # update dynamics bit vector
    if update_proj_action
        # update in-place flags switching affine dynamics based on projection action
        # ws.proj_state.nn_mask = D_k in Goodnotes handwritten notes
        update_proj_flags!(ws.proj_state, ws.vars.preproj_vec, ws.scratch.base.temp_m_vec1, ws.p.K)
    end

    # now we go to "bulk of" x and q_n update
    @views mul!(ws.scratch.base.temp_n_vec1, ws.p.P, state_in[1:ws.p.n]) # compute P * x

    if confirm_residual_update
        Px_norm = norm(ws.scratch.base.temp_n_vec1, Inf)

        @views xTPx = dot(state_in[1:ws.p.n], ws.scratch.base.temp_n_vec1) # x^T P x
        @views cTx  = dot(ws.p.c, state_in[1:ws.p.n]) # c^T x
        @views bTy  = dot(ws.p.b, state_in[ws.p.n+1:end]) # b^T y

        # can now update gap and objective metrics
        ws.res.gap_abs = abs(xTPx + cTx + bTy) # primal-dual gap
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(abs(0.5xTPx + cTx), abs(0.5xTPx + bTy))) # relative gap
        ws.res.obj_primal = 0.5xTPx + cTx
        ws.res.obj_dual = - 0.5xTPx - bTy
    end

    # add c
    ws.scratch.base.temp_n_vec1 .+= ws.p.c

    # compute y_bar. WARNING specialised to ws.method.θ == 1.0 case
    ws.scratch.method.y_bar .= ws.scratch.base.temp_m_vec1
    # ws.scratch.method.y_bar .*= (1 + ws.method.θ)
    ws.scratch.method.y_bar .*= 2.0
    # @views ws.scratch.method.y_bar .-= ws.method.θ * state_in[ws.p.n+1:end]
    @views ws.scratch.method.y_bar .-= state_in[ws.p.n+1:end]

    # compute A' * y_bar, into temp_n_vec2
    mul!(ws.scratch.base.temp_n_vec2, ws.p.A', ws.scratch.method.y_bar)

    # compute A' * y_bar norm
    if confirm_residual_update
        ATybar_norm = norm(ws.scratch.base.temp_n_vec2, Inf)
    end

    # compute P x + A' * y_bar
    ws.scratch.base.temp_n_vec1 .+= ws.scratch.base.temp_n_vec2 # this is to be pre-multiplied by W^{-1}

    # if updating dual residual
    # NOTE use of y_bar in this residual computation, not y
    if confirm_residual_update
        @views ws.res.r_dual .= ws.scratch.base.temp_n_vec1 # assign P * x + A' * y_bar + c

        # at this point the dual residual vector is updated

        # update dual residual metrics
        ws.res.rd_abs = norm(ws.res.r_dual, Inf)
        ws.res.rd_rel = ws.res.rd_abs / (1 + max(Px_norm, ATybar_norm, ws.p.c_norm_inf)) # update relative dual residual metric
    end
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to ws.scratch.base.temp_n_vec1
    apply_inv!(ws.method.W_inv, ws.scratch.base.temp_n_vec1, ws.scratch.base.temp_n_vec2)

    # assign new iterates
    @views state_out[1:ws.p.n] .= state_in[1:ws.p.n] - ws.scratch.base.temp_n_vec1
    state_out[ws.p.n+1:end] .= ws.scratch.base.temp_m_vec1
    
    # reset residual check count
    if confirm_residual_update
        ws.res.residual_check_count[] = 0
    end

    return nothing
end

function restart_trigger(restart_period::Union{Real, Symbol}, k::Integer,
    cumsum_angles::Float64...)
    if restart_period == Inf
        return false
    elseif restart_period == :adaptive
        return all(v >= 2π for v in cumsum_angles)
        # return any(v >= 2π for v in cumsum_angles)
    else
        return k > 0 && k % restart_period == 0
    end
end

"""
(PrePPM) This function is used to modify the explicitly formed linearised operator of
the method. This is an expensive computation obviously not to be used when
the code is being run for "real" purposes.
"""
function construct_explicit_operator!(
    ws::AbstractWorkspace{T, I, V, M},
    ws_diag::DiagnosticsWorkspace
    ) where {T, I, V, M <: PrePPM} # note dispatch on PrePPM
    
    D_A = Diagonal(ws.proj_state.nn_mask)

    ws_diag.tilde_A[1:ws.p.n, 1:ws.p.n] .= I(ws.p.n) - ws_diag.W_inv_mat * ws.p.P - 2 * ws.method.ρ * ws_diag.W_inv_mat * ws.p.A' * D_A * ws.p.A
    ws_diag.tilde_A[1:ws.p.n, ws.p.n+1:end] .= - ws_diag.W_inv_mat * ws.p.A' * (2 * D_A - I(ws.p.m))
    ws_diag.tilde_A[ws.p.n+1:end, 1:ws.p.n] .= ws.method.ρ * D_A * ws.p.A
    ws_diag.tilde_A[ws.p.n+1:end, ws.p.n+1:end] .= D_A

    ws_diag.tilde_b[1:ws.p.n] .= ws_diag.W_inv_mat * (2 * ws.method.ρ * ws.p.A' * D_A * ws.p.b - ws.p.c)
    @views ws_diag.tilde_b[ws.p.n+1:end] .= - ws.method.ρ * D_A * ws.p.b

    # # compute fixed-points, ie solutions (if existing, and potentially
    # # non-unique) to the system (tilde_A - I) z = -tilde_b
    # println("Actual fixed point is: ")
    # println((tilde_A - I(ws.p.m + ws.p.n)) \ (-tilde_b))
end

"""
We use multiple dispatch to avoid branching at run time in the main loop
in optimise!.
"""
function step!(
    ws::VanillaWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    config::SolverConfig,
    record::AbstractRecord,
    full_diagnostics::Bool,
    timer::TimerOutput,
    )
    vanilla_step!(ws, record)
end

function step!(
    ws::KrylovWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    config::SolverConfig,
    record::AbstractRecord,
    full_diagnostics::Bool,
    timer::TimerOutput,
    )
    krylov_step!(ws, ws_diag, config, record, full_diagnostics, timer)
end

function step!(
    ws::AndersonWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    config::SolverConfig,
    record::AbstractRecord,
    full_diagnostics::Bool,
    timer::TimerOutput,
    )
    anderson_step!(ws, ws_diag, config, record, full_diagnostics, timer)
end

"""
Run the optimiser for the initial inputs and solver options given.
acceleration is a Symbol in {:none, :krylov, :anderson}
"""
function optimise!(
    ws::AbstractWorkspace,
    config::SolverConfig;
    setup_time::Float64 = 0.0, # time spent in set-up (seconds)
    state_ref::Union{Nothing, Vector{Float64}} = nothing,
    timer::TimerOutput,
    full_diagnostics::Bool = false,
    spectrum_plot_period::Real = Inf)

    # create views into x and y variables, along with "Arnoldi vector" q
    if ws.vars isa KrylovVariables
        @views view_state = ws.vars.state_q[:, 1]
    elseif ws.vars isa VanillaVariables || ws.vars isa AndersonVariables
        view_state = ws.vars.state # point to same place in memory
    else
        error("Unknown variable type in workspace.")
    end

    # if we are to ever explicitly form the linearised method operator
    if full_diagnostics
        ws_diag = DiagnosticsWorkspace(ws)
    else
        ws_diag = nothing
    end

    # data containers for metrics (if return_run_data == true).
    if config.run_fast
        record = NullRecord()
    else
        record = IterationRecord(ws)
    end

    # start main loop
    termination = false
    exit_status = :unknown
    # record iter start time
    loop_start_ns = time_ns()

    while !termination
        # print info and save data if requested
        push_res_to_record!(ws, record)
        push_ref_dist_to_record!(ws, record, view_state, state_ref)

        print_results(ws, config.print_mod, relative = config.print_res_rel)

        # iteration update
        # dispatches to appropriate step function,
        # which handles all logic internally
        step!(ws, ws_diag, config, record, full_diagnostics, timer)
        
        push_cosines_projs!(ws, record)
        
        if full_diagnostics && ws.k[] % spectrum_plot_period == 0
            # construct explicit operator for spectrum plotting
            construct_explicit_operator!(ws, ws_diag)
            # plot spectrum of the linearised operator
            plot_spectrum(ws_diag.tilde_A, ws.k[])
        end

        # increment iter counter
        ws.k[] += 1


        # update stopwatches
        loop_time = (time_ns() - loop_start_ns) / 1e9
        global_time = loop_time + setup_time

        # check termination conditions
        if kkt_criterion(ws, config.rel_kkt_tol)
            termination = true
            exit_status = :kkt_solved
        elseif ws isa KrylovWorkspace && ws.fp_found[]
            termination = true
            exit_status = :exact_fp_found
        elseif ws isa VanillaWorkspace && ws.k[] > config.max_iter # note this only applies to VanillaWorkspace (no acceleration!)
            termination = true
            exit_status = :max_iter
        elseif (ws isa AndersonWorkspace || ws isa KrylovWorkspace) && ws.k_operator[] > config.max_k_operator # note distinctness from ordinary :max_iter above
            termination = true
            exit_status = :max_k_operator
        elseif loop_time > config.loop_timeout
            termination = true
            exit_status = :loop_timeout
        elseif global_time > config.global_timeout
            termination = true
            exit_status = :global_timeout
        end

        if !termination
            maybe_update_rho!(ws, config, timer)
        end
    end

    # initialise results with common fields
    # ie for both run-fast set to true and to false
    metrics_final = ReturnMetrics(ws.res)
    results = Results(Dict{Symbol, Any}(), metrics_final, exit_status, ws.k[], ws isa VanillaWorkspace ? ws.k[] : ws.k_operator[])

    # store final records if run-fast is set to true
    # TODO perhaps refactor to use record structs for later plots?
    if !config.run_fast
        push_res_to_record!(ws, record)
        push_ref_dist_to_record!(ws, record, view_state, state_ref)

        print_results(ws, config.print_mod, relative = config.print_res_rel, terminated = true, exit_status = exit_status)

        # assign results
        for key in HISTORY_KEYS
            # pull record.key via getfield since key is a symbol
            results.metrics_history[key] = getfield(record, key)
        end
    else
        print_results(ws, config.print_mod, relative = config.print_res_rel, terminated = true, exit_status = exit_status)
    end

    return results, ws_diag
end

optimise!(ws::AbstractWorkspace, config::AbstractDict; kwargs...) = optimise!(ws, SolverConfig(config); kwargs...)
