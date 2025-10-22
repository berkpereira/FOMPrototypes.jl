using LinearAlgebra
using Random
using Printf
using Infiltrator
using TimerOutputs
import COSMOAccelerators

# data to store history of when a run uses args["run-fast"] == false
const HISTORY_KEYS = [
    :primal_obj_vals,
    :dual_obj_vals,
    :pri_res_norms,
    :dual_res_norms,
    :record_proj_flags,
    :x_dist_to_sol,
    :y_dist_to_sol,
    :xy_chardist,
    :update_mat_iters,
    :update_mat_ranks,
    :update_mat_singval_ratios,
    :acc_step_iters,
    :linesearch_iters,
    :xy_step_norms,
    :xy_step_char_norms,
    :xy_update_cosines,

    :fp_metric_ratios,
    :acc_attempt_iters,
]

"""
Efficiently computes FOM iteration at point xy_in, and stores it in xy_out.
Vector xy_in is left unchanged.

Set update_res_flags to true iff the method is being used for a bona fide
vanilla FOM iteration, where the active set flags and working residual metrics
should be updated by recycling the computational work (mostly mat-vec
products involving A, A', and P) which is carried out here regardless.

This function is somewhat analogous to twocol_method_operator!, but it is
concerned only with applying the iteration to some actual iterate, ie
it does not consider any sort of concurrent Arnoldi-like process.

This is convenient whenever we need the method's operator but without
updating an Arnoldi vector, eg when solving the upper Hessenber linear least
squares problem following from the Krylov acceleration subproblem.

Note that xy_in is the "current" iterate as a single vector in R^{n+m}.
New iterate is written (in-place) into xy_out input vector.
"""
function onecol_method_operator!(
    ws::AbstractWorkspace,
    xy_in::AbstractVector{Float64},
    xy_out::AbstractVector{Float64},
    update_res_flags::Bool = false
    )

    @views mul!(ws.scratch.temp_m_vec, ws.p.A, xy_in[1:ws.p.n]) # compute A * x

    if update_res_flags
        @views Ax_norm = norm(ws.scratch.temp_m_vec, Inf)
    end

    ws.scratch.temp_m_vec .-= ws.p.b # subtract b from A * x

    # if updating primal residual
    if update_res_flags
        @views ws.res.r_primal .= ws.scratch.temp_m_vec # assign A * x - b
        project_to_dual_K!(ws.res.r_primal, ws.p.K) # project to dual cone (TODO: sort out for more general cones than just QP case)

        # at this point the primal residual is updated
        ws.res.rp_abs = norm(ws.res.r_primal, Inf)
        ws.res.rp_rel = ws.res.rp_abs / (1 + max(Ax_norm, ws.p.b_norm_inf))
    end

    ws.scratch.temp_m_vec .*= ws.ρ
    @views ws.scratch.temp_m_vec .+= xy_in[ws.p.n+1:end] # add current
    if update_res_flags
        @views ws.vars.preproj_y .= ws.scratch.temp_m_vec # this is what's fed into dual cone projection operator
    end
    @views project_to_dual_K!(ws.scratch.temp_m_vec, ws.p.K) # ws.scratch.temp_m_vec now stores y_{k+1}

    if update_res_flags
        # update in-place flags switching affine dynamics based on projection action
        # ws.proj_flags = D_k in Goodnotes handwritten notes
        update_proj_flags!(ws.proj_flags, ws.vars.preproj_y, ws.scratch.temp_m_vec)
    end

    # now we go to "bulk of" x and q_n update
    @views mul!(ws.scratch.temp_n_vec1, ws.p.P, xy_in[1:ws.p.n]) # compute P * x

    if update_res_flags
        Px_norm = norm(ws.scratch.temp_n_vec1, Inf)

        @views xTPx = dot(xy_in[1:ws.p.n], ws.scratch.temp_n_vec1) # x^T P x
        @views cTx  = dot(ws.p.c, xy_in[1:ws.p.n]) # c^T x
        @views bTy  = dot(ws.p.b, xy_in[ws.p.n+1:end]) # b^T y

        # can now update gap and objective metrics
        ws.res.gap_abs = abs(xTPx + cTx + bTy) # primal-dual gap
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(abs(0.5xTPx + cTx), abs(0.5xTPx + bTy))) # relative gap
        ws.res.obj_primal = 0.5xTPx + cTx
        ws.res.obj_dual = - 0.5xTPx - bTy
    end

    ws.scratch.temp_n_vec1 .+= ws.p.c # add linear part of objective, c, to P * x

    # TODO reduce allocations in this mul! call: pass another temp vector?
    # consider dedicated scratch storage for y_bar, akin to what
    # we do in twocol_method_operator!
    @views mul!(ws.scratch.temp_n_vec2, ws.p.A', (1 + ws.θ) * ws.scratch.temp_m_vec - ws.θ * xy_in[ws.p.n+1:end]) # compute A' * y_bar

    if update_res_flags
        ATybar_norm = norm(ws.scratch.temp_n_vec2, Inf)
    end

    # now compute P x + A^T y_bar
    ws.scratch.temp_n_vec1 .+= ws.scratch.temp_n_vec2 # this is to be pre-multiplied by W^{-1}

    # if updating dual residual (NOTE use of y_bar)
    if update_res_flags
        @views ws.res.r_dual .= ws.scratch.temp_n_vec1 # assign P * x + A' * y_bar + c

        # at this point the dual residual vector is updated

        # update dual residual metrics
        ws.res.rd_abs = norm(ws.res.r_dual, Inf)
        ws.res.rd_rel = ws.res.rd_abs / (1 + max(Px_norm, ATybar_norm, ws.p.c_norm_inf)) # update relative dual residual metric
    end
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to ws.scratch.temp_n_vec1
    apply_inv!(ws.W_inv, ws.scratch.temp_n_vec1)

    # assign new iterates
    @views xy_out[1:ws.p.n] .= xy_in[1:ws.p.n] - ws.scratch.temp_n_vec1
    xy_out[ws.p.n+1:end] .= ws.scratch.temp_m_vec
    
    return nothing
end

function iter_x!(x::AbstractVector{Float64},
    gradient_preop::AbstractInvOp,
    P::AbstractMatrix{Float64},
    c::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    y_bar::AbstractVector{Float64},
    prev_x::AbstractVector{Float64},
    store_step_vec::AbstractVector{Float64})

    prev_x .= x # update prev_x
    
    # store_step_vec is used to store the latest x iterate delta
    store_step_vec .= - gradient_preop(P * x + c + A' * y_bar)

    x .+= store_step_vec

    return
end

function iter_y!(
    y::AbstractVector{Float64},
    unproj_y::AbstractVector{Float64},
    x::AbstractVector{Float64},
    A::AbstractMatrix{Float64},
    b::AbstractVector{Float64},
    K::Vector{Clarabel.SupportedCone},
    ρ::Float64,
    temp_store_step_vec::AbstractVector{Float64},
    store_step_vec::AbstractVector{Float64})

    unproj_y .= y + ρ * (A * x - b)
    temp_store_step_vec .= project_to_dual_K(unproj_y, K)
    store_step_vec .= temp_store_step_vec - y
    y .= temp_store_step_vec

    return
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

function preallocate_record(ws::AbstractWorkspace, run_fast::Bool,
    x_sol::Union{Nothing, AbstractVector{Float64}})
    if run_fast
        return nothing
    else
        return (
            primal_obj_vals = Float64[],
            pri_res_norms = Float64[],
            dual_obj_vals = Float64[],
            dual_res_norms = Float64[],
            record_proj_flags = Vector{Vector{Bool}}(),
            update_mat_ranks = Float64[],
            update_mat_singval_ratios = Float64[],
            update_mat_iters = Int[],
            acc_step_iters = Int[],
            acc_attempt_iters = Int[],
            linesearch_iters = Int[],
            xy_step_norms = Float64[],
            xy_step_char_norms = Float64[], # record method's "char norm" of the updates
            xy_update_cosines = Float64[],
            x_dist_to_sol = !isnothing(x_sol) ? Float64[] : nothing,
            y_dist_to_sol = !isnothing(x_sol) ? Float64[] : nothing,
            xy_chardist = !isnothing(x_sol) ? Float64[] : nothing,
            current_update_mat_col = Ref(1),
            updates_matrix = zeros(ws.p.n + ws.p.m, 20),

            fp_metric_ratios = Float64[],

        )
    end
end

"""
This function is used to modify the explicitly formed linearised operator of
the method. This is an expensive computation obviously not to be used when
the code is being run for "real" purposes.
"""
function construct_explicit_operator!(
    ws::AbstractWorkspace,
    ws_diag::DiagnosticsWorkspace
    )
    
    D_A = Diagonal(ws.proj_flags)

    ws_diag.tilde_A[1:ws.p.n, 1:ws.p.n] .= I(ws.p.n) - ws_diag.W_inv_mat * ws.p.P - 2 * ws.ρ * ws_diag.W_inv_mat * ws.p.A' * D_A * ws.p.A
    ws_diag.tilde_A[1:ws.p.n, ws.p.n+1:end] .= - ws_diag.W_inv_mat * ws.p.A' * (2 * D_A - I(ws.p.m))
    ws_diag.tilde_A[ws.p.n+1:end, 1:ws.p.n] .= ws.ρ * D_A * ws.p.A
    ws_diag.tilde_A[ws.p.n+1:end, ws.p.n+1:end] .= D_A

    ws_diag.tilde_b[1:ws.p.n] .= ws_diag.W_inv_mat * (2 * ws.ρ * ws.p.A' * D_A * ws.p.b - ws.p.c)
    @views ws_diag.tilde_b[ws.p.n+1:end] .= - ws.ρ * D_A * ws.p.b

    # # compute fixed-points, ie solutions (if existing, and potentially
    # # non-unique) to the system (tilde_A - I) z = -tilde_b
    # println("Actual fixed point is: ")
    # println((tilde_A - I(ws.p.m + ws.p.n)) \ (-tilde_b))
end

"""
Run the optimiser for the initial inputs and solver options given.
acceleration is a Symbol in {:none, :krylov, :anderson}
"""
function optimise!(
    ws::AbstractWorkspace,
    args::Dict{String, T};
    setup_time::Float64 = 0.0, # time spent in set-up (seconds)
    x_sol::Union{Nothing, Vector{Float64}} = nothing,
    y_sol::Union{Nothing, Vector{Float64}} = nothing,
    timer::TimerOutput,
    full_diagnostics::Bool = false,
    spectrum_plot_period::Real = Inf) where T

    # create views into x and y variables, along with "Arnoldi vector" q
    if ws.vars isa KrylovVariables
        @views view_x = ws.vars.xy_q[1:ws.p.n, 1]
        @views view_y = ws.vars.xy_q[ws.p.n+1:end, 1]
        @views view_q = ws.vars.xy_q[:, 2]
    elseif ws.vars isa VanillaVariables || ws.vars isa AndersonVariables
        @views view_x = ws.vars.xy[1:ws.p.n]
        @views view_y = ws.vars.xy[ws.p.n+1:end]
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
    if args["run-fast"]
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
        push_ref_dist_to_record!(ws, record, view_x, view_y, x_sol, y_sol)

        print_results(ws, args["print-mod"], relative = args["print-res-rel"])

        # krylov setup
        if args["acceleration"] == :krylov
            krylov_step!(ws, ws_diag, args, record, full_diagnostics, timer)
        elseif args["acceleration"] == :anderson
            anderson_step!(ws, ws_diag, args, record, full_diagnostics, timer)
        else # no acceleration!
            vanilla_step!(ws, args, record)
        end
        
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
        if kkt_criterion(ws, args["rel-kkt-tol"])
            termination = true
            exit_status = :kkt_solved
        elseif ws isa KrylovWorkspace && ws.fp_found[]
            termination = true
            exit_status = :exact_fp_found
        elseif ws isa VanillaWorkspace && ws.k[] > args["max-iter"] # note this only applies to VanillaWorkspace (no acceleration!)
            termination = true
            exit_status = :max_iter
        elseif (ws isa AndersonWorkspace || ws isa KrylovWorkspace) && ws.k_operator[] > args["max-k-operator"] # note distinctness from ordinary :max_iter above
            termination = true
            exit_status = :max_k_operator
        elseif loop_time > args["loop-timeout"]
            termination = true
            exit_status = :loop_timeout
        elseif global_time > args["global-timeout"]
            termination = true
            exit_status = :global_timeout
        end
    end

    # initialise results with common fields
    # ie for both run-fast set to true and to false
    metrics_final = ReturnMetrics(ws.res)
    results = Results(Dict{Symbol, Any}(), metrics_final, exit_status, ws.k[], ws isa VanillaWorkspace ? ws.k[] : ws.k_operator[])

    # store final records if run-fast is set to true
    # TODO perhaps refactor to use record structs for later plots?
    if !args["run-fast"]
        push_res_to_record!(ws, record)
        push_ref_dist_to_record!(ws, record, view_x, view_y, x_sol, y_sol)
        
        print_results(ws, args["print-mod"], relative = args["print-res-rel"], terminated = true, exit_status = exit_status)

        # assign results
        for key in HISTORY_KEYS
            # pull record.key via getfield since key is a symbol
            results.metrics_history[key] = getfield(record, key)
        end
    else
        print_results(ws, args["print-mod"], relative = args["print-res-rel"], terminated = true, exit_status = exit_status)
    end

    return results, ws_diag
end