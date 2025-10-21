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

"""
Compute FOM(xy) into `fom_out` and set `fp_out .= fom_out - xy`.
Both `fom_out` and `fp_out` must be preallocated vectors of length m+n.
This centralises the small sequence of operations and the associated scratch usage.
"""
function compute_fom_and_fp!(
    ws::AbstractWorkspace,
    xy_in::AbstractVector{Float64},
    fom_out::AbstractVector{Float64},
    fp_out::AbstractVector{Float64},
    )
    # compute FOM(xy_in) into fom_out (in-place)
    onecol_method_operator!(ws, xy_in, fom_out)
    # fp_out := FOM(xy_in) - xy_in
    fp_out .= fom_out .- xy_in
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

function push_to_record!(ws::AbstractWorkspace, record::NamedTuple, run_fast::Bool,
    x_sol::Union{Nothing, AbstractVector{Float64}},
    curr_x_dist::Union{Nothing, Float64},
    curr_y_dist::Union{Nothing, Float64},
    curr_xy_chardist::Union{Nothing, Float64})
    
    if run_fast
        return nothing
    else
        if ws.k[] > 0
            # NB: effective rank counts number of normalised singular values
            # larger than a specified small tolerance (eg 1e-8).
            # curr_effective_rank, curr_singval_ratio = effective_rank(record.updates_matrix, 1e-8)
            # push!(record.update_mat_ranks, curr_effective_rank)
            # push!(record.update_mat_singval_ratios, curr_singval_ratio)
            # push!(record.update_mat_iters, ws.k[])

            push!(record.pri_res_norms, ws.res.rp_abs)
            push!(record.dual_res_norms, ws.res.rd_abs)
            push!(record.primal_obj_vals, ws.res.obj_primal)
            push!(record.dual_obj_vals, ws.res.obj_dual)
        end
        if !isnothing(x_sol)
            push!(record.x_dist_to_sol, curr_x_dist)
            push!(record.y_dist_to_sol, curr_y_dist)
            push!(record.xy_chardist, curr_xy_chardist)
        end
    end
end

"""
This function returns a Boolean indicating whether the KKT relative error
has gone below the require tolerance, a common termination criterion.

For instance, PDQP preprint uses tol = 1e-3 for low accuracy and 1e-6 for high
accuracy solutions.
"""
function kkt_criterion(ws::AbstractWorkspace, kkt_tol::Float64)
    max_err = max(ws.res.rp_rel, ws.res.rd_rel, ws.res.gap_rel)
    return max_err <= kkt_tol
end

# We also define a series of helper functions defining the behaviour in
# different "types" of iterations
function krylov_usual_step!(
    ws::KrylovWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    timer::TimerOutput;
    )
    # apply method operator (to both (x, y) and Arnoldi (q) vectors)
    twocol_method_operator!(ws, true)

    # Arnoldi "step", orthogonalises the incoming basis vector
    # and updates the Hessenberg matrix appropriately, all in-place
    @timeit timer "krylov arnoldi" @views ws.arnoldi_breakdown[] = arnoldi_step!(ws.krylov_basis, ws.vars.xy_q[:, 2], ws.H, ws.givens_count)

    # if ws.krylov_operator == :tilde_A, the QR factorisation   
    # we require for the Krylov LLS subproblem is that of
    # the Hessenberg matrix MINUS an identity matrix,
    # so we must apply this immediately
    if ws.krylov_operator == :tilde_A
        ws.H[ws.givens_count[]+1, ws.givens_count[]+1] -= 1.0
    end

    # this is only when running full (expensive) diagnostics
    if !isnothing(ws_diag)
        ws_diag.H_unmod[:, ws.givens_count[] + 1] .= ws.H[:, ws.givens_count[] + 1]
    end
    
    @timeit timer "krylov givens" begin
        # apply previously existing Givens rotations to the brand new column
        # in H. if H has a single column here, then
        # ws.givens_count[] == 0 and this does nothing
        apply_existing_rotations_new_col!(ws.H, ws.givens_rotations, ws.givens_count)

        # generate new Givens rotation, store it in
        # ws.givens_rotations, and apply it in-place to the new
        # column of H
        # also increment ws.givens_count by 1
        # the EXCEPTION is if Arnoldi broke down (found invariant subspace
        # and set H(givens_count[] + 2, givens_count[] + 1) equal to 0)
        if !ws.arnoldi_breakdown[]
            # ws.givens_count[] gets incremented by 1 inside this function
            generate_store_apply_rotation!(ws.H, ws.givens_rotations, ws.givens_count)
        else
            # NOTE: in this case we increment ws.givens_count[] so as to signal
            # that we have gathered a new column to solve, but this rotation
            # is not actually generated/stored, since the Arnoldi breakdown
            # has given us a triangular principal subblock of ws.H
            # even without it
            ws.givens_count[] += 1
        end
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
    

    # we use ws.control_flags.back_to_building_krylov_basis as follows
    # as we build Krylov basis up, ws.givens_count is incremented
    # when it gets to a given trigger point, we attempt acceleration and set
    # ws.control_flags.just_tried_accel = true. after the following, recycling iteration, 
    # the counter ws.givens_count will still be at the trigger point and
    # ws.control_flags.just_tried_accel is false. however, we wish to keep builing
    # the krylov basis --- NOT attempt acceleration again. to distinguish these
    # cases we use ws.control_flags.back_to_building_krylov_basis
    # ws.control_flags.back_to_building_krylov_basis = true
    
    # allocate storage for current and previous updates
    curr_xy_update = zeros(Float64, ws.p.n + ws.p.m)
    prev_xy_update = zeros(Float64, ws.p.n + ws.p.m)

    # characteristic PPM preconditioner of the method
    if !args["run-fast"]
        char_norm_mat = [(ws.W - ws.p.P) ws.p.A'; ws.p.A I(ws.p.m) / ws.ρ]
        function char_norm_func(vector::AbstractArray{Float64})
            return sqrt(dot(vector, char_norm_mat * vector))
        end
    else
        char_norm_func = nothing
    end

    # data containers for metrics (if return_run_data == true).
    record = preallocate_record(ws, args["run-fast"], x_sol)

    # notion of iteration for COSMOAccelerators may differ!
    # we shall increment it manually only when appropriate, for use
    # with functions from COSMOAccelerators

    # start main loop
    termination = false
    exit_status = :unknown
    # record iter start time
    loop_start_ns = time_ns()

    while !termination
        # compute distance to real solution
        if !args["run-fast"]
            if x_sol !== nothing
                ws.scratch.temp_mn_vec1[1:ws.p.n] .= view_x - x_sol
                ws.scratch.temp_mn_vec1[ws.p.n+1:end] .= view_y - (-y_sol)
                curr_xy_chardist = char_norm_func(ws.scratch.temp_mn_vec1)

                @views curr_x_dist = norm(ws.scratch.temp_mn_vec1[1:ws.p.n])
                @views curr_y_dist = norm(ws.scratch.temp_mn_vec1[ws.p.n+1:end])
                curr_xy_dist = sqrt.(curr_x_dist .^ 2 .+ curr_y_dist .^ 2)
            else
                curr_x_dist = NaN
                curr_y_dist = NaN
                curr_xy_dist = NaN
                curr_xy_chardist = NaN
            end

            # print info and save data if requested
            print_results(ws, args["print-mod"], curr_xy_dist=curr_xy_dist, relative = args["print-res-rel"])
            push_to_record!(ws, record, args["run-fast"], x_sol, curr_x_dist, curr_y_dist, curr_xy_chardist)

        else
            print_results(ws, args["print-mod"], relative = args["print-res-rel"])
        end

        # krylov setup
        if args["acceleration"] == :krylov
            # copy older iterate
            @views ws.vars.xy_prev .= ws.vars.xy_q[:, 1]
            
            # acceleration attempt step
            if (ws.givens_count[] in ws.trigger_givens_counts && !ws.control_flags.back_to_building_krylov_basis) || ws.arnoldi_breakdown[]
                @timeit timer "krylov sol" ws.control_flags.krylov_status = compute_krylov_accelerant!(ws, ws.scratch.accelerated_point)

                ws.k_operator[] += 1 # one operator application in computing Krylov point (GMRES-equivalence, see Walker and Ni 2011)

                # use safeguard if the Krylov procedure suffered no breakdowns
                if ws.control_flags.krylov_status == :success

                    if full_diagnostics # best to update for use here
                        construct_explicit_operator!(ws, ws_diag)

                        # NOTE CARE might want to override with true fixed point!
                        # @warn "trying override Krylov with (PINV?) true fixed-point of current affine operation!"
                        # try
                        #     # ws.scratch.accelerated_point .= (ws_diag.tilde_A - I) \ (-ws_diag.tilde_b)
                        #     ws.scratch.accelerated_point .= pinv(ws_diag.tilde_A - I) * (-ws_diag.tilde_b)
                        #     @info "✅ successful override"
                        # catch e
                        #     @info "❌ unsuccessful override, proceeding w GNRES candidate"
                        # end
                        
                        BQ = (ws_diag.tilde_A - I) * ws.krylov_basis[:, 1:ws.givens_count[]]
                        arnoldi_relation_err = BQ - ws.krylov_basis[:, 1:ws.givens_count[] + 1] * ws_diag.H_unmod[1:ws.givens_count[] + 1, 1:ws.givens_count[]]
                        norm_arnoldi_err = norm(arnoldi_relation_err)

                        println()
                        println("---- ACCEL INFO ITER $(ws.k[]), GIVENS COUNT $(ws.givens_count[]) ----")
                        if norm_arnoldi_err > 1e-10
                            println("❌ Arnoldi relation error (size $(size(arnoldi_relation_err))) at iter $(ws.k[]): ", norm_arnoldi_err, ". norm of B*Q = $(norm(BQ))")
                        else
                            println("✅ Arnoldi relation error (size $(size(arnoldi_relation_err))) at iter $(ws.k[]): ", norm_arnoldi_err, ". norm of B*Q = $(norm(BQ))")
                        end
                    end

                    @timeit timer "fixed-point safeguard" @views ws.control_flags.accepted_accel = accel_fp_safeguard!(ws, ws_diag, ws.vars.xy_q[:, 1], ws.scratch.accelerated_point, args["safeguard-factor"], record, full_diagnostics)

                    ws.k_operator[] += 1 # note: only 1 because we also count another when assigning recycled iterate in the following iteration
                else
                    ws.control_flags.accepted_accel = false
                end

                if ws.control_flags.accepted_accel
                    # increment effective iter counter (ie excluding unsuccessful acc attempts)
                    ws.k_eff[] += 1
                    
                    if !args["run-fast"]
                        @views curr_xy_update .= ws.scratch.accelerated_point - ws.vars.xy_q[:, 1]
                        push!(record.acc_step_iters, ws.k[])
                        record.updates_matrix .= 0.0
                        record.current_update_mat_col[] = 1

                        push!(record.xy_step_norms, norm(curr_xy_update))
                        push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                    end

                    # assign actually
                    ws.vars.xy_q[:, 1] .= ws.scratch.accelerated_point
                else
                    if !args["run-fast"]
                        @views curr_xy_update .= 0.0
                        push!(record.xy_step_norms, NaN)
                        push!(record.xy_step_char_norms, NaN)
                    end
                end

                # # prevent recording 0 or very large update norm in any case
                # if !args["run-fast"]
                #     push!(record.xy_step_norms, NaN)
                #     push!(record.xy_step_char_norms, NaN)
                # end

                # reset krylov acceleration data when memory is fully
                # populated, regardless of success/failure
                if ws.givens_count[] + 1 == ws.mem || ws.arnoldi_breakdown[]
                    ws.krylov_basis .= 0.0
                    ws.H .= 0.0
                    ws.givens_count[] = 0

                    ws.arnoldi_breakdown[] = false
                end

                ws.control_flags.just_tried_accel = true
                ws.control_flags.back_to_building_krylov_basis = true
            # standard step in the krylov setup (two columns)
            else
                # increment effective iter counter (ie excluding unsuccessful acc attempts)
                ws.k_eff[] += 1

                ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
                
                # if we just had an accel attempt, we recycle work carried out
                # during the acceleration acceptance criterion check
                # we also reinit the Krylov orthogonal basis 
                if ws.k[] == 0 # special case in initial iteration
                    # TODO sort carefully what to with these separate scratch vectors
                    @views onecol_method_operator!(ws, ws.vars.xy_q[:, 1], ws.scratch.initial_vec, true)
                    @views ws.vars.xy_q[:, 1] .= ws.scratch.initial_vec

                    # fills in first column of ws.krylov_basis
                    init_krylov_basis!(ws) # ws.H is zeros at this point still
                
                elseif ws.control_flags.just_tried_accel
                    # recycle working (x, y) iterate
                    ws.vars.xy_q[:, 1] .= ws.scratch.xy_recycled
                    
                    # also re-initialise Krylov basis IF we'd filled up
                    # the entire memory
                    if ws.givens_count[] == 0 # just wiped Krylov basis clean
                        # fills in first column of ws.krylov_basis
                        init_krylov_basis!(ws, ws.control_flags.krylov_status) # ws.H is zeros at this point still
                        
                        if full_diagnostics
                            ws_diag.H_unmod .= 0.0
                        end
                    end
                else # the usual case

                    krylov_usual_step!(ws, ws_diag, timer)

                    # ws.givens_count has had "time" to be incremented past
                    # a trigger point, so we can relax this flag in order
                    # to allow the next acceleration attempt when it comes up
                    ws.control_flags.back_to_building_krylov_basis = false
                end

                # record iteration data if requested
                if !args["run-fast"]
                    @views curr_xy_update .= ws.vars.xy_q[:, 1] - ws.vars.xy_prev
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                    insert_update_into_matrix!(record.updates_matrix, curr_xy_update, record.current_update_mat_col)
                end

                # reset krylov acceleration flag
                ws.control_flags.just_tried_accel = false
            end
        elseif args["acceleration"] == :anderson
            # Anderson acceleration attempt
            if ws.composition_counter[] == args["anderson-interval"]
                # ws.vars.xy might be overwritten, so we take note of it here
                # this is for the sole purpose of checking the norm of the
                # step just below in this code branch
                ws.scratch.xy_pre_overwrite .= ws.vars.xy

                # attempt acceleration step. if successful (ie no numerical
                # problems), this overwites ws.vars.xy with accelerated step
                @timeit timer "anderson accel" COSMOAccelerators.accelerate!(ws.vars.xy, ws.vars.xy_prev, ws.accelerator, 0)
                
                ws.scratch.accelerated_point .= ws.vars.xy # accelerated point if no numerics blowups
                ws.vars.xy .= ws.scratch.xy_pre_overwrite # retake the iterate from which we are attempting acceleration right now

                # ws.accelerator.success only indicates there was no
                # numerical blowup in the COSMOAccelerators.accelerate!
                # internals
                if ws.accelerator.success
                    # at this point:
                    # ws.vars.xy contains the iterate at which we stopped
                    # when we began to pull the accelerate! levers;
                    # ws.scratch.accelerated_point contains the candidate
                    # acceleration point, which is legitimately different
                    # from ws.vars.xy
                    @timeit timer "fixed-point safeguard" @views ws.control_flags.accepted_accel = accel_fp_safeguard!(ws, ws_diag, ws.vars.xy, ws.scratch.accelerated_point, args["safeguard-factor"], record, full_diagnostics)

                    ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application

                    if ws.control_flags.accepted_accel
                        ws.k_eff[] += 1 # increment effective iter counter (ie excluding unsuccessful acc attempts)

                        if !args["run-fast"]
                            curr_xy_update .= ws.scratch.accelerated_point - ws.vars.xy
                            push!(record.acc_step_iters, ws.k[])
                            record.updates_matrix .= 0.0
                            record.current_update_mat_col[] = 1

                            push!(record.xy_step_norms, norm(curr_xy_update))
                            push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                        end

                        # assign actually
                        ws.vars.xy .= ws.scratch.accelerated_point
                    elseif !args["run-fast"] # prevent recording zero update norm
                        push!(record.xy_step_norms, NaN)
                        push!(record.xy_step_char_norms, NaN)
                    end


                    # note that this flag serves for us to use recycled
                    # work from the fixed-point safeguard step when
                    # the next iteration comes around
                    ws.control_flags.just_tried_accel = true
                elseif !args["run-fast"] # prevent recording zero update norm
                    push!(record.xy_step_norms, NaN)
                    push!(record.xy_step_char_norms, NaN)
                end

                # set up the first x to be composed anderon-interval times
                # before being fed to accelerator's update! method
                ws.vars.xy_into_accelerator .= ws.vars.xy # this is T^0(ws.vars.xy)
                ws.composition_counter[] = 0
            else # standard iteration with Anderson acceleration
                ws.k_eff[] += 1
                ws.k_vanilla[] += 1
                ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
                
                # copy older iterate before iterating
                ws.vars.xy_prev .= ws.vars.xy

                # we compute new iterate: if we just tried acceleration,
                # we use the recycled variable stored during the fixed-point
                # safeguarding step
                if ws.control_flags.just_tried_accel
                    ws.vars.xy .= ws.scratch.xy_recycled
                else
                    # TODO sort carefully what to with these separate scratch vectors
                    onecol_method_operator!(ws, ws.vars.xy, ws.scratch.swap_vec, true)
                    # swap contents of ws.vars.xy and ws.scratch.swap_vec
                    custom_swap!(ws.vars.xy, ws.scratch.swap_vec, ws.scratch.temp_mn_vec1)
                end

                # just applied onecol operator, so we increment
                # composition counter
                ws.composition_counter[] += 1

                if ws.composition_counter[] == args["anderson-interval"]
                    @timeit timer "anderson update" COSMOAccelerators.update!(ws.accelerator, ws.vars.xy, ws.vars.xy_into_accelerator, 0)
                end

                if !args["run-fast"]
                    curr_xy_update .= ws.vars.xy - ws.vars.xy_prev

                    # record iteration data here
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                end

                # cannot recycle anything at next iteration
                ws.control_flags.just_tried_accel = false
            end
        else # no acceleration!
            # copy older iterate before iterating
            ws.vars.xy_prev .= ws.vars.xy

            # TODO sort carefully what to with these separate scratch vectors
            onecol_method_operator!(ws, ws.vars.xy, ws.scratch.swap_vec, true)
            # swap contents of ws.vars.xy and ws.scratch.swap_vec
            custom_swap!(ws.vars.xy, ws.scratch.swap_vec, ws.scratch.temp_mn_vec1)
            # now ws.vars.xy contains newer iterate,  while
            # ws.scratch.swap_vec contains older one

            if !args["run-fast"]
                curr_xy_update .= ws.vars.xy - ws.scratch.swap_vec

                # record iteration data here
                push!(record.xy_step_norms, norm(curr_xy_update))
                push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
            end
        end
        
        if !args["run-fast"]
            # store cosine between last two iterate updates
            if ws.k[] >= 1
                xy_prev_updates_cos = abs(dot(curr_xy_update, prev_xy_update) / (norm(curr_xy_update) * norm(prev_xy_update)))
                push!(record.xy_update_cosines, xy_prev_updates_cos)
            end
            prev_xy_update .= curr_xy_update

            # store active set flags
            push!(record.record_proj_flags, ws.proj_flags)
        end

        
        if full_diagnostics && ws.k[] % spectrum_plot_period == 0
            # construct explicit operator for spectrum plotting
            construct_explicit_operator!(ws, ws_diag)
            # plot spectrum of the linearised operator
            plot_spectrum(ws_diag.tilde_A, ws.k[])
        end

        # Notion of a previous iterate step makes sense (again).
        just_restarted = false

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
    if !args["run-fast"]
        curr_xy_chardist = !isnothing(x_sol) ? char_norm_func([view_x - x_sol; view_y + y_sol]) : nothing
        curr_x_dist = !isnothing(x_sol) ? norm(view_x - x_sol) : nothing
        curr_y_dist = !isnothing(y_sol) ? norm(view_y - (-y_sol)) : nothing

        push!(record.primal_obj_vals, ws.res.obj_primal)
        push!(record.dual_obj_vals, ws.res.obj_dual)
        push!(record.pri_res_norms, ws.res.rp_abs)
        push!(record.dual_res_norms, ws.res.rd_abs)
        if !isnothing(x_sol)
            push!(record.x_dist_to_sol, curr_x_dist)
            push!(record.y_dist_to_sol, curr_y_dist)
            push!(record.xy_chardist, curr_xy_chardist)
            curr_xy_dist = sqrt.(curr_x_dist .^ 2 .+ curr_y_dist .^ 2)
        end
        
        print_results(ws, args["print-mod"], curr_xy_dist=curr_xy_dist, relative = args["print-res-rel"], terminated = true, exit_status = exit_status)

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