using LinearAlgebra
using Plots
using Random
using Printf
using Infiltrator
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
]

"""
In our framing of the FOM as an affine operator with generally varying
dynamics, we want to apply the operator concurrently to two vectors.
One of them is the iterate itself, where the way in which to do this is obvious
and follows from the method's definition.
The other is a so-called (by us) Arnoldi vector, which is used in building a
basis for a Krylov subspace for GMRES/Anderson acceleration. The application
of the FOM operator to this Arnoldi vector requires the interpretation of the
method's non-affine dynamics --- namely, the projection to the dual cone --- as
an affine operation. The action of this projection for some given current y
iterate is what defines the affine dynamics we interpret the FOM to have at a
given iteration.

Determining the dynamics of the affine operator happens at the same time as we
should be applying some of the operator's action to both the vectors of
interest. Mainly for this reason, it makes sense to encapsulate both of these
tasks within this single function.

Depending on arguments given, function can also recycle A * x, A' * y, and

temp_n_mat1 should be of dimension n by 2
temp_m_mat should be of dimension m by 2
temp_m_vec should be of dimension m
"""
function twocol_method_operator!(ws::KrylovWorkspace,
    temp_n_mat1::Matrix{Float64},
    temp_n_mat2::Matrix{Float64},
    temp_m_mat::Matrix{Float64},
    temp_n_vec_complex1::Vector{ComplexF64},
    temp_n_vec_complex2::Vector{ComplexF64},
    temp_m_vec_complex::Vector{ComplexF64},
    update_res_flags::Bool = false)
    
    # working variable is ws.vars.xy_q, with two columns
    
    @views two_col_mul!(temp_m_mat, ws.p.A, ws.vars.xy_q[1:ws.p.n, :], temp_n_vec_complex1, temp_m_vec_complex) # compute A * [x, q_n]

    if update_res_flags
        @views Ax_norm = norm(temp_m_mat[:, 1], Inf)
    end

    @views temp_m_mat[:, 1] .-= ws.p.b # subtract b from A * x (but NOT from A * q_n)

    # if updating primal residual
    if update_res_flags
        @views ws.res.r_primal .= temp_m_mat[:, 1] # assign A * x - b
        project_to_dual_K!(ws.res.r_primal, ws.p.K) # project to dual cone (TODO: sort out for more general cones than just QP case)

        # at this point the primal residual vector is updated

        # update primal residual metrics
        ws.res.rp_abs = norm(ws.res.r_primal, Inf)
        ws.res.rp_rel = ws.res.rp_abs / (1 + max(Ax_norm, ws.p.b_norm_inf))
    end

    temp_m_mat .*= ws.ρ
    @views temp_m_mat .+= ws.vars.xy_q[ws.p.n+1:end, :] # add current y
    @views ws.vars.preproj_y .= temp_m_mat[:, 1] # this is what's fed into dual cone projection operator
    @views project_to_dual_K!(temp_m_mat[:, 1], ws.p.K) # temp_m_mat[:, 1] now stores y_{k+1}
    
    # update in-place flags switching affine dynamics based on projection action
    # ws.proj_flags = D_k in Goodnotes handwritten notes
    if update_res_flags
        @views update_proj_flags!(ws.proj_flags, ws.vars.preproj_y, temp_m_mat[:, 1])
    end

    # can now compute the bit of q corresponding to the y iterate
    @views temp_m_mat[:, 2] .*= ws.proj_flags

    # now compute bar iterates (y and q_m) concurrently
    # TODO reduce memory allocation in this line...
    ws.vars.y_qm_bar .= temp_m_mat
    ws.vars.y_qm_bar .*= (1 + ws.θ)
    @views ws.vars.y_qm_bar .+= -ws.θ .* ws.vars.xy_q[ws.p.n+1:end, :]

    if ws.krylov_operator == :B # ie use B = A - I as krylov operator
        @views temp_m_mat[:, 2] .-= ws.vars.xy_q[ws.p.n+1:end, 2] # add -I component
    end
    # NOTE: temp_m_mat[:, 2] now stores UPDATED q_m
    
    # ASSIGN new y and q_m to ws variables
    ws.vars.xy_q[ws.p.n+1:end, :] .= temp_m_mat

    # now we go to "bulk of" x and q_n update
    @views two_col_mul!(temp_n_mat1, ws.p.P, ws.vars.xy_q[1:ws.p.n, :], temp_n_vec_complex1, temp_n_vec_complex2) # compute P * [x, q_n]

    if update_res_flags
        @views Px_norm = norm(temp_n_mat1[:, 1], Inf)
        
        @views xTPx = dot(ws.vars.xy_q[1:ws.p.n, 1], temp_n_mat1[:, 1]) # x^T P x
        @views cTx  = dot(ws.p.c, ws.vars.xy_q[1:ws.p.n, 1]) # c^T x
        @views bTy  = dot(ws.p.b, ws.vars.xy_q[ws.p.n+1:end, 1]) # b^T y

        # can now update gap and objective metrics
        ws.res.gap_abs = abs(xTPx + cTx + bTy) # primal-dual gap
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(0.5xTPx + cTx, 0.5xTPx + bTy)) # relative gap
        ws.res.obj_primal = 0.5xTPx + cTx
        ws.res.obj_dual = - 0.5xTPx - bTy
    end

    @views temp_n_mat1[:, 1] .+= ws.p.c # add linear part of objective, c, to P * x (but NOT to P * q_n)

    @views two_col_mul!(temp_n_mat2, ws.p.A', ws.vars.y_qm_bar, temp_m_vec_complex, temp_n_vec_complex1) # compute A' * [y_bar, q_m_bar]

    if update_res_flags
        @views ATybar_norm = norm(temp_n_mat2[:, 1], Inf)
    end

    # now compute P x + A^T y_bar
    temp_n_mat1 .+= temp_n_mat2 # this is what is pre-multiplied by W^{-1}

    # if updating dual residual (NOTE use of y_bar)
    if update_res_flags
        @views ws.res.r_dual .= temp_n_mat1[:, 1] # assign P * x + A' * y_bar + c

        # at this point the dual residual vector is updated

        # update dual residual metrics
        ws.res.rd_abs = norm(ws.res.r_dual, Inf)
        ws.res.rd_rel = ws.res.rd_abs / (1 + max(Px_norm, ATybar_norm, ws.p.c_norm_inf)) # update relative dual residual metric
    end
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to temp_n_mat1
    if ws.W_inv isa CholeskyInvOp # eg in ADMM, PDHG
        # need a working complex vector for efficient Cholesky inversion
        # of two columns simultaneously
        apply_inv!(ws.W_inv, temp_n_mat1, temp_n_vec_complex1)
    else
        # diagonal inversion is simpler
        apply_inv!(ws.W_inv, temp_n_mat1)
    end

    # ASSIGN new x and q_n: subtract off what we just computed, both columns
    if ws.krylov_operator == :tilde_A
        @views ws.vars.xy_q[1:ws.p.n, :] .-= temp_n_mat1
    else # ie use B = A - I as krylov operator
        @views ws.vars.xy_q[1:ws.p.n, 1] .-= temp_n_mat1[:, 1] # as above
        @views ws.vars.xy_q[1:ws.p.n, 2] .= -temp_n_mat1[:, 2] # simpler
    end

    return nothing
end

"""
This function is somewhat analogous to twocol_method_operator!, but it is
concerned only with applying the iteration to some actual iterate, ie
it does not consider any sort of concurrent Arnoldi-like process.

This is convenient whenever we need the method's operator but without
updating an Arnoldi vector, eg when solving the upper Hessenber linear least
squares problem following from the Krylov acceleration subproblem.

Note that xy is the "current" iterate as a single vector in R^{n+m}.
New iterate is written (in-place) into result_vec input vector.
"""
function onecol_method_operator!(ws::AbstractWorkspace,
    xy::AbstractVector{Float64},
    result_vec::AbstractVector{Float64},
    temp_n_vec1::AbstractVector{Float64},
    temp_n_vec2::AbstractVector{Float64},
    temp_m_vec::AbstractVector{Float64},
    update_res_flags::Bool = false)

    @views mul!(temp_m_vec, ws.p.A, xy[1:ws.p.n]) # compute A * x

    if update_res_flags
        @views Ax_norm = norm(temp_m_vec, Inf)
    end

    temp_m_vec .-= ws.p.b # subtract b from A * x

    # if updating primal residual
    if update_res_flags
        @views ws.res.r_primal .= temp_m_vec # assign A * x - b
        project_to_dual_K!(ws.res.r_primal, ws.p.K) # project to dual cone (TODO: sort out for more general cones than just QP case)

        # at this point the primal residual is updated
        ws.res.rp_abs = norm(ws.res.r_primal, Inf)
        ws.res.rp_rel = ws.res.rp_abs / (1 + max(Ax_norm, ws.p.b_norm_inf))
    end

    temp_m_vec .*= ws.ρ
    @views temp_m_vec .+= xy[ws.p.n+1:end] # add current
    if update_res_flags
        @views ws.vars.preproj_y .= temp_m_vec # this is what's fed into dual cone projection operator
    end
    @views project_to_dual_K!(temp_m_vec, ws.p.K) # temp_m_vec now stores y_{k+1}

    if update_res_flags
        # update in-place flags switching affine dynamics based on projection action
        # ws.proj_flags = D_k in Goodnotes handwritten notes
        update_proj_flags!(ws.proj_flags, ws.vars.preproj_y, temp_m_vec)
    end

    # now we go to "bulk of" x and q_n update
    @views mul!(temp_n_vec1, ws.p.P, xy[1:ws.p.n]) # compute P * x

    if update_res_flags
        Px_norm = norm(temp_n_vec1, Inf)

        @views xTPx = dot(xy[1:ws.p.n], temp_n_vec1) # x^T P x
        @views cTx  = dot(ws.p.c, xy[1:ws.p.n]) # c^T x
        @views bTy  = dot(ws.p.b, xy[ws.p.n+1:end]) # b^T y

        # can now update gap and objective metrics
        ws.res.gap_abs = abs(xTPx + cTx + bTy) # primal-dual gap
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(0.5xTPx + cTx, 0.5xTPx + bTy)) # relative gap
        ws.res.obj_primal = 0.5xTPx + cTx
        ws.res.obj_dual = - 0.5xTPx - bTy
    end

    temp_n_vec1 .+= ws.p.c # add linear part of objective, c, to P * x

    # TODO reduce allocations in this mul! call: pass another temp vector?
    # consider dedicated scratch storage for y_bar, akin to what
    # we do in twocol_method_operator!
    @views mul!(temp_n_vec2, ws.p.A', (1 + ws.θ) * temp_m_vec - ws.θ * xy[ws.p.n+1:end]) # compute A' * y_bar

    if update_res_flags
        ATybar_norm = norm(temp_n_vec2, Inf)
    end

    # now compute P x + A^T y_bar
    temp_n_vec1 .+= temp_n_vec2 # this is to be pre-multiplied by W^{-1}

    # if updating dual residual (NOTE use of y_bar)
    if update_res_flags
        @views ws.res.r_dual .= temp_n_vec1 # assign P * x + A' * y_bar + c

        # at this point the dual residual vector is updated

        # update dual residual metrics
        ws.res.rd_abs = norm(ws.res.r_dual, Inf)
        ws.res.rd_rel = ws.res.rd_abs / (1 + max(Px_norm, ATybar_norm, ws.p.c_norm_inf)) # update relative dual residual metric
    end
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to temp_n_vec1
    apply_inv!(ws.W_inv, temp_n_vec1)

    # assign new iterates
    @views result_vec[1:ws.p.n] .= xy[1:ws.p.n] - temp_n_vec1
    result_vec[ws.p.n+1:end] .= temp_m_vec
    
    return nothing
end

function iter_x!(x::AbstractVector{Float64},
    gradient_preop::AbstractInvOp,
    P::Symmetric,
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

function iter_y!(y::AbstractVector{Float64},
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

# check for acceptance of a Krylov acceleration candidate.
function accept_acc_candidate(ws::KrylovWorkspace,
    current_xy::AbstractVector{Float64},
    accelerated_xy::AbstractVector{Float64},
    temp_mn_vec1::AbstractVector{Float64},
    temp_mn_vec2::AbstractVector{Float64},
    temp_n_vec1::AbstractVector{Float64},
    temp_n_vec2::AbstractVector{Float64},
    temp_m_vec::AbstractVector{Float64};
    char_norm_func::Union{Function, Nothing} = nothing,)

    # TODO sort this fixed-point residual norm checking out
    res_norm_func = char_norm_func === nothing ? norm : char_norm_func

    # compute fixed-point residual at standard next iterate
    onecol_method_operator!(ws, current_xy, temp_mn_vec1, temp_n_vec1, temp_n_vec2, temp_m_vec) # iterate from current_xy
    onecol_method_operator!(ws, temp_mn_vec1, temp_mn_vec2, temp_n_vec1, temp_n_vec2, temp_m_vec) # iterate from temp_mn_vec1, which follows from current_xy

    # this computes fixed point residual OF THE standard iterate from
    # the current iterate
    fp_res_standard = res_norm_func(temp_mn_vec2 - temp_mn_vec1)

    # now compute fixed-point residual at accelerated iterate
    onecol_method_operator!(ws, accelerated_xy, temp_mn_vec1, temp_n_vec1, temp_n_vec2, temp_m_vec)
    fp_res_accel = res_norm_func(temp_mn_vec1 - accelerated_xy)

    # accept candidate if it reduces fixed-point residual
    # println("Accel FP residual over standard FP residual: $(fp_res_accel / fp_res_standard)")
    
    return fp_res_accel < fp_res_standard
end

"""
When solver is run with (prototype) adaptive restart mechanism, this function 
determines whether it is time for a restart or not, based on the cumulative
angle data for each iterate sequence.
"""
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

function preallocate_scratch(ws::AbstractWorkspace, acceleration::Symbol)
    if acceleration == :krylov
        return (
            temp_n_mat1 = zeros(Float64, ws.p.n, 2),
            temp_n_mat2 = zeros(Float64, ws.p.n, 2),
            temp_m_mat = zeros(Float64, ws.p.m, 2),
            temp_m_vec = zeros(Float64, ws.p.m),
            temp_n_vec1 = zeros(Float64, ws.p.n),
            temp_n_vec2 = zeros(Float64, ws.p.n), 
            temp_mn_vec1 = zeros(Float64, ws.p.n + ws.p.m),
            temp_mn_vec2 = zeros(Float64, ws.p.n + ws.p.m),
            temp_n_vec_complex1 = zeros(ComplexF64, ws.p.n),
            temp_n_vec_complex2 = zeros(ComplexF64, ws.p.n),
            temp_m_vec_complex = zeros(ComplexF64, ws.p.m),
            prev_xy = zeros(Float64, ws.p.n + ws.p.m),
            accelerated_point = zeros(Float64, ws.p.n + ws.p.m),
        )
    elseif acceleration == :none || acceleration == :anderson
        return (
            temp_n_mat1 = zeros(Float64, ws.p.n, 2),
            temp_n_mat2 = zeros(Float64, ws.p.n, 2),
            temp_m_mat = zeros(Float64, ws.p.m, 2),
            temp_m_vec = zeros(Float64, ws.p.m),
            temp_n_vec1 = zeros(Float64, ws.p.n),
            temp_n_vec2 = zeros(Float64, ws.p.n), 
            temp_mn_vec1 = zeros(Float64, ws.p.n + ws.p.m),
            temp_mn_vec2 = zeros(Float64, ws.p.n + ws.p.m),
            temp_n_vec_complex = zeros(ComplexF64, ws.p.n),
            temp_m_vec_complex = zeros(ComplexF64, ws.p.m),
            prev_xy = zeros(Float64, ws.p.n + ws.p.m),
        )
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
            linesearch_iters = Int[],
            xy_step_norms = Float64[],
            xy_step_char_norms = Float64[], # record method's "char norm" of the updates
            xy_update_cosines = Float64[],
            x_dist_to_sol = !isnothing(x_sol) ? Float64[] : nothing,
            y_dist_to_sol = !isnothing(x_sol) ? Float64[] : nothing,
            xy_chardist = !isnothing(x_sol) ? Float64[] : nothing,
            current_update_mat_col = Ref(1),
            updates_matrix = zeros(ws.p.n + ws.p.m, 20),
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

"""
Run the optimiser for the initial inputs and solver options given.
acceleration is a Symbol in {:none, :krylov, :anderson}
"""
function optimise!(ws::AbstractWorkspace,
    args::Dict{String, T};
    setup_time::Float64 = 0.0, # time spent in set-up (seconds)
    x_sol::Union{Nothing, Vector{Float64}} = nothing,
    y_sol::Union{Nothing, Vector{Float64}} = nothing,
    explicit_affine_operator::Bool = false,
    spectrum_plot_period::Int = 17,) where T

    # create views into x and y variables, along with "Arnoldi vector" q
    if ws.vars isa TwocolVariables
        @views view_x = ws.vars.xy_q[1:ws.p.n, 1]
        @views view_y = ws.vars.xy_q[ws.p.n+1:end, 1]
        @views view_q = ws.vars.xy_q[:, 2]
    elseif ws.vars isa OnecolVariables
        @views view_x = ws.vars.xy[1:ws.p.n]
        @views view_y = ws.vars.xy[ws.p.n+1:end]
    else
        error("Unknown variable type in workspace.")
    end
    
    # init acceleration trigger flag
    just_accelerated = true
    # set restart counter
    j_restart = 0

    # pre-allocate vectors for intermediate results in in-place computations
    scratch = preallocate_scratch(ws, args["acceleration"]) # scratch is a named tuple
    prev_xy = zeros(Float64, ws.p.n + ws.p.m)
    curr_xy_update = zeros(Float64, ws.p.n + ws.p.m)
    prev_xy_update = zeros(Float64, ws.p.n + ws.p.m)

    # characteristic PPM preconditioner of the method
    if !args["run-fast"]
        char_norm_mat = [(ws.W - ws.p.P) -ws.p.A'; -ws.p.A I(ws.p.m) / ws.ρ]
        function char_norm_func(vector::AbstractArray{Float64})
            return sqrt(dot(vector, char_norm_mat * vector))
        end
        # char_norm_func = norm
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
        # Update average iterates
        if args["restart-period"] != Inf
            x_avg .= (j_restart * x_avg + view_x) / (j_restart + 1)
            y_avg .= (j_restart * y_avg + view_y) / (j_restart + 1)
        end

        # compute distance to real solution
        if !args["run-fast"]
            if x_sol !== nothing
                scratch.temp_mn_vec1[1:ws.p.n] .= view_x - x_sol
                scratch.temp_mn_vec1[ws.p.n+1:end] .= view_y - (-y_sol)
                curr_xy_chardist = char_norm_func(scratch.temp_mn_vec1)
                
                @views curr_x_dist = norm(scratch.temp_mn_vec1[1:ws.p.n])
                @views curr_y_dist = norm(scratch.temp_mn_vec1[ws.p.n+1:end])
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
            @views prev_xy .= ws.vars.xy_q[:, 1]
            
            # acceleration attempt step
            if ws.k[] % (ws.mem + 1) == 0 && ws.k[] > 0
                custom_acceleration_candidate!(ws, scratch.accelerated_point, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec)

                # TODO: sort out norm to use in fixed-point residual safeguard
                # step --- this is superfluous kwarg to accept_acc_candidate
                # at the moment
                @views if accept_acc_candidate(ws, ws.vars.xy_q[:, 1], scratch.accelerated_point, scratch.temp_mn_vec1, scratch.temp_mn_vec2, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, char_norm_func=char_norm_func)
                    # increment effective iter counter (ie excluding unsuccessful acc attempts)
                    ws.k_eff[] += 1
                    
                    # println("Accepted Krylov acceleration candidate at iteration $(ws.k[]).")
                    
                    # assign actually
                    ws.vars.xy_q[:, 1] .= scratch.accelerated_point

                    # record stuff
                    if !args["run-fast"]
                        @views curr_xy_update .= scratch.accelerated_point - ws.vars.xy_q[:, 1]
                        push!(record.acc_step_iters, ws.k[])
                        record.updates_matrix .= 0.0
                        record.current_update_mat_col[] = 1
                    end
                end
                # prevent recording 0 or very large update norm
                if !args["run-fast"]
                    push!(record.xy_step_norms, NaN)
                    push!(record.xy_step_char_norms, NaN)
                end

                # reset krylov acceleration data at each attempt regardless of success
                ws.krylov_basis .= 0.0
                ws.H .= 0.0

                just_accelerated = true
            # standard step in the krylov setup (two columns)
            else
                # increment effective iter counter (ie excluding unsuccessful acc attempts)
                ws.k_eff[] += 1
                
                # apply method operator (to both (x, y) and Arnoldi (q) vectors)
                twocol_method_operator!(ws, scratch.temp_n_mat1, scratch.temp_n_mat2, scratch.temp_m_mat, scratch.temp_n_vec_complex1, scratch.temp_n_vec_complex2, scratch.temp_m_vec_complex, true)


                # record iteration data
                if !args["run-fast"]
                    @views curr_xy_update .= ws.vars.xy_q[:, 1] - prev_xy
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                    insert_update_into_matrix!(record.updates_matrix, curr_xy_update, record.current_update_mat_col)
                end

                if just_accelerated
                    # assign initial vector in the Krylov basis as initial
                    # fixed-point residual
                    @views ws.vars.xy_q[:, 2] .= ws.vars.xy_q[:, 1] - prev_xy
                    # normalise
                    @views ws.vars.xy_q[:, 2] ./= norm(ws.vars.xy_q[:, 2])
                    # store in Krylov basis
                    @views ws.krylov_basis[:, 1] .= ws.vars.xy_q[:, 2]
                else
                    # Arnoldi "step", orthogonalises the incoming basis vector
                    # and updates the Hessenberg matrix appropriately, all in-place
                    @views arnoldi_step!(ws.krylov_basis, ws.vars.xy_q[:, 2], ws.H)
                end

                # reset krylov acceleration flag
                just_accelerated = false

                j_restart += 1
            end
        else # NOT krylov set-up, so working variable is one-column
            # Anderson acceleration attempt
            if args["acceleration"] == :anderson && ws.k[] % ws.attempt_period == 0 && ws.k[] > 0
                # ws.vars.xy might be overwritten, so we take note of it here
                scratch.temp_mn_vec1 .= ws.vars.xy

                # attempt acceleration step. if successful, this
                # overwites ws.vars.xy
                COSMOAccelerators.accelerate!(ws.vars.xy, prev_xy, ws.accelerator, ws.k_vanilla[])

                if !args["run-fast"]
                    # record step (might be zero if acceleration failed)
                    curr_xy_update .= ws.vars.xy - scratch.temp_mn_vec1

                    # account for records as appropriate
                    if any(curr_xy_update .!= 0.0) # ie if acceleration was successful
                        ws.k_eff[] += 1
                        # println("Accepted Anderson acceleration candidate at iteration $(ws.k[]).")
                        push!(record.acc_step_iters, ws.k[])
                        record.updates_matrix .= 0.0
                        record.current_update_mat_col[] = 1
                    end
                    
                    # prevent recording large or zero update norm
                    push!(record.xy_step_norms, NaN)
                    push!(record.xy_step_char_norms, NaN)
                elseif any(ws.vars.xy .!= scratch.temp_mn_vec1) # ie if acceleration was successful
                    ws.k_eff[] += 1
                    # println("Accepted Anderson acceleration candidate at iteration $(ws.k[]).")
                end
            
            else # standard onecol iteration
                # copy older iterate before iterating
                prev_xy .= ws.vars.xy

                onecol_method_operator!(ws, ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, true)

                # swap contents of ws.vars.xy and scratch.temp_mn_vec1
                custom_swap!(ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_mn_vec2)
                # now ws.vars.xy contains newer iterate,  while
                # scratch.temp_mn_vec1 contains older one

                # if using Anderson accel, now update accelerator standard
                # iterate/successor pair history
                if args["acceleration"] == :anderson
                    COSMOAccelerators.update!(ws.accelerator, ws.vars.xy, scratch.temp_mn_vec1, ws.k_vanilla[])
                    
                    # note COSMOAccelerators functions expect just vanilla
                    # iteration count (excluding all acceleration attempts)
                    ws.k_vanilla[] += 1
                    
                    ws.k_eff[] += 1
                end

                if !args["run-fast"]
                    # record step (might be zero if acceleration failed)
                    curr_xy_update .= ws.vars.xy - scratch.temp_mn_vec1

                    # record iteration data here
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                end
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

        # Notion of a previous iterate step makes sense (again).
        just_restarted = false

        # increment iter counter
        ws.k[] += 1

        # update stopwatches
        loop_time = (time_ns() - loop_start_ns) / 1e9
        global_time = loop_time + setup_time

        # check termination conditions
        if ws.k[] > args["max-iter"]
            termination = true
            exit_status = :max_iter
        elseif kkt_criterion(ws, args["rel-kkt-tol"])
            termination = true
            exit_status = :kkt_solved
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
    results = Results(Dict{Symbol, Any}(), metrics_final, exit_status)

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
        
        print_results(ws, args["print-mod"], curr_xy_dist=curr_xy_dist, relative = args["print-res-rel"], terminated = true)

        # assign results
        for key in HISTORY_KEYS
            # pull record.key via getfield since key is a symbol
            results.metrics_history[key] = getfield(record, key)
        end
    else
        print_results(ws, args["print-mod"], relative = args["print-res-rel"], terminated = true)
    end

    return results
end