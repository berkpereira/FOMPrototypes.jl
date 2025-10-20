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
        # @views ws.vars.xy_q[1:ws.p.n, :] .-= temp_n_mat1 # seems to incur a lot of materialize! calls costs
        @views ws.vars.xy_q[1:ws.p.n, 1] .-= temp_n_mat1[:, 1]
        @views ws.vars.xy_q[1:ws.p.n, 2] .-= temp_n_mat1[:, 2]
    else # ie use B = A - I as krylov operator
        @views ws.vars.xy_q[1:ws.p.n, 1] .-= temp_n_mat1[:, 1] # as above
        @views ws.vars.xy_q[1:ws.p.n, 2] .= -temp_n_mat1[:, 2] # simpler
    end

    return nothing
end

"""
Efficiently computes FOM iteration at point xy, and stores it in result_vec.
Vector xy is left unchanged.

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
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(abs(0.5xTPx + cTx), abs(0.5xTPx + bTy))) # relative gap
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

"""
Check for acceptance of a Krylov acceleration candidate.
Recycles work done in matrix-vector products involving A, A', and P.

Writes to to_recycle_xy.
Returns Boolean indicating acceleration success/failure.

Source code has similarities to that of onecol_method_operator!
"""
function accel_fp_safeguard!(
    ws::Union{KrylovWorkspace, AndersonWorkspace},
    current_xy::AbstractVector{Float64},
    accelerated_xy::AbstractVector{Float64},
    to_recycle_xy::AbstractVector{Float64},
    safeguard_factor::Float64,
    temp_mn_vec1::AbstractVector{Float64},
    temp_mn_vec2::AbstractVector{Float64},
    temp_n_vec1::AbstractVector{Float64},
    temp_n_vec2::AbstractVector{Float64},
    temp_m_vec::AbstractVector{Float64},
    record::Union{NamedTuple, Nothing} = nothing,
    tilde_A::Union{AbstractMatrix{Float64}, Nothing} = nothing,
    tilde_b::Union{AbstractVector{Float64}, Nothing} = nothing,
    full_diagnostics::Bool = false,
    )

    # we catch the unguarded case early, since it is the simplest by far
    if ws.safeguard_norm == :none
        # store FOM(accelerated_xy) in to_recycle_xy right away
        onecol_method_operator!(ws, accelerated_xy, to_recycle_xy, temp_n_vec1, temp_n_vec2, temp_m_vec)

        # always successful acceleration
        return true
    end
    
    fp_metric_vanilla = 0.0
    fp_metric_acc = 0.0

    # store FOM(current_xy) in temp_mn_vec1
    onecol_method_operator!(ws, current_xy, temp_mn_vec1, temp_n_vec1, temp_n_vec2, temp_m_vec)

    # (temporarily) store FOM(current_xy) --- vanilla iterate --- in xy_recycled
    # this is overwritten later in this function if the acceleration
    # is successful, namely with xy_recycled .= FOM(accelerated_xy)
    to_recycle_xy .= temp_mn_vec1 # TODO avoid this copying by simply using to_recycle_xy in some of the lines around this one

    # store fixed-point residual of current_xy in temp_mn_vec2
    temp_mn_vec2 .= temp_mn_vec1 - current_xy

    if ws.safeguard_norm == :char
        # store M1 * FOM_x(current x) in temp_n_vec1
        @views M1_op!(temp_mn_vec2[1:ws.p.n], temp_n_vec1, ws, ws.variant, ws.A_gram, temp_n_vec2)

        # increment fp_metric_vanilla with < M1 * fp_res_x(current x), fp_res_x(current x) >
        @views fp_metric_vanilla += dot(temp_n_vec1, temp_mn_vec2[1:ws.p.n])
        # increment fp_metric_vanilla with < M2 * fp_res_y(current y), fp_res_y(current y) >
        @views fp_metric_vanilla += norm(temp_mn_vec2[ws.p.n+1:end])^2 / ws.ρ
        # increment fp_metric_vanilla with 2 < A fp_res_x(current x), fp_res_y(current y) >
        @views mul!(temp_m_vec, ws.p.A, temp_mn_vec2[1:ws.p.n]) # temp_m_vec stores A * fp_res_x(current x)
        @views fp_metric_vanilla += 2 * dot(temp_m_vec, temp_mn_vec2[ws.p.n+1:end])

        fp_metric_vanilla = sqrt(fp_metric_vanilla)
    else # ws.safeguard_norm == :euclid
        fp_metric_vanilla = norm(temp_mn_vec2)
    end

    # fp_metric_vanilla is fully computed by now
    # now we move on to fp_metric_acc
    # NOTE ALL TEMP/working vectors can be used cleanly again by now
    if tilde_A !== nothing
        try
            # pinv_sol = (tilde_A - I) \ (-tilde_b)
            pinv_sol = pinv(tilde_A - I) * (-tilde_b)
            pinv_residual = norm(tilde_A * pinv_sol + tilde_b - pinv_sol)
            if pinv_residual > 1e-9
                println("❌ no true fixed-point at this affine operator! pinv sol residual: ", pinv_residual)
            else
                println("✅ true fixed-point at this affine operator! pinv sol residual: ",  pinv_residual)
            end
            println("Norm of pinv solution: ", norm(pinv_sol))
            println("Norm of accelerated_xy: ", norm(accelerated_xy))

            error_to_est = pinv_sol - accelerated_xy
            rel_err = norm(error_to_est) / norm(pinv_sol)
            if rel_err < 1e-2
                println("✅ good Krylov-approximation of pinv-fixed point: rel error at iter $(ws.k[]): ", rel_err)
            else
                println("❌ bad Krylov-approximation of pinv-fixed point: rel error at iter $(ws.k[]): ", rel_err)
            end

            char_mat = Matrix([(ws.W - ws.p.P) ws.p.A'; ws.p.A I(ws.p.m) / ws.ρ])
            true_lookahead = zeros(ws.p.m + ws.p.n)
            onecol_method_operator!(
                ws, pinv_sol, true_lookahead, temp_n_vec1, temp_n_vec2,
                temp_m_vec)
            # take a step from the linear system ≈solution, as in our Krylov method
            onecol_method_operator!(
                ws, true_lookahead, pinv_sol, temp_n_vec1, temp_n_vec2,
                temp_m_vec)
            
            # swap: after this, true_lookahead stores T(pinv_sol)
            custom_swap!(true_lookahead, pinv_sol, temp_mn_vec1)
            
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
    if full_diagnostics && ws isa KrylovWorkspace
        krylov_basis_svdvals = svdvals(ws.krylov_basis)
        krylov_basis_svdvals = krylov_basis_svdvals[findall(krylov_basis_svdvals .>= 1e-10)]
        max_sval = maximum(krylov_basis_svdvals)
        min_sval = minimum(krylov_basis_svdvals)
        cond_no = max_sval / min_sval
        if cond_no > 1.1
            println("❌ condition number of Krylov basis at iter $(ws.k[]) is ", cond_no)
            println("largest singular value: ", max_sval)
            println("smallest singular value: ", min_sval)
            @show krylov_basis_svdvals
        else
            println("✅ condition number of Krylov basis at iter $(ws.k[]) is ", cond_no)
        end
    end


    # store FOM(accelerated_xy) in temp_mn_vec1
    # NOTE: we must keep temp_mn_vec1 INTACT from this on UNTIL we decide whether
    # or not to overwrite to_recycle_xy with FOM(accelerated_xy)
    onecol_method_operator!(ws, accelerated_xy, temp_mn_vec1, temp_n_vec1, temp_n_vec2, temp_m_vec)

    # store fixed-point residual of accelerated_xy in temp_mn_vec2
    temp_mn_vec2 .= temp_mn_vec1 - accelerated_xy

    if ws.safeguard_norm == :char
        # store M1 * FOM_x(current_x) in temp_n_vec1
        @views M1_op!(temp_mn_vec2[1:ws.p.n], temp_n_vec1, ws, ws.variant, ws.A_gram, temp_n_vec2)

        # increment fp_metric_acc with < M1 * fp_res_x(accelerated x), fp_res_x(accelerated x) >
        @views fp_metric_acc += dot(temp_n_vec1, temp_mn_vec2[1:ws.p.n])
        # increment fp_metric_acc with < M2 * fp_res_y(accelerated y), fp_res_y(accelerated y) >
        @views fp_metric_acc += norm(temp_mn_vec2[ws.p.n+1:end])^2 / ws.ρ
        # increment fp_metric_acc with 2 < A fp_res_x(accelerated x), fp_res_y(accelerated y) >
        @views mul!(temp_m_vec, ws.p.A, temp_mn_vec2[1:ws.p.n]) # temp_m_vec stores A * fp_res_x(current x)
        @views fp_metric_acc += 2 * dot(temp_m_vec, temp_mn_vec2[ws.p.n+1:end])

        fp_metric_acc = sqrt(fp_metric_acc)
    else # ws.safeguard_norm == :euclid
        fp_metric_acc = norm(temp_mn_vec2)
    end

    # fp_metric_acc is also fully computed by now
    # recall that temp_mn_vec1 STILL stores FOM(accelerated_xy) right now

    
    metric_ratio = fp_metric_acc / fp_metric_vanilla
    
    # report success/failure of Krylov acceleration attempt
    if record !== nothing
        push!(record.acc_attempt_iters, ws.k[])
        push!(record.fp_metric_ratios, metric_ratio)
    end

    # determine acceptance
    acceleration_success = fp_metric_acc <= safeguard_factor * fp_metric_vanilla
    # if acceleration_success && metric_ratio < 0.9
    #     println("✅ acceleration attempt at iter $(ws.k[]) was successful. fp_metric_acc / fp_metric_vanilla: $metric_ratio")
    # elseif acceleration_success && metric_ratio >= 0.9
    #     println("⚠️ acceleration attempt at iter $(ws.k[]) was successful but not great. fp_metric_acc / fp_metric_vanilla: $metric_ratio")
    # else
    #     println("❌ acceleration attempt at iter $(ws.k[]) was UNsuccessful. fp_metric_acc / fp_metric_vanilla: $metric_ratio")
    # end
    # println()

    # if acceleration was a success, to_recycle_xy takes FOM(accelerated_xy)
    # else, we leave it as it was assigned to above in this function, namely
    # to_recycle_xy == FOM(current_xy)
    if acceleration_success
        to_recycle_xy .= temp_mn_vec1
    end

    return acceleration_success
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

function preallocate_scratch(ws::VanillaWorkspace)
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
    )
end

function preallocate_scratch(ws::AndersonWorkspace)
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
        accelerated_point = zeros(Float64, ws.p.n + ws.p.m),
    )
end

function preallocate_scratch(ws::KrylovWorkspace)
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
        accelerated_point = zeros(Float64, ws.p.n + ws.p.m),
    )
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
    scratch,
    timer::TimerOutput;
    H_unmod::Union{Nothing, AbstractMatrix{Float64}} = nothing,
    tilde_A::Union{AbstractMatrix{Float64}, Nothing} = nothing,
    )
    # apply method operator (to both (x, y) and Arnoldi (q) vectors)
    twocol_method_operator!(ws, scratch.temp_n_mat1, scratch.temp_n_mat2, scratch.temp_m_mat, scratch.temp_n_vec_complex1, scratch.temp_n_vec_complex2, scratch.temp_m_vec_complex, true)

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
    if H_unmod !== nothing
        H_unmod[:, ws.givens_count[] + 1] .= ws.H[:, ws.givens_count[] + 1]
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
    W_inv_mat::AbstractMatrix{Float64},
    tilde_A::AbstractMatrix{Float64},
    tilde_b::AbstractVector{Float64},
    )
    
    D_A = Diagonal(ws.proj_flags)

    tilde_A[1:ws.p.n, 1:ws.p.n] .= I(ws.p.n) - W_inv_mat * ws.p.P - 2 * ws.ρ * W_inv_mat * ws.p.A' * D_A * ws.p.A
    tilde_A[1:ws.p.n, ws.p.n+1:end] .= - W_inv_mat * ws.p.A' * (2 * D_A - I(ws.p.m))
    tilde_A[ws.p.n+1:end, 1:ws.p.n] .= ws.ρ * D_A * ws.p.A
    tilde_A[ws.p.n+1:end, ws.p.n+1:end] .= D_A

    tilde_b[1:ws.p.n] .= W_inv_mat * (2 * ws.ρ * ws.p.A' * D_A * ws.p.b - ws.p.c)
    @views tilde_b[ws.p.n+1:end] .= - ws.ρ * D_A * ws.p.b

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
        tilde_A = zeros(ws.p.m + ws.p.n, ws.p.m + ws.p.n)
        tilde_b = zeros(ws.p.m + ws.p.n)

        if ws isa KrylovWorkspace
            H_unmod = UpperHessenberg(zeros(ws.mem, ws.mem - 1))
        else
            H_unmod = nothing
        end

        # form dense identity
        dense_I = Matrix{Float64}(I, ws.p.n, ws.p.n)

        # form matrix inverse of W (= P + M_1)
        if ws.W_inv isa CholeskyInvOp
            W_inv_mat = ws.W_inv.F \ dense_I
            W_inv_mat = Symmetric(W_inv_mat)
        elseif ws.W_inv isa DiagInvOp
            W_inv_mat = Diagonal(ws.W_inv.inv_diag)
        end
    else
        tilde_A = nothing
        tilde_b = nothing
        H_unmod = nothing
    end
    
    # init acceleration trigger flag
    just_tried_acceleration = false

    krylov_status = :init

    # we use back_to_building_krylov_basis as follows
    # as we build Krylov basis up, ws.givens_count is incremented
    # when it gets to a given trigger point, we attempt acceleration and set
    # just_tried_acceleration = true. after the following, recycling iteration, 
    # the counter ws.givens_count will still be at the trigger point and
    # just_tried_acceleration is false. however, we wish to keep builing
    # the krylov basis --- NOT attempt acceleration again. to distinguish these
    # cases we use back_to_building_krylov_basis
    back_to_building_krylov_basis = true
    
    # set restart counter
    j_restart = 0

    # pre-allocate vectors for intermediate results in in-place computations
    scratch = preallocate_scratch(ws) # scratch is a named tuple
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
        # Update average iterates
        # if args["restart-period"] != Inf
        #     x_avg .= (j_restart * x_avg + view_x) / (j_restart + 1)
        #     y_avg .= (j_restart * y_avg + view_y) / (j_restart + 1)
        # end

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
            @views ws.vars.xy_prev .= ws.vars.xy_q[:, 1]
            
            # acceleration attempt step
            if (ws.givens_count[] in ws.trigger_givens_counts && !back_to_building_krylov_basis) || ws.arnoldi_breakdown[]
                @timeit timer "krylov sol" krylov_status = compute_krylov_accelerant!(ws, scratch.accelerated_point, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec)

                ws.k_operator[] += 1 # one operator application in computing Krylov point (GMRES-equivalence, see Walker and Ni 2011)

                # use safeguard if the Krylov procedure suffered no breakdowns
                if krylov_status == :success

                    if full_diagnostics # best to update for use here
                        construct_explicit_operator!(ws, W_inv_mat, tilde_A, tilde_b)

                        # NOTE CARE might want to override with true fixed point!
                        # @warn "trying override Krylov with (PINV?) true fixed-point of current affine operation!"
                        # try
                        #     # scratch.accelerated_point .= (tilde_A - I) \ (-tilde_b)
                        #     scratch.accelerated_point .= pinv(tilde_A - I) * (-tilde_b)
                        #     @info "✅ successful override"
                        # catch e
                        #     @info "❌ unsuccessful override, proceeding w GNRES candidate"
                        # end
                        
                        BQ = (tilde_A - I) * ws.krylov_basis[:, 1:ws.givens_count[]]
                        arnoldi_relation_err = BQ - ws.krylov_basis[:, 1:ws.givens_count[] + 1] * H_unmod[1:ws.givens_count[] + 1, 1:ws.givens_count[]]
                        norm_arnoldi_err = norm(arnoldi_relation_err)

                        println()
                        println("---- ACCEL INFO ITER $(ws.k[]), GIVENS COUNT $(ws.givens_count[]) ----")
                        if norm_arnoldi_err > 1e-10
                            println("❌ Arnoldi relation error (size $(size(arnoldi_relation_err))) at iter $(ws.k[]): ", norm_arnoldi_err, ". norm of B*Q = $(norm(BQ))")
                        else
                            println("✅ Arnoldi relation error (size $(size(arnoldi_relation_err))) at iter $(ws.k[]): ", norm_arnoldi_err, ". norm of B*Q = $(norm(BQ))")
                        end
                    end

                    @timeit timer "fixed-point safeguard" @views accept_krylov = accel_fp_safeguard!(ws, ws.vars.xy_q[:, 1], scratch.accelerated_point, ws.vars.xy_recycled, args["safeguard-factor"], scratch.temp_mn_vec1, scratch.temp_mn_vec2, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, record, tilde_A, tilde_b, full_diagnostics)

                    ws.k_operator[] += 1 # note: only 1 because we also count another when assigning recycled iterate in the following iteration
                else
                    accept_krylov = false
                end

                if accept_krylov
                    # increment effective iter counter (ie excluding unsuccessful acc attempts)
                    ws.k_eff[] += 1
                    
                    if !args["run-fast"]
                        @views curr_xy_update .= scratch.accelerated_point - ws.vars.xy_q[:, 1]
                        push!(record.acc_step_iters, ws.k[])
                        record.updates_matrix .= 0.0
                        record.current_update_mat_col[] = 1

                        push!(record.xy_step_norms, norm(curr_xy_update))
                        push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                    end

                    # assign actually
                    ws.vars.xy_q[:, 1] .= scratch.accelerated_point
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

                just_tried_acceleration = true
                back_to_building_krylov_basis = true
            # standard step in the krylov setup (two columns)
            else
                # increment effective iter counter (ie excluding unsuccessful acc attempts)
                ws.k_eff[] += 1

                ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
                
                # if we just had an accel attempt, we recycle work carried out
                # during the acceleration acceptance criterion check
                # we also reinit the Krylov orthogonal basis 
                if ws.k[] == 0 # special case in initial iteration
                    @views onecol_method_operator!(ws, ws.vars.xy_q[:, 1], scratch.temp_mn_vec1, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, true)
                    @views ws.vars.xy_q[:, 1] .= scratch.temp_mn_vec1

                    # fills in first column of ws.krylov_basis
                    init_krylov_basis!(ws) # ws.H is zeros at this point still
                
                elseif just_tried_acceleration
                    # recycle working (x, y) iterate
                    ws.vars.xy_q[:, 1] .= ws.vars.xy_recycled
                    
                    # also re-initialise Krylov basis IF we'd filled up
                    # the entire memory
                    if ws.givens_count[] == 0 # just wiped Krylov basis clean
                        # fills in first column of ws.krylov_basis
                        init_krylov_basis!(ws, krylov_status) # ws.H is zeros at this point still
                        
                        if full_diagnostics
                            H_unmod .= 0.0
                        end
                    end
                else # the usual case

                    krylov_usual_step!(ws, scratch, timer, H_unmod=H_unmod, tilde_A=tilde_A)

                    # ws.givens_count has had "time" to be increment past
                    # a trigger point, so we can relax this flag in order
                    # to allow the next acceleration attempt when it comes up
                    back_to_building_krylov_basis = false
                end

                # record iteration data if requested
                if !args["run-fast"]
                    @views curr_xy_update .= ws.vars.xy_q[:, 1] - ws.vars.xy_prev
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                    insert_update_into_matrix!(record.updates_matrix, curr_xy_update, record.current_update_mat_col)
                end

                # reset krylov acceleration flag
                just_tried_acceleration = false
                j_restart += 1
            end
        elseif args["acceleration"] == :anderson
            # Anderson acceleration attempt
            if ws.composition_counter[] == args["anderson-interval"]
                # ws.vars.xy might be overwritten, so we take note of it here
                # this is for the sole purpose of checking the norm of the
                # step just below in this code branch
                scratch.temp_mn_vec1 .= ws.vars.xy

                # attempt acceleration step. if successful (ie no numerical
                # problems), this overwites ws.vars.xy with accelerated step
                @timeit timer "anderson accel" COSMOAccelerators.accelerate!(ws.vars.xy, ws.vars.xy_prev, ws.accelerator, 0)
                
                scratch.accelerated_point .= ws.vars.xy # accelerated point if no numerics blowups
                ws.vars.xy .= scratch.temp_mn_vec1 # retake the iterate from which we are attempting acceleration right now

                # ws.accelerator.success only indicates there was no
                # numerical blowup in the COSMOAccelerators.accelerate!
                # internals
                if ws.accelerator.success
                    # at this point:
                    # ws.vars.xy contains the iterate at which we stopped
                    # when we began to pull the accelerate! levers;
                    # scratch.accelerated_point contains the candidate
                    # acceleration point, which is legitimately different
                    # from ws.vars.xy
                    @timeit timer "fixed-point safeguard" @views accept_anderson = accel_fp_safeguard!(ws, ws.vars.xy, scratch.accelerated_point, ws.vars.xy_recycled, args["safeguard-factor"], scratch.temp_mn_vec1, scratch.temp_mn_vec2, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, record, tilde_A, tilde_b, full_diagnostics)

                    ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application

                    if accept_anderson
                        ws.k_eff[] += 1 # increment effective iter counter (ie excluding unsuccessful acc attempts)

                        if !args["run-fast"]
                            curr_xy_update .= scratch.accelerated_point - ws.vars.xy
                            push!(record.acc_step_iters, ws.k[])
                            record.updates_matrix .= 0.0
                            record.current_update_mat_col[] = 1

                            push!(record.xy_step_norms, norm(curr_xy_update))
                            push!(record.xy_step_char_norms, char_norm_func(curr_xy_update))
                        end

                        # assign actually
                        ws.vars.xy .= scratch.accelerated_point
                    elseif !args["run-fast"] # prevent recording zero update norm
                        push!(record.xy_step_norms, NaN)
                        push!(record.xy_step_char_norms, NaN)
                    end


                    # note that this flag serves for us to use recycled
                    # work from the fixed-point safeguard step when
                    # the next iteration comes around
                    just_tried_acceleration = true
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
                if just_tried_acceleration
                    ws.vars.xy .= ws.vars.xy_recycled
                else
                    onecol_method_operator!(ws, ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, true)
                    # swap contents of ws.vars.xy and scratch.temp_mn_vec1
                    custom_swap!(ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_mn_vec2)
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
                just_tried_acceleration = false
            end
        else # no acceleration!
            # copy older iterate before iterating
            ws.vars.xy_prev .= ws.vars.xy

            onecol_method_operator!(ws, ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec, true)

            # swap contents of ws.vars.xy and scratch.temp_mn_vec1
            custom_swap!(ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_mn_vec2)
            # now ws.vars.xy contains newer iterate,  while
            # scratch.temp_mn_vec1 contains older one

            if !args["run-fast"]
                curr_xy_update .= ws.vars.xy - scratch.temp_mn_vec1

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
            construct_explicit_operator!(ws, W_inv_mat, tilde_A, tilde_b)
            # plot spectrum of the linearised operator
            plot_spectrum(tilde_A, ws.k[])
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

    return results, tilde_A, tilde_b, H_unmod
end