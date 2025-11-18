using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Clarabel

"""
For LINEAR CONSTRAINTS DO ENTRY BY ENTRY, not cone by cone. Check for
entry-wise equality/inequality.
"""
function update_proj_flags!(
    proj_flags::BitVector,
    preproj_vec::AbstractVector{Float64},
    postproj_y::AbstractVector{Float64}
    )

    proj_flags .= (preproj_vec .== postproj_y)
end

# helper to form and store the first Krylov basis vector
function init_krylov_basis!(
    ws::KrylovWorkspace,
    krylov_status::Symbol=:success)

    if krylov_status == :success
        # form fixed‐point residual in slot 2
        @views ws.vars.state_q[:, 2] .= ws.vars.state_q[:, 1] .- ws.vars.state_prev
        # normalise
        @views ws.vars.state_q[:, 2] ./= norm(ws.vars.state_q[:, 2])

        if ws.vars.state_q[1, 2] === NaN
            @info "🟢 Have NaNs in initial Krylov basis vector, indicates a fixed-point has been found already!"
            ws.fp_found[] = true
        end
        # store in krylov_basis[:,1]
        @views ws.krylov_basis[:, 1] .= ws.vars.state_q[:, 2]
    elseif krylov_status == :B_nullvec
        # if we have landed on a fp residual which is a null vector of B,
        # ie a 1-eigenvector of tilde_A, this is likely to lead to breakdowns
        # in successive tries as the FOM is stuck on a slow convergence zone.
        # for this reason it might be a good idea to just try something
        # different from this residual vector, otherwise we can get stuck
        # solving trivial Krylov systems every iteration
        @info "Found a null vector of B in previous Krylov solution, so am initialising Krylov basis with a random vector this time. This is iteration $(ws.k[])."
        @views ws.vars.state_q[:, 2] .= randn(ws.p.m + ws.p.n)
        ws.vars.state_q[:, 2] ./= norm(ws.vars.state_q[:, 2])
        @views ws.krylov_basis[:, 1] .= ws.vars.state_q[:, 2]
    else
        # throw exception because this is not yet implemented.
        throw(ArgumentError("Non-package Krylov initialisation not yet implemented."))
    end
end

"""
This function implements a mat-mat free specification of the application of
tilde_A to a vector, at some iteration.

NOTE: one must pass in a bit vector of enforced constraints as well as the
AbstractWorkspace struct ws, even though AbstractWorkspace already has a field storing a
bit vector of this kind.
The issue is that the AbstractWorkspace field corresponds to the actual current
iterate of the method, whereas we may want to evaluate the method's output
from some arbitrary, different iterate!
"""
function tilde_A_prod(ws::AbstractWorkspace,
    enforced_constraints::BitVector,
    q::AbstractArray{Float64})

    @views top_left = q[1:ws.p.n, :] - (ws.method.W_inv * (ws.p.P * q[1:ws.p.n, :] + ws.method.ρ * ws.A_gram * q[1:ws.p.n, :]))
    bot_left = - ws.p.A * top_left
    @views top_right = ws.method.ρ * ws.method.W_inv * (ws.p.A' * ((enforced_constraints - .!enforced_constraints) .* q[ws.p.n+1:end, :]))
    @views bot_right = - ws.p.A * top_right + enforced_constraints .* q[ws.p.n+1:end, :]

    # NB: matrix, NOT vector, is returned.
    # This is to allow for multiplication by a matrix of 2 columns (maintain
    # two sequences of vector iterates simultaneously).
    return [top_left + top_right; bot_left + bot_right]
end


"""
Compute the Krylov accelerant candidate, write to result_vec.

This function performs an accelerated update of the candidate solution vector by:
    1. Evaluating the operator on an initial column vector from the workspace.
    2. Computing the residual in the Krylov subspace from the difference between the evaluated result and the reference vector.
    3. Solving a least squares problem based on a (possibly shifted) Hessenberg matrix derived from the workspace.
    4. Constructing a full-dimensional approximate solution using the Krylov basis, and re-applying the operator it to obtain the acceleration candidate.

# Parameters besides working vectors
- ws::KrylovWorkspace: 
        The workspace containing the Krylov subspace information, including the Hessenberg matrix, 
        Krylov basis, and other variables required for the computation.
- result_vec::Vector{Float64}: 
        Vector where the computed accelerated candidate solution will be stored.

# Returns
- Nothing. The function updates `result_vec` with the computed acceleration candidate.
"""
function compute_krylov_accelerant!(
    ws::KrylovWorkspace,
    result_vec::Vector{Float64},
    )

    # slice of H to use is H[1:ws.givens_count[], 1:ws.givens_count[]]

    # compute FOM(state_q[:, 1]) and store it in result_vec
    @views onecol_method_operator!(ws, Val{ws.method.variant}(), ws.vars.state_q[:, 1], result_vec)

    # to recycle this FOM(current state) in the subsequent safeguard call,
    # avoiding an additional operator call
    ws.scratch.extra.step_when_computing_krylov .= result_vec

    @views rhs_res = ws.scratch.extra.rhs_res[1:ws.givens_count[] + 1]
    ws.scratch.base.temp_mn_vec1 .= result_vec
    @views ws.scratch.base.temp_mn_vec1 .-= ws.vars.state_q[:, 1]
    @views mul!(
        rhs_res,
        (ws.krylov_basis[:, 1:ws.givens_count[] + 1])',
        ws.scratch.base.temp_mn_vec1
        )
    
    # ws.H is passed as triangular, so we now need to apply the
    # Givens rotations to Q_{krylov + 1}^T * residual
    # and solve the appropriate triangular system to obtain the LLS solution
    
    # compute Hessenber LLS solution, reduced dimension y
    # the difference between ws.krylov_operator == tilde_A
    # and ws.krylov_operator == :B is accounted for in the ws.H matrix
    # itself, which in the former case is shifted by -I at every step of the
    # way leading up to here (see Obsidian 2024/week51 notes)
    lls_status = solve_current_least_squares!(
        ws.H,
        ws.givens_rotations,
        ws.givens_count,
        rhs_res, # NB solution overwrites rhs_res
        ws.arnoldi_breakdown[] # indicates whether to apply the last APPARENT Givens rotation to the rhs vector or not
        )

    # compute full-dimension LLS solution
    if lls_status == :success
        @views mul!(
            ws.scratch.extra.gmres_increment,
            ws.krylov_basis[:, 1:ws.givens_count[]],
            rhs_res[1:ws.givens_count[]]
        )
        
        @views ws.scratch.base.temp_mn_vec1 .= ws.vars.state_q[:, 1]

        ws.scratch.base.temp_mn_vec1 .+= ws.scratch.extra.gmres_increment

        # obtain actual acceleration candidate by applying FOM to
        # this, and write candidate to result_vec
        onecol_method_operator!(ws, Val{ws.method.variant}(), ws.scratch.base.temp_mn_vec1, result_vec)
    else
        @info "❌ Krylov acceleration failed with status: $lls_status"
    end

    return lls_status
end


function krylov_usual_step!(
    ws::KrylovWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    timer::TimerOutput;
    )
    # apply method operator (to both state and Arnoldi (q) vectors)
    twocol_method_operator!(ws, Val{ws.method.variant}(), true, true)

    # Arnoldi "step", orthogonalises the incoming basis vector
    # and updates the Hessenberg matrix appropriately, all in-place
    @timeit timer "krylov arnoldi" @views ws.arnoldi_breakdown[] = arnoldi_step!(ws.krylov_basis, ws.vars.state_q[:, 2], ws.H, ws.givens_count)

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

Depending on arguments given, function can also recycle A * x, A' * y, ...

temp_n_mat1 should be of dimension n by 2
temp_m_mat1 should be of dimension m by 2
temp_m_vec should be of dimension m
"""
function twocol_method_operator!(
    ws::KrylovWorkspace{T, I, M},
    ::Val{P},
    update_proj_action::Bool = false,
    update_residuals::Bool = false,
    ) where {T, I, M <: PrePPM, P} # note dispatch on PrePPM
    
    confirm_residual_update = update_residuals && ws.res.residual_check_count[] >= ws.residual_period

    # working variable is ws.vars.state_q, with two columns

    @views two_col_mul!(ws.scratch.extra.temp_m_mat1, ws.p.A, ws.vars.state_q[1:ws.p.n, :], ws.scratch.extra.temp_n_vec_complex1, ws.scratch.extra.temp_m_vec_complex) # compute A * [x, q_n]

    if confirm_residual_update
        @views Ax_norm = norm(ws.scratch.extra.temp_m_mat1[:, 1], Inf)
    end

    @views ws.scratch.extra.temp_m_mat1[:, 1] .-= ws.p.b # subtract b from A * x (but NOT from A * q_n)

    # if updating primal residual
    if confirm_residual_update
        @views ws.res.r_primal .= ws.scratch.extra.temp_m_mat1[:, 1] # assign A * x - b
        project_to_dual_K!(ws.res.r_primal, ws.p.K) # project to dual cone (TODO: sort out for more general cones than just QP case)

        # at this point the primal residual vector is updated

        # update primal residual metrics
        ws.res.rp_abs = norm(ws.res.r_primal, Inf)
        ws.res.rp_rel = ws.res.rp_abs / (1 + max(Ax_norm, ws.p.b_norm_inf))
    end

    ws.scratch.extra.temp_m_mat1 .*= ws.method.ρ
    @views ws.scratch.extra.temp_m_mat1 .+= ws.vars.state_q[ws.p.n+1:end, :] # add current y
    @views ws.vars.preproj_vec .= ws.scratch.extra.temp_m_mat1[:, 1] # this is what's fed into dual cone projection operator
    @views project_to_dual_K!(ws.scratch.extra.temp_m_mat1[:, 1], ws.p.K) # ws.scratch.extra.temp_m_mat1[:, 1] now stores y_{k+1}

    # update in-place flags switching affine dynamics based on projection action
    # ws.proj_flags = D_k in Goodnotes handwritten notes
    if update_proj_action
        @views update_proj_flags!(ws.proj_flags, ws.vars.preproj_vec, ws.scratch.extra.temp_m_mat1[:, 1])
    end

    # can now compute the bit of q corresponding to the y iterate
    @views ws.scratch.extra.temp_m_mat1[:, 2] .*= ws.proj_flags

    # now compute bar iterates (y and q_m) concurrently
    # TODO reduce memory allocation in this line (only in general ws.method.θ != 1.0 cases)
    ws.scratch.method_mats.y_qm_bar .= ws.scratch.extra.temp_m_mat1
    # ws.scratch.method_mats.y_qm_bar .*= (1 + ws.method.θ)
    ws.scratch.method_mats.y_qm_bar .*= 2.0 # WARNING specialised to ws.method.θ == 1.0
    # @views ws.scratch.method_mats.y_qm_bar .+= -ws.method.θ .* ws.vars.state_q[ws.p.n+1:end, :] # WARNING specialised this to ws.method.θ == 1.0 (all we have anyway for now), see below:
    @views ws.scratch.method_mats.y_qm_bar .-= ws.vars.state_q[ws.p.n+1:end, :]

    if ws.krylov_operator == :B # ie use B = A - I as krylov operator
        @views ws.scratch.extra.temp_m_mat1[:, 2] .-= ws.vars.state_q[ws.p.n+1:end, 2] # add -I component
    end
    # NOTE: ws.scratch.extra.temp_m_mat1[:, 2] now stores UPDATED q_m
    
    # ASSIGN new y and q_m to ws variables
    ws.vars.state_q[ws.p.n+1:end, :] .= ws.scratch.extra.temp_m_mat1

    # now we go to "bulk of" x and q_n update
    @views two_col_mul!(ws.scratch.extra.temp_n_mat1, ws.p.P, ws.vars.state_q[1:ws.p.n, :], ws.scratch.extra.temp_n_vec_complex1, ws.scratch.extra.temp_n_vec_complex2) # compute P * [x, q_n]

    if confirm_residual_update
        @views Px_norm = norm(ws.scratch.extra.temp_n_mat1[:, 1], Inf)

        @views xTPx = dot(ws.vars.state_q[1:ws.p.n, 1], ws.scratch.extra.temp_n_mat1[:, 1]) # x^T P x
        @views cTx  = dot(ws.p.c, ws.vars.state_q[1:ws.p.n, 1]) # c^T x
        @views bTy  = dot(ws.p.b, ws.vars.state_q[ws.p.n+1:end, 1]) # b^T y

        # can now update gap and objective metrics
        ws.res.gap_abs = abs(xTPx + cTx + bTy) # primal-dual gap
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(0.5xTPx + cTx, 0.5xTPx + bTy)) # relative gap
        ws.res.obj_primal = 0.5xTPx + cTx
        ws.res.obj_dual = - 0.5xTPx - bTy
    end

    @views ws.scratch.extra.temp_n_mat1[:, 1] .+= ws.p.c # add linear part of objective, c, to P * x (but NOT to P * q_n)

    @views two_col_mul!(ws.scratch.extra.temp_n_mat2, ws.p.A', ws.scratch.method_mats.y_qm_bar, ws.scratch.extra.temp_m_vec_complex, ws.scratch.extra.temp_n_vec_complex1) # compute A' * [y_bar, q_m_bar]

    if confirm_residual_update
        @views ATybar_norm = norm(ws.scratch.extra.temp_n_mat2[:, 1], Inf)
    end

    # now compute P x + A^T y_bar
    ws.scratch.extra.temp_n_mat1 .+= ws.scratch.extra.temp_n_mat2 # this is what is pre-multiplied by W^{-1}

    # if updating dual residual (NOTE use of y_bar)
    if confirm_residual_update
        @views ws.res.r_dual .= ws.scratch.extra.temp_n_mat1[:, 1] # assign P * x + A' * y_bar + c

        # at this point the dual residual vector is updated

        # update dual residual metrics
        ws.res.rd_abs = norm(ws.res.r_dual, Inf)
        ws.res.rd_rel = ws.res.rd_abs / (1 + max(Px_norm, ATybar_norm, ws.p.c_norm_inf)) # update relative dual residual metric
    end
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to temp_n_mat1
    if ws.method.W_inv isa CholeskyInvOp # eg in ADMM, PDHG
        # need a working complex vector for efficient Cholesky inversion
        # of two columns simultaneously
        apply_inv!(ws.method.W_inv, ws.scratch.extra.temp_n_mat1, ws.scratch.extra.temp_n_vec_complex1, ws.scratch.extra.temp_n_vec_complex2)
    else
        # diagonal inversion is simpler
        apply_inv!(ws.method.W_inv, ws.scratch.extra.temp_n_mat1)
    end

    # ASSIGN new x and q_n: subtract off what we just computed, both columns
    if ws.krylov_operator == :tilde_A
        # @views ws.vars.state_q[1:ws.p.n, :] .-= temp_n_mat1 # seems to incur a lot of materialize! calls costs
        @views ws.vars.state_q[1:ws.p.n, 1] .-= ws.scratch.extra.temp_n_mat1[:, 1]
        @views ws.vars.state_q[1:ws.p.n, 2] .-= ws.scratch.extra.temp_n_mat1[:, 2]
    else # ie use B = A - I as krylov operator
        @views ws.vars.state_q[1:ws.p.n, 1] .-= ws.scratch.extra.temp_n_mat1[:, 1] # as above, FOM iterate column
        @views ws.vars.state_q[1:ws.p.n, 2] .= -ws.scratch.extra.temp_n_mat1[:, 2] # simpler Arnoldi vector update in B case
    end

    # reset residual check count
    if confirm_residual_update
        ws.res.residual_check_count[] = 0
    end
end

# specialised to ADMM variant of PrePPM
function twocol_method_operator!(
    ws::KrylovWorkspace{T, I, M},
    ::Val{:ADMM}, # just for ADMM variant of PrePPM method
    update_proj_action::Bool = false,
    update_residuals::Bool = false,
    ) where {T, I, M <: PrePPM} # note dispatch on PrePPM

    confirm_residual_update = update_residuals && ws.res.residual_check_count[] >= ws.residual_period
    
    # working variable is ws.vars.state_q, with two column
    @views two_col_mul!(ws.scratch.extra.temp_m_mat1, ws.p.A, ws.vars.state_q[1:ws.p.n, :], ws.scratch.extra.temp_n_vec_complex1, ws.scratch.extra.temp_m_vec_complex) # compute A * [x, q_x]

    ws.scratch.extra.Ax_mat .= ws.scratch.extra.temp_m_mat1 # store A * [x, q_x] for reuse in y update

    if confirm_residual_update
        @views Ax_norm = norm(ws.scratch.extra.Ax_mat[:, 1], Inf)
    end

    # subtract b (optimisation iterate only)
    @views ws.scratch.extra.temp_m_mat1[:, 1] .-= ws.p.b

    if confirm_residual_update
        # assign A x_k - b
        @views ws.res.r_primal .= ws.scratch.extra.temp_m_mat1[:, 1]
        # project to dual cone TODO general cone considerations
        project_to_dual_K!(ws.res.r_primal, ws.p.K) 

        # update primal residual notions
        ws.res.rp_abs = norm(ws.res.r_primal, Inf)
        ws.res.rp_rel = ws.res.rp_abs / (1 + max(Ax_norm, ws.p.b_norm_inf))
    end

    # multiply by ρ
    ws.scratch.extra.temp_m_mat1 .*= ws.method.ρ
    # add to current y_k
    @views ws.scratch.extra.temp_m_mat1 .+= ws.vars.state_q[ws.p.n+1:end, :]

    if update_proj_action
        # this is fed through cone projection
        @views ws.vars.preproj_vec .= ws.scratch.extra.temp_m_mat1[:, 1]
    end

    # project to dual cone K (optimisation iterate only)
    @views project_to_dual_K!(ws.scratch.extra.temp_m_mat1[:, 1], ws.p.K)

    if update_proj_action
        @views update_proj_flags!(ws.proj_flags, ws.vars.preproj_vec, ws.scratch.extra.temp_m_mat1[:, 1])
    end

    # can now multiply arnoldi column through projection bit vector
    @views ws.scratch.extra.temp_m_mat1[:, 2] .*= ws.proj_flags

    # compute y_bar (specialised to ws.θ == 1.0 case)
    ws.scratch.method_mats.y_qm_bar .= ws.scratch.extra.temp_m_mat1
    ws.scratch.method_mats.y_qm_bar .*= 2.0
    @views ws.scratch.method_mats.y_qm_bar .-= ws.vars.state_q[ws.p.n+1:end, :]

    # temp_m_mat1 is taken up with updated y subvectors which we
    # have to keep until the end of this function body
    ws.scratch.extra.temp_m_mat2 .= ws.scratch.extra.Ax_mat
    ws.scratch.extra.temp_m_mat2 .*= -ws.method.ρ
    ws.scratch.extra.temp_m_mat2 .+= ws.scratch.method_mats.y_qm_bar # now stores y_bar -ρ A q_x

    # multiply this by A'
    @views two_col_mul!(ws.scratch.extra.temp_n_mat1, ws.p.A', ws.scratch.extra.temp_m_mat2, ws.scratch.extra.temp_m_vec_complex, ws.scratch.extra.temp_n_vec_complex1)

    # add c (optimisation iterate only)
    @views ws.scratch.extra.temp_n_mat1[:, 1] .+= ws.p.c

    # short-lived working container for -δ x_k term (often 0)
    @views ws.scratch.extra.temp_n_mat2 .= ws.vars.state_q[1:ws.p.n, :]
    ws.scratch.extra.temp_n_mat2 .*= ws.method.W_inv.shift

    # subtract this. ws.scratch.extra.temp_n_mat2 free after this
    ws.scratch.extra.temp_n_mat1 .-= ws.scratch.extra.temp_n_mat2

    # apply inverse of W
    apply_inv!(ws.method.W_inv, ws.scratch.extra.temp_n_mat1, ws.scratch.extra.temp_n_vec_complex1, ws.scratch.extra.temp_n_vec_complex2)

    # negate
    ws.scratch.extra.temp_n_mat1 .*= -1.0
    # this concludes [x, q_x] update

    # dual residual and gap measures
    if confirm_residual_update
        # compute P * x_k
        @views mul!(ws.res.r_dual, ws.p.P, ws.vars.state_q[1:ws.p.n, 1])
        Px_norm = norm(ws.res.r_dual, Inf)

        @views xTPx = dot(ws.vars.state_q[1:ws.p.n, 1], ws.res.r_dual) # x^T P x
        @views cTx  = dot(ws.p.c, ws.vars.state_q[1:ws.p.n, 1]) # c^T x
        @views bTy  = dot(ws.p.b, ws.vars.state_q[ws.p.n+1:end, 1])

        # compute A' y_k
        @views mul!(ws.scratch.base.temp_n_vec2 , ws.p.A', ws.vars.state_q[ws.p.n+1:end, 1])
        ATy_norm = norm(ws.scratch.base.temp_n_vec2, Inf)

        # compute P * x_k + A' * y_k
        ws.res.r_dual .+= ws.scratch.base.temp_n_vec2
        # add c to complete dual residual
        ws.res.r_dual .+= ws.p.c

        ws.res.rd_abs = norm(ws.res.r_dual, Inf)
        ws.res.rd_rel = ws.res.rd_abs / (1 + max(Px_norm, ATy_norm, ws.p.c_norm_inf))

        ws.res.obj_primal = 0.5xTPx + cTx
        ws.res.obj_dual = - 0.5xTPx - bTy

        ws.res.gap_abs = abs(xTPx + cTx + bTy)
        ws.res.gap_rel = ws.res.gap_abs / (1 + max(abs(ws.res.obj_primal), abs(ws.res.obj_dual)))
    end

    # assign new iterates
    # consumes temp_n_mat1 and temp_m_mat1
    ws.vars.state_q[1:ws.p.n, :] .= ws.scratch.extra.temp_n_mat1
    ws.vars.state_q[ws.p.n+1:end, :] .= ws.scratch.extra.temp_m_mat1

    # reset residual check count
    if confirm_residual_update
        ws.res.residual_check_count[] = 0
    end
end

function krylov_step!(
    ws::KrylovWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    args::Dict{String, Any},
    record::AbstractRecord,
    full_diagnostics::Bool,
    timer::TimerOutput,    
    )
    # copy older iterate
    @views ws.vars.state_prev .= ws.vars.state_q[:, 1]
    
    # acceleration attempt step
    if (ws.givens_count[] in ws.trigger_givens_counts && !ws.control_flags.back_to_building_krylov_basis) || ws.arnoldi_breakdown[]
        @timeit timer "krylov sol" ws.control_flags.krylov_status = compute_krylov_accelerant!(ws, ws.scratch.extra.accelerated_point)

        ws.k_operator[] += 1 # one operator application in computing Krylov point (GMRES-equivalence, see Walker and Ni 2011)

        # use safeguard if the Krylov procedure suffered no breakdowns
        if ws.control_flags.krylov_status == :success

            if full_diagnostics # best to update for use here
                construct_explicit_operator!(ws, ws_diag)
                
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

            @timeit timer "fixed-point safeguard" @views ws.control_flags.accepted_accel = accel_fp_safeguard!(ws, ws_diag, ws.vars.state_q[:, 1], ws.scratch.extra.accelerated_point, args["safeguard-factor"], record, full_diagnostics)

            ws.k_operator[] += 1 # note: only 1 because we also count another when assigning recycled iterate in the following iteration
        else
            ws.control_flags.accepted_accel = false
        end

        if ws.control_flags.accepted_accel
            # increment effective iter counter (ie excluding unsuccessful acc attempts)
            ws.k_eff[] += 1
            ws.res.residual_check_count[] += 1
            
            push_update_to_record!(ws, record, false)

            # assign actually
            ws.vars.state_q[:, 1] .= ws.scratch.extra.accelerated_point
        else
            push_update_to_record!(ws, record, false)
        end

        # reset krylov acceleration data when memory is fully
        # populated, regardless of success/failure
        if ws.givens_count[] + 1 == ws.mem || ws.arnoldi_breakdown[]
            ws.krylov_basis .= 0.0
            ws.H .= 0.0
            ws.givens_count[] = 0

            ws.arnoldi_breakdown[] = false
        end

        ws.control_flags.back_to_building_krylov_basis = true
    # vanilla step
    else
        # increment effective iter counter (ie excluding unsuccessful acc attempts)
        ws.k_eff[] += 1
        ws.res.residual_check_count[] += 1

        ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
        
        # if we just had an accel attempt, we recycle work carried out
        # during the acceleration acceptance criterion check
        # we also reinit the Krylov orthogonal basis 
        if ws.k[] == 0 # special case in initial iteration

            @views onecol_method_operator!(ws, Val{ws.method.variant}(), ws.vars.state_q[:, 1], ws.scratch.extra.initial_vec, true, true)
            @views ws.vars.state_q[:, 1] .= ws.scratch.extra.initial_vec

            # fills in first column of ws.krylov_basis
            init_krylov_basis!(ws) # ws.H is zeros at this point still
        
        elseif ws.control_flags.recycle_next
            # RECYCLE working state iterate
            ws.vars.state_q[:, 1] .= ws.scratch.extra.state_recycled
            
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
        push_update_to_record!(ws, record, true)

        # reset flag for recycling
        ws.control_flags.recycle_next = false
    end
end