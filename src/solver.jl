using LinearAlgebra
using Plots
using Random
using Printf
using Infiltrator
import COSMOAccelerators
include("types.jl")
include("utils.jl")
include("residuals.jl")
include("krylov_acceleration.jl")
include("linesearch.jl")
include("printing.jl")

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

temp_n_mat1 should be of dimension n by 2
temp_m_mat should be of dimension m by 2
temp_m_vec should be of dimension m
"""
function twocol_method_operator!(ws::Workspace,
    krylov_operator_tilde_A::Bool,
    temp_n_mat1::AbstractMatrix{Float64},
    temp_n_mat2::AbstractMatrix{Float64},
    temp_m_mat::AbstractMatrix{Float64},
    temp_n_vec_complex1::AbstractVector{ComplexF64},
    temp_n_vec_complex2::AbstractVector{ComplexF64},
    temp_m_vec_complex::AbstractVector{ComplexF64})
    
    # working variable is ws.vars.xy_q, with two columns
    
    @views two_col_mul!(temp_m_mat, ws.p.A, ws.vars.xy_q[1:ws.p.n, :], temp_n_vec_complex1, temp_m_vec_complex) # compute A * [x, q_n]
    temp_m_mat[:, 1] .-= ws.p.b # subtract b from A * x (but NOT from A * q_n)
    temp_m_mat .*= ws.ρ
    @views temp_m_mat .+= ws.vars.xy_q[ws.p.n+1:end, :] # add current
    @views ws.vars.preproj_y .= temp_m_mat[:, 1] # this is what's fed into dual cone projection operator
    @views project_to_dual_K!(temp_m_mat[:, 1], ws.p.K) # temp_m_mat[:, 1] now stores y_{k+1}
    
    # update in-place flags switching affine dynamics based on projection action
    # ws.proj_flags = D_k in Goodnotes handwritten notes
    @views update_proj_flags!(ws.proj_flags, ws.vars.preproj_y, temp_m_mat[:, 1])

    # can now compute the bit of q corresponding to the y iterate
    temp_m_mat[:, 2] .*= ws.proj_flags

    # now compute bar iterates (y and q_m) concurrently
    # TODO reduce memory allocation in this line...
    @views ws.vars.y_qm_bar .= (1 + ws.θ) * temp_m_mat - ws.θ * ws.vars.xy_q[ws.p.n+1:end, :]

    if !krylov_operator_tilde_A # ie use B = A - I as krylov operator
        @views temp_m_mat[:, 2] .-= ws.vars.xy_q[ws.p.n+1:end, 2] # add -I component
    end
    # NOTE: temp_m_mat[:, 2] now stores UPDATED q_m
    
    # ASSIGN new y and q_m to Workspace variables
    ws.vars.xy_q[ws.p.n+1:end, :] .= temp_m_mat

    # now we go to "bulk of" x and q_n update
    @views two_col_mul!(temp_n_mat1, ws.p.P, ws.vars.xy_q[1:ws.p.n, :], temp_n_vec_complex1, temp_n_vec_complex2) # compute P * [x, q_n]
    temp_n_mat1[:, 1] .+= ws.p.c # add linear part of objective to P * x (but NOT to P * q_n)
    @views two_col_mul!(temp_n_mat2, ws.p.A', ws.vars.y_qm_bar, temp_m_vec_complex, temp_n_vec_complex1) # compute A' * [y_bar, q_m_bar]
    temp_n_mat1 .+= temp_n_mat2 # this is what is pre-multiplied by W^{-1}
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to temp_n_mat1
    apply_inv!(ws.cache[:W_inv], temp_n_mat1)

    # ASSIGN new x and q_n: subtract off what we just computed, both columns
    if krylov_operator_tilde_A
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
function onecol_method_operator!(ws::Workspace,
    xy::AbstractVector{Float64},
    result_vec::AbstractVector{Float64},
    temp_n_vec1::AbstractVector{Float64},
    temp_n_vec2::AbstractVector{Float64},
    temp_m_vec::AbstractVector{Float64})

    @views mul!(temp_m_vec, ws.p.A, xy[1:ws.p.n]) # compute A * x
    temp_m_vec .-= ws.p.b # subtract b from A * x
    temp_m_vec .*= ws.ρ
    @views temp_m_vec .+= xy[ws.p.n+1:end] # add current
    @views project_to_dual_K!(temp_m_vec, ws.p.K) # temp_m_vec now stores y_{k+1}

    # now we go to "bulk of" x and q_n update
    @views mul!(temp_n_vec1, ws.p.P, xy[1:ws.p.n]) # compute P * x
    temp_n_vec1 .+= ws.p.c # add linear part of objective to P * x
    @views mul!(temp_n_vec2, ws.p.A', (1 + ws.θ) * temp_m_vec - ws.θ * xy[ws.p.n+1:end]) # compute A' * y_bar
    temp_n_vec1 .+= temp_n_vec2 # this is what is pre-multiplied by W^{-1}
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to temp_n_mat1
    apply_inv!(ws.cache[:W_inv], temp_n_vec1)

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
function accept_acc_candidate(ws::Workspace,
    current_xy::AbstractVector{Float64},
    accelerated_xy::AbstractVector{Float64},
    temp_mn_vec1::AbstractVector{Float64},
    temp_mn_vec2::AbstractVector{Float64},
    temp_n_vec1::AbstractVector{Float64},
    temp_n_vec2::AbstractVector{Float64},
    temp_m_vec::AbstractVector{Float64})

    # compute fixed-point residual at standard next iterate
    onecol_method_operator!(ws, current_xy, temp_mn_vec1, temp_n_vec1, temp_n_vec2, temp_m_vec) # iterate from current_xy
    onecol_method_operator!(ws, temp_mn_vec1, temp_mn_vec2, temp_n_vec1, temp_n_vec2, temp_m_vec) # iterate from temp_mn_vec1, which follows from current_xy

    # this computes fixed point residual OF THE standard iterate from
    # the current iterate
    fp_res_standard = norm(temp_mn_vec2 - temp_mn_vec1)

    # now compute fixed-point residual at accelerated iterate
    onecol_method_operator!(ws, accelerated_xy, temp_mn_vec1, temp_n_vec1, temp_n_vec2, temp_m_vec)
    fp_res_accel = norm(temp_mn_vec1 - accelerated_xy)

    # accept candidate if it reduces fixed-point residual
    println("Accel FP residual over standard FP residual: $(fp_res_accel / fp_res_standard)")
    
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

function preallocate_scratch(ws::Workspace, acceleration::Symbol)
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

function preallocate_record(ws::Workspace, run_fast::Bool,
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

function push_to_record!(ws::Workspace, record::NamedTuple, run_fast::Bool,
    k::Int, x_sol::Union{Nothing, AbstractVector{Float64}},
    curr_x_dist::Union{Nothing, Float64},
    curr_y_dist::Union{Nothing, Float64},
    curr_xy_chardist::Union{Nothing, Float64},
    primal_obj::Float64,
    dual_obj::Float64,
    curr_pri_res_norm::Float64,
    curr_dual_res_norm::Float64)
    if run_fast
        return nothing
    else
        if k > 0
            # NB: effective rank counts number of normalised singular values
            # larger than a specified small tolerance (eg 1e-8).
            # curr_effective_rank, curr_singval_ratio = effective_rank(record.updates_matrix, 1e-8)
            # push!(record.update_mat_ranks, curr_effective_rank)
            # push!(record.update_mat_singval_ratios, curr_singval_ratio)
            # push!(record.update_mat_iters, k)
            
            nothing
        end
        if !isnothing(x_sol)
            push!(record.x_dist_to_sol, curr_x_dist)
            push!(record.y_dist_to_sol, curr_y_dist)
            push!(record.xy_chardist, curr_xy_chardist)
        end
        push!(record.primal_obj_vals, primal_obj)
        push!(record.dual_obj_vals, dual_obj)
        push!(record.pri_res_norms, curr_pri_res_norm)
        push!(record.dual_res_norms, curr_dual_res_norm)
    end
end

"""
Run the optimiser for the initial inputs and solver options given.

Note on krylov_operator_tilde_A: if this is true, we use the Krylov subproblem
derived from considering tilde_A as the operator generating the Krylov
subspace in the Arnoldi process. If it is false, we use the operator
B := tilde_A - I instead (my first implementation used the latter, B approach).

acceleration is a symbol in {:none, :krylov, :anderson}

anderson_periods sets how often to attempt Anderson acceleration
"""
function optimise!(ws::Workspace,
    max_iter::Integer, print_modulo::Integer, run_fast::Bool,
    acceleration::Symbol;
    restart_period::Union{Real, Symbol} = Inf, residual_norm::Real = Inf,
    acceleration_memory::Integer,
    anderson_period::Integer,
    krylov_operator_tilde_A::Bool,
    linesearch_period::Union{Real, Symbol},
    linesearch_α_max::Float64 = 100.0, # NB: default linesearch params inspired by Giselsson et al. 2016 paper.
    linesearch_β::Float64 = 0.7,
    linesearch_ϵ::Float64 = 0.03,
    x_sol::AbstractVector{Float64} = nothing,
    y_sol::AbstractVector{Float64} = nothing,
    explicit_affine_operator::Bool = false,
    spectrum_plot_period::Int = 17,)

    # initialise dict to store results
    results = Results{Float64}()

    # create views into x and y variables, along with "Arnoldi vector" q
    if ws.vars isa Variables
        @views view_x = ws.vars.xy_q[1:ws.p.n, 1]
        @views view_y = ws.vars.xy_q[ws.p.n+1:end, 1]
        @views view_q = ws.vars.xy_q[:, 2]
    elseif ws.vars isa OnecolVariables
        @views view_x = ws.vars.xy[1:ws.p.n]
        @views view_y = ws.vars.xy[ws.p.n+1:end]
    else
        error("Unknown variable type in workspace.")
    end

    # prepare the pre-gradient linear operator for the x update
    ws.cache[:W] = W_operator(ws.variant, ws.p.P, ws.p.A, ws.cache[:A_gram], ws.τ, ws.ρ)
    ws.cache[:W_inv] = prepare_inv(ws.cache[:W])

    # if using acceleration, init krylov basis and Hessenberg arrays
    if acceleration == :krylov
        ws.cache[:krylov_basis] = zeros(Float64, ws.p.n + ws.p.m, acceleration_memory)
        ws.cache[:H] = init_upper_hessenberg(acceleration_memory)
    elseif acceleration == :anderson
        # default types:
        # COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.QRDecomp}, COSMOAccelerators.RestartedMemory, COSMOAccelerators.NoRegularizer}
        aa = COSMOAccelerators.AndersonAccelerator{Float64, COSMOAccelerators.Type2{COSMOAccelerators.NormalEquations}, COSMOAccelerators.RollingMemory, COSMOAccelerators.NoRegularizer}(ws.p.n + ws.p.m, mem = acceleration_memory)
    end
    
    # init acceleration trigger flag
    just_accelerated = true
    # set restart counter
    j_restart = 0

    # pre-allocate vectors to store residuals
    pri_res = zeros(Float64, ws.p.m)
    dual_res = zeros(Float64, ws.p.n)
    acc_pri_res = zeros(Float64, ws.p.m)
    acc_dual_res = zeros(Float64, ws.p.n)

    # pre-allocate vectors for intermediate results in in-place computations
    scratch = preallocate_scratch(ws, acceleration) # scratch is a named tuple
    prev_xy = zeros(Float64, ws.p.n + ws.p.m)
    curr_xy_update = zeros(Float64, ws.p.n + ws.p.m)
    prev_xy_update = zeros(Float64, ws.p.n + ws.p.m)

    # characteristic PPM preconditioner of the method
    if !run_fast
        ws.cache[:char_norm_mat] = [(ws.cache[:W] - ws.p.P) -ws.p.A'; -ws.p.A I(ws.p.m) / ws.ρ]
        function char_norm(vector::AbstractArray{Float64})
            return sqrt(dot(vector, ws.cache[:char_norm_mat] * vector))
        end
    end

    # data containers for metrics (if return_run_data == true).
    record = preallocate_record(ws, run_fast, x_sol)

    # notion of iteration for COSMOAccelerators may differ!
    # we shall increment it manually only when appropriate, for use
    # with functions from COSMOAccelerators
    k_anderson = 0

    # start main loop
    for k in 0:max_iter
        # Update average iterates
        if restart_period != Inf
            x_avg .= (j_restart * x_avg + view_x) / (j_restart + 1)
            y_avg .= (j_restart * y_avg + view_y) / (j_restart + 1)
        end

        # Compute residuals, their norms, and objective values.
        primal_residual!(ws, pri_res, ws.p.A, view_x, ws.p.b)
        dual_residual!(dual_res, scratch.temp_n_vec1, ws.p.P, ws.p.A, view_x, view_y, ws.p.c)
        curr_pri_res_norm = norm(pri_res, residual_norm)
        curr_dual_res_norm = norm(dual_res, residual_norm)
        primal_obj = primal_obj_val(ws.p.P, ws.p.c, view_x)
        dual_obj = dual_obj_val(ws.p.P, ws.p.b, view_x, view_y)
        gap = duality_gap(primal_obj, dual_obj)

        # compute distance to real solution
        if !run_fast
            curr_xy_chardist = !isnothing(x_sol) ? char_norm([view_x - x_sol; view_y + y_sol]) : nothing
            curr_x_dist = !isnothing(x_sol) ? norm(view_x - x_sol) : nothing
            curr_y_dist = !isnothing(y_sol) ? norm(view_y - (-y_sol)) : nothing
            curr_xy_dist = sqrt.(curr_x_dist .^ 2 .+ curr_y_dist .^ 2)

            # print info and save data if requested
            print_results(k, print_modulo, primal_obj, curr_pri_res_norm, curr_dual_res_norm, abs(gap), curr_xy_dist=curr_xy_dist)
            push_to_record!(ws, record, run_fast, k, x_sol, curr_x_dist, curr_y_dist, curr_xy_chardist, primal_obj, dual_obj, curr_pri_res_norm, curr_dual_res_norm)
        end

        # krylov setup
        if acceleration == :krylov
            # copy older iterate
            @views prev_xy .= ws.vars.xy_q[:, 1]
            
            # acceleration attempt step
            if k % (acceleration_memory + 1) == 0 && k > 0
                custom_acceleration_candidate!(ws, krylov_operator_tilde_A, acceleration_memory, scratch.accelerated_point, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec)

                @views if accept_acc_candidate(ws, ws.vars.xy_q[:, 1], scratch.accelerated_point, scratch.temp_mn_vec1, scratch.temp_mn_vec2, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec)
                    println("Accepted acceleration candidate at iteration $k.")
                    
                    # assign actually
                    ws.vars.xy_q[:, 1] .= scratch.accelerated_point

                    # record stuff
                    if !run_fast
                        @views curr_xy_update .= scratch.accelerated_point - ws.vars.xy_q[:, 1]
                        push!(record.acc_step_iters, k)
                        record.updates_matrix .= 0.0
                        record.current_update_mat_col[] = 1
                    end
                end
                # prevent recording 0 or very large update norm
                if !run_fast
                    push!(record.xy_step_norms, NaN)
                    push!(record.xy_step_char_norms, NaN)
                end

                # reset krylov acceleration data at each attempt regardless of success
                ws.cache[:krylov_basis] .= 0.0
                ws.cache[:H] .= 0.0

                just_accelerated = true
            # standard step in the krylov setup (two columns)
            else
                # apply method operator (to both (x, y) and Arnoldi (q) vectors)
                twocol_method_operator!(ws, krylov_operator_tilde_A, scratch.temp_n_mat1, scratch.temp_n_mat2, scratch.temp_m_mat, scratch.temp_n_vec_complex1, scratch.temp_n_vec_complex2, scratch.temp_m_vec_complex)


                # record iteration data
                if !run_fast
                    curr_xy_update .= ws.vars.xy_q[:, 1] - prev_xy
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm(curr_xy_update))
                    insert_update_into_matrix!(record.updates_matrix, curr_xy_update, record.current_update_mat_col)
                end

                if just_accelerated
                    # assign initial vector in the Krylov basis as initial
                    # fixed-point residual
                    @views ws.vars.xy_q[:, 2] .= ws.vars.xy_q[:, 1] - prev_xy
                    # normalise
                    @views ws.vars.xy_q[:, 2] ./= norm(ws.vars.xy_q[:, 2])
                    # store in Krylov basis
                    @views ws.cache[:krylov_basis][:, 1] .= ws.vars.xy_q[:, 2]
                else
                    # Arnoldi "step", orthogonalises the incoming basis vector
                    # and updates the Hessenberg matrix appropriately, all in-place
                    @views arnoldi_step!(ws.cache[:krylov_basis], ws.vars.xy_q[:, 2], ws.cache[:H])
                end

                # reset krylov acceleration flag
                just_accelerated = false

                j_restart += 1
            end
        else # NOT krylov set-up, so working variable is one-column
            if acceleration == :anderson && k % anderson_period == 0 && k > 0
                # ws.vars.xy might be overwritten, so we take note of it here
                scratch.temp_mn_vec1 .= ws.vars.xy

                # attempt acceleration step. if successful, this
                # overwites ws.vars.xy
                COSMOAccelerators.accelerate!(ws.vars.xy, prev_xy, aa, k_anderson)

                if !run_fast
                    # record step (might be zero if acceleration failed)
                    curr_xy_update .= ws.vars.xy - scratch.temp_mn_vec1

                    # account for records as appropriate
                    if any(x -> x != 0.0, curr_xy_update)
                        push!(record.acc_step_iters, k)
                        record.updates_matrix .= 0.0
                        record.current_update_mat_col[] = 1
                    end
                    
                    # prevent recording large or zero update norm
                    push!(record.xy_step_norms, NaN)
                    push!(record.xy_step_char_norms, NaN)
                end
            else # standard onecol iteration
                # copy older iterate before iterating
                prev_xy .= ws.vars.xy

                onecol_method_operator!(ws, ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_n_vec1, scratch.temp_n_vec2, scratch.temp_m_vec)

                # swap contents of ws.vars.xy and scratch.temp_mn_vec1
                custom_swap!(ws.vars.xy, scratch.temp_mn_vec1, scratch.temp_mn_vec2)
                # now ws.vars.xy contains newer iterate,  while
                # scratch.temp_mn_vec1 contains older one

                # if using Anderson accel, now update accelerator standard
                # iterate/successor pair history
                if acceleration == :anderson
                    COSMOAccelerators.update!(aa, ws.vars.xy, scratch.temp_mn_vec1, k_anderson)
                    
                    # note that we do NOT increment the COSMOAccelerators iteration
                    # on acceleration attempts --- only on standard ones
                    k_anderson += 1
                end

                if !run_fast
                    # record step (might be zero if acceleration failed)
                    curr_xy_update .= ws.vars.xy - scratch.temp_mn_vec1

                    # record iteration data here
                    push!(record.xy_step_norms, norm(curr_xy_update))
                    push!(record.xy_step_char_norms, char_norm(curr_xy_update))
                end
            end
        end
        
        if !run_fast
            # store cosine between last two iterate updates
            if k >= 1
                xy_prev_updates_cos = abs(dot(curr_xy_update, prev_xy_update) / (norm(curr_xy_update) * norm(prev_xy_update)))
                push!(record.xy_update_cosines, xy_prev_updates_cos)
            end
            prev_xy_update .= curr_xy_update

            # store active set flags
            push!(record.record_proj_flags, ws.proj_flags)
        end

        # Notion of a previous iterate step makes sense (again).
        just_restarted = false
    end

    # TERMINATION: Compute residuals, their norms, and objective values
    # TODO: make this stuff modular as opposed to copied from the main loop
    primal_residual!(ws, pri_res, ws.p.A, view_x, ws.p.b)
    dual_residual!(dual_res, scratch.temp_n_vec1, ws.p.P, ws.p.A, view_x, view_y, ws.p.c)
    curr_pri_res_norm = norm(pri_res, residual_norm)
    curr_dual_res_norm = norm(dual_res, residual_norm)
    primal_obj = primal_obj_val(ws.p.P, ws.p.c, view_x)
    dual_obj = dual_obj_val(ws.p.P, ws.p.b, view_x, view_y)
    gap = duality_gap(primal_obj, dual_obj)
    
    # END: Store metrics if requested.
    if !run_fast
        curr_xy_chardist = !isnothing(x_sol) ? sqrt(dot([view_x - x_sol; view_y + y_sol], ws.cache[:char_norm_mat] * [view_x - x_sol; view_y + y_sol])) : nothing
        curr_x_dist = !isnothing(x_sol) ? norm(view_x - x_sol) : nothing
        curr_y_dist = !isnothing(y_sol) ? norm(view_y - (-y_sol)) : nothing

        push!(record.primal_obj_vals, primal_obj)
        push!(record.dual_obj_vals, dual_obj)
        push!(record.pri_res_norms, curr_pri_res_norm)
        push!(record.dual_res_norms, curr_dual_res_norm)
        if !isnothing(x_sol)
            push!(record.x_dist_to_sol, curr_x_dist)
            push!(record.y_dist_to_sol, curr_y_dist)
            push!(record.xy_chardist, curr_xy_chardist)
        end

        # END: print iteration info
        curr_xy_dist = sqrt.(curr_x_dist .^ 2 .+ curr_y_dist .^ 2)
        print_results(max_iter, print_modulo, primal_obj, curr_pri_res_norm, curr_dual_res_norm, abs(gap), curr_xy_dist = curr_xy_dist, terminated = true)

        # assign results as appropriate
        results.data = Dict(
            :primal_obj_vals => record.primal_obj_vals,
            :dual_obj_vals => record.dual_obj_vals,
            :pri_res_norms => record.pri_res_norms,
            :dual_res_norms => record.dual_res_norms,
            :record_proj_flags => record.record_proj_flags,
            :x_dist_to_sol => record.x_dist_to_sol,
            :y_dist_to_sol => record.y_dist_to_sol,
            :xy_chardist => record.xy_chardist,
            :update_mat_iters => record.update_mat_iters,
            :update_mat_ranks => record.update_mat_ranks,
            :update_mat_singval_ratios => record.update_mat_singval_ratios,
            :acc_step_iters => record.acc_step_iters,
            :linesearch_iters => record.linesearch_iters,
            :xy_step_norms => record.xy_step_norms,
            :xy_step_char_norms => record.xy_step_char_norms,
            :xy_update_cosines => record.xy_update_cosines
        )
    end

    return results
end