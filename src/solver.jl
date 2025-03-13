using LinearAlgebra
using Plots
using Random
using Printf
using Infiltrator
include("types.jl")
include("utils.jl")
include("residuals.jl")
include("acceleration.jl")
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

temp_n_mat should be of dimension n by 2
temp_m_mat should be of dimension m by 2
temp_m_vec should be of dimension m
"""
function method_operator!(ws::Workspace,
    temp_n_mat::AbstractMatrix{Float64},
    temp_n_mat2::AbstractMatrix{Float64},
    temp_m_mat::AbstractMatrix{Float64},
    temp_m_vec::AbstractVector{Float64},
    temp_n_vec_complex::AbstractVector{ComplexF64},
    temp_m_vec_complex::AbstractVector{ComplexF64})
    
    # working variable is ws.vars.xy_q, with two columns
    
    @views two_col_mul!(temp_m_mat, ws.p.A, ws.vars.xy_q[1:ws.p.n, :], temp_n_vec_complex, temp_m_vec_complex) # A * x
    temp_m_mat .-= ws.p.b
    temp_m_mat .*= ws.ρ
    @views temp_m_mat .+= ws.vars.xy_q[ws.p.n+1:end, :]
    @views ws.vars.preproj_y .= temp_m_mat[:, 1] # this is what's fed into dual cone projection operator
    @views project_to_dual_K!(temp_m_mat[:, 1], ws.p.K) # result of this is y_{k+1}
    
    # update in-place flags switching affine dynamics based on projection action
    # ws.proj_flags = D_k in Goodnotes notes
    update_proj_flags!(ws.proj_flags, ws.vars.preproj_y, temp_m_mat[:, 1])

    # can now compute the bit of q corresponding to the y iterate
    temp_m_mat[:, 2] .*= ws.proj_flags
    @views temp_m_vec .= (.!ws.proj_flags) .* temp_m_mat[:, 1] # temp_m_mat[:, 1] at this point stores y_{k+1}
    temp_m_mat[:, 2] .+= temp_m_vec # result of this is the new q_m vector
    
    # now compute bar iterates (y and q_m) concurrently
    @views ws.vars.y_qm_bar .= (1 + ws.θ) * temp_m_mat - ws.θ * ws.vars.xy_q[ws.p.n+1:end, :]
    
    # ASSIGN new y and q_m
    ws.vars.xy_q[ws.p.n+1:end, :] .= temp_m_mat

    # now onto the bulk of x and q_n update
    @views two_col_mul!(temp_n_mat, ws.p.P, ws.vars.xy_q[1:ws.p.n, :], temp_n_vec_complex, temp_m_vec_complex) # P * x
    temp_n_mat .+= ws.p.c # add linear part of objective
    @views two_col_mul!(temp_n_mat2, ws.p.A', ws.vars.y_qm_bar, temp_n_vec_complex, temp_m_vec_complex) # A' * y_bar
    temp_n_mat .+= temp_n_mat2 # this is to be pre-multiplied by W^{-1}
    
    # in-place, efficiently apply W^{-1} = (P + \tilde{M}_1)^{-1} to temp_n_mat
    apply_inv!(ws.cache[:W_inv], temp_n_mat)

    # ASSIGN new x and q_n
    ws.vars.xy_q[1:ws.p.n, :] .-= temp_n_mat
    
    # TODO: figure out different operations based on the Krylov operator to use
    # for q: tilde_A or B := tilde_A - I.
    # TAKE OUT non-linear affine bit, i.e. tilde_b, for q sequence.

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

# Crude check for acceptance of an acceleration candidate.
function accept_acc_candidate(ws::Workspace,
    acc_pri_res::AbstractVector{Float64},
    acc_dual_res::AbstractVector{Float64},
    curr_pri_res::AbstractVector{Float64},
    curr_dual_res::AbstractVector{Float64},
    residual_norm::Real)
    # Compute residual norms.
    acc_pri_res_norm = norm(acc_pri_res, residual_norm)
    acc_dual_res_norm = norm(acc_dual_res, residual_norm)
    curr_pri_res_norm = norm(curr_pri_res, residual_norm)
    curr_dual_res_norm = norm(curr_dual_res, residual_norm)

    # Accept candidate if reduced residuals.
    return acc_pri_res_norm < curr_pri_res_norm && acc_dual_res_norm < curr_dual_res_norm
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

"""
Run the optimiser for the initial inputs and solver options given.

Note on krylov_operator_tilde_A: if this is true, we use the Krylov subproblem
derived from considering tilde_A as the operator generating the Krylov
subspace in the Arnoldi process. If it is false, we use the operator
B := tilde_A - I instead (my first implementation used the latter, B approach).
"""
function optimise!(ws::Workspace, max_iter::Integer, print_modulo::Integer;
    restart_period::Union{Real, Symbol} = Inf, residual_norm::Real = Inf,
    acceleration::Bool = false,
    acceleration_memory::Integer = 20,
    linesearch_period::Union{Real, Symbol} = 20,
    linesearch_α_max::Float64 = 100.0, # NB: default linesearch params inspired by Giselsson et al. 2016 paper.
    linesearch_β::Float64 = 0.7,
    linesearch_ϵ::Float64 = 0.03,
    krylov_operator_tilde_A::Bool = false,
    x_sol::AbstractVector{Float64} = nothing,
    y_sol::AbstractVector{Float64} = nothing,
    explicit_affine_operator::Bool = false,
    spectrum_plot_period::Int = 17,)

    # initialise dict to store results
    results = Results{Float64}()

    # create views into x and y variables, along with "Arnoldi vector" q
    @views view_x = ws.vars.xy_q[1:ws.p.n, 1]
    @views view_y = ws.vars.xy_q[ws.p.n+1:end, 1]
    @views view_q = ws.vars.xy_q[:, 2]

    # We maintain a matrix whose columns are formed by the past
    # acceleration_memory iterate updates, normalised to unit l2 norm.
    # Init this matrix.
    no_stored_updates_cond = 20
    updates_matrix = zeros(ws.p.n + ws.p.m, no_stored_updates_cond)
    current_update_mat_col = Ref(1) # This index is updated in circular fashion.
    # This variable determines how often we compute and store this rank.
    # updates_cond_period = Int(floor((acceleration_memory + 1) / 4))
    updates_cond_period = 1

    # prepare the pre-gradient linear operator for the x update
    ws.cache[:W] = W_operator(ws.variant, ws.p.P, ws.p.A, ws.cache[:A_gram], ws.τ, ws.ρ)
    ws.cache[:W_inv] = prepare_inv(ws.cache[:W])

    # characteristic PPM preconditioner of the method
    ws.cache[:char_norm_mat] = [(ws.cache[:W] - ws.p.P) -ws.p.A'; -ws.p.A I(ws.p.m) / ws.ρ]

    # implement function to compute charcteristic norm of the method
    function char_norm(vector::AbstractArray{Float64})
        return sqrt(dot(vector, ws.cache[:char_norm_mat] * vector))
    end
    
    # for restarts, keep average iterates
    if restart_period != Inf
        x_avg = copy(view_x)
        y_avg = copy(view_y)
    end

    # pre-allocate vectors to store auxiliary quantities, residuals, etc.
    pri_res = zeros(Float64, ws.p.m)
    dual_res = zeros(Float64, ws.p.n)
    temp_n_mat = zeros(Float64, ws.p.n, 2)
    temp_n_mat2 = zeros(Float64, ws.p.n, 2)
    temp_m_mat = zeros(Float64, ws.p.m, 2)
    temp_n_vec = zeros(Float64, ws.p.n) # memory for intermediate computations
    temp_m_vec = zeros(Float64, ws.p.m)
    temp_n_vec_complex = zeros(ComplexF64, ws.p.n)
    temp_m_vec_complex = zeros(ComplexF64, ws.p.m)
    acc_pri_res = zeros(Float64, ws.p.m)
    acc_dual_res = zeros(Float64, ws.p.n)
    prev_xy = similar(ws.vars.xy_q[:, 1])
    j_restart = 0

    # data containers for metrics (if return_run_data == true).
    primal_obj_vals = Float64[]
    dual_obj_vals = Float64[]
    pri_res_norms = Float64[]
    dual_res_norms = Float64[]
    record_proj_flags = Vector{Vector{Bool}}()
    update_mat_ranks = Float64[]
    update_mat_singval_ratios = Float64[]
    update_mat_iters = Int[]
    acc_step_iters = Int[]
    linesearch_iters = Int[]
    xy_step_norms = Float64[]
    xy_step_char_norms = Float64[] #record method's "char norm" of the updates
    xy_update_cosines = Float64[]
    
    # If actual solution provided.
    if !isnothing(x_sol)
        x_dist_to_sol = Float64[]
        s_dist_to_sol = Float64[]
        y_dist_to_sol = Float64[]
        v_dist_to_sol = Float64[]
        xy_semidist = Float64[]
    end

    # I want these vectors to persist across iterations of the main loop,
    # so I have to inialise them beforehand.
    curr_xy_update = Vector{Float64}(undef, ws.p.n + ws.p.m)
    prev_xy_update = Vector{Float64}(undef, ws.p.n + ws.p.m)

    # This flag helps us keep track of whether we should forgo the notion of a
    # "previous" step, including in the very first iteration.
    just_restarted = true
    just_accelerated = true

    for k in 0:max_iter
        # Update average iterates
        if restart_period != Inf
            x_avg .= (j_restart * x_avg + view_x) / (j_restart + 1)
            y_avg .= (j_restart * y_avg + view_y) / (j_restart + 1)
        end

        # Compute residuals, their norms, and objective values.
        primal_residual!(ws, pri_res, ws.p.A, view_x, ws.p.b)
        dual_residual!(dual_res, temp_n_vec, ws.p.P, ws.p.A, view_x, view_y, ws.p.c)
        curr_pri_res_norm = norm(pri_res, residual_norm)
        curr_dual_res_norm = norm(dual_res, residual_norm)
        primal_obj = primal_obj_val(ws.p.P, ws.p.c, view_x)
        dual_obj = dual_obj_val(ws.p.P, ws.p.b, view_x, view_y)
        gap = duality_gap(primal_obj, dual_obj)

        if !isnothing(x_sol)
            # TODO: compute this in terms of norm function defined at
            # the beginning of this function definition
            curr_xy_semidist = !isnothing(x_sol) ? sqrt(dot([view_x - x_sol; view_y + y_sol], ws.cache[:char_norm_mat] * [view_x - x_sol; view_y + y_sol])) : nothing
            curr_x_dist = !isnothing(x_sol) ? norm(view_x - x_sol) : nothing
            curr_y_dist = !isnothing(y_sol) ? norm(view_y - (-y_sol)) : nothing
        end

        # Print iteration info.
        if k % print_modulo == 0
            curr_xy_dist = sqrt.(curr_x_dist .^ 2 .+ curr_y_dist .^ 2)
            print_results(k, primal_obj, curr_pri_res_norm, curr_dual_res_norm, abs(gap), curr_xy_dist=curr_xy_dist)
        end

        # Note period of computation of the rank of updates.
        if k % updates_cond_period == 0 && k > 0
            # NB: effective rank counts number of normalised singular values
            # larger than a specified small tolerance (eg 1e-8).
            curr_effective_rank, curr_singval_ratio = effective_rank(updates_matrix, 1e-8)
            push!(update_mat_ranks, curr_effective_rank)
            push!(update_mat_singval_ratios, curr_singval_ratio)
            push!(update_mat_iters, k)
        end

        # Store metrics.
        push!(primal_obj_vals, primal_obj)
        push!(dual_obj_vals, dual_obj)
        push!(pri_res_norms, curr_pri_res_norm)
        push!(dual_res_norms, curr_dual_res_norm)
        if !isnothing(x_sol)
            push!(x_dist_to_sol, curr_x_dist)
            push!(y_dist_to_sol, curr_y_dist)
            push!(xy_semidist, curr_xy_semidist)
        end

        ### ITERATE ###
        
        # STANDARD OR LINESEARCH ITERATION.
    
        prev_xy .= ws.vars.xy_q[:, 1]

        linesearch_success = false #init to access outside conditional
        if k % linesearch_period == 0 && k > 0 # LINESEARCH ATTEMPT
            linesearch_success, linesearch_update, linesearch_lookahead = fixed_point_linesearch!(ws, linesearch_α_max, linesearch_β, linesearch_ϵ, char_norm, krylov_operator_tilde_A, k)
            
            #TODO: get rid of this line:
            # supposed_lookahead .= linesearch_lookahead

            if linesearch_success
                push!(linesearch_iters, k)
            end
        else # STANDARD (no line search) ITERATION
            method_operator!(ws, temp_n_mat, temp_n_mat2,
            temp_m_mat, temp_m_vec, temp_n_vec_complex, temp_m_vec_complex)

            # assign to xy_q variable
            ws.vars.xy_q[:, 1] .= [view_x; view_y]

            curr_xy_update .= ws.vars.xy_q[:, 1] - prev_xy

            push!(xy_step_norms, norm(curr_xy_update))
            push!(xy_step_char_norms, char_norm(curr_xy_update))
            insert_update_into_matrix!(updates_matrix, curr_xy_update, current_update_mat_col)


            just_accelerated = false
            j_restart += 1

        end
        
        # Store cosine between last two iterate updates.
        if k >= 1
            xy_prev_updates_cos = abs(dot(curr_xy_update, prev_xy_update) / (norm(curr_xy_update) * norm(prev_xy_update)))
            push!(xy_update_cosines, xy_prev_updates_cos)
        end
        prev_xy_update .= curr_xy_update

        push!(record_proj_flags, ws.proj_flags)

        # Notion of a previous iterate step makes sense (again).
        just_restarted = false
    end

    # END: Compute residuals, their norms, and objective values.
    # TODO: make this stuff modular as opposed to copied from the main loop.
    primal_residual!(ws, pri_res, ws.p.A, view_x, ws.p.b)
    dual_residual!(dual_res, temp_n_vec, ws.p.P, ws.p.A, view_x, view_y, ws.p.c)
    curr_pri_res_norm = norm(pri_res, residual_norm)
    curr_dual_res_norm = norm(dual_res, residual_norm)
    primal_obj = primal_obj_val(ws.p.P, ws.p.c, view_x)
    dual_obj = dual_obj_val(ws.p.P, ws.p.b, view_x, view_y)
    gap = duality_gap(primal_obj, dual_obj)
    
    curr_xy_semidist = !isnothing(x_sol) ? sqrt(dot([view_x - x_sol; view_y + y_sol], ws.cache[:char_norm_mat] * [view_x - x_sol; view_y + y_sol])) : nothing
    curr_x_dist = !isnothing(x_sol) ? norm(view_x - x_sol) : nothing
    curr_y_dist = !isnothing(y_sol) ? norm(view_y - (-y_sol)) : nothing

    # END: Store metrics if requested.
    # TODO: make this stuff modular as opposed to copied from the main loop.
    push!(primal_obj_vals, primal_obj)
    push!(dual_obj_vals, dual_obj)
    push!(pri_res_norms, curr_pri_res_norm)
    push!(dual_res_norms, curr_dual_res_norm)
    if !isnothing(x_sol)
        push!(x_dist_to_sol, curr_x_dist)
        push!(y_dist_to_sol, curr_y_dist)
        push!(xy_semidist, curr_xy_semidist)
    end

    # END: print iteration info.
    curr_xy_dist = sqrt.(curr_x_dist .^ 2 .+ curr_y_dist .^ 2)
    print_results(max_iter, primal_obj, curr_pri_res_norm, curr_dual_res_norm, abs(gap), curr_xy_dist = curr_xy_dist, terminated = true)

    # assign results as appropriate
    results.data = Dict(
        :primal_obj_vals => primal_obj_vals,
        :dual_obj_vals => dual_obj_vals,
        :pri_res_norms => pri_res_norms,
        :dual_res_norms => dual_res_norms,
        :record_proj_flags => record_proj_flags,
        :x_dist_to_sol => x_dist_to_sol,
        :y_dist_to_sol => y_dist_to_sol,
        :xy_semidist => xy_semidist,
        :update_mat_iters => update_mat_iters,
        :update_mat_ranks => update_mat_ranks,
        :update_mat_singval_ratios => update_mat_singval_ratios,
        :acc_step_iters => acc_step_iters,
        :linesearch_iters => linesearch_iters,
        :xy_step_norms => xy_step_norms,
        :xy_step_char_norms => xy_step_char_norms,
        :xy_update_cosines => xy_update_cosines
    )

    return results
end