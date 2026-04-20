# Results plotting for FOMPrototypes

#############################
# Refactored Plotting Block #
#############################

function plot_results(
    ws::AbstractWorkspace,
    results,
    problem_set::String,
    problem_name::String,
    config::SolverConfig,
    backend::Symbol = :plotlyjs)

    active = resolve_plot_set(config.plot_set)

    newline_char = initialise_misc(backend)

    println("Backend is $(Plots.backend_name())")

    k_final = length(results.metrics_history[:primal_obj_vals])

    # plotting constants
    LINEWIDTH = 2.5
    VERT_LINEWIDTH = 1
    ALPHA = 0.9

    # Common title components
    title_common = "Problem: $(problem_set) $(problem_name).$newline_char Variant $(config.variant) $newline_char"
    # title_common *= "Restart period = $(config.restart_period).$newline_char Linesearch period = $(config.linesearch_period)$newline_char"
    if config.acceleration == :none
        title_common *= "Acceleration: none.$newline_char"
        krylov_operator_str = ""
    elseif config.acceleration == :anderson
        title_common *= "Anderson acceleration: mem = $(config.accel_memory), interval = $(config.anderson_interval),$newline_char broyden = $(config.anderson_broyden_type), mem_type = $(config.anderson_mem_type).$newline_char"
        krylov_operator_str = ""
    elseif config.acceleration == :krylov
        title_common *= "Krylov acceleration: mem = $(config.accel_memory), op = $(config.krylov_operator).$newline_char"
    end

    # Lazily compute active-set data only when needed
    constraint_lines = Int[]
    nn_history = Vector{Vector{Bool}}()
    soc_history = Vector{Vector{SOCAction}}()
    needs_constraint_data = any(s -> should_plot(active, s),
        [:proj_diffs, :enforced_constraints, :active_set_deviation, :unseen_deviations, :soc_angles,
         :preproj_late_flippers, :scaled_preproj_late_flippers])
    if needs_constraint_data
        nn_history = get(results.metrics_history, :record_proj_flags, Vector{Vector{Bool}}())
        soc_history = get(results.metrics_history, :record_soc_states, Vector{Vector{SOCAction}}())
        constraint_lines = constraint_changes(nn_history, soc_history)
    end

    # Lazily load pre-projection history when needed
    preproj_history = Vector{Vector{Float64}}()
    if any(s -> should_plot(active, s), [:preproj_late_flippers, :scaled_preproj_late_flippers,
                                         :flip_prediction_quality, :flip_prediction_quality_rate])
        preproj_history = get(results.metrics_history, :record_preproj_vecs, Vector{Vector{Float64}}())
    end

    # Helper function to add common vertical lines, only if show_vlines is true.
    function add_vlines!(plt; include_active_set_changes::Bool = true)
        if config.show_vlines
            vline!(plt, results.metrics_history[:acc_step_iters], line = (:dash, ALPHA, :red, VERT_LINEWIDTH * 1.5), label="Accelerated Steps")
            vline!(plt, results.metrics_history[:linesearch_iters], line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH), label="Line Search Steps")
            if include_active_set_changes
                vline!(plt, constraint_lines, line = (:solid, ALPHA, :green, VERT_LINEWIDTH), label="Active set changes")
            end
        end
        return plt
    end

    # Primal objective plot.
    if should_plot(active, :primal_obj)
        primal_obj_plot = plot(0:k_final-1, results.metrics_history[:primal_obj_vals], linewidth=LINEWIDTH,
        label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value",
        title="$title_common Objective")
        add_vlines!(primal_obj_plot)
        display(primal_obj_plot)
    end

    # Dual objective plot.
    if should_plot(active, :dual_obj)
        dual_obj_plot = plot(0:k_final-1, results.metrics_history[:dual_obj_vals], linewidth=LINEWIDTH,
        label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value",
        title="$title_common Dual Objective")
        add_vlines!(dual_obj_plot)
        display(dual_obj_plot)
    end

    # Duality gap plot.
    if should_plot(active, :duality_gap)
        gap_plot = plot(0:k_final-1, results.metrics_history[:primal_obj_vals] - results.metrics_history[:dual_obj_vals], linewidth=LINEWIDTH,
        label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap",
        title="$title_common Duality Gap")
        add_vlines!(gap_plot)
        display(gap_plot)
    end

    # Primal residual plot.
    if should_plot(active, :primal_res)
        pres_plot = plot(0:k_final-1, results.metrics_history[:pri_res_norms], linewidth=LINEWIDTH,
        label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual",
        title="$title_common Primal Residual Norm", yaxis=:log10)
        add_vlines!(pres_plot)
        display(pres_plot)
    end

    # Dual residual plot.
    if should_plot(active, :dual_res)
        dres_plot = plot(0:k_final-1, results.metrics_history[:dual_res_norms], linewidth=LINEWIDTH,
        label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual",
        title="$title_common Dual Residual Norm", yaxis=:log10)
        add_vlines!(dres_plot)
        display(dres_plot)
    end

    if length(results.metrics_history[:x_dist_to_sol]) != 0
        # state distance to solution plot.
        if should_plot(active, :state_dist)
            state_dist_to_sol = sqrt.(results.metrics_history[:x_dist_to_sol] .^ 2 .+ results.metrics_history[:y_dist_to_sol] .^ 2)
            state_dist_plot = plot(0:k_final, state_dist_to_sol, linewidth=LINEWIDTH,
                label="Prototype state Distance", xlabel="Iteration", ylabel="Distance to Solution",
                title="$title_common state Distance to Solution", yaxis=:log10)
            add_vlines!(state_dist_plot)
            display(state_dist_plot)
        end

        # state characteristic norm distance to solution plot.
        if should_plot(active, :state_chardist)
            seminorm_plot = plot(0:k_final, results.metrics_history[:state_chardist], linewidth=LINEWIDTH,
            label="state Seminorm Distance (Theory)", xlabel="Iteration", ylabel="Distance to Solution",
            title="$title_common state Characteristic Norm Distance to Solution", yaxis=:log10)
            add_vlines!(seminorm_plot)
            display(seminorm_plot)
        end
    end

    # state step norms plot.
    if should_plot(active, :step_l2)
        state_step_norms_plot = plot(0:k_final-1, results.metrics_history[:state_step_norms], linewidth=LINEWIDTH,
            label="state Step l2 Norm", xlabel="Iteration", ylabel="Step Norm",
            title="$title_common state l2 Step Norm", yaxis=:log10)
        add_vlines!(state_step_norms_plot)
        display(state_step_norms_plot)
    end

    # state step CHAR norms plot.
    if should_plot(active, :step_char)
        state_step_char_norms_plot = plot(0:k_final-1, results.metrics_history[:state_step_char_norms], linewidth=LINEWIDTH,
            label="state Step Char Norm", xlabel="Iteration", ylabel="Step CHAR Norm",
            title="$title_common state CHAR Step Norm", yaxis=:log10)
        add_vlines!(state_step_char_norms_plot)
        display(state_step_char_norms_plot)
    end

    # Consecutive update state angles plot (in degrees).
    if should_plot(active, :update_angles)
        state_update_angles_deg = rad2deg.(results.metrics_history[:state_update_angles])
        state_update_angles_plot = plot(1:k_final-1, state_update_angles_deg, linewidth=LINEWIDTH,
            label="Prototype Update Angle", xlabel="Iteration", ylabel="Angle between Consecutive Updates (degrees)",
            title="$title_common Consecutive State Update Angles")
        add_vlines!(state_update_angles_plot)
        display(state_update_angles_plot)
    end

    # plot count of flipped constraints
    if should_plot(active, :proj_diffs)
        proj_diffs_plot = plot_projection_diffs(nn_history, soc_history)
        add_vlines!(proj_diffs_plot)
        display(proj_diffs_plot)
    end

    # plot count of enforced constraints
    if should_plot(active, :enforced_constraints)
        enforced_constraints_plot = plot_enforced_constraints_count(nn_history, soc_history)
        add_vlines!(enforced_constraints_plot)
        display(enforced_constraints_plot)
    end

    # plot active set deviation from final iteration
    if should_plot(active, :active_set_deviation)
        deviation_plot = plot_active_set_deviation_from_final(nn_history, soc_history)
        add_vlines!(deviation_plot)
        display(deviation_plot)
    end

    # plot unseen deviations from final (constraints that need to flip but never have)
    if should_plot(active, :unseen_deviations)
        unseen_plot = plot_unseen_deviations_from_final(nn_history, soc_history)
        add_vlines!(unseen_plot)
        display(unseen_plot)
    end

    # plot |u_i| for late-flipping constraints
    if should_plot(active, :preproj_late_flippers) && !isempty(preproj_history)
        preproj_plot = plot_preproj_late_flippers(
            preproj_history, nn_history, ws.p.K;
            title_prefix = title_common,
        )
        if !isnothing(preproj_plot)
            add_vlines!(preproj_plot)
            display(preproj_plot)
        end
    end

    # plot scaled |ũ_i| for late-flipping constraints
    if should_plot(active, :scaled_preproj_late_flippers) && !isempty(preproj_history)
        scaled_preproj_plot = plot_scaled_preproj_late_flippers(
            preproj_history, nn_history, ws.p.K, ws.p.A, ws.method.ρ[1];
            title_prefix = title_common,
        )
        if !isnothing(scaled_preproj_plot)
            add_vlines!(scaled_preproj_plot)
            display(scaled_preproj_plot)
        end
    end

    # plot flip-prediction quality for the min-|u_i| predictor
    if should_plot(active, :flip_prediction_quality) && !isempty(preproj_history)
        flip_pred_plot = plot_flip_prediction_quality(
            preproj_history, nn_history, ws.p.K;
            title_prefix = title_common,
        )
        if !isnothing(flip_pred_plot)
            display(flip_pred_plot)
        end
    end

    # plot flip-prediction quality for the rate-of-change Δ|u_i| predictor
    if should_plot(active, :flip_prediction_quality_rate) && !isempty(preproj_history)
        flip_pred_rate_plot = plot_flip_prediction_quality_rate(
            preproj_history, nn_history, ws.p.K;
            title_prefix = title_common,
        )
        if !isnothing(flip_pred_rate_plot)
            display(flip_pred_rate_plot)
        end
    end

    # FP Metric Ratio plot.
    if should_plot(active, :fp_metric)
        acc_attempt_iters = results.metrics_history[:acc_attempt_iters]
        fp_metric_ratios = results.metrics_history[:fp_metric_ratios]
        fp_metric_plot = plot(acc_attempt_iters, fp_metric_ratios, linewidth=LINEWIDTH,
            label="FP Metric Ratio", xlabel="Acceleration Attempt Iterations", ylabel="FP Metric Ratio",
            title="$title_common FP Metric Ratio",
            lw=2, # Set line width for better visibility
            marker=:circle, # Add markers to each data point
            markersize=3,
            yscale=:log10)
        add_vlines!(fp_metric_plot)
        display(fp_metric_plot)
    end

    # SOC normal direction angles plot (only if data available)
    if should_plot(active, :soc_angles) &&
       haskey(results.metrics_history, :soc_normal_angles) &&
       !isempty(results.metrics_history[:soc_normal_angles])
        soc_angles_plot = plot_soc_normal_angles(
            results.metrics_history[:soc_normal_angles],
            title_common
        )
        if !isnothing(soc_angles_plot)
            add_vlines!(soc_angles_plot; include_active_set_changes=false)
            display(soc_angles_plot)
        end
    end
end

plot_results(ws::AbstractWorkspace, results, problem_set::String, problem_name::String, config::AbstractDict, backend::Symbol = :plotlyjs) =
    plot_results(ws, results, problem_set, problem_name, SolverConfig(config), backend)
