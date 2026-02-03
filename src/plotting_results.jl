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

    # Add Krylov operator string if acceleration is :krylov


    nn_history = get(results.metrics_history, :record_proj_flags, Vector{Vector{Bool}}())
    soc_history = get(results.metrics_history, :record_soc_states, Vector{Vector{SOCAction}}())
    constraint_lines = constraint_changes(nn_history, soc_history)

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
    primal_obj_plot = plot(0:k_final-1, results.metrics_history[:primal_obj_vals], linewidth=LINEWIDTH,
    label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value",
    title="$title_common Objective")
    add_vlines!(primal_obj_plot)
    display(primal_obj_plot)

    # Dual objective plot.
    dual_obj_plot = plot(0:k_final-1, results.metrics_history[:dual_obj_vals], linewidth=LINEWIDTH,
    label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value",
    title="$title_common Dual Objective")
    add_vlines!(dual_obj_plot)
    display(dual_obj_plot)

    # Duality gap plot.
    gap_plot = plot(0:k_final-1, results.metrics_history[:primal_obj_vals] - results.metrics_history[:dual_obj_vals], linewidth=LINEWIDTH,
    label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap",
    title="$title_common Duality Gap")
    add_vlines!(gap_plot)
    display(gap_plot)

    # Primal residual plot.
    pres_plot = plot(0:k_final-1, results.metrics_history[:pri_res_norms], linewidth=LINEWIDTH,
    label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual",
    title="$title_common Primal Residual Norm", yaxis=:log10)
    add_vlines!(pres_plot)
    display(pres_plot)

    # Dual residual plot.
    dres_plot = plot(0:k_final-1, results.metrics_history[:dual_res_norms], linewidth=LINEWIDTH,
    label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual",
    title="$title_common Dual Residual Norm", yaxis=:log10)
    add_vlines!(dres_plot)
    display(dres_plot)

    if length(results.metrics_history[:x_dist_to_sol]) != 0
        # state distance to solution plot.
        state_dist_to_sol = sqrt.(results.metrics_history[:x_dist_to_sol] .^ 2 .+ results.metrics_history[:y_dist_to_sol] .^ 2)
        state_dist_plot = plot(0:k_final, state_dist_to_sol, linewidth=LINEWIDTH,
            label="Prototype state Distance", xlabel="Iteration", ylabel="Distance to Solution",
            title="$title_common state Distance to Solution", yaxis=:log10)
        add_vlines!(state_dist_plot)
        display(state_dist_plot)

        # state characteristic norm distance to solution plot.
        seminorm_plot = plot(0:k_final, results.metrics_history[:state_chardist], linewidth=LINEWIDTH,
        label="state Seminorm Distance (Theory)", xlabel="Iteration", ylabel="Distance to Solution",
        title="$title_common state Characteristic Norm Distance to Solution", yaxis=:log10)
        add_vlines!(seminorm_plot)
        display(seminorm_plot)
    end

    # state step norms plot.
    state_step_norms_plot = plot(0:k_final-1, results.metrics_history[:state_step_norms], linewidth=LINEWIDTH,
        label="state Step l2 Norm", xlabel="Iteration", ylabel="Step Norm",
        title="$title_common state l2 Step Norm", yaxis=:log10)
    add_vlines!(state_step_norms_plot)
    display(state_step_norms_plot)

    # state step CHAR norms plot.
    state_step_char_norms_plot = plot(0:k_final-1, results.metrics_history[:state_step_char_norms], linewidth=LINEWIDTH,
        label="state Step Char Norm", xlabel="Iteration", ylabel="Step CHAR Norm",
        title="$title_common state CHAR Step Norm", yaxis=:log10)
    add_vlines!(state_step_char_norms_plot)
    display(state_step_char_norms_plot)

    # # Singular values ratio plot.
    # sing_vals_ratio_plot = plot(results.metrics_history[:update_mat_iters], results.metrics_history[:update_mat_singval_ratios], linewidth=LINEWIDTH,
    # label="Prototype Update Matrix", xlabel="Iteration", ylabel="First Two Singular Values' Ratio",
    # title="$title_beginning Update Matrix Singular Value Ratio  $title_end",
    # yaxis=:log10, marker=:circle)
    # add_vlines!(sing_vals_ratio_plot)
    # display(sing_vals_ratio_plot)

    # # Update matrix rank plot.
    # update_ranks_plot = plot(results.metrics_history[:update_mat_iters], results.metrics_history[:update_mat_ranks],
    # label="Prototype Update Matrix", xlabel="Iteration", ylabel="Rank",
    # title="$title_beginning Update Matrix Rank  $title_end",
    # linewidth=LINEWIDTH, xticks=0:100:MAX_ITER)
    # add_vlines!(update_ranks_plot)
    # display(update_ranks_plot)

    # Consecutive update state angles plot (in degrees).
    state_update_angles_deg = rad2deg.(results.metrics_history[:state_update_angles])
    state_update_angles_plot = plot(1:k_final-1, state_update_angles_deg, linewidth=LINEWIDTH,
        label="Prototype Update Angle", xlabel="Iteration", ylabel="Angle between Consecutive Updates (degrees)",
        title="$title_common Consecutive State Update Angles")
    add_vlines!(state_update_angles_plot)
    display(state_update_angles_plot)

    # Projection flags plot (often intensive)
    # enforced_constraints_plot(nn_history, soc_history)

    # plot count of flipped constraints
    proj_diffs_plot = plot_projection_diffs(nn_history, soc_history)
    add_vlines!(proj_diffs_plot)
    display(proj_diffs_plot)

    # plot count of enforced constraints
    enforced_constraints_plot = plot_enforced_constraints_count(nn_history, soc_history)
    add_vlines!(enforced_constraints_plot)
    display(enforced_constraints_plot)

    # plot active set deviation from final iteration
    deviation_plot = plot_active_set_deviation_from_final(nn_history, soc_history)
    add_vlines!(deviation_plot)
    display(deviation_plot)

    # plot unseen deviations from final (constraints that need to flip but never have)
    unseen_plot = plot_unseen_deviations_from_final(nn_history, soc_history)
    add_vlines!(unseen_plot)
    display(unseen_plot)

    # FP Metric Ratio plot.
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

    # SOC normal direction angles plot (only if data available)
    if haskey(results.metrics_history, :soc_normal_angles) &&
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
