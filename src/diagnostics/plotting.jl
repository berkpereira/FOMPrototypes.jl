using Plots

function enforced_constraints_plot(
    nn_flags::Vector{Vector{Bool}},
    soc_states::Vector{Vector{SOCAction}},
    iter_gap::Int = 10,
    )
    total_iters = max(length(nn_flags), length(soc_states))
    if total_iters == 0
        println("Warning: No projection history available to plot.")
        return
    end

    outer_indices = union(collect(1:iter_gap:total_iters), [1, total_iters]) |> sort
    nn_dim = _first_vector_length(nn_flags)
    soc_dim = _first_vector_length(soc_states)

    p = plot(
        xlabel="Solver Iteration",
        ylabel="Constraint index",
        title="Constraint Activity (NN + SOC)",
        legend=:topright,
    )

    if nn_dim > 0
        x_qp = Int[]
        y_qp = Int[]
        for i in outer_indices
            if i > length(nn_flags)
                continue
            end
            vec = nn_flags[i]
            for (j, val) in enumerate(vec)
                if !val
                    push!(x_qp, i)
                    push!(y_qp, j)
                end
            end
        end

        if !isempty(x_qp)
            scatter!(
                p,
                x_qp,
                y_qp;
                markersize=2.5,
                marker=:circle,
                mc=:limegreen,
                lc=:limegreen,
                lw=0,
                label="Inactive NN",
            )
        end
    end

    if soc_dim > 0
        offset = nn_dim
        color_map = Dict(
            soc_zero => :firebrick,
            soc_identity => :dodgerblue,
            soc_interesting => :darkorange,
        )
        marker_map = Dict(
            soc_zero => :utriangle,
            soc_identity => :square,
            soc_interesting => :diamond,
        )
        label_map = Dict(
            soc_zero => "SOC interior",
            soc_identity => "SOC identity",
            soc_interesting => "SOC boundary",
        )

        for action in (soc_zero, soc_identity, soc_interesting)
            xs = Int[]
            ys = Int[]
            for i in outer_indices
                if i > length(soc_states)
                    continue
                end
                vec = soc_states[i]
                for (j, state) in enumerate(vec)
                    if state == action
                        push!(xs, i)
                        push!(ys, offset + j)
                    end
                end
            end

            if !isempty(xs)
                scatter!(
                    p,
                    xs,
                    ys;
                    markersize=3.0,
                    marker=marker_map[action],
                    mc=color_map[action],
                    lc=color_map[action],
                    lw=0,
                    label=label_map[action],
                )
            end
        end
    end

    display(p)
    return p
end

# another function, similar to plot_projection_diffs, but instead of plotting
# count of differences, it simply plots the counts
# of enforced constraints per iteration.
function plot_enforced_constraints_count(
    nn_flags::Vector{Vector{Bool}},
    soc_states::Vector{Vector{SOCAction}},
    )
    total_iters = max(length(nn_flags), length(soc_states))
    if total_iters == 0
        println("Warning: No projection history available to plot.")
        return
    end

    nn_counts = _nn_counts(nn_flags, total_iters)
    soc_zero_counts = _soc_counts(soc_states, total_iters, soc_zero)
    soc_id_counts = _soc_counts(soc_states, total_iters, soc_identity)
    soc_interesting_counts = _soc_counts(soc_states, total_iters, soc_interesting)
    iter_axis = 1:total_iters

    p = plot(
        iter_axis,
        nn_counts;
        seriestype=:line,
        xlabel="Solver Iteration",
        ylabel="Count",
        title="Constraint Status per Iteration",
        legend=:topright,
        lw=2,
        marker=:circle,
        markersize=3,
        label="NN enforced",
        color=:seagreen,
    )

    nn_dim = _first_vector_length(nn_flags)
    if nn_dim > 0
        hline!(p, [nn_dim]; color=:purple, linestyle=:dash, label="NN total")
    end

    plot!(
        p,
        iter_axis,
        soc_zero_counts;
        seriestype=:line,
        lw=2,
        marker=:cross,
        markersize=3,
        label="SOC interior",
        color=:firebrick,
    )
    plot!(
        p,
        iter_axis,
        soc_id_counts;
        seriestype=:line,
        lw=2,
        marker=:hexagon,
        markersize=3,
        label="SOC identity",
        color=:dodgerblue,
    )
    plot!(
        p,
        iter_axis,
        soc_interesting_counts;
        seriestype=:line,
        lw=2,
        marker=:diamond,
        markersize=3,
        label="SOC boundary",
        color=:darkorange,
    )

    return p
end

function plot_projection_diffs(
    nn_flags::Vector{Vector{Bool}},
    soc_states::Vector{Vector{SOCAction}},
    )
    total_iters = max(length(nn_flags), length(soc_states))
    if total_iters < 2
        println("Warning: Need at least two iteration vectors to compute differences. Nothing to plot.")
        return
    end

    iter_axis = 2:total_iters
    nn_diffs = _diff_counts(nn_flags, total_iters)
    soc_diffs = _diff_counts(soc_states, total_iters)

    p = plot(
        iter_axis,
        nn_diffs;
        seriestype=:line,
        xlabel="Solver Iteration",
        ylabel="Number of Changes",
        title="Constraint Changes per Iteration",
        legend=:topright,
        lw=2,
        marker=:circle,
        markersize=3,
        label="NN changes",
        color=:seagreen,
    )
    plot!(
        p,
        iter_axis,
        soc_diffs;
        seriestype=:line,
        lw=2,
        marker=:diamond,
        markersize=3,
        label="SOC changes",
        color=:darkorange,
    )

    return p
end

function _first_vector_length(history)
    for vec in history
        return length(vec)
    end
    return 0
end

function _nn_counts(history, total_iters)
    counts = fill(NaN, total_iters)
    for i in 1:total_iters
        if i <= length(history)
            counts[i] = sum(history[i])
        end
    end
    return counts
end

function _soc_counts(history, total_iters, target::SOCAction)
    counts = fill(NaN, total_iters)
    for i in 1:total_iters
        if i <= length(history)
            counts[i] = count(==(target), history[i])
        end
    end
    return counts
end

function _diff_counts(history, total_iters)
    if total_iters < 2
        return Float64[]
    end
    counts = fill(0.0, total_iters - 1)
    for i in 2:total_iters
        if i <= length(history) && i - 1 <= length(history)
            counts[i - 1] = sum(history[i - 1] .!= history[i])
        end
    end
    return counts
end

"""
Plots angular changes in SOC projection normal directions over iterations.

Creates a line plot with one series per SOC cone, showing how the normal
direction changes (in radians) between consecutive iterations when the SOC
is in the interesting state.

# Arguments
- `soc_normal_angles::Vector{Vector{Float64}}`: Outer vector = iterations,
  inner vector = angles for each SOC at that iteration
- `title_prefix::String`: Prefix for plot title (default: "")

# Returns
- Plot object or nothing if no data available

# Notes
- Handles cases with zero SOCs gracefully (returns nothing)
- NaN values are automatically handled by Plots.jl (gaps in lines)
- Only shows SOCs that have at least one non-NaN value
- Uses log scale on y-axis to show small angle changes
"""
function plot_soc_normal_angles(
    soc_normal_angles::Vector{Vector{Float64}},
    title_prefix::String = "",
    )

    # Handle empty input
    if isempty(soc_normal_angles)
        println("Warning: No SOC normal angle history available to plot.")
        return nothing
    end

    total_iters = length(soc_normal_angles)
    num_socs = isempty(soc_normal_angles) ? 0 : length(soc_normal_angles[1])

    # Handle zero SOCs case
    if num_socs == 0
        println("Info: No SOC cones in problem, skipping normal angles plot.")
        return nothing
    end

    iter_axis = 1:total_iters

    # Create base plot
    p = plot(
        xlabel="Solver Iteration",
        ylabel="Angular Change (radians)",
        title="$(title_prefix)SOC Normal Direction Angular Changes",
        legend=:outerright,
        ylims=(-π, π),
        minorgrid=true,
    )

    # Add horizontal reference line at π radians
    hline!(p, [π]; color=:gray, linestyle=:dash,
           linewidth=1, label="π rad (reversal)", alpha=0.5)

    # Define color palette for SOCs
    colors = [:dodgerblue, :crimson, :forestgreen, :darkorange, :purple,
              :deeppink, :teal, :gold, :brown, :navy]

    # Plot one series per SOC
    for soc_idx in 1:num_socs
        # Extract angle history for this SOC
        angles_for_soc = [soc_normal_angles[iter][soc_idx]
                          for iter in 1:total_iters]

        # Only plot if there's at least one non-NaN value
        if any(!isnan, angles_for_soc)
            color = colors[(soc_idx - 1) % length(colors) + 1]
            plot!(
                p,
                iter_axis,
                angles_for_soc;
                seriestype=:line,
                linewidth=2,
                marker=:circle,
                markersize=2,
                label="SOC $soc_idx",
                color=color,
                alpha=0.8,
            )
        end
    end

    return p
end
