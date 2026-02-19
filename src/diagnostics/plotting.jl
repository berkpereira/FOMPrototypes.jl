using Plots

function enforced_constraints_plot(
    nn_flags::Union{Vector{Vector{Bool}}, Matrix{Bool}},
    soc_states::Union{Vector{Vector{SOCAction}}, Matrix{SOCAction}},
    iter_gap::Int = 10,
    )
    total_iters = max(_total_iters(nn_flags), _total_iters(soc_states))
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
            if i > _total_iters(nn_flags)
                continue
            end
            vec = _get_at_iter(nn_flags, i)
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
                if i > _total_iters(soc_states)
                    continue
                end
                vec = _get_at_iter(soc_states, i)
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
    nn_flags::Union{Vector{Vector{Bool}}, Matrix{Bool}},
    soc_states::Union{Vector{Vector{SOCAction}}, Matrix{SOCAction}},
    )
    total_iters = max(_total_iters(nn_flags), _total_iters(soc_states))
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
        legend=:topleft,
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
    nn_flags::Union{Vector{Vector{Bool}}, Matrix{Bool}},
    soc_states::Union{Vector{Vector{SOCAction}}, Matrix{SOCAction}},
    )
    total_iters = max(_total_iters(nn_flags), _total_iters(soc_states))
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

function plot_active_set_deviation_from_final(
    nn_flags::Union{Vector{Vector{Bool}}, Matrix{Bool}},
    soc_states::Union{Vector{Vector{SOCAction}}, Matrix{SOCAction}},
    )
    total_iters = max(_total_iters(nn_flags), _total_iters(soc_states))
    if total_iters < 1
        println("Warning: No projection history available to plot.")
        return
    end

    iter_axis = 1:total_iters
    nn_diffs = _diff_counts_to_final(nn_flags, total_iters)
    soc_diffs = _diff_counts_to_final(soc_states, total_iters)

    p = plot(
        iter_axis,
        nn_diffs;
        seriestype=:line,
        xlabel="Solver Iteration",
        ylabel="Number of Differences",
        title="Active Set Deviation from Final",
        legend=:topright,
        lw=2,
        marker=:circle,
        markersize=3,
        label="NN deviation",
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
        label="SOC deviation",
        color=:darkorange,
    )

    return p
end

function plot_unseen_deviations_from_final(
    nn_flags::Union{Vector{Vector{Bool}}, Matrix{Bool}},
    soc_states::Union{Vector{Vector{SOCAction}}, Matrix{SOCAction}},
    )
    total_iters = max(_total_iters(nn_flags), _total_iters(soc_states))
    if total_iters < 1
        println("Warning: No projection history available to plot.")
        return
    end

    iter_axis = 1:total_iters
    nn_unseen = _unseen_deviation_counts(nn_flags, total_iters)
    soc_unseen = _unseen_deviation_counts(soc_states, total_iters)

    p = plot(
        iter_axis,
        nn_unseen;
        seriestype=:line,
        xlabel="Solver Iteration",
        ylabel="Count",
        title="Unseen Deviations from Final Active Set",
        legend=:topright,
        lw=2,
        marker=:circle,
        markersize=3,
        label="NN unseen",
        color=:seagreen,
    )
    plot!(
        p,
        iter_axis,
        soc_unseen;
        seriestype=:line,
        lw=2,
        marker=:diamond,
        markersize=3,
        label="SOC unseen",
        color=:darkorange,
    )

    return p
end

function _first_vector_length(history::Matrix)
    if isempty(history)
        return 0
    end
    return size(history, 1)  # Return number of rows (number of entities)
end

function _first_vector_length(history::Vector{Vector{T}}) where T
    for vec in history
        return length(vec)
    end
    return 0
end

# Helper to get total iterations from either format
_total_iters(history::Matrix) = size(history, 2)
_total_iters(history::Vector) = length(history)

# Helper to get data at iteration i
_get_at_iter(history::Matrix, i::Int) = history[:, i]
_get_at_iter(history::Vector{Vector{T}}, i::Int) where T = history[i]

function _nn_counts(history::Matrix{Bool}, total_iters)
    counts = fill(NaN, total_iters)
    _, num_cols = size(history)
    for i in 1:total_iters
        if i <= num_cols
            counts[i] = sum(history[:, i])
        end
    end
    return counts
end

function _nn_counts(history::Vector{Vector{Bool}}, total_iters)
    counts = fill(NaN, total_iters)
    for i in 1:total_iters
        if i <= length(history)
            counts[i] = sum(history[i])
        end
    end
    return counts
end

function _soc_counts(history::Matrix{SOCAction}, total_iters, target::SOCAction)
    counts = fill(NaN, total_iters)
    _, num_cols = size(history)
    for i in 1:total_iters
        if i <= num_cols
            counts[i] = count(==(target), history[:, i])
        end
    end
    return counts
end

function _soc_counts(history::Vector{Vector{SOCAction}}, total_iters, target::SOCAction)
    counts = fill(NaN, total_iters)
    for i in 1:total_iters
        if i <= length(history)
            counts[i] = count(==(target), history[i])
        end
    end
    return counts
end

function _diff_counts(history::Matrix, total_iters)
    if total_iters < 2
        return Float64[]
    end
    counts = fill(0.0, total_iters - 1)
    _, num_cols = size(history)
    for i in 2:total_iters
        if i <= num_cols && i - 1 <= num_cols
            counts[i - 1] = sum(history[:, i - 1] .!= history[:, i])
        end
    end
    return counts
end

function _diff_counts(history::Vector{Vector{T}}, total_iters) where T
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

function _diff_counts_to_final(history::Matrix, total_iters)
    if total_iters < 1
        return Float64[]
    end
    _, num_cols = size(history)
    if num_cols == 0
        return fill(0.0, total_iters)
    end
    final_col = history[:, num_cols]
    counts = fill(0.0, total_iters)
    for i in 1:total_iters
        if i <= num_cols
            counts[i] = sum(history[:, i] .!= final_col)
        end
    end
    return counts
end

function _diff_counts_to_final(history::Vector{Vector{T}}, total_iters) where T
    if total_iters < 1 || isempty(history)
        return fill(0.0, total_iters)
    end
    final_vec = history[end]
    counts = fill(0.0, total_iters)
    for i in 1:total_iters
        if i <= length(history)
            counts[i] = sum(history[i] .!= final_vec)
        end
    end
    return counts
end

function _unseen_deviation_counts(history::Matrix, total_iters)
    if total_iters < 1
        return Float64[]
    end
    num_constraints, num_cols = size(history)
    if num_cols == 0 || num_constraints == 0
        return fill(0.0, total_iters)
    end
    final_col = history[:, num_cols]
    has_ever_flipped = falses(num_constraints)
    counts = fill(0.0, total_iters)
    for i in 1:total_iters
        if i <= num_cols
            if i > 1
                has_ever_flipped .|= (history[:, i] .!= history[:, i - 1])
            end
            differs_from_final = history[:, i] .!= final_col
            counts[i] = sum(differs_from_final .& .!has_ever_flipped)
        end
    end
    return counts
end

function _unseen_deviation_counts(history::Vector{Vector{T}}, total_iters) where T
    if total_iters < 1 || isempty(history)
        return fill(0.0, total_iters)
    end
    num_constraints = length(history[1])
    if num_constraints == 0
        return fill(0.0, total_iters)
    end
    final_vec = history[end]
    has_ever_flipped = falses(num_constraints)
    counts = fill(0.0, total_iters)
    for i in 1:total_iters
        if i <= length(history)
            if i > 1
                has_ever_flipped .|= (history[i] .!= history[i - 1])
            end
            differs_from_final = history[i] .!= final_vec
            counts[i] = sum(differs_from_final .& .!has_ever_flipped)
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
- `soc_normal_angles::Matrix{Float64}`: Matrix of size (num_socs × num_iterations)
  where each row corresponds to one SOC and each column to one iteration
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
    soc_normal_angles::Matrix{Float64},
    title_prefix::String = "",
    )

    # Handle empty input
    if isempty(soc_normal_angles)
        println("Warning: No SOC normal angle history available to plot.")
        return nothing
    end

    num_socs, total_iters = size(soc_normal_angles)

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
        yaxis=:log10,
        ylims=(1e-9, π),
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
        # Extract angle history for this SOC (row of the matrix)
        angles_for_soc = soc_normal_angles[soc_idx, :]

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

"""
    nn_to_full_indices(K) -> Vector{Int}

Map nn_mask index space to full m-dimensional preproj_vec index space.
Returns a vector where entry j gives the position in the full m-vector
corresponding to nn_mask[j].
"""
function nn_to_full_indices(K)
    indices = Int[]
    full_idx = 1
    for cone in K
        if cone isa Clarabel.NonnegativeConeT
            append!(indices, full_idx:(full_idx + cone.dim - 1))
        end
        full_idx += cone.dim
    end
    return indices
end

"""
    find_late_flipping_nn_constraints(nn_flags, n_constraints=10) -> Vector{Int}

Identify the NN constraint indices (in nn_mask index space) that were the
last to settle to the final active set.

For each constraint, finds the last iteration where it disagreed with the
final value. Returns up to `n_constraints` indices sorted by settling
iteration (latest-settling first). Constraints that never disagreed are excluded.
"""
function find_late_flipping_nn_constraints(
    nn_flags::Vector{Vector{Bool}},
    n_constraints::Int = 5,
    )

    if isempty(nn_flags)
        return Int[]
    end

    num_constraints = length(nn_flags[1])
    total_iters = length(nn_flags)
    final_flags = nn_flags[end]

    # For each constraint, find the last iteration where it differed from final.
    settling_iters = zeros(Int, num_constraints)

    for j in 1:num_constraints
        for i in total_iters:-1:1
            if nn_flags[i][j] != final_flags[j]
                settling_iters[j] = i + 1
                break
            end
        end
        # settling_iters[j] == 0 means it never disagreed with the final value.
    end

    # Sort by settling iteration (descending), only include those that flipped
    flipped_indices = findall(s -> s > 0, settling_iters)
    sort!(flipped_indices, by = j -> settling_iters[j], rev = true)

    return flipped_indices[1:min(n_constraints, length(flipped_indices))]
end

"""
    plot_preproj_late_flippers(preproj_history, nn_flags, K; ...) -> Plot

Plot |u_i| over iterations for the last `n_constraints` NN constraints
to settle to the final active set. Log scale on y-axis.
"""
function plot_preproj_late_flippers(
    preproj_history::Vector{Vector{Float64}},
    nn_flags::Vector{Vector{Bool}},
    K;
    n_constraints::Int = 5,
    title_prefix::String = "",
    )

    if isempty(preproj_history) || isempty(nn_flags)
        println("Warning: No pre-projection or flag history available.")
        return nothing
    end

    late_flippers = find_late_flipping_nn_constraints(nn_flags, n_constraints)

    if isempty(late_flippers)
        println("Info: No NN constraints flipped during the solve; skipping pre-projection plot.")
        return nothing
    end

    idx_map = nn_to_full_indices(K)

    total_iters = length(preproj_history)
    iter_axis = 1:total_iters

    p = plot(
        xlabel = "Solver Iteration",
        ylabel = "|u_i| (pre-projection magnitude)",
        title = "$(title_prefix)|u_i| for Late-Settling NN Constraints",
        legend = :outerright,
        yaxis = :log10,
        minorgrid = true,
    )

    colors = [:dodgerblue, :crimson, :forestgreen, :darkorange, :purple,
              :deeppink, :teal, :gold, :brown, :navy]

    for (plot_idx, nn_idx) in enumerate(late_flippers)
        full_idx = idx_map[nn_idx]

        abs_u = [max(abs(preproj_history[k][full_idx]), 1e-16) for k in 1:total_iters]

        color = colors[(plot_idx - 1) % length(colors) + 1]
        plot!(p, iter_axis, abs_u;
            seriestype = :line,
            linewidth = 1.5,
            label = "NN #$(nn_idx)",
            color = color,
            alpha = 0.85,
        )
    end

    return p
end

"""
    plot_scaled_preproj_late_flippers(preproj_history, nn_flags, K, A, ρ; ...) -> Plot

Plot |ũ_i| = |u_i| / (1 + ρ‖a_i‖₂) over iterations for the last
`n_constraints` NN constraints to settle. Log scale on y-axis.

Uses the final ρ value for scaling; this is approximate if ρ changed mid-solve.
"""
function plot_scaled_preproj_late_flippers(
    preproj_history::Vector{Vector{Float64}},
    nn_flags::Vector{Vector{Bool}},
    K,
    A::AbstractMatrix,
    ρ::Float64;
    n_constraints::Int = 5,
    title_prefix::String = "",
    )

    if isempty(preproj_history) || isempty(nn_flags)
        println("Warning: No pre-projection or flag history available.")
        return nothing
    end

    late_flippers = find_late_flipping_nn_constraints(nn_flags, n_constraints)

    if isempty(late_flippers)
        println("Info: No NN constraints flipped during the solve; skipping scaled pre-projection plot.")
        return nothing
    end

    idx_map = nn_to_full_indices(K)

    # Compute row norms of A once
    m = size(A, 1)
    row_norms = [norm(A[i, :]) for i in 1:m]

    total_iters = length(preproj_history)
    iter_axis = 1:total_iters

    p = plot(
        xlabel = "Solver Iteration",
        ylabel = "|ũ_i| (scaled pre-projection)",
        title = "$(title_prefix)Scaled |ũ_i| for Late-Settling NN Constraints",
        legend = :outerright,
        yaxis = :log10,
        minorgrid = true,
    )

    colors = [:dodgerblue, :crimson, :forestgreen, :darkorange, :purple,
              :deeppink, :teal, :gold, :brown, :navy]

    for (plot_idx, nn_idx) in enumerate(late_flippers)
        full_idx = idx_map[nn_idx]
        scale_factor = 1.0 + ρ * row_norms[full_idx]

        abs_u_scaled = [max(abs(preproj_history[k][full_idx]) / scale_factor, 1e-16)
                        for k in 1:total_iters]

        color = colors[(plot_idx - 1) % length(colors) + 1]
        plot!(p, iter_axis, abs_u_scaled;
            seriestype = :line,
            linewidth = 1.5,
            label = "NN #$(nn_idx)",
            color = color,
            alpha = 0.85,
        )
    end

    return p
end
