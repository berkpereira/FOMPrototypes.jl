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
    _compute_nn_aggregate_signals(preproj_history, K; row_norms, ρ)

Compute per-iteration aggregate statistics (min, 2nd-min, median, 25th/75th
percentiles) of the pre-projection signal across ALL NN constraints.

If `row_norms` and `ρ` are provided, the signal is scaled:
`|u_j| / (1 + ρ ‖a_j‖₂)`. Otherwise it is unscaled: `|u_j|`.

Returns a `NamedTuple` `(min, min2, median, q25, q75)`, each `Vector{Float64}`
of length `total_iters`, or `nothing` if there are no NN constraints.
`min2` is the second-smallest value (equal to `min` when there is only one NN
constraint).
All values are floored at `1e-16`.
"""
function _compute_nn_aggregate_signals(
    preproj_history::Vector{Vector{Float64}},
    K;
    row_norms::Union{Nothing, Vector{Float64}} = nothing,
    ρ::Union{Nothing, Float64} = nothing,
)
    idx_map = nn_to_full_indices(K)
    num_nn = length(idx_map)
    num_nn == 0 && return nothing

    total_iters = length(preproj_history)
    scaled = !isnothing(row_norms) && !isnothing(ρ)

    signals_k = Vector{Float64}(undef, num_nn)
    agg_min    = Vector{Float64}(undef, total_iters)
    agg_min2   = Vector{Float64}(undef, total_iters)
    agg_median = Vector{Float64}(undef, total_iters)
    agg_q25    = Vector{Float64}(undef, total_iters)
    agg_q75    = Vector{Float64}(undef, total_iters)

    for k in 1:total_iters
        u_k = preproj_history[k]
        for (j, full_idx) in enumerate(idx_map)
            val = abs(u_k[full_idx])
            if scaled
                val /= (1.0 + ρ * row_norms[full_idx])
            end
            signals_k[j] = max(val, 1e-16)
        end

        agg_min[k]    = minimum(signals_k)
        agg_min2[k]   = num_nn >= 2 ? partialsort(signals_k, 2) : agg_min[k]
        agg_median[k] = median(signals_k)
        agg_q25[k]    = quantile(signals_k, 0.25)
        agg_q75[k]    = quantile(signals_k, 0.75)
    end

    return (min = agg_min, min2 = agg_min2, median = agg_median,
            q25 = agg_q25, q75 = agg_q75)
end

"""
    _plot_nn_aggregates!(p, iter_axis, agg)

Overlay aggregate statistics (IQR band, median, min, 2nd-min) onto an existing
plot. `agg` is the NamedTuple returned by `_compute_nn_aggregate_signals`.
"""
function _plot_nn_aggregates!(p, iter_axis, agg)
    # IQR shaded band (25th–75th percentile)
    plot!(p, iter_axis, agg.q25;
        seriestype = :line,
        linewidth = 0,
        fillrange = agg.q75,
        fillalpha = 0.15,
        fillcolor = :gray,
        label = "IQR (all NN)",
    )

    # Median line
    plot!(p, iter_axis, agg.median;
        seriestype = :line,
        linewidth = 1.8,
        linestyle = :dashdot,
        color = :gray55,
        alpha = 0.9,
        label = "Median (all NN)",
    )

    # 2nd-smallest line
    plot!(p, iter_axis, agg.min2;
        seriestype = :line,
        linewidth = 2.0,
        linestyle = :dot,
        color = :gray40,
        alpha = 0.9,
        label = "2nd min (all NN)",
    )


    # Minimum line
    plot!(p, iter_axis, agg.min;
        seriestype = :line,
        linewidth = 2.0,
        linestyle = :dash,
        color = :black,
        alpha = 0.9,
        label = "Min (all NN)",
    )

end

"""
    plot_preproj_late_flippers(preproj_history, nn_flags, K; ...) -> Plot

Plot |u_i| over iterations for the last `n_constraints` NN constraints
to settle to the final active set. Log scale on y-axis.

Overlays aggregate statistics (median, minimum, IQR band) computed across
all NN constraints. Set `show_aggregates=false` to suppress the overlay.
"""
function plot_preproj_late_flippers(
    preproj_history::Vector{Vector{Float64}},
    nn_flags::Vector{Vector{Bool}},
    K;
    n_constraints::Int = 10,
    title_prefix::String = "",
    show_aggregates::Bool = true,
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

    if show_aggregates
        agg = _compute_nn_aggregate_signals(preproj_history, K)
        if !isnothing(agg)
            _plot_nn_aggregates!(p, iter_axis, agg)
        end
    end

    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Flip-prediction quality assessment
# ──────────────────────────────────────────────────────────────────────────────

"""
    _nn_flip_sequence(nn_flags) -> Vector{NamedTuple}

Return the ordered list of NN-only flip events.  Each entry is
`(iter = k, flipped = [j, ...])` where `k` is the iteration at which ≥1 NN
constraint changed relative to the previous iteration, and `flipped` is the
vector of nn_mask indices (1-based within NN space) that changed.
"""
function _nn_flip_sequence(nn_flags::Vector{Vector{Bool}})
    total_iters = length(nn_flags)
    events = NamedTuple{(:iter, :flipped), Tuple{Int, Vector{Int}}}[]
    for k in 2:total_iters
        changed = findall(j -> nn_flags[k][j] != nn_flags[k-1][j], eachindex(nn_flags[1]))
        if !isempty(changed)
            push!(events, (iter = k, flipped = changed))
        end
    end
    return events
end

"""
    assess_min_signal_predictor(preproj_history, nn_flags, K; row_norms, ρ, n_history)

For each pair of consecutive NN flip events (prev → next), evaluate the
"minimum |u_i| predictor": at the moment of flip event `e`, which constraint
has the smallest |u_i| among those *not* in the exclusion set?

**Exclusion set**: the union of flipped indices from the last `n_history` flip
events (i.e. event `e` itself plus the `n_history - 1` events before it).
Defaults to `n_history = 5`, so the 5 most-recently-flipped constraints are
excluded before taking the minimum.

Optional scaling: if `row_norms` (length-m vector of A row norms) and `ρ` are
provided, signals are scaled as `|u_i| / (1 + ρ‖a_i‖₂)`.

Returns a `Vector` of `NamedTuple` with fields:
- `prev_flip_iter`  – iteration when the prediction was made (flip event `e`)
- `next_flip_iter`  – iteration of the next flip (what we're predicting)
- `predicted_idx`   – nn_mask index predicted to flip next (lowest signal)
- `actual_nn_idxs`  – nn_mask indices that actually flipped next
- `n_excluded`      – size of the exclusion set at this prediction step
- `n_candidates`    – how many constraints were eligible (total NN minus excluded)
- `rank_of_actual`  – rank of the best actual flipper in ascending signal order
                      (1 = perfect; `n_candidates + 1` if all actual flippers
                      were in the exclusion set)
- `correct`         – true iff `rank_of_actual == 1`
- `predicted_signal`– signal value of the predicted (minimum) constraint
- `actual_signal`   – signal value of the best-ranked actual flipper (NaN if
                      the actual flipper was entirely within the exclusion set)
- `signal_ratio`    – `actual_signal / predicted_signal` (≥ 1 when wrong, = 1
                      when correct; NaN when `actual_signal` is NaN)
"""
function assess_min_signal_predictor(
    preproj_history::Vector{Vector{Float64}},
    nn_flags::Vector{Vector{Bool}},
    K;
    row_norms::Union{Nothing, Vector{Float64}} = nothing,
    ρ::Union{Nothing, Float64} = nothing,
    n_history::Int = 8,
)
    events = _nn_flip_sequence(nn_flags)
    length(events) < 2 && return NamedTuple[]

    idx_map = nn_to_full_indices(K)
    num_nn  = length(idx_map)
    scaled  = !isnothing(row_norms) && !isnothing(ρ)

    function signal_at(k)
        u_k = preproj_history[k]
        Float64[let val = abs(u_k[idx_map[j]])
                    scaled ? max(val / (1.0 + ρ * row_norms[idx_map[j]]), 1e-16) :
                             max(val, 1e-16)
                end for j in 1:num_nn]
    end

    results = NamedTuple[]
    for e in 1:(length(events) - 1)
        prev_event = events[e]
        next_event = events[e + 1]
        k_prev     = prev_event.iter

        signals = signal_at(k_prev)

        # Exclusion set: union of flipped indices from the last n_history events
        window_start = max(1, e - n_history + 1)
        excluded = Set(vcat([events[i].flipped for i in window_start:e]...))

        # Candidates: all NN constraints not in the exclusion set, sorted by signal
        candidates = sort(
            [(nn_idx = j, sig = signals[j]) for j in 1:num_nn if j ∉ excluded],
            by = c -> c.sig,
        )
        isempty(candidates) && continue

        predicted_idx = candidates[1].nn_idx

        # Find the best (lowest) rank among the actual next flippers,
        # ignoring any that were in the exclusion set.
        best_rank       = length(candidates) + 1
        best_actual_sig = NaN
        for j in next_event.flipped
            j ∈ excluded && continue
            r = findfirst(c -> c.nn_idx == j, candidates)
            isnothing(r) && continue
            if r < best_rank
                best_rank       = r
                best_actual_sig = signals[j]
            end
        end

        push!(results, (
            prev_flip_iter   = k_prev,
            next_flip_iter   = next_event.iter,
            predicted_idx    = predicted_idx,
            actual_nn_idxs   = next_event.flipped,
            n_excluded       = length(excluded),
            n_candidates     = length(candidates),
            rank_of_actual   = best_rank,
            correct          = (best_rank == 1),
            predicted_signal = candidates[1].sig,
            actual_signal    = best_actual_sig,
            signal_ratio     = best_actual_sig / candidates[1].sig,
        ))
    end
    return results
end

"""
    plot_flip_prediction_quality(preproj_history, nn_flags, K; ...) -> Plot

Visualise how well the "min-|u_i|" predictor would have forecast each successive
NN constraint flip, using `assess_min_signal_predictor`.

**Predictor definition**: at flip event `e`, exclude the union of flipped
constraints from the last `n_history` events (default 5), then predict that
the constraint with the smallest |u_i| among the remaining ones will flip next.

**Two-panel layout**:
- *Top* – rank of the actual next flipper in the ascending-signal ordering
  (rank 1 = correct, higher = wrong).  Bars coloured green/red.
- *Bottom* – signal ratio = actual_flipper_signal / predicted_min_signal (log
  scale).  Ratio = 1 for correct predictions; ratio > 1 for wrong ones, with
  larger values indicating the predictor was *more* confidently misleading.

Optional scaling via `row_norms` (A row norms) and `ρ`: applies
`|u_i| / (1 + ρ‖a_i‖₂)` before ranking, matching the scaled pre-projection
plots.
"""
function plot_flip_prediction_quality(
    preproj_history::Vector{Vector{Float64}},
    nn_flags::Vector{Vector{Bool}},
    K;
    title_prefix::String = "",
    row_norms::Union{Nothing, Vector{Float64}} = nothing,
    ρ::Union{Nothing, Float64} = nothing,
    n_history::Int = 8,
)
    if isempty(preproj_history) || isempty(nn_flags)
        println("Warning: No pre-projection or flag history available.")
        return nothing
    end

    preds = assess_min_signal_predictor(preproj_history, nn_flags, K;
        row_norms = row_norms, ρ = ρ, n_history = n_history)

    if isempty(preds)
        println("Info: Insufficient NN flip events for prediction quality assessment (need ≥ 2).")
        return nothing
    end

    n_preds   = length(preds)
    x_iters   = [p.prev_flip_iter  for p in preds]
    ranks     = [p.rank_of_actual  for p in preds]
    ratios    = [p.signal_ratio    for p in preds]
    correct   = [p.correct         for p in preds]

    n_correct = count(identity, correct)
    accuracy  = round(100 * n_correct / n_preds; digits = 1)
    scale_str = (!isnothing(row_norms) && !isnothing(ρ)) ? " (scaled)" : ""

    # Split into correct / wrong for separate coloured series
    c_mask = correct
    w_mask = .!correct
    c_x = x_iters[c_mask];  c_r = ranks[c_mask]
    w_x = x_iters[w_mask];  w_r = ranks[w_mask]

    # Bar width: aim for bars that are visible but don't merge
    x_span   = n_preds > 1 ? (maximum(x_iters) - minimum(x_iters)) : 1
    bw       = max(1, x_span / (3 * n_preds))

    # ── Top panel: rank ──────────────────────────────────────────────────────
    p1 = plot(;
        ylabel  = "Rank of actual next flipper",
        title   = "$(title_prefix)Min-|u_i|$(scale_str) flip predictor (excl. last $(n_history)) — " *
                  "$(n_correct)/$(n_preds) correct ($(accuracy)%)",
        legend  = :topright,
        xlims   = (minimum(x_iters) - bw, maximum(x_iters) + bw),
        ylims   = (0, max(maximum(ranks) + 1, 3)),
        minorgrid = true,
    )
    if any(c_mask)
        bar!(p1, c_x, c_r; color = :forestgreen, linecolor = :forestgreen,
             bar_width = bw, label = "Correct (rank 1)", alpha = 0.85)
    end
    if any(w_mask)
        bar!(p1, w_x, w_r; color = :crimson, linecolor = :crimson,
             bar_width = bw, label = "Wrong (rank > 1)", alpha = 0.85)
    end
    hline!(p1, [1.0]; linestyle = :dash, color = :black, linewidth = 1.5,
           label = "Rank 1 (perfect)")

    # ── Bottom panel: signal ratio (log) ────────────────────────────────────
    valid = .!isnan.(ratios)
    c_valid = c_mask .& valid
    w_valid = w_mask .& valid

    p2 = plot(;
        xlabel  = "Solver iteration (when prediction was made)",
        ylabel  = "actual / predicted signal",
        legend  = :topright,
        yaxis   = :log10,
        minorgrid = true,
        xlims   = (minimum(x_iters) - bw, maximum(x_iters) + bw),
    )
    # Connect all valid points with a faint dotted line for continuity
    if any(valid)
        sort_order = sortperm(x_iters[valid])
        plot!(p2, x_iters[valid][sort_order], ratios[valid][sort_order];
              seriestype = :line, linewidth = 0.8, linestyle = :dot,
              color = :gray50, alpha = 0.5, label = "")
    end
    if any(c_valid)
        scatter!(p2, x_iters[c_valid], ratios[c_valid];
                 markersize = 7, color = :forestgreen, markerstrokewidth = 0,
                 label = "Correct")
    end
    if any(w_valid)
        scatter!(p2, x_iters[w_valid], ratios[w_valid];
                 markersize = 7, color = :crimson, markerstrokewidth = 0,
                 label = "Wrong")
    end
    hline!(p2, [1.0]; linestyle = :dash, color = :black, linewidth = 1.5,
           label = "ratio = 1")

    return plot(p1, p2; layout = (2, 1), size = (900, 620), left_margin = 5Plots.mm)
end

"""
    plot_scaled_preproj_late_flippers(preproj_history, nn_flags, K, A, ρ; ...) -> Plot

Plot |ũ_i| = |u_i| / (1 + ρ‖a_i‖₂) over iterations for the last
`n_constraints` NN constraints to settle. Log scale on y-axis.

Overlays aggregate statistics (median, minimum, IQR band) computed across
all NN constraints. Set `show_aggregates=false` to suppress the overlay.

Uses the final ρ value for scaling; this is approximate if ρ changed mid-solve.
"""
function plot_scaled_preproj_late_flippers(
    preproj_history::Vector{Vector{Float64}},
    nn_flags::Vector{Vector{Bool}},
    K,
    A::AbstractMatrix,
    ρ::Float64;
    n_constraints::Int = 10,
    title_prefix::String = "",
    show_aggregates::Bool = true,
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

    if show_aggregates
        agg = _compute_nn_aggregate_signals(preproj_history, K;
            row_norms = row_norms, ρ = ρ)
        if !isnothing(agg)
            _plot_nn_aggregates!(p, iter_axis, agg)
        end
    end

    return p
end
