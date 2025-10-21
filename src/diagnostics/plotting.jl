using Plots

function enforced_constraints_plot(enforced_constraint_flags::Vector{Vector{Bool}}, iter_gap::Int = 10)
    # Prepare x, y, and color data
    x = Int[]   # Horizontal indices
    y = Int[]   # Vertical indices
    color = String[]  # Colors for markers

    outer_indices = union(1:iter_gap:length(enforced_constraint_flags), [1, length(enforced_constraint_flags)]) |> sort

    for i in outer_indices
        vec = enforced_constraint_flags[i]
        for (j, val) in enumerate(vec)
            if !val # If constraint not enforced
                push!(x, i)           # Horizontal index corresponds to the outer vector index
                push!(y, j)           # Vertical index corresponds to the inner vector index
                push!(color, "#07f72f")  # Green for false
            end
        end
    end

    # Create the plot
    display(scatter(x, y, 
        markersize=2.5, 
        mc=color,
        lc=color,
        lw=0., 
        xlabel="Solver Iteration", 
        ylabel="Constraint index", 
        legend=false,
        title="UNEnforced Constraints"
    ))
end


# another function, similar to plot_projection_diffs, but instead of plotting
# count of differences, it simply plots the counts
# of enforced constraints per iteration.
function plot_enforced_constraints_count(enforced_constraint_flags::Vector{Vector{Bool}}, no_constraints::Int)
    # Count the number of enforced constraints for each iteration.
    counts = Int[]
    for vec in enforced_constraint_flags
        push!(counts, sum(vec))
    end

    # Create and display the bar plot.
    # Create and display the line plot.
    p = plot(counts,
        seriestype=:line,
        xlabel="Solver Iteration",
        ylabel="Number of Enforced Constraints",
        title="Enforced Constraints Count per Iteration",
        legend=false,
        lw=2, # Set line width for better visibility
        marker=:circle, # Add markers to each data point
        markersize=3
    )

    # add horizontal line at number of constraints in total
    hline!([no_constraints], color=:purple, linestyle=:solid, lw=3)

    return p
end

function plot_projection_diffs(enforced_constraint_flags::Vector{Vector{Bool}})
    # Ensure there are at least two vectors to enable comparison.
    if length(enforced_constraint_flags) < 2
        println("Warning: Need at least two iteration vectors to compute differences. Nothing to plot.")
        return
    end

    # Calculate the number of differing flags between each consecutive iteration.
    # This is equivalent to the Hamming distance between the boolean vectors.
    diff_counts = Int[]
    for i in 2:length(enforced_constraint_flags)
        # The .!= operator performs an element-wise "not equal" comparison,
        # resulting in a boolean vector. The sum() of a boolean vector in Julia
        # efficiently counts the number of 'true' values.
        diff = sum(enforced_constraint_flags[i-1] .!= enforced_constraint_flags[i])
        push!(diff_counts, diff)
    end

    # The x-axis represents the iteration number where the change from the previous one is recorded.
    iterations = 1:length(enforced_constraint_flags)-1

    # Create and display the line plot.
    p = plot(iterations, diff_counts,
        seriestype=:line,
        xlabel="Solver Iteration",
        ylabel="Number of Flipped Constraints",
        title="Change in Enforced Constraints per Iteration",
        legend=false,
        lw=2, # Set line width for better visibility
        marker=:circle, # Add markers to each data point
        markersize=3
    )

    return p
end