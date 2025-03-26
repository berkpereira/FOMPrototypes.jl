using Plots
plotlyjs()

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
        title="UNenforced Constraints"
    ))
end
