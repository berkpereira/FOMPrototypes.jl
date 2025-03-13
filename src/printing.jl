using Printf

# Function below prints in the format given

function print_results(
    k::Int,
    curr_obj::Float64,
    primal_res_norm::Float64,
    dual_res_norm::Float64,
    curr_duality_gap::Float64;
    curr_xy_dist::Union{Float64, Nothing} = nothing,
    obj_sol::Union{Float64, Nothing} = nothing,
    x_sol::Union{Vector{Float64}, Nothing} = nothing,
    y_sol::Union{Vector{Float64}, Nothing} = nothing,
    x::Union{Vector{Float64}, Nothing} = nothing,
    y::Union{Vector{Float64}, Nothing} = nothing,
    terminated::Bool = false
)
    if terminated
        println("--------------------")
        println("     TERMINATED     ")
        println("--------------------")
    end

    # Start with iteration, objective, and primal/dual residuals
    print_output = @sprintf("Iter %4.1d | obj: %12.5e", k, curr_obj)

    # Include objective error if obj_sol is provided
    if obj_sol !== nothing
        print_output *= @sprintf(" (err: %12.5e)", curr_obj - obj_sol)
    end

    # Include x error if both x and x_sol are provided
    if x !== nothing && x_sol !== nothing
        print_output *= @sprintf(" | x err: %12.5e", norm(x - x_sol))
    end

    # Include y error if both y and y_sol are provided
    if y !== nothing && y_sol !== nothing
        print_output *= @sprintf(" | y err: %12.5e", norm(y - (-y_sol)))
    end

    # Include primal residual
    print_output *= @sprintf(" | pri res: %12.5e", primal_res_norm)

    # Include dual residual
    print_output *= @sprintf(" | dua res: %12.5e", dual_res_norm)

    # Include duality gap
    print_output *= @sprintf(" | gap: %12.5e", curr_duality_gap)

    # (x, y) distance to solution
    if !isnothing(curr_xy_dist)
        print_output *= @sprintf(" | (x, y) dist: %12.5e", curr_xy_dist)
    end
        

    # Print the final output
    println(print_output)

    if terminated
        println()
    end
end
