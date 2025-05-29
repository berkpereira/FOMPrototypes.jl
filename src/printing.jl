using Printf

# Function below prints in the format given

function print_results(
    ws::AbstractWorkspace,
    print_modulo::Int;
    curr_xy_dist::Union{Float64, Nothing} = nothing,
    obj_sol::Union{Float64, Nothing} = nothing,
    x_sol::Union{Vector{Float64}, Nothing} = nothing,
    y_sol::Union{Vector{Float64}, Nothing} = nothing,
    x::Union{Vector{Float64}, Nothing} = nothing,
    y::Union{Vector{Float64}, Nothing} = nothing,
    relative::Bool = false,
    terminated::Bool = false)

    # if not at print_modulo nor termination, do nothing
    if !terminated && ws.k[] % print_modulo != 0
        return
    end

    if terminated
        println("--------------------")
        println("     TERMINATED     ")
        println("--------------------")
    end

    # Start with iteration, objective, and primal/dual residuals
    if ws isa NoneWorkspace # no acceleration
        print_output = @sprintf("k %4.1d | obj: %12.5e", ws.k[], ws.res.obj_primal)
    elseif ws isa KrylovWorkspace # NB: k effective excludes count of unsuccessful acceleration attempts
        print_output = @sprintf("k %4.1d | k eff %4.1d | obj: %12.5e", ws.k[], ws.k_eff[], ws.res.obj_primal)
    elseif ws isa AndersonWorkspace # NB k van(illa) counts only vanilla iterations, as the COSMO papers do
        print_output = @sprintf("k %4.1d | k eff %4.1d | k van %4.1d | obj: %12.5e", ws.k[], ws.k_eff[], ws.k_vanilla[], ws.res.obj_primal)
    end

    # objective error if obj_sol is provided
    if obj_sol !== nothing
        print_output *= @sprintf(" (err: %12.5e)", ws.res.obj_primal - obj_sol)
    end

    # x error if both x and x_sol are provided
    if x !== nothing && x_sol !== nothing
        print_output *= @sprintf(" | x l2 err: %12.5e", norm(x - x_sol))
    end

    # y error if both y and y_sol are provided
    if y !== nothing && y_sol !== nothing
        print_output *= @sprintf(" | y l2 err: %12.5e", norm(y - (-y_sol)))
    end

    # primal and dual residual, gap
    if relative
        print_output *= @sprintf(" | rp rel: %12.5e", ws.res.rp_rel)
        print_output *= @sprintf(" | rd rel: %12.5e", ws.res.rd_rel)
        print_output *= @sprintf(" | gap rel: %12.5e", ws.res.gap_rel)
    else
        print_output *= @sprintf(" | rp abs: %12.5e", ws.res.rp_abs)
        print_output *= @sprintf(" | rd abs: %12.5e", ws.res.rd_abs)
        print_output *= @sprintf(" | gap abs: %12.5e", ws.res.gap_abs)
    end

    # (x, y) distance to solution
    if !isnothing(curr_xy_dist) && curr_xy_dist !== NaN
        print_output *= @sprintf(" | (x, y) dist: %12.5e", curr_xy_dist)
    end

    # Print the final output
    println(print_output)

    if terminated
        println()
    end
end
