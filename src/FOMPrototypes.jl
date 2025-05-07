module FOMPrototypes

export main, run_cli, run_prototype, solve_reference, fetch_data

# Import packages.
using ArgParse
using Infiltrator
using LinearMaps
using Infiltrator
using Profile
using BenchmarkTools
using Plots
using SparseArrays
using SCS
using Random

# Include all project source files.
include(joinpath(@__DIR__, "custom_nla.jl"))
include(joinpath(@__DIR__, "types.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "residuals.jl"))
include(joinpath(@__DIR__, "krylov_acceleration.jl"))
include(joinpath(@__DIR__, "linesearch.jl"))
include(joinpath(@__DIR__, "printing.jl"))
include(joinpath(@__DIR__, "solver.jl"))
include(joinpath(@__DIR__, "problem_data.jl"))

########################
# Initialization Block #
########################
    
function initialise_misc()
    # Set Plots backend.
    # For interactive plots: plotlyjs()
    # For faster plotting: gr()
    gr()

    # Determine newline character based on backend.
    local newline_char = Plots.backend_name() in [:gr, :pythonplot] ? "\n" : "<br>"

    # Set default plot size (in pixels)
    default(size=(1100, 700)) # for desktop
    # default(size=(1200, 800)) # for laptop

    return newline_char
end

##################################
# Parsing command line arguments #
##################################

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--ref-solver"
        help = "Reference solver to use: SCS or Clarabel."
        arg_type = Symbol
        required = true

        "--variant", "-v"
        help = "Variant to use: ADMM, PDHG, 1, 2, 3, or 4."
        arg_type = Symbol
        required = true

        "--problem-set"
        help = "Problem identifier to run"
        arg_type = String
        required = true

        "--problem-name"
        help = "Name of the problem to run"
        arg_type = String
        required = true

        "--res-norm"
        help = "Residual p-norm to use for various solver purposes."
        arg_type = Float64
        default = Inf

        "--max-iter"
        help = "Maximum number of solver iterations"
        arg_type = Int
        default = 1000

        "--print-mod"
        help = "How many iterations between printing info."
        arg_type = Int
        default = 50

        "--rho"
        help = "PrePDHG ρ step size"
        arg_type = Float64
        default = 1.0

        "--theta"
        help = "PrePDHG θ parameter"
        arg_type = Float64
        default = 1.0

        "--acceleration", "-a"
        help = "Acceleration type: none, anderson, or krylov."
        arg_type = Symbol
        default = :none

        "--accel-memory"
        help = "Memory size for acceleration methods."
        arg_type = Int
        default = 20

        "--krylov-operator"
        help = "Krylov operator type: tilde_A or B."
        arg_type = Symbol
        default = :tilde_A

        "--anderson-period"
        help = "Period for Anderson acceleration."
        arg_type = Int
        default = 10

        "--anderson-broyden-type"
        help = "Which type of Broyden update to use: 1, normal2, or QR2."
        arg_type = Symbol
        default = :normal2

        "--anderson-mem-type"
        help = "Memory type for Anderson acceleration: rolling or restarted."
        arg_type = Symbol
        default = :rolling

        "--anderson-reg"
        help = "Regulariser for Anderson least-squares problem: none, Tikonov, or Frobenius."
        arg_type = Symbol
        default = :none

        "--run-fast"
        help = "Run fast mode (no plotting, less data recording during run)."
        arg_type = Bool
        default = false

        "--residuals-relative"
        help = "Use relative metrics when printing iter info."
        arg_type = Bool
        default = true

        "--show-vlines"
        help = "Show relevant vertical dashed lines in plots."
        arg_type = Bool
        default = false

        ### ignoring these at the moment... ###

        "--restart-period"
        help = "Restart period for the solver."
        arg_type = Real
        default = Inf

        "--linesearch-period"
        help = "Period for performing line search."
        arg_type = Real
        default = Inf

        "--linesearch-eps"
        help = "Epsilon parameter for line search."
        arg_type = Float64
        default = 0.001
    end

    return parse_args(s)
end

##################################
# Problem Selection & Data Fetch #
##################################

function choose_problem(problem_option::Symbol)
    # Choose problem option. Valid options: :LASSO, :HUBER, :MAROS, :GISELSSON

    if problem_option === :LASSO
        problem_set = "sslsq"
        problem_name = "NYPA_Maragal_5_lasso"; # large, challenging
        # problem_name = "HB_abb313_lasso"  # (m, n) = (665, 665)
        # problem_name = "HB_ash219_lasso" # (m, n) = (389, 389)
    elseif problem_option === :HUBER
        problem_set = "sslsq"
        problem_name = "HB_ash958_huber"  # (m, n) = (3419, 3099)
    elseif problem_option === :MAROS
        problem_set = "maros"
        # problem_name = "DUAL3"; # large
        problem_name = "QSCSD8"   # not as large, (m, n) = (3147, 2750)
        # Other MAROS options commented out...
    elseif problem_option === :GISELSSON
        problem_set = "giselsson"
        problem_name = "giselsson_problem"
    else
        error("Invalid problem option")
    end

    return problem_option, problem_set, problem_name
end

function fetch_data(args)
    problem_set = args["problem-set"]
    problem_name = args["problem-name"]
    
    if problem_name != "giselsson"
        data = load_clarabel_benchmark_prob_data(problem_set, problem_name)
    else
        repo_root = dirname(Pkg.project().path)
        giselsson_path = joinpath(repo_root, "synthetic_problem_data/giselsson_problem.jld2")
        data = load(giselsson_path)["data"]
    end

    # Unpack the data.
    P, c, A, b, K = data.P, data.c, data.A, data.b, data.K

    # Create a problem instance.
    problem = ProblemData(problem_set, problem_name, P, c, A, b, K)
    return problem
end

######################################
# Solve the Reference (Clarabel/SCS) #
######################################

function solve_reference(problem::ProblemData, args)
    # Choose the reference solver in {:SCS, :Clarabel}
    reference_solver = args["ref-solver"]
    problem_set = args["problem-set"]
    problem_name = args["problem-name"]

    println()
    if reference_solver == :SCS
        println("RUNNING SCS...")
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "eps_abs", 1e-10)
        set_optimizer_attribute(model, "eps_rel", 1e-10)

        # set acceleration_lookback to 0 to disable Anderson acceleration
        set_optimizer_attribute(model, "acceleration_lookback", 0) # default 10, set to 0 to DISABLE acceleration
        # set_optimizer_attribute(model, "acceleration_interval", 10) # default 10
        set_optimizer_attribute(model, "max_iters", 150) # default 1e5
        # set_optimizer_attribute(model, "normalize", 0) # whether to scale data, default 1
        set_optimizer_attribute(model, "scale", 1) # initial dual scale factor, default 0.1
        set_optimizer_attribute(model, "adaptive_scale", 0) # whether to heuristically adapt dual scale, default 1
        set_optimizer_attribute(model, "rho_x", 1) # primal scale factor, default 1e-6
        set_optimizer_attribute(model, "alpha", 1) # relaxation parameter, default 1.5
    elseif reference_solver == :Clarabel
        println("RUNNING CLARABEL...")
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "tol_infeas_rel", 1e-12)
    else
        error("Invalid reference solver option. Choose between :SCS and :Clarabel.")
    end
    println("Problem set/name: $problem_set/$problem_name")

    # Define primal and slack variables.
    @variable(model, x_ref[1:problem.n])
    @variable(model, s_ref[1:problem.m])

    # Add the equality constraint: A*x_ref + s_ref == b.
    @constraint(model, con, problem.A * x_ref + s_ref .== problem.b)

    # Add cone constraints.
    add_cone_constraints!(model, s_ref, problem.K)

    # Define the quadratic objective.
    @objective(model, Min, 0.5 * dot(x_ref, problem.P * x_ref) + dot(problem.c, x_ref))

    # Solve the problem.
    JuMP.optimize!(model)

    # Extract solutions.
    x_ref = value.(x_ref)
    s_ref = value.(s_ref)
    y_ref = dual.(con)  # Dual variables (Lagrange multipliers)
    obj_ref = objective_value(model)

    return x_ref, s_ref, y_ref, obj_ref
end

#######################################
# Run the Prototype Optimization      #
#######################################

function run_prototype(problem::ProblemData, args::Dict{String, Any};
    x_ref::Union{Nothing, Vector{Float64}} = nothing, y_ref::Union{Nothing, Vector{Float64}} = nothing)

    # NB we do not compute A' * A, just store its specification as a linear map
    A_gram = LinearMap(x -> problem.A' * (problem.A * x), size(problem.A, 2), size(problem.A, 2); issymmetric = true)

    if args["variant"] != :ADMM
        take_away_op = build_operator(args["variant"], problem.P, problem.A, A_gram, args["rho"])
        Random.seed!(42)  # seed for reproducibility
        max_τ = 1 / dom_λ_power_method(take_away_op, 30)
        τ = 0.90 * max_τ # 90% of max_τ is used in PDLP paper, for instance
    else # ADMM does not use τ step size
        τ = nothing
    end

    println("RUNNING PROTOTYPE VARIANT $(args["variant"])...")
    println("Problem set/name: $(args["problem-set"])/$(args["problem-name"])")
    println("Acceleration: $(args["acceleration"])")
    if args["acceleration"] in [:krylov, :anderson]
        println("Acceleration memory: $(args["accel-memory"])")
    end

    # initialise the workspace
    if args["acceleration"] == :krylov
        ws = KrylovWorkspace(problem, args["variant"], τ, args["rho"], args["theta"], args["accel-memory"], args["krylov-operator"], A_gram = A_gram)
    elseif args["acceleration"] == :anderson
        ws = AndersonWorkspace(problem, args["variant"], τ, args["rho"], args["theta"], args["accel-memory"], args["anderson-period"], A_gram = A_gram, broyden_type = args["anderson-broyden-type"], memory_type = args["anderson-mem-type"], regulariser_type = args["anderson-reg"])
    else
        ws = NoneWorkspace(problem, args["variant"], τ, args["rho"], args["theta"], A_gram = A_gram)
    end

    # Run the solver (time or profile execution)
    results = optimise!(ws,
    args["max-iter"],
    args["print-mod"],
    args["residuals-relative"],
    args["run-fast"],
    args["acceleration"],
    args["rel-kkt-tol"],
    restart_period = args["restart-period"],
    residual_norm = args["res-norm"],
    linesearch_period = args["linesearch-period"],
    linesearch_ϵ = args["linesearch-eps"],
    x_sol = x_ref, y_sol = y_ref,
    explicit_affine_operator = false)

    return ws, results
end

#############################
# Refactored Plotting Block #
#############################

function plot_results(results, args, newline_char)

    k_final = length(results.data[:primal_obj_vals])
    
    # plotting constants
    LINEWIDTH = 2.5
    VERT_LINEWIDTH = 1.5
    ALPHA = 0.9

    # Common title components
    title_common = "Problem: $(args["problem-set"]) $(args["problem-name"]).$newline_char Variant $(args["variant"]) $newline_char"
    title_common *= "Restart period = $(args["restart-period"]).$newline_char Linesearch period = $(args["linesearch-period"])$newline_char"
    if args["acceleration"] == :none
        title_common *= "Acceleration: none.$newline_char"
        krylov_operator_str = ""
    elseif args["acceleration"] == :anderson
        title_common *= "Anderson acceleration: mem = $(args["accel-memory"]), period = $(args["anderson-period"]),$newline_char broyden = $(args["anderson-broyden-type"]), mem_type = $(args["anderson-mem-type"]).$newline_char"
        krylov_operator_str = ""
    elseif args["acceleration"] == :krylov
        title_common *= "Krylov acceleration: mem = $(args["accel-memory"]), op = $(args["krylov-operator"]).$newline_char"
    end

    # Add Krylov operator string if acceleration is :krylov
    
    
    constraint_lines = constraint_changes(results.data[:record_proj_flags])

    # Helper function to add common vertical lines, only if show_vlines is true.
    function add_vlines!(plt; constraint_style=(:dash, ALPHA, :green, VERT_LINEWIDTH))
        if args["show-vlines"]
            vline!(plt, results.data[:acc_step_iters], line = (:dash, ALPHA, :red, VERT_LINEWIDTH), label="Accelerated Steps")
            vline!(plt, results.data[:linesearch_iters], line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH), label="Line Search Steps")
            vline!(plt, constraint_lines, line = constraint_style, label="Active set changes")
        end
        return plt
    end

    # Primal objective plot.
    primal_obj_plot = plot(0:k_final-1, results.data[:primal_obj_vals], linewidth=LINEWIDTH,
    label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value",
    title="$title_common Objective")
    add_vlines!(primal_obj_plot)
    display(primal_obj_plot)

    # Dual objective plot.
    dual_obj_plot = plot(0:k_final-1, results.data[:dual_obj_vals], linewidth=LINEWIDTH,
    label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value",
    title="$title_common Dual Objective")
    add_vlines!(dual_obj_plot)
    display(dual_obj_plot)

    # Duality gap plot.
    gap_plot = plot(0:k_final-1, results.data[:primal_obj_vals] - results.data[:dual_obj_vals], linewidth=LINEWIDTH,
    label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap",
    title="$title_common Duality Gap")
    add_vlines!(gap_plot)
    display(gap_plot)

    # Primal residual plot.
    pres_plot = plot(0:k_final-1, results.data[:pri_res_norms], linewidth=LINEWIDTH,
    label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual",
    title="$title_common Primal Residual Norm", yaxis=:log)
    add_vlines!(pres_plot)
    display(pres_plot)

    # Dual residual plot.
    dres_plot = plot(0:k_final-1, results.data[:dual_res_norms], linewidth=LINEWIDTH,
    label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual",
    title="$title_common Dual Residual Norm", yaxis=:log)
    add_vlines!(dres_plot)
    display(dres_plot)

    # (x, y) distance to solution plot.
    xy_dist_to_sol = sqrt.(results.data[:x_dist_to_sol] .^ 2 .+ results.data[:y_dist_to_sol] .^ 2)
    xy_dist_plot = plot(0:k_final, xy_dist_to_sol, linewidth=LINEWIDTH,
        label="Prototype (x, y) Distance", xlabel="Iteration", ylabel="Distance to Solution",
        title="$title_common (x, y) Distance to Solution", yaxis=:log)
    add_vlines!(xy_dist_plot)
    display(xy_dist_plot)

    # (x, y) characteristic norm distance to solution plot.
    seminorm_plot = plot(0:k_final, results.data[:xy_chardist], linewidth=LINEWIDTH,
    label="(x, y) Seminorm Distance (Theory)", xlabel="Iteration", ylabel="Distance to Solution",
    title="$title_common (x, y) Characteristic Norm Distance to Solution", yaxis=:log)
    add_vlines!(seminorm_plot)
    display(seminorm_plot)

    # (x, y) step norms plot.
    xy_step_norms_plot = plot(0:k_final-1, results.data[:xy_step_norms], linewidth=LINEWIDTH,
        label="(x, y) Step l2 Norm", xlabel="Iteration", ylabel="Step Norm",
        title="$title_common (x, y) l2 Step Norm", yaxis=:log)
    add_vlines!(xy_step_norms_plot)
    display(xy_step_norms_plot)

    # (x, y) step CHAR norms plot.
    xy_step_char_norms_plot = plot(0:k_final-1, results.data[:xy_step_char_norms], linewidth=LINEWIDTH,
        label="(x, y) Step Char Norm", xlabel="Iteration", ylabel="Step CHAR Norm",
        title="$title_common (x, y) CHAR Step Norm", yaxis=:log)
    add_vlines!(xy_step_char_norms_plot)
    display(xy_step_char_norms_plot)

    # # Singular values ratio plot.
    # sing_vals_ratio_plot = plot(results.data[:update_mat_iters], results.data[:update_mat_singval_ratios], linewidth=LINEWIDTH,
    # label="Prototype Update Matrix", xlabel="Iteration", ylabel="First Two Singular Values' Ratio",
    # title="$title_beginning Update Matrix Singular Value Ratio  $title_end",
    # yaxis=:log, marker=:circle)
    # add_vlines!(sing_vals_ratio_plot)
    # display(sing_vals_ratio_plot)

    # # Update matrix rank plot.
    # update_ranks_plot = plot(results.data[:update_mat_iters], results.data[:update_mat_ranks],
    # label="Prototype Update Matrix", xlabel="Iteration", ylabel="Rank",
    # title="$title_beginning Update Matrix Rank  $title_end",
    # linewidth=LINEWIDTH, xticks=0:100:MAX_ITER)
    # add_vlines!(update_ranks_plot)
    # display(update_ranks_plot)

    # Projection flags plot (often intensive)
    # enforced_constraints_plot(results.data[:record_proj_flags])

    # Consecutive update (x, y) cosines plot.
    xy_update_cosines_plot = plot(1:k_final-1, results.data[:xy_update_cosines], linewidth=LINEWIDTH,
        label="Prototype Update Cosine", xlabel="Iteration", ylabel="Cosine of Consecutive Updates",
        title="$title_common Consecutive (x, y) Update Cosines")
    add_vlines!(xy_update_cosines_plot, constraint_style = (:dashdot, ALPHA, :green, VERT_LINEWIDTH))
    display(xy_update_cosines_plot)
end

##########################
# Command line interface #
##########################
"""
    run_cli()

Parse `ARGS`, call `parse_command_line`, then `main`, and exit.
"""
function run_cli()
    config = parse_command_line()
    ws, results, x_ref, y_ref = main(config)

    # maybe print a summary, write outputs, etc.
    return
end


##########################
# Main Execution Block #
###########################

function main(config::Dict{String, Any})
    # Choose the problem and fetch data.
    println()
    println("About to import problem data...")
    problem = fetch_data(config)

    # Solve the reference problem (Clarabel/SCS).
    println()
    println("About to solve problem with reference solver...")

    x_ref, s_ref, y_ref, obj_ref = solve_reference(problem, config)

    # Run the prototype optimization.
    println()
    println("About to run prototype solver...")
    @time ws, results = run_prototype(problem, config, x_ref = x_ref, y_ref = y_ref)

    if !config["run-fast"]
        newline_char = initialise_misc()
        println()
        println("About to plot results...")
        # plot_results(results, config["variant"], config["restart-period"], config["acceleration"], config["accel-memory"], config["linesearch-period"], newline_char, config["problem-set"], config["problem-name"], config["krylov-operator"], show_vlines = config["show-vlines"])
        plot_results(results, config, newline_char)
    end
    
    #return data of interest to inspect
    return ws, results, x_ref, y_ref
end

end # module FOMPrototypes