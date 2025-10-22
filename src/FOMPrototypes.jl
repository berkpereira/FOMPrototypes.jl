module FOMPrototypes

export main, run_cli, run_prototype, solve_reference, fetch_data,
    plot_results

# Import packages.
using Revise
using ArgParse
using Infiltrator
using TimerOutputs
using LinearMaps
using Infiltrator
using Profile
using BenchmarkTools
using SparseArrays
using SCS
using Random
using JLD2

# Include all project source files.
include(joinpath(@__DIR__, "core/types.jl"))
include(joinpath(@__DIR__, "core/problem_data.jl"))
include(joinpath(@__DIR__, "core/utils.jl"))
include(joinpath(@__DIR__, "linops/custom_nla.jl"))
include(joinpath(@__DIR__, "linops/types_utils.jl"))
include(joinpath(@__DIR__, "alg/cones.jl"))
include(joinpath(@__DIR__, "linops/alg_utils.jl"))
include(joinpath(@__DIR__, "linops/residuals.jl"))
include(joinpath(@__DIR__, "alg/record.jl"))
include(joinpath(@__DIR__, "alg/vanilla.jl"))
include(joinpath(@__DIR__, "alg/krylov.jl"))
include(joinpath(@__DIR__, "alg/anderson.jl"))
include(joinpath(@__DIR__, "alg/linesearch.jl"))
include(joinpath(@__DIR__, "alg/safeguard.jl"))
include(joinpath(@__DIR__, "diagnostics/printing.jl"))
include(joinpath(@__DIR__, "diagnostics/plotting.jl"))
include(joinpath(@__DIR__, "solver.jl"))

########################
# Initialization Block #
########################
    
function initialise_misc(backend::Symbol = :plotlyjs)
    # Set Plots backend.
    # For interactive plots: plotlyjs()
    # For faster plotting: gr()
    if backend == :plotlyjs
        plotlyjs()
    elseif backend == :pyplot
        pyplot()
    elseif backend == :gr
        gr()
    else
        error("Invalid backend specified. Use :plotlyjs, :pyplot, or :gr.")
    end

    # Determine newline character based on backend.
    local newline_char = Plots.backend_name() in [:gr, :pythonplot] ? "\n" : "<br>"

    # Set default plot size (in pixels)
    # default(size=(2000, 450)) # for desktop
    default(size=(800, 600)) # for laptop

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

        "--anderson-interval"
        help = "Anderson acceleration is applied to the operator obtained from composing the optimiser operator THIS many times."
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

        "--rel-kkt-tol"
        help = "Relative KKT tolerance for stopping criterion."
        arg_type = Float64
        default = 1e-6

        "--run-fast"
        help = "Run fast mode (no plotting, less data recording during run)."
        arg_type = Bool
        default = true

        "--print-res-rel"
        help = "Use relative metrics when printing iter info."
        arg_type = Bool
        default = true

        "--show-vlines"
        help = "Show relevant vertical dashed lines in plots."
        arg_type = Bool
        default = false

        "--global-timeout"
        help = "Global timeout for the solver (in seconds)."
        arg_type = Float64
        default = 60.0

        "--loop-timeout"
        help = "Timeout for iterative loop (in seconds)."
        arg_type = Float64
        default = 30.0

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

function fetch_data(problem_set::String, problem_name::String)
    if problem_name == "giselsson" || problem_name == "toy"
        repo_root = normpath(joinpath(@__DIR__, ".."))
        file = "synthetic_problem_data/$(problem_name)_problem.jld2"
        data = load(joinpath(repo_root, file))
        # Unpack the data.
        P, c, A, b, K = data["P"], data["c"], data["A"], data["b"], data["K"]
    else
        data = load_clarabel_benchmark_prob_data(problem_set, problem_name)
        # Unpack the data.
        P, c, A, b, K = data.P, data.c, data.A, data.b, data.K
    end

    # Create a problem instance.
    problem = ProblemData(problem_set, problem_name, P, c, A, b, K)
    return problem
end

######################################
# Solve the Reference (Clarabel/SCS) #
######################################

function solve_reference(problem::ProblemData,
    problem_set::String,
    problem_name::String,
    args::Dict{String, T}) where T
    # Choose the reference solver in {:SCS, :Clarabel}
    reference_solver = args["ref-solver"]

    println()
    if reference_solver == :SCS
        println("RUNNING SCS...")
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "eps_rel", 1e-6)
        set_optimizer_attribute(model, "eps_abs", 1e-6)

        # set acceleration_lookback to 0 to disable Anderson acceleration
        # set_optimizer_attribute(model, "acceleration_lookback", 0) # default 10, set to 0 to DISABLE acceleration
        # set_optimizer_attribute(model, "acceleration_interval", 10) # default 10
        # set_optimizer_attribute(model, "max_iters", 150) # default 1e5
        set_optimizer_attribute(model, "normalize", 0) # whether to scale data, default 1
        set_optimizer_attribute(model, "scale", 1) # initial dual scale factor, default 0.1
        set_optimizer_attribute(model, "adaptive_scale", 0) # whether to heuristically adapt dual scale, default 1
        # set_optimizer_attribute(model, "rho_x", 1) # primal scale factor, default 1e-6
        set_optimizer_attribute(model, "alpha", 1) # relaxation parameter, default 1.5
    elseif reference_solver == :Clarabel
        println("RUNNING CLARABEL...")
        model = Model(Clarabel.Optimizer)
        # set_optimizer_attribute(model, "tol_infeas_rel", 1e-12)
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

    state_ref = [x_ref; y_ref]

    return model, state_ref, obj_ref
end

#######################################
# Run the Prototype Optimization      #
#######################################

function run_prototype(problem::ProblemData,
    problem_set::String,
    problem_name::String,
    args::Dict{String, T};
    state_ref::Union{Nothing, Vector{Float64}} = nothing,
    full_diagnostics::Bool = false,
    spec_plot_period::Real = Inf) where T

    # simple args consistency check
    if args["anderson-interval"] < 1
        error("Anderson interval must be 1 or more.")
    end

    # initialise timer object
    to = TimerOutput()

    @timeit to "setup" begin
        # NB we do not compute A' * A, just store its specification as a linear map
        A_gram = LinearMap(x -> problem.A' * (problem.A * x), size(problem.A, 2), size(problem.A, 2); issymmetric = true)

        @timeit to "build operator" if args["variant"] != :ADMM
            take_away_op = build_takeaway_op(args["variant"], problem.P, problem.A, A_gram, args["rho"])
            Random.seed!(42)  # seed for reproducibility
            max_τ = 1 / dom_λ_power_method(take_away_op, 30)

            @info "Maximum τ: $(max_τ)"
            
            if max_τ !== NaN
                τ = 0.90 * max_τ # 90% of max_τ is used in PDLP paper, for instance
            else # max_τ === NaN can happen eg in variant Symbol(1) if R(P + ρ * A' * A) is zero. in these cases we can use any τ > 0
                τ = 1.0 # fallback value
            end
        else # ADMM does not use τ step size
            τ = nothing
        end

        println("RUNNING PROTOTYPE VARIANT $(args["variant"])...")
        println("Problem set/name: $(problem_set)/$(problem_name)")
        println("Acceleration: $(args["acceleration"])")
        if args["acceleration"] in [:krylov, :anderson]
            println("Acceleration memory: $(args["accel-memory"])")
        end

        @timeit to "init workspace" begin
            # initialise the workspace
            if args["acceleration"] == :krylov
                ws = KrylovWorkspace(problem, args["variant"], τ, args["rho"], args["theta"], args["accel-memory"], args["krylov-tries-per-mem"], args["safeguard-norm"], args["krylov-operator"], A_gram = A_gram, to = to)
            elseif args["acceleration"] == :anderson
                anderson_log = !args["run-fast"]
                ws = AndersonWorkspace(problem, args["variant"], τ, args["rho"], args["theta"], args["accel-memory"], args["anderson-interval"], args["safeguard-norm"], A_gram = A_gram, broyden_type = args["anderson-broyden-type"], memory_type = args["anderson-mem-type"], regulariser_type = args["anderson-reg"], anderson_log = anderson_log, to = to)
            else
                ws = VanillaWorkspace(problem, PrePPM, args["variant"], τ, args["rho"], args["theta"], A_gram = A_gram, to = to)
            end
        end
    end

    @timeit to "solver" begin
        # Run the solver
        results, ws_diag = optimise!(
            ws,
            args,
            setup_time = to.inner_timers["setup"].accumulated_data.time / 1e9,
            state_ref = state_ref,
            timer = to,
            full_diagnostics = full_diagnostics,
            spectrum_plot_period = spec_plot_period)
    end

    return ws, ws_diag, results, to
end

#############################
# Refactored Plotting Block #
#############################

function plot_results(
    ws::AbstractWorkspace,
    results,
    problem_set::String,
    problem_name::String,
    args::Dict{String, T},
    backend::Symbol = :plotlyjs) where T

    newline_char = initialise_misc(backend)

    println("Backend is $(Plots.backend_name())")

    k_final = length(results.metrics_history[:primal_obj_vals])
    
    # plotting constants
    LINEWIDTH = 2.5
    VERT_LINEWIDTH = 1
    ALPHA = 0.9

    # Common title components
    title_common = "Problem: $(problem_set) $(problem_name).$newline_char Variant $(args["variant"]) $newline_char"
    # title_common *= "Restart period = $(args["restart-period"]).$newline_char Linesearch period = $(args["linesearch-period"])$newline_char"
    if args["acceleration"] == :none
        title_common *= "Acceleration: none.$newline_char"
        krylov_operator_str = ""
    elseif args["acceleration"] == :anderson
        title_common *= "Anderson acceleration: mem = $(args["accel-memory"]), interval = $(args["anderson-interval"]),$newline_char broyden = $(args["anderson-broyden-type"]), mem_type = $(args["anderson-mem-type"]).$newline_char"
        krylov_operator_str = ""
    elseif args["acceleration"] == :krylov
        title_common *= "Krylov acceleration: mem = $(args["accel-memory"]), op = $(args["krylov-operator"]).$newline_char"
    end

    # Add Krylov operator string if acceleration is :krylov
    
    
    constraint_lines = constraint_changes(results.metrics_history[:record_proj_flags])

    # Helper function to add common vertical lines, only if show_vlines is true.
    function add_vlines!(plt; include_active_set_changes::Bool = true)
        if args["show-vlines"]
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
    title="$title_common Primal Residual Norm", yaxis=:log)
    add_vlines!(pres_plot)
    display(pres_plot)

    # Dual residual plot.
    dres_plot = plot(0:k_final-1, results.metrics_history[:dual_res_norms], linewidth=LINEWIDTH,
    label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual",
    title="$title_common Dual Residual Norm", yaxis=:log)
    add_vlines!(dres_plot)
    display(dres_plot)

    if length(results.metrics_history[:x_dist_to_sol]) != 0
        # state distance to solution plot.
        state_dist_to_sol = sqrt.(results.metrics_history[:x_dist_to_sol] .^ 2 .+ results.metrics_history[:y_dist_to_sol] .^ 2)
        state_dist_plot = plot(0:k_final, state_dist_to_sol, linewidth=LINEWIDTH,
            label="Prototype state Distance", xlabel="Iteration", ylabel="Distance to Solution",
            title="$title_common state Distance to Solution", yaxis=:log)
        add_vlines!(state_dist_plot)
        display(state_dist_plot)

        # state characteristic norm distance to solution plot.
        seminorm_plot = plot(0:k_final, results.metrics_history[:state_chardist], linewidth=LINEWIDTH,
        label="state Seminorm Distance (Theory)", xlabel="Iteration", ylabel="Distance to Solution",
        title="$title_common state Characteristic Norm Distance to Solution", yaxis=:log)
        add_vlines!(seminorm_plot)
        display(seminorm_plot)
    end

    # state step norms plot.
    state_step_norms_plot = plot(0:k_final-1, results.metrics_history[:state_step_norms], linewidth=LINEWIDTH,
        label="state Step l2 Norm", xlabel="Iteration", ylabel="Step Norm",
        title="$title_common state l2 Step Norm", yaxis=:log)
    add_vlines!(state_step_norms_plot)
    display(state_step_norms_plot)

    # state step CHAR norms plot.
    state_step_char_norms_plot = plot(0:k_final-1, results.metrics_history[:state_step_char_norms], linewidth=LINEWIDTH,
        label="state Step Char Norm", xlabel="Iteration", ylabel="Step CHAR Norm",
        title="$title_common state CHAR Step Norm", yaxis=:log)
    add_vlines!(state_step_char_norms_plot)
    display(state_step_char_norms_plot)

    # # Singular values ratio plot.
    # sing_vals_ratio_plot = plot(results.metrics_history[:update_mat_iters], results.metrics_history[:update_mat_singval_ratios], linewidth=LINEWIDTH,
    # label="Prototype Update Matrix", xlabel="Iteration", ylabel="First Two Singular Values' Ratio",
    # title="$title_beginning Update Matrix Singular Value Ratio  $title_end",
    # yaxis=:log, marker=:circle)
    # add_vlines!(sing_vals_ratio_plot)
    # display(sing_vals_ratio_plot)

    # # Update matrix rank plot.
    # update_ranks_plot = plot(results.metrics_history[:update_mat_iters], results.metrics_history[:update_mat_ranks],
    # label="Prototype Update Matrix", xlabel="Iteration", ylabel="Rank",
    # title="$title_beginning Update Matrix Rank  $title_end",
    # linewidth=LINEWIDTH, xticks=0:100:MAX_ITER)
    # add_vlines!(update_ranks_plot)
    # display(update_ranks_plot)

    # Consecutive update state cosines plot.
    state_update_cosines_plot = plot(1:k_final-1, results.metrics_history[:state_update_cosines], linewidth=LINEWIDTH,
        label="Prototype Update Cosine", xlabel="Iteration", ylabel="Cosine of Consecutive Updates",
        title="$title_common Consecutive state Update Cosines")
    add_vlines!(state_update_cosines_plot)
    display(state_update_cosines_plot)

    # Projection flags plot (often intensive)
    # enforced_constraints_plot(results.metrics_history[:record_proj_flags])

    # plot count of flipped constraints
    proj_diffs_plot = plot_projection_diffs(results.metrics_history[:record_proj_flags])
    add_vlines!(proj_diffs_plot, include_active_set_changes = false)
    display(proj_diffs_plot)

    # plot count of enforced constraints
    enforced_constraints_plot = plot_enforced_constraints_count(results.metrics_history[:record_proj_flags], ws.p.m)
    add_vlines!(enforced_constraints_plot, include_active_set_changes = false)
    display(enforced_constraints_plot)

    # FP Metric Ratio plot.
    acc_attempt_iters = results.metrics_history[:acc_attempt_iters]
    fp_metric_ratios = results.metrics_history[:fp_metric_ratios]
    fp_metric_plot = plot(acc_attempt_iters, fp_metric_ratios, linewidth=LINEWIDTH,
        label="FP Metric Ratio", xlabel="Acceleration Attempt Iterations", ylabel="FP Metric Ratio",
        title="$title_common FP Metric Ratio",
        lw=2, # Set line width for better visibility
        marker=:circle, # Add markers to each data point
        markersize=3)
    add_vlines!(fp_metric_plot)
    display(fp_metric_plot)
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
    ws, ws_diag, results, x_ref, y_ref = main(config)

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
    problem = fetch_data(config["problem-set"], config["problem-name"])

    # Solve the reference problem (Clarabel/SCS).
    println()
    println("About to solve problem with reference solver...")

    model, state_ref, obj_ref = solve_reference(problem,
    config["problem-set"], config["problem-name"], config)

    # Run the prototype optimization.
    println()
    println("About to run prototype solver...")
    
    ws, ws_diag, results, to = run_prototype(problem,
    config["problem-set"], config["problem-name"],
    config, state_ref = state_ref)

    if !config["run-fast"]
        println()
        println("About to plot results...")
        plot_results(ws, results, config["problem-set"], config["problem-name"], config)
    end
    
    #return data of interest to inspect
    return ws, ws_diag, results, to, x_ref, y_ref
end

end # module FOMPrototypes