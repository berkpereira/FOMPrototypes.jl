# Imports
# Include all project source files.
include("types.jl")
include("solver.jl")
include("utils.jl")
include("problem_data.jl")
include("solver.jl")
include("printing.jl")
include("plotting.jl")
include("acceleration.jl")

# Import packages.
using Revise
using LinearMaps
using Infiltrator
using Profile
using StatProfilerHTML
using BenchmarkTools
using Printf
using Plots
using SparseArrays
using SCS
using Random
using JLD2
using Pkg

###########################
# 1. Initialization Block #
###########################
    
function initialize_project()
    # Set Plots backend.
    # For interactive plots: plotlyjs()
    # For faster plotting: gr()
    plotlyjs()

    # Determine newline character based on backend.
    local newline_char = Plots.backend_name() in [:gr, :pythonplot] ? "\n" : "<br>"

    # Set default plot size (in pixels)
    default(size=(1100, 700)) # for desktop
    # default(size=(1200, 800)) # for laptop

    return newline_char
end

#####################################
# 2. Problem Selection & Data Fetch #
#####################################

function choose_problem()
    # Choose problem option. Valid options: :LASSO, :HUBER, :MAROS, :GISELSSON
    problem_option = :LASSO

    if problem_option === :LASSO
        problem_set = "sslsq"
        # problem_name = "NYPA_Maragal_5_lasso"; # large, challenging
        problem_name = "HB_abb313_lasso"  # (m, n) = (665, 665)
        # Additional LASSO options commented out...
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

function fetch_data(problem_option, problem_set, problem_name)
    if problem_option !== :GISELSSON
        data = load_clarabel_benchmark_prob_data(problem_set, problem_name)
    else
        repo_root = dirname(Pkg.project().path)
        giselsson_path = joinpath(repo_root, "synthetic_problem_data/giselsson_problem.jld2")
        data = load(giselsson_path)["data"]
    end

    # Unpack the data.
    P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K

    # Create a problem instance.
    problem = ProblemData(P, c, A, b, K)
    return problem, P, c, A, b, m, n, K
end

#########################################
# 3. Solve the Reference (Clarabel/SCS)   #
#########################################

function solve_reference(problem, A, b, P, c, m, n, K, problem_set, problem_name)
    # Choose the reference solver: :SCS or :Clarabel
    reference_solver = :Clarabel

    println()
    if reference_solver === :SCS
        println("RUNNING SCS...")
        model = Model(SCS.Optimizer)
    elseif reference_solver === :Clarabel
        println("RUNNING CLARABEL...")
        model = Model(Clarabel.Optimizer)
    end
    println("Problem set/name: $problem_set/$problem_name")

    # Optionally, you can suppress solver output:
    # set_silent(model)

    # Define primal and slack variables.
    @variable(model, x_ref[1:n])
    @variable(model, s_ref[1:m])

    # Add the equality constraint: A*x_ref + s_ref == b.
    @constraint(model, con, A * x_ref + s_ref .== b)

    # Add cone constraints.
    add_cone_constraints!(model, s_ref, K)

    if reference_solver === :SCS
        set_optimizer_attribute(model, "eps_abs", 1e-12)
        set_optimizer_attribute(model, "eps_rel", 1e-12)
    end

    # Define the quadratic objective.
    @objective(model, Min, 0.5 * dot(x_ref, P * x_ref) + dot(c, x_ref))

    # Solve the problem.
    JuMP.optimize!(model)

    # Extract solutions.
    x_ref = value.(x_ref)
    s_ref = value.(s_ref)
    y_ref = dual.(con)  # Dual variables (Lagrange multipliers)
    obj_ref = objective_value(model)

    return x_ref, s_ref, y_ref, obj_ref
end

##########################################
# 4. Run the Prototype Optimization      #
##########################################

function run_prototype(problem, A, P, c, b, m, n, x_ref, s_ref, y_ref, problem_set, problem_name)
    #basic params
    ρ = 1.0
    θ = 1.0 # NB this ought to be fixed = 1.0 until we change many other things
    variant = 1  #in {-1, 0, 1, 2, 3, 4}
    
    MAX_ITER = 1000
    PRINT_MOD = 50
    RES_NORM = Inf
    
    #restarts
    RESTART_PERIOD = Inf
    
    #krylov acceleration
    ACCEL_MEMORY = 19
    ACCELERATION = false
    KRYLOV_OPERATOR_TILDE_A = false
    
    #line search
    LINESEARCH_PERIOD = Inf
    LINESEARCH_ϵ = 0.001
    
    #monitoring "affine" operator structure
    COMPUTE_OPERATOR_EXPLICITLY = false
    SPEC_PLOT_PERIOD = 26
    
    # NB we do not compute A' * A, just store its specification as a linear map
    A_gram = LinearMap(x -> A' * (A * x), size(A, 2), size(A, 2); issymmetric = true)

    if variant != 0 # we skip this if using ADMM
        take_away_op = build_operator(variant, P, A, A_gram, ρ)
        Random.seed!(42)  # seed for reproducibility
        max_τ = 1 / dom_λ_power_method(take_away_op, 30)
    end
    
    τ = 0.90 * max_τ # 90% of max_τ is used in PDLP paper, for instance

    println("RUNNING PROTOTYPE VARIANT $variant...")
    println("Problem set/name: $problem_set/$problem_name")
    println("Restart period: $RESTART_PERIOD")
    println("Acceleration: $ACCELERATION")
    if ACCELERATION
        println("Acceleration memory: $ACCEL_MEMORY")
    end

    # Initialize the workspace.
    ws = Workspace(problem, variant, τ, ρ, θ)
    ws.cache[:A_gram] = A_gram

    # Run the solver (time the execution).
    @infiltrate

    results = nothing
    @time begin
        results = optimise!(ws,
            MAX_ITER,
            PRINT_MOD,
            restart_period = RESTART_PERIOD,
            residual_norm = RES_NORM,
            acceleration = ACCELERATION,
            acceleration_memory = ACCEL_MEMORY,
            linesearch_period = LINESEARCH_PERIOD,
            linesearch_ϵ = LINESEARCH_ϵ,
            krylov_operator_tilde_A = KRYLOV_OPERATOR_TILDE_A,
            x_sol = x_ref, y_sol = y_ref,
            explicit_affine_operator = COMPUTE_OPERATOR_EXPLICITLY,
            spectrum_plot_period = SPEC_PLOT_PERIOD)
    end

    return ws, results, variant, MAX_ITER, RESTART_PERIOD, ACCELERATION, ACCEL_MEMORY, LINESEARCH_PERIOD, LINESEARCH_ϵ, KRYLOV_OPERATOR_TILDE_A
end

###############################
# 5. Refactored Plotting Block #
###############################

function plot_results(results, variant, MAX_ITER, RESTART_PERIOD, ACCELERATION,
    ACCEL_MEMORY, LINESEARCH_PERIOD, newline_char,
    problem_set, problem_name, KRYLOV_OPERATOR_TILDE_A; show_vlines::Bool = true)
# Plotting constants.
LINEWIDTH = 2.5
VERT_LINEWIDTH = 1.5
ALPHA = 0.9

title_beginning = "Problem: $problem_set $problem_name.$newline_char Variant $variant $newline_char"
title_end = "$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION (memory = period = $ACCEL_MEMORY).$newline_char Linesearch period = $LINESEARCH_PERIOD."

if KRYLOV_OPERATOR_TILDE_A
krylov_operator_str = "$newline_char Krylov operator is A"
else
krylov_operator_str = "$newline_char Krylov operator is B = A – I"
end

constraint_lines = constraint_changes(results.enforced_set_flags)

# Helper function to add common vertical lines, only if show_vlines is true.
function add_vlines!(plt; constraint_style=(:dash, ALPHA, :green, VERT_LINEWIDTH))
if show_vlines
vline!(plt, results.acc_step_iters, line = (:dash, ALPHA, :red, VERT_LINEWIDTH), label="Accelerated Steps")
vline!(plt, results.linesearch_iters, line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH), label="Line Search Steps")
vline!(plt, constraint_lines, line = constraint_style, label="Active set changes")
end
return plt
end

# Primal objective plot.
primal_obj_plot = plot(0:MAX_ITER+1, results.primal_obj_vals, linewidth=LINEWIDTH,
label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value",
title="$title_beginning Objective $krylov_operator_str $title_end")
add_vlines!(primal_obj_plot)
display(primal_obj_plot)

# Dual objective plot.
dual_obj_plot = plot(0:MAX_ITER+1, results.dual_obj_vals, linewidth=LINEWIDTH,
label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value",
title="$title_beginning Dual objective $krylov_operator_str $title_end")
add_vlines!(dual_obj_plot)
display(dual_obj_plot)

# Duality gap plot.
gap_plot = plot(0:MAX_ITER+1, results.primal_obj_vals - results.dual_obj_vals, linewidth=LINEWIDTH,
label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap",
title="$title_beginning Duality Gap $krylov_operator_str $title_end")
add_vlines!(gap_plot)
display(gap_plot)

# Primal residual plot.
pres_plot = plot(0:MAX_ITER+1, results.pri_res_norms, linewidth=LINEWIDTH,
label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual",
title="$title_beginning Primal Residual Norm $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(pres_plot)
display(pres_plot)

# Dual residual plot.
dres_plot = plot(0:MAX_ITER+1, results.dual_res_norms, linewidth=LINEWIDTH,
label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual",
title="$title_beginning Dual Residual Norm $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(dres_plot)
display(dres_plot)

# (x, y) distance to solution plot.
xy_dist_to_sol = sqrt.(results.x_dist_to_sol .^ 2 .+ results.y_dist_to_sol .^ 2)
xy_dist_plot = plot(0:MAX_ITER+1, xy_dist_to_sol, linewidth=LINEWIDTH,
    label="Prototype (x, y) Distance", xlabel="Iteration", ylabel="Distance to Solution",
    title="$title_beginning (x, y) Distance to Solution $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(xy_dist_plot)
display(xy_dist_plot)

# (x, y) characteristic norm distance to solution plot.
seminorm_plot = plot(0:MAX_ITER+1, results.xy_semidist, linewidth=LINEWIDTH,
label="(x, y) Seminorm Distance (Theory)", xlabel="Iteration", ylabel="Distance to Solution",
title="$title_beginning (x, y) Characteristic Norm Distance to Solution $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(seminorm_plot)
display(seminorm_plot)

# (x, v) distance plot.
xv_dist_to_sol = sqrt.(results.x_dist_to_sol .^ 2 .+ results.v_dist_to_sol .^ 2)
xv_dist_plot = plot(0:MAX_ITER+1, xv_dist_to_sol, linewidth=LINEWIDTH,
label="Prototype (x, v) Distance", xlabel="Iteration", ylabel="Distance to Solution",
title="$title_beginning (x, v) Distance to Solution $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(xv_dist_plot)
display(xv_dist_plot)

# (x, y) step norms plot.
xy_step_norms_plot = plot(0:MAX_ITER, results.xy_step_norms, linewidth=LINEWIDTH,
    label="(x, y) Step l2 Norm", xlabel="Iteration", ylabel="Step Norm",
    title="$title_beginning (x, y) l2 Step Norm $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(xy_step_norms_plot)
display(xy_step_norms_plot)

# (x, y) step CHAR norms plot.
xy_step_char_norms_plot = plot(0:MAX_ITER, results.xy_step_char_norms, linewidth=LINEWIDTH,
    label="(x, y) Step Char Norm", xlabel="Iteration", ylabel="Step CHAR Norm",
    title="$title_beginning (x, y) CHAR Step Norm $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(xy_step_char_norms_plot)
display(xy_step_char_norms_plot)

# (x, v) step norms plot.
xv_step_norms_plot = plot(0:MAX_ITER, results.xv_step_norms, linewidth=LINEWIDTH,
label="(x, v) Step l2 Norm", xlabel="Iteration", ylabel="Step Norm",
title="$title_beginning (x, v) l2 Step Norm $krylov_operator_str $title_end", yaxis=:log)
add_vlines!(xv_step_norms_plot)
display(xv_step_norms_plot)

# Singular values ratio plot.
sing_vals_ratio_plot = plot(results.update_mat_iters, results.update_mat_singval_ratios, linewidth=LINEWIDTH,
label="Prototype Update Matrix", xlabel="Iteration", ylabel="First Two Singular Values' Ratio",
title="$title_beginning Update Matrix Singular Value Ratio $krylov_operator_str $title_end",
yaxis=:log, marker=:circle)
add_vlines!(sing_vals_ratio_plot)
display(sing_vals_ratio_plot)

# Update matrix rank plot.
update_ranks_plot = plot(results.update_mat_iters, results.update_mat_ranks,
label="Prototype Update Matrix", xlabel="Iteration", ylabel="Rank",
title="$title_beginning Update Matrix Rank $krylov_operator_str $title_end",
linewidth=LINEWIDTH, xticks=0:100:MAX_ITER)
add_vlines!(update_ranks_plot)
display(update_ranks_plot)

# Consecutive update (x, y) cosines plot.
xy_update_cosines_plot = plot(1:MAX_ITER, results.xy_update_cosines, linewidth=LINEWIDTH,
    label="Prototype Update Cosine", xlabel="Iteration", ylabel="Cosine of Consecutive Updates",
    title="$title_beginning Consecutive (x, y) Update Cosines $krylov_operator_str $title_end")
add_vlines!(xy_update_cosines_plot, constraint_style = (:dashdot, ALPHA, :green, VERT_LINEWIDTH))
display(xy_update_cosines_plot)

# Consecutive update (x, v) cosines plot.
xv_update_cosines_plot = plot(1:MAX_ITER, results.xv_update_cosines, linewidth=LINEWIDTH,
label="Prototype Update Cosine", xlabel="Iteration", ylabel="Cosine of Consecutive Updates",
title="$title_beginning Consecutive (x, v) Update Cosines $krylov_operator_str $title_end")
add_vlines!(xv_update_cosines_plot, constraint_style = (:dashdot, ALPHA, :green, VERT_LINEWIDTH))
display(xv_update_cosines_plot)
end

###########################
# 6. Main Execution Block #
###########################

function main()
    # Initialize the project (includes files, packages, and plotting settings).
    newline_char = initialize_project()

    # Choose the problem and fetch data.
    println()
    println("About to import problem data...")
    problem_option, problem_set, problem_name = choose_problem()
    problem, P, c, A, b, m, n, K = fetch_data(problem_option, problem_set, problem_name)

    # Solve the reference problem (Clarabel/SCS).
    println()
    println("About so solve problem with reference solver...")
    x_ref, s_ref, y_ref, obj_ref = solve_reference(problem, A, b, P, c, m, n, K, problem_set, problem_name)

    # Run the prototype optimization.
    println()
    println("About to run prototype solver...")
    ws, results, variant, MAX_ITER, RESTART_PERIOD, ACCELERATION, ACCEL_MEMORY,
    LINESEARCH_PERIOD, LINESEARCH_ϵ, KRYLOV_OPERATOR_TILDE_A =
        run_prototype(problem, A, P, c, b, m, n, x_ref, s_ref, y_ref, problem_set, problem_name)

    # Generate refactored plots.
    println()
    println("About to plot results...")
    plot_results(results, variant, MAX_ITER, RESTART_PERIOD, ACCELERATION,
                 ACCEL_MEMORY, LINESEARCH_PERIOD, newline_char,
                 problem_set, problem_name, KRYLOV_OPERATOR_TILDE_A,
                 show_vlines = false)
    
    #return data of interest to inspect
    return ws, results
end

# Execute main()
# ws, results = main();