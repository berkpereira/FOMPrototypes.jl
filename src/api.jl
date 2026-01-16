# High-level API for FOMPrototypes

##################################
# Problem Selection & Data Fetch #
##################################

function fetch_data(problem_set::String, problem_name::String)
    if problem_name in ["giselsson", "toy", "zhang_socp"]
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

function solve_reference(
    problem::ProblemData,
    problem_set::String,
    problem_name::String,
    config::SolverConfig)
    # Choose the reference solver in {:SCS, :Clarabel}
    reference_solver = config.ref_solver

    println()
    if reference_solver == :SCS
        println("RUNNING SCS...")
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "eps_rel", config.rel_kkt_tol)
        set_optimizer_attribute(model, "eps_abs", config.rel_kkt_tol)

        # set acceleration_lookback to 0 to disable Anderson acceleration
        # set_optimizer_attribute(model, "acceleration_lookback", 0) # default 10, set to 0 to DISABLE acceleration
        # set_optimizer_attribute(model, "acceleration_interval", 10) # default 10
        set_optimizer_attribute(model, "max_iters", 10_000) # default 100_000
        set_optimizer_attribute(model, "normalize", 0) # whether to scale data, default 1
        # set_optimizer_attribute(model, "adaptive_scale", 0) # whether to heuristically adapt dual scale, default 1
        # set_optimizer_attribute(model, "rho_x", 1) # primal scale factor, default 1e-6
        set_optimizer_attribute(model, "alpha", 1) # relaxation parameter, default 1.5
    elseif reference_solver == :COSMO
        println("RUNNING COSMO...")
        model = Model(COSMO.Optimizer)
        set_optimizer_attribute(model, "eps_rel", config.rel_kkt_tol)
        set_optimizer_attribute(model, "eps_abs", config.rel_kkt_tol)
        # set_optimizer_attribute(model, "check_termination", 1)
        set_optimizer_attribute(model, "max_iter", 10_000)
        set_optimizer_attribute(model, "alpha", 1.0)
        set_optimizer_attribute(model, "scaling", 0)
        # set_optimizer_attribute(model, "rho", 0.1)
    elseif reference_solver == :Clarabel
        println("RUNNING CLARABEL...")
        model = Model(Clarabel.Optimizer)
        # set_optimizer_attribute(model, "tol_infeas_rel", 1e-12)
    else
        error("Invalid reference solver option. Choose between :SCS, :COSMO, and :Clarabel.")
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

solve_reference(problem::ProblemData, problem_set::String, problem_name::String, config::AbstractDict) =
    solve_reference(problem, problem_set, problem_name, SolverConfig(config))

#######################################
# Run the Prototype Optimization      #
#######################################

function run_prototype(problem::ProblemData,
    problem_set::String,
    problem_name::String,
    config::SolverConfig;
    state_ref::Union{Nothing, Vector{Float64}} = nothing,
    full_diagnostics::Bool = false,
    spec_plot_period::Real = Inf)

    # simple args consistency check
    if config.anderson_interval < 1
        error("Anderson interval must be 1 or more.")
    end

    # initialise timer object
    to = TimerOutput()

    @timeit to "setup" begin
        # NB we do not compute A' * A, just store its specification as a linear map
        A_gram = LinearMap(x -> problem.A' * (problem.A * x), size(problem.A, 2), size(problem.A, 2); issymmetric = true)

        @timeit to "build operator" if config.variant != :ADMM
            take_away_op = build_takeaway_op(config.variant, problem.P, problem.A, A_gram, config.rho)
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

        println("RUNNING PROTOTYPE VARIANT $(config.variant)...")
        println("Problem set/name: $(problem_set)/$(problem_name)")
        println("Acceleration: $(config.acceleration)")
        if config.acceleration in [:krylov, :anderson, :randomized]
            println("Acceleration memory: $(config.accel_memory)")
        end

        @timeit to "init workspace" begin
            # initialise the workspace
            if config.acceleration == :krylov
                ws = KrylovWorkspace(problem, PrePPM, config.variant, τ, config.rho, config.theta, config.accel_memory, config.krylov_tries_per_mem, config.safeguard_norm, config.krylov_operator, A_gram = A_gram, residual_period = config.residual_period, to = to)
            elseif config.acceleration == :anderson
                anderson_log = !config.run_fast
                ws = AndersonWorkspace(problem, PrePPM, config.variant, τ, config.rho, config.theta, config.accel_memory, config.anderson_interval, config.safeguard_norm, A_gram = A_gram, residual_period = config.residual_period, broyden_type = config.anderson_broyden_type, memory_type = config.anderson_mem_type, regulariser_type = config.anderson_reg, anderson_log = anderson_log, to = to)
            elseif config.acceleration == :randomized
                ws = RandomizedWorkspace(problem, PrePPM, config.variant, τ, config.rho, config.theta, config.accel_memory, config.randomized_regularization, config.safeguard_norm, config.randomized_operator, config.randomized_augment_fp, A_gram = A_gram, residual_period = config.residual_period, to = to)
            else
                ws = VanillaWorkspace(problem, PrePPM, config.variant, τ, config.rho, config.theta, A_gram = A_gram, residual_period = config.residual_period, config = config, to = to)
            end
        end
    end

    @timeit to "solver" begin
        # Run the solver
        results, ws_diag = optimise!(
            ws,
            config,
            setup_time = to.inner_timers["setup"].accumulated_data.time / 1e9,
            state_ref = state_ref,
            timer = to,
            full_diagnostics = full_diagnostics,
            spectrum_plot_period = spec_plot_period)
    end

    return ws, ws_diag, results, to
end

run_prototype(problem::ProblemData, problem_set::String, problem_name::String, config::AbstractDict; kwargs...) =
    run_prototype(problem, problem_set, problem_name, SolverConfig(config); kwargs...)

##########################
# Main Execution Block   #
##########################

function main(config::SolverConfig)
    # Choose the problem and fetch data.
    println()
    println("About to import problem data...")
    problem = fetch_data(config.problem_set, config.problem_name)

    # Solve the reference problem (Clarabel/SCS).
    println()
    println("About to solve problem with reference solver...")

    model, state_ref, obj_ref = solve_reference(problem,
    config.problem_set, config.problem_name, config)

    # Run the prototype optimization.
    println()
    println("About to run prototype solver...")

    ws, ws_diag, results, to = run_prototype(problem,
    config.problem_set, config.problem_name,
    config, state_ref = state_ref)

    if !config.run_fast
        println()
        println("About to plot results...")
        plot_results(ws, results, config.problem_set, config.problem_name, config)
    end

    #return data of interest to inspect
    return ws, ws_diag, results, to, x_ref, y_ref
end

# convenience fallback for dict inputs
main(config::AbstractDict) = main(SolverConfig(config))
