# this file is used to run solver code directly from a terminal command
import FOMPrototypes

if abspath(PROGRAM_FILE) == @__FILE__
    config = FOMPrototypes.parse_command_line()
    println()

    warmup_config = FOMPrototypes.SolverConfig(config;
        problem_set = "sslsq",
        problem_name = "NYPA_Maragal_3_huber",
        max_iter = 300,
        global_timeout = 10.0,
        loop_timeout = 10.0,
        rel_kkt_tol = 0.0)

    println("WARMUP run:")
    problem = FOMPrototypes.fetch_data(warmup_config.problem_set, warmup_config.problem_name);
    ws, results, to = FOMPrototypes.run_prototype(problem, warmup_config.problem_set, warmup_config.problem_name, warmup_config);

    println("--------------------")
    println("--------------------")

    println("ACTUAL run:")
    problem = FOMPrototypes.fetch_data(config.problem_set, config.problem_name);
    ws, results, to = FOMPrototypes.run_prototype(problem, config.problem_set, config.problem_name, config);
end
