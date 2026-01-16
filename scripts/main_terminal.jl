# This file is used to run solver code directly from a terminal command
import FOMPrototypes
using JLD2

include("cli_parser.jl")   # CLI argument parsing
include("script_utils.jl") # Shared utilities (save_diagnostic_matrix)

const SAVE_MATRIX = false  # Set to true when you want to save
const MATRIX_TAG = "optimal"  # Set to "optimal" or "non-optimal"

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_command_line()
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
    ws, ws_diag, results, to = FOMPrototypes.run_prototype(problem, warmup_config.problem_set, warmup_config.problem_name, warmup_config);

    println("--------------------")
    println("--------------------")

    println("ACTUAL run:")
    problem = FOMPrototypes.fetch_data(config.problem_set, config.problem_name);
    ws, ws_diag, results, to = FOMPrototypes.run_prototype(
        problem,
        config.problem_set,
        config.problem_name,
        config,
        full_diagnostics = SAVE_MATRIX
    );

    # save diagnostic matrices if requested:
    if SAVE_MATRIX
        save_diagnostic_matrix(ws_diag, config.problem_set, config.problem_name, MATRIX_TAG, config.variant, config.rho, config)
    end
end
