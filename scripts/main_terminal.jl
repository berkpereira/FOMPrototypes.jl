# this file is used to run solver code directly from a terminal command
using Revise
import FOMPrototypes

if abspath(PROGRAM_FILE) == @__FILE__
    config = FOMPrototypes.parse_command_line()
    println()

    warmup_config = deepcopy(config)
    warmup_config["problem-set"] = "sslsq"
    warmup_config["problem-name"] = "NYPA_Maragal_3_huber"
    warmup_config["max-iter"] = 300
    warmup_config["global-timeout"] = 10.0
    warmup_config["loop-timeout"] = 10.0
    warmup_config["rel-kkt-tol"] = 0.0

    println("WARMUP run:")
    problem = FOMPrototypes.fetch_data(warmup_config["problem-set"], warmup_config["problem-name"]);
    ws, results, to = FOMPrototypes.run_prototype(problem, warmup_config["problem-set"], warmup_config["problem-name"], warmup_config);

    println("--------------------")
    println("--------------------")

    println("ACTUAL run:")
    problem = FOMPrototypes.fetch_data(config["problem-set"], config["problem-name"]);
    ws, results, to = FOMPrototypes.run_prototype(problem, config["problem-set"], config["problem-name"], config);
end