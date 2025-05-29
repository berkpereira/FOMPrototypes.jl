include("problem_data.jl")

search_problem_set = "mpc" # in keys(ClarabelBenchmarks.PROBLEMS)

# Define search criteria
MIN_M = 1
MAX_M = 100_000
MIN_N = 1
MAX_N = 100_000

suitable_problems = filter_clarabel_problems(
    search_problem_set,
    min_m = MIN_M,
    max_m = MAX_M,
    min_n = MIN_N,
    max_n = MAX_N
)

# Write results to a text file including an info header line.
names_file = "./problem_search_results/search_results_$search_problem_set.txt"
open(names_file, "w") do f
    # Write header with search criteria
    header = "#SEARCH{set:$search_problem_set,min_m:$MIN_M,max_m:$MAX_M,min_n:$MIN_N,max_n:$MAX_N}"
    println(f, header)
    
    # Write problem names
    for problem in suitable_problems
        println(f, problem)
    end
end

retrieved_problems = readlines(names_file)