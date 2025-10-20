include("../src/problem_data.jl")

SEARCH_PROBLEM_SET = "netlib_feasible" # in keys(ClarabelBenchmarks.PROBLEMS)

# Define search criteria
MIN_M = 1
MAX_M = 100_000
MIN_N = 1
MAX_N = 100_000

suitable_problems = filter_clarabel_problems(
    SEARCH_PROBLEM_SET,
    min_m = MIN_M,
    max_m = MAX_M,
    min_n = MIN_N,
    max_n = MAX_N
)

# Write results to a text file including an info header line.
bench_type = :spmv # bench_type in {:fom, :spmv}
if bench_type == :fom
    names_file = "./problem_search_results_fom/search_results_$SEARCH_PROBLEM_SET.txt"
elseif bench_type == :spmv
    names_file = "./problem_search_results_spmv/search_results_$SEARCH_PROBLEM_SET.txt"
else
    @error "Unrecognised bench type for problem search: $bench_type"
end

parent_dir = dirname(names_file)
if !isdir(parent_dir)
    mkpath(parent_dir)
end
open(names_file, "w") do f
    # Write header with search criteria
    header = "#SEARCH{set:$SEARCH_PROBLEM_SET,min_m:$MIN_M,max_m:$MAX_M,min_n:$MIN_N,max_n:$MAX_N}"
    println(f, header)
    
    # Write problem names
    for problem in suitable_problems
        println(f, problem)
    end
end

retrieved_problems = readlines(names_file)