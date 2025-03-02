include("problem_data.jl")
include("types.jl")
include("optimal_diagonal_shift.jl")
include("utils.jl")
using CSV, DataFrames, Clarabel, ClarabelBenchmarks
using Printf
using Plots
using SparseArrays
using SCS
using Random
using JLD2
using Pkg

function process_problem(problem_set, problem_name; ρ::Float64)
    # Load the problem data
    data = load_clarabel_benchmark_prob_data(problem_set, problem_name)
    P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K
    problem = ProblemData(P, c, A, b, K)
    
    # Set up common computations
    A_gram = A' * A
    
    # Compute eigenvalue differences for each variant
    eig_diffs = zeros(4)
    for variant in 1:4
        # Compute take_away_mat based on variant
        take_away_mat = if variant == 1
            off_diag_part(P + ρ * A_gram)
        elseif variant == 2
            P + ρ * A_gram
        elseif variant == 3
            P + off_diag_part(ρ * A_gram)
        else # variant == 4
            off_diag_part(P) + ρ * A_gram
        end
        
        S = -take_away_mat
        spec_S = eigvals(Matrix(S))
        eig_diffs[variant] = maximum(spec_S) - minimum(spec_S)
    end
    
    # Solve SDP using variant 1 (arbitrary choice as the 
    # SDP optimal value is independent of the variant chosen from the 4)
    S = -(P + ρ * A_gram)  # variant 1
    
    # Note that the max_τ also does NOT affect the optimal objective value,
    # so it is irrelevant in this context. This is why we set it to 1.0 here.
    println("About to call solve_spectral_radius_sdp")
    result = solve_spectral_radius_sdp(S, 1.0, solver=:Clarabel, time_limit = 5.0)
    
    # Return results
    return (
        problem_set = problem_set,
        problem_name = problem_name,
        opt_spectral_radius = result.status == OPTIMAL ? result.objective : NaN,
        eig_diff_v1 = eig_diffs[1],
        eig_diff_v2 = eig_diffs[2],
        eig_diff_v3 = eig_diffs[3],
        eig_diff_v4 = eig_diffs[4]
    )
end

function run_preprocessing_analysis(results_filename; ρ::Float64)
    # Read the search results file
    lines = readlines(results_filename)
    
    # SKIP header line (starts with #SEARCH)
    problem_names = lines[2:end]
    
    # Extract problem set from header, also convert to String type
    # (otherwise this is of SubString type).
    problem_set = String(match(r"set:(\w+)", lines[1])[1])
    
    # Process each problem and collect results
    results = []
    for problem_name in problem_names
        @info "Processing $problem_name"
        try
            result = process_problem(problem_set, problem_name, ρ=ρ)
            push!(results, result)
        catch e
            @warn "Failed to process $problem_name" exception=e
        end
    end
    
    # Convert results to DataFrame and save
    df = DataFrame(results)
    output_file = "preprocessing_results_$problem_set.csv"
    CSV.write(output_file, df)
    
    return df
end

########################## RUN SPECTRAL RADII ANALYSIS #########################

problem_set = "maros"
results = run_preprocessing_analysis("./search_results_$problem_set.txt", ρ = 1.0)