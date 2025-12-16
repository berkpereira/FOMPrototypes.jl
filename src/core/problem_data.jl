using JuMP, ClarabelBenchmarks, Clarabel, LinearAlgebra, SparseArrays

function load_clarabel_benchmark_prob_data(problem_set::String, problem_name::String)

    # Select problem and load data
    # Outer (first) dictionary is the problem class. Inner (second) is the problem name.
    println(keys(ClarabelBenchmarks.PROBLEMS))
    println(keys(ClarabelBenchmarks.PROBLEMS[problem_set]))
    problem = ClarabelBenchmarks.PROBLEMS[problem_set][problem_name]

    # This disables problem data-scaling.
    # We could enable it to get a better conditioned problem, but then any
    # but then any convergence checks would be checking something different 
    # compared to other solvers.
    optimizer = optimizer_with_attributes(Clarabel.Optimizer, "equilibrate_enable" => true)

    # create and populate a solver, but don't solve
    model = problem(Model(optimizer); solve = false)

    # extract the Clarabel solver object from the JuMP `model` wrapper 
    solver = model.moi_backend.optimizer.model.optimizer.solver

    # extract the problem data 
    P = solver.data.P # Clarabel stores diag + upper triangle only
    # Fill lower triangle to make a full symmetric sparse matrix
    # (mirror strictly upper part; keep diagonal as-is)
    P = P + transpose(triu(P, 1))
    A = solver.data.A
    (m, n) = size(A)
    c = solver.data.q
    b = solver.data.b
    K = solver.data.cones # K is a vector of cones, each cone is a dictionary

    # Return data as a named tuple for ease of access
    return (P = P, A = A, m = m, n = n, c = c, b = b, K = K)
end;

"""
Filters a list of `Problem` objects, returning only those that are suitable for the Clarabel solver.

# Arguments
- `problems::Vector{Problem}`: A vector of `Problem` objects to be filtered.

# Returns
- A vector of `Problem` objects that are compatible with the Clarabel solver.
"""
function filter_clarabel_problems(
    problem_set::String;
    min_m::Union{Int, Nothing} = nothing,
    max_m::Union{Int, Nothing} = nothing,
    min_n::Union{Int, Nothing} = nothing,
    max_n::Union{Int, Nothing} = nothing,
    min_num_cones::Union{Int, Nothing} = nothing,
    max_num_cones::Union{Int, Nothing} = nothing,
    admissible_cones::Union{Vector{DataType}, Nothing} = [Clarabel.NonnegativeConeT, Clarabel.ZeroConeT, Clarabel.SecondOrderConeT]
)
    # Retrieve all problems in the specified problem set
    all_problems = ClarabelBenchmarks.PROBLEMS[problem_set]

    # Initialize an empty list for matching problem names
    matching_problems = []

    # Loop through each problem in the problem set
    for (problem_name, problem) in all_problems
        # Load the problem data
        data = load_clarabel_benchmark_prob_data(problem_set, problem_name)
        P, A, m, n, c, b, K = data.P, data.A, data.m, data.n, data.c, data.b, data.K

        # Check dimensions (m and n)
        if !isnothing(min_m) && m < min_m
            continue
        end
        if !isnothing(max_m) && m > max_m
            continue
        end
        if !isnothing(min_n) && n < min_n
            continue
        end
        if !isnothing(max_n) && n > max_n
            continue
        end

        # Check the number of cones
        num_cones = length(K)
        if !isnothing(min_num_cones) && num_cones < min_num_cones
            continue
        end
        if !isnothing(max_num_cones) && num_cones > max_num_cones
            continue
        end

        # Check admissible cone types
        if !isnothing(admissible_cones)
            if any(cone -> !(typeof(cone) in admissible_cones), K)
                continue
            end
        end

        # If all conditions pass, add the problem name to the list
        push!(matching_problems, problem_name)
    end

    # Return the list of matching problems
    return matching_problems
end

"""
This function serves to parse the header line of the output .txt files
containing problem names as given by a search in the Clarabel benchmark
problems, as output by the function `filter_clarabel_problems`.
"""
function parse_search_header(filename)
    lines = readlines(filename)
    header = lines[1]
    if !startswith(header, "#SEARCH")
        error("Invalid header format")
    end
    
    # Extract criteria string between curly braces
    criteria = match(r"#SEARCH(\{.*\})", header)[1]
    # Parse criteria here as needed
    
    return lines[2:end]  # Return problem names
end;
;
