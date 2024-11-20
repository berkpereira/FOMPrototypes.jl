using JuMP, ClarabelBenchmarks

function load_clarabel_benchmark_prob_data(problem_set::String, problem_name::String)

    # Select problem and load data
    # Outer (first) dictionary is the problem class. Inner (second) is the problem name.
    problem = ClarabelBenchmarks.PROBLEMS[problem_set][problem_name]

    # This disables problem data-scaling.
    # We could enable it to get a better conditioned problem, but then any
    # but then any convergence checks would be checking something different 
    # compared to other solvers.
    optimizer = optimizer_with_attributes(Clarabel.Optimizer,"equilibrate_enable"=>false)

    # create and populate a solver, but don't solve
    model = problem(Model(optimizer); solve = false) 

    # extract the Clarabel solver object from the JuMP `model` wrapper 
    solver = model.moi_backend.optimizer.model.optimizer.solver

    # extract the problem data 
    P = solver.data.P
    A = solver.data.A
    (m, n) = size(A)
    c = solver.data.q
    b = solver.data.b
    K = solver.data.cones # K is a list of cones, each cone is a dictionary

    # Return data as a named tuple for ease of access
    return (P = P, A = A, m = m, n = n, c = c, b = b, K = K)
end;
;