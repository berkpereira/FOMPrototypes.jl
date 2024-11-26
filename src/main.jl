include("utils.jl")
include("problem_data.jl")
include("solver.jl")
include("printing.jl")

using .PrototypeMethod
using BenchmarkTools
using Printf
using Plots

############################## FIND PROBLEMS ###################################

search_problem_set = "sslsq"

suitable_problems = filter_clarabel_problems(search_problem_set, min_m = 1, min_n = 100)

# Write to a file whose name depends on search_problem_set string
names_file = "./search_results_$search_problem_set.txt"
open(names_file, "w") do f
    for problem in suitable_problems
        println(f, problem)
    end
end

retrieved_problems = readlines(names_file)


############## CHOOSE PROBLEM SET AND PROBLEM NAME #############################

# problem_set = "maros";
# problem_name = "HUESTIS";

problem_set = "sslsq";
problem_name = "NYPA_Maragal_5_lasso"
# problem_name = "HB_ash219_lasso" # pretty good solution from prototype
# problem_name = "HB_ash85_huber" # very slow convergence w prototype
# problem_name = "HB_ash85_lasso" # stagnates badly at about 2e-3 objective error from SCS solution


############################## FETCH DATA ######################################

data = load_clarabel_benchmark_prob_data(problem_set, problem_name);
P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K;

############################### SCS SOLUTION ###################################

using SCS

println("\nRunning SCS...")

# Define the SCS optimizer
model = Model(SCS.Optimizer)
# set_silent(model) # Suppress output
# Define primal variables x and s
@variable(model, x_scs[1:n]);    # Primal variables in R^n
@variable(model, s_scs[1:m]);    # Slack variables in R^m

# Add constraints for Ax + s = b
@constraint(model, con, A * x_scs + s_scs .== b);

# Add cone constraints to the model
add_cone_constraints_scs!(model, s_scs, K)

# Add the quadratic objective to the model
@objective(model, Min, 0.5 * dot(x_scs, P * x_scs) + dot(c, x_scs));

# Solve the problem
JuMP.optimize!(model);

# Extract solutions
x_scs = value.(x_scs);
s_scs = value.(s_scs);
y_scs = dual.(con);  # Dual variables (Lagrange multipliers)
obj_scs = objective_value(model);
;

############################# PROTOTYPE SOLUTION ###############################


# Step size parameters chosen by the user
ρ = 1.0; # (dual)

# Choose algorithm based on M1 construction
# int in {1, 2, 3, 4}
variant = 1
A_gram = A' * A
take_away = take_away_matrix(variant, A_gram);

max_iters = 500;
print_modulo = 10;

# Choose primal step size as a proportion of maximum allowable to keep M1 PSD
max_τ = 1 / dom_λ_power_method(Matrix(take_away), 10)
τ = 0.9 * max_τ

# Create a problem instance
problem = PrototypeMethod.QPProblem(data.P, data.c, data.A, data.b, data.K);

# Initialize sequences x, s, and y
x = zeros(n);
s = zeros(m);
y = zeros(m);

# Initialise metrics storage
# obj_vals = Float64[];
# primal_residuals = Float64[];
# dual_residuals = Float64[];

# Main iteration loop
println("Running prototype variant $variant...")

PrototypeMethod.optimise!(problem, variant, x, s, y, τ, ρ,
A_gram, max_iters, print_modulo, 50);


################################################################################
################################# PLOT STUFF ###################################
################################################################################

# Plot objective values
plot(0:max_iters, obj_vals, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="Variant $variant: Objective")

# Plot primal residuals
plot(0:max_iters, primal_residuals, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm")

############################# ANALYSE RESIDUAL DATA ############################

using FFTW

# Compute the FFT of primal_residuals
fft_result = fft(primal_residuals)

# Compute the magnitudes of the FFT components
magnitudes = abs.(fft_result)

# Find the index of the frequency with the largest magnitude
max_index = argmax(magnitudes)

# Determine the corresponding frequency (in normalized units)
N = length(primal_residuals)  # Number of data points
frequencies = (0:N-1) / N     # Normalized frequencies
dominant_frequency = frequencies[max_index]

println("Dominant Frequency (normalized): $dominant_frequency")