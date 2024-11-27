begin

include("utils.jl")
include("problem_data.jl")
include("solver.jl")
include("printing.jl")

using Revise
using .PrototypeMethod
using BenchmarkTools
using Printf
using Plots
using SparseArrays
plotlyjs();

# Set default plot size (in pixels)
default(size=(800, 700));

end

############################## FIND PROBLEMS ###################################

# search_problem_set = "sslsq"

# suitable_problems = filter_clarabel_problems(search_problem_set, min_m = 1, min_n = 100)

# # Write to a file whose name depends on search_problem_set string
# names_file = "./search_results_$search_problem_set.txt"
# open(names_file, "w") do f
#     for problem in suitable_problems
#         println(f, problem)
#     end
# end

# retrieved_problems = readlines(names_file)


############## CHOOSE PROBLEM SET AND PROBLEM NAME #############################

# problem_set = "maros";
# problem_name = "HUESTIS";

problem_option = :LASSO; # in {:LASSO, :HUBER, :MAROS}

if problem_option === :LASSO
    problem_set = "sslsq";
    problem_name = "NYPA_Maragal_5_lasso"; # good size, challenging
elseif problem_option === :HUBER
    problem_set = "sslsq";
    problem_name = "NYPA_Maragal_5_huber"; # good size, challenging
elseif problem_option === :MAROS
    problem_set = "maros";
    problem_name = "QSCSD8"; # not as large, n = 1500, m = 900
else
    error("Invalid problem option")
end


############################## FETCH DATA ######################################

begin

data = load_clarabel_benchmark_prob_data(problem_set, problem_name);
P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K;

# Create a problem instance.
problem = PrototypeMethod.QPProblem(data.P, data.c, data.A, data.b, data.K);

end;
############################### SCS SOLUTION ###################################

begin
using SCS

println("\nRunning SCS...")

# Define the SCS optimizer
model = Model(SCS.Optimizer);
# set_silent(model) # Suppress output
# Define primal variables x and s
@variable(model, x_scs[1:n]);    # Primal variables in R^n
@variable(model, s_scs[1:m]);    # Slack variables in R^m

# Add constraints for Ax + s = b
@constraint(model, con, A * x_scs + s_scs .== b);

# Add cone constraints to the model
add_cone_constraints_scs!(model, s_scs, K);

# Add the quadratic objective to the model
@objective(model, Min, 0.5 * dot(x_scs, P * x_scs) + dot(c, x_scs));

# Solve the problem
JuMP.optimize!(model);

# Extract solutions
x_scs = value.(x_scs);
s_scs = value.(s_scs);
y_scs = dual.(con);  # Dual variables (Lagrange multipliers)
obj_scs = objective_value(model);

end


############################# PROTOTYPE SOLUTION ###############################

begin

# Step size parameters chosen by the user
ρ = 1.0; # (dual)

# Choose algorithm based on M1 construction
# int in {1, 2, 3, 4}
variant = 1;
A_gram = A' * A;
take_away = take_away_matrix(variant, A_gram);

MAX_ITERS = 600;
PRINT_MOD = 50;
RESTART_PERIOD = 40;
RETURN_RUN_DATA = true;

# Choose primal step size as a proportion of maximum allowable to keep M1 PSD
max_τ = 1 / dom_λ_power_method(Matrix(take_away), 30);
τ = 0.9 * max_τ

println("Running prototype variant $variant...")
println("Restart period: $RESTART_PERIOD")

x = zeros(n);
s = zeros(m);
y = zeros(m);

if RETURN_RUN_DATA
    primal_objs, dual_objs, primal_residuals, dual_residuals, x_step_angles, s_step_angles, y_step_angles, concat_step_angles, normalised_concat_step_angles = PrototypeMethod.optimise!(problem, variant, x, s, y, τ, ρ, A_gram, MAX_ITERS, PRINT_MOD, RESTART_PERIOD, Inf, RETURN_RUN_DATA);
else
    final_primal_obj, final_dual_obj = PrototypeMethod.optimise!(problem, variant, x, s, y, τ, ρ, A_gram, MAX_ITERS, PRINT_MOD, RESTART_PERIOD, Inf, RETURN_RUN_DATA);
end

end;


################################################################################
################################# PLOT STUFF ###################################
################################################################################

begin

EXP_SMOOTHING_PARAMETER = 0.9

display(plot(0:MAX_ITERS, primal_objs, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="Variant $variant: Objective. Restart period = $RESTART_PERIOD"))

display(plot(0:MAX_ITERS, dual_objs, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value", title="Variant $variant: Dual Objective. Restart period = $RESTART_PERIOD"))

plot(0:MAX_ITERS, primal_residuals, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm. Restart period = $RESTART_PERIOD")
# overlay plot of same signal but with exp_moving_average applied
display(plot!(exp_moving_average(primal_residuals, EXP_SMOOTHING_PARAMETER), label="Prototype Residual (Exp Moving Average)", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm. Restart period = $RESTART_PERIOD"))

display(plot(0:MAX_ITERS, dual_residuals, label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual", title="Variant $variant: Dual Residual Norm. Restart period = $RESTART_PERIOD"))

end

# STEP ANGLE STUFF

begin

plot(exp_moving_average(x_step_angles, EXP_SMOOTHING_PARAMETER), label="x Step Angle", xlabel="Iteration", ylabel="Angle (radians)", title="Variant $variant: Step Angles. Restart period = $RESTART_PERIOD")
plot!(s_step_angles, label="s Step Angle")
plot!(exp_moving_average(y_step_angles, EXP_SMOOTHING_PARAMETER), label="y Step Angle")
plot!(exp_moving_average(concat_step_angles, EXP_SMOOTHING_PARAMETER), label="Concatenated Step Angle")
display(plot!(exp_moving_average(normalised_concat_step_angles, EXP_SMOOTHING_PARAMETER), label="NORMALISED Concatenated Step Angle"))
end

# PLOT RUNNING SUMS OF ANGLES
begin
    plot(cumsum(x_step_angles), label="x Step Angle", xlabel="Iteration", ylabel="Cumulative Angle (radians)", title="Variant $variant: Cumulative Step Angles. Restart period = $RESTART_PERIOD")

    plot!(cumsum(s_step_angles), label="s Step")

    plot!(cumsum(y_step_angles), label="y Step")
end

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