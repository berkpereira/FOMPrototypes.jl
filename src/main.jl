begin

include("utils.jl")
include("problem_data.jl")
include("solver.jl")
include("printing.jl")
include("acceleration.jl")

using Revise
using .PrototypeMethod
using BenchmarkTools
using Printf
using Plots
using SparseArrays
using SCS
plotlyjs();

# Set default plot size (in pixels)
# default(size=(800, 900)); # large; for laptop
default(size=(600, 800))

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

begin

problem_option = :MAROS; # in {:LASSO, :HUBER, :MAROS}

if problem_option === :LASSO
    problem_set = "sslsq";
    problem_name = "NYPA_Maragal_5_lasso"; # good size, challenging
    # problem_name = "NYPA_Maragal_1_lasso"; # smallest in SSLSQ set
    # problem_name = "HB_ash219_lasso"; # smaller
elseif problem_option === :HUBER
    problem_set = "sslsq";
    problem_name = "NYPA_Maragal_5_huber"; # good size, challenging
elseif problem_option === :MAROS
    problem_set = "maros";
    problem_name = "QSCSD8"; # not as large, n = 1500, m = 900
    # problem_name = "DUAL3";
else
    error("Invalid problem option")
end

end;

############################## FETCH DATA ######################################

begin

# Load the problem data.
data = load_clarabel_benchmark_prob_data(problem_set, problem_name);
P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K;

# Create a problem instance.
problem = PrototypeMethod.QPProblem(P, c, A, b, K);

end;
############################### SCS SOLUTION ###################################

begin

println("\nRunning SCS...")
println()

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

end;


############################# PROTOTYPE SOLUTION ###############################

begin

# Step size parameters chosen by the user
ρ = 1.0; # (dual)

# Choose algorithm based on M1 construction
# int in {1, 2, 3, 4}
variant = 1;
A_gram = A' * A;
take_away = take_away_matrix(variant, A_gram);

MAX_ITERS = 3_000;
PRINT_MOD = 50;
RES_NORM = Inf;
RESTART_PERIOD = 840;
RETURN_RUN_DATA = true;
ACCELERATION = true;

# Choose primal step size as a proportion of maximum allowable to keep M1 PSD
max_τ = 1 / dom_λ_power_method(Matrix(take_away), 30);
# max_τ = 1 / maximum(eigvals(Matrix(take_away)));
τ = 0.9 * max_τ;

println("Running prototype variant $variant...")
println("Restart period: $RESTART_PERIOD")

x = zeros(n);
s = zeros(m);
y = zeros(m);

if RETURN_RUN_DATA
    primal_objs, dual_objs, primal_residuals, dual_residuals, x_step_angles, s_step_angles, y_step_angles, concat_step_angles, normalised_concat_step_angles = PrototypeMethod.optimise!(problem, variant, x, s, y, τ, ρ, A_gram, MAX_ITERS, PRINT_MOD, RESTART_PERIOD, RES_NORM, RETURN_RUN_DATA, ACCELERATION);
else
    final_primal_obj, final_dual_obj = PrototypeMethod.optimise!(problem, variant, x, s, y, τ, ρ, A_gram, MAX_ITERS, PRINT_MOD, RESTART_PERIOD, RES_NORM, RETURN_RUN_DATA, ACCELERATION);
end

end;


################################################################################
################################# PLOT STUFF ###################################
################################################################################

begin

EXP_SMOOTHING_PARAMETER = 0.95

display(plot(primal_objs, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="Variant $variant: Objective.<br>Restart period = $RESTART_PERIOD"))

display(plot(dual_objs, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value", title="Variant $variant: Dual objective.<br>Restart period = $RESTART_PERIOD"))

display(plot(primal_objs - dual_objs, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap", title="Variant $variant: Duality Gap.<br>Restart period = $RESTART_PERIOD"))

plot(primal_residuals, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm.<br>Restart period = $RESTART_PERIOD", yaxis=:log)
# overlay plot of same signal but with exp_moving_average applied
display(plot!(exp_moving_average(primal_residuals, EXP_SMOOTHING_PARAMETER), label="Prototype Residual (Exp Moving Average)", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm.<br>Restart period = $RESTART_PERIOD", yaxis=:log))

display(plot(dual_residuals, label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual", title="Variant $variant: Dual Residual Norm.<br>Restart period = $RESTART_PERIOD", yaxis=:log))

end

# STEP ANGLE STUFF

begin



plot(exp_moving_average(x_step_angles, EXP_SMOOTHING_PARAMETER), label="x Step Angle", xlabel="Iteration", ylabel="Angle (radians)", title="Variant $variant: Step Angles.<br>Restart period = $RESTART_PERIOD")
plot!(exp_moving_average(s_step_angles, EXP_SMOOTHING_PARAMETER), label="s Step Angle")
plot!(exp_moving_average(y_step_angles, EXP_SMOOTHING_PARAMETER), label="y Step Angle")
plot!(exp_moving_average(concat_step_angles, EXP_SMOOTHING_PARAMETER), label="Concatenated Step Angle")
display(plot!(exp_moving_average(normalised_concat_step_angles, EXP_SMOOTHING_PARAMETER), label="NORMALISED Concatenated Step Angle"))


# Running sums of angles

plot(cumsum(x_step_angles), label="x Step Angle", xlabel="Iteration", ylabel="Cumulative Angle (radians)", title="Variant $variant: Cumulative Step Angles.<br>Restart period = $RESTART_PERIOD")

plot!(cumsum(s_step_angles), label="s Step")

plot!(cumsum(y_step_angles), label="y Step")

plot!(cumsum(concat_step_angles), label="Concatenated Step")

display(plot!(cumsum(normalised_concat_step_angles), label="NORMALISED Concatenated Step"))
end

################################################################################
############################# ANALYSE RESIDUAL DATA? ###########################
################################################################################

# using FFTW

# # Compute the FFT of primal_residuals
# fft_result = fft(primal_residuals)

# # Compute the magnitudes of the FFT components
# magnitudes = abs.(fft_result)

# # Find the index of the frequency with the largest magnitude
# max_index = argmax(magnitudes)

# # Determine the corresponding frequency (in normalized units)
# N = length(primal_residuals)  # Number of data points
# frequencies = (0:N-1) / N     # Normalized frequencies
# dominant_frequency = frequencies[max_index]

# println("Dominant Frequency (normalized): $dominant_frequency")#