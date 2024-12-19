begin

include("types.jl")
include("solver.jl")
include("utils.jl")
include("problem_data.jl")
include("solver.jl")
include("printing.jl")
include("plotting.jl")
include("acceleration.jl")


using Revise
using Profile
using StatProfilerHTML
using BenchmarkTools
using Printf
using Plots
using SparseArrays
using SCS
using Random

# Set Plots backend.
# Use plotlyjs() for interactive plots. Use gr() for faster generation.
plotlyjs();

newline_char = Plots.backend_name() == :gr ? "\n" : "<br>"

# Set default plot size (in pixels)
# default(size=(800, 900)); # large; for laptop
default(size=(800, 700))

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

problem_option = :LASSO; # in {:LASSO, :HUBER, :MAROS}

if problem_option === :LASSO
    problem_set = "sslsq";
    # problem_name = "NYPA_Maragal_5_lasso"; # good size, challenging
    # problem_name = "NYPA_Maragal_1_lasso"; # smallest in SSLSQ set
    problem_name = "HB_ash219_lasso"; # smallish
elseif problem_option === :HUBER
    problem_set = "sslsq";
    problem_name = "HB_ash958_huber"; # (m, n) = (3419, 3099)
    # problem_name = "NYPA_Maragal_5_huber"; # large, challenging
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
problem = ProblemData(P, c, A, b, K);

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

# set_optimizer_attribute(model, "eps_abs", 1e-10)
# set_optimizer_attribute(model, "eps_rel", 1e-10)

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

MAX_ITERS = 600;
PRINT_MOD = 50;
RES_NORM = Inf;
RESTART_PERIOD = Inf;
ACCEL_MEMORY = 39;
ACCELERATION = true;

# Choose primal step size as a proportion of maximum allowable to keep M1 PSD
Random.seed!(42) # Seed for reproducible power iteration results.
max_τ = 1 / dom_λ_power_method(Matrix(take_away), 10);
τ = 0.90 * max_τ;


println("Running prototype variant $variant...")
println("Restart period: $RESTART_PERIOD")

end

# @profview begin # For built-in profiler view in VS Code.
@time begin

# Initialise workspace (initial iterates are set to zero by default).
ws = Workspace(problem, variant, τ, ρ)
ws.cache[:A_gram] = A_gram

# Call the solver.
primal_objs, dual_objs, primal_residuals, dual_residuals, enforced_set_flags, x_dist_to_sol, s_dist_to_sol, y_dist_to_sol = optimise!(ws, MAX_ITERS, PRINT_MOD, RESTART_PERIOD, RES_NORM, ACCELERATION,
ACCEL_MEMORY, x_scs, s_scs, y_scs, false);

end;


################################################################################
################################# PLOT STUFF ###################################
################################################################################

begin

display(plot(primal_objs, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="Variant $variant: Objective.$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION"))

display(plot(dual_objs, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value", title="Variant $variant: Dual objective.$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION"))

display(plot(primal_objs - dual_objs, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap", title="Variant $variant: Duality Gap.$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION"))

display(plot(primal_residuals, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm.$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION", yaxis=:log))

display(plot(dual_residuals, label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual", title="Variant $variant: Dual Residual Norm.$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION", yaxis=:log))

# display(plot(x_dist_to_sol / sqrt(ws.p.n), label="Prototype x Distance", xlabel="Iteration", ylabel="Distance to Solution", title="Variant $variant.$newline_char 'Normalised' x Distance to Solution.$newline_char Restart period = $RESTART_PERIOD$newline_char Acceleration: $ACCELERATION", yaxis=:log))

# display(plot(s_dist_to_sol / sqrt(ws.p.m), label="Prototype s Distance", xlabel="Iteration", ylabel="Distance to Solution", title="Variant $variant.$newline_char 'Normalised' s Distance to Solution.$newline_char Restart period = $RESTART_PERIOD$newline_char Acceleration: $ACCELERATION", yaxis=:log))

# display(plot(y_dist_to_sol / sqrt(ws.p.m), label="Prototype y Distance", xlabel="Iteration", ylabel="Distance to Solution", title="Variant $variant.$newline_char 'Normalised' y Distance to Solution.$newline_char Restart period = $RESTART_PERIOD$newline_char Acceleration: $ACCELERATION", yaxis=:log))

# concat_dist_to_sol = sqrt.(x_dist_to_sol .^ 2 .+ s_dist_to_sol .^ 2 .+ y_dist_to_sol .^ 2)
# display(plot(concat_dist_to_sol / sqrt(ws.p.n + 2 * ws.p.m), label="Prototype Concatenated Distance", xlabel="Iteration", ylabel="Distance to Solution", title="Variant $variant.$newline_char 'Normalised' Concatenated Distance to Solution.$newline_char Restart period = $RESTART_PERIOD$newline_char Acceleration: $ACCELERATION", yaxis=:log))

# enforced_constraints_plot(enforced_set_flags, 10)

display(plot_equal_segments(enforced_set_flags))

end




################################################################################
# STEP ANGLE STUFF
################################################################################

begin

EXP_SMOOTHING_PARAMETER = 0.95

plot(exp_moving_average(x_step_angles, EXP_SMOOTHING_PARAMETER), label="x Step Angle", xlabel="Iteration", ylabel="Angle (radians)", title="Variant $variant: Step Angles.$newline_char Restart period = $RESTART_PERIOD")
plot!(exp_moving_average(s_step_angles, EXP_SMOOTHING_PARAMETER), label="s Step Angle")
plot!(exp_moving_average(y_step_angles, EXP_SMOOTHING_PARAMETER), label="y Step Angle")
plot!(exp_moving_average(concat_step_angles, EXP_SMOOTHING_PARAMETER), label="Concatenated Step Angle")
display(plot!(exp_moving_average(normalised_concat_step_angles, EXP_SMOOTHING_PARAMETER), label="NORMALISED Concatenated Step Angle"))


# Running sums of angles

plot(cumsum(x_step_angles), label="x Step Angle", xlabel="Iteration", ylabel="Cumulative Angle (radians)", title="Variant $variant: Cumulative Step Angles.$newline_char Restart period = $RESTART_PERIOD")

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