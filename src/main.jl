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
using JLD2
using Pkg

# Set Plots backend.
# For interactive plots: plotlyjs()
# For faster plotting: gr()
gr()

newline_char = Plots.backend_name() in [:gr, :pythonplot] ? "\n" : "<br>"

# Set default plot size (in pixels)
default(size=(1100, 700)) # for desktop
# default(size=(1200, 800)) # for laptop

end

############## CHOOSE PROBLEM SET AND PROBLEM NAME #############################

begin

problem_option = :GISELSSON # in {:LASSO, :HUBER, :MAROS, :GISELSSON}

if problem_option === :LASSO
    problem_set = "sslsq";
    # problem_name = "NYPA_Maragal_5_lasso"; # large, challenging
    problem_name = "HB_abb313_lasso" # (m, n) = (665, 665)
    # problem_name = "HB_ash219_lasso"; # (m, n) = (389, 389)
    # problem_name = "NYPA_Maragal_1_lasso"; # smallest in SSLSQ set
elseif problem_option === :HUBER
    problem_set = "sslsq";
    problem_name = "HB_ash958_huber"; # (m, n) = (3419, 3099)
    # problem_name = "NYPA_Maragal_5_huber"; # large, challenging
elseif problem_option === :MAROS
    problem_set = "maros";
    # problem_name = "DUAL3"; # large
    problem_name = "QSCSD8"; # not as large, (m, n) = (3147, 2750)
    # problem_name = "HS76" # VERY tiny
elseif problem_option === :GISELSSON
    problem_set = "giselsson";
    problem_name = "giselsson_problem";
else
    error("Invalid problem option")
end

end;

############################## FETCH DATA ######################################

begin

# Load the problem data.
if problem_option !== :GISELSSON
    data = load_clarabel_benchmark_prob_data(problem_set, problem_name);
else
    repo_root = dirname(Pkg.project().path)
    giselsson_path = joinpath(repo_root, "synthetic_problem_data/giselsson_problem.jld2")
    data = load(giselsson_path)["data"];
end
P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K;

# Create a problem instance.
problem = ProblemData(P, c, A, b, K);

end;

########################### Clarabel/SCS SOLUTION ##############################

begin

reference_solver = :Clarabel # in {:SCS, :Clarabel}

if reference_solver === :SCS
    println()
    println("RUNNING SCS...")
    println()
    model = Model(SCS.Optimizer);
elseif reference_solver === :Clarabel
    println()
    println("RUNNING CLARABEL...")
    println()
    model = Model(Clarabel.Optimizer);
end
println("Problem set/name: $problem_set/$problem_name")

# set_silent(model) # Suppress output
# Define primal variables x and s
@variable(model, x_ref[1:n]);    # Primal variables in R^n
@variable(model, s_ref[1:m]);    # Slack variables in R^m

# Add constraints for Ax + s = b
@constraint(model, con, A * x_ref + s_ref .== b);

# Add cone constraints to the model
add_cone_constraints!(model, s_ref, K);

if reference_solver === :SCS
    set_optimizer_attribute(model, "eps_abs", 1e-12)
    set_optimizer_attribute(model, "eps_rel", 1e-12)
end

# Add the quadratic objective to the model
@objective(model, Min, 0.5 * dot(x_ref, P * x_ref) + dot(c, x_ref));

# Solve the problem
JuMP.optimize!(model);

# Extract solutions
x_ref = value.(x_ref);
s_ref = value.(s_ref);
y_ref = dual.(con);  # Dual variables (Lagrange multipliers)
obj_ref = objective_value(model);
end;


############################# PROTOTYPE SOLUTION ###############################

begin

# Step size parameters chosen by the user
ρ = 1.0;

# Choose algorithm based on M1 construction
variant = 1; # int in {1, 2, 3, 4}
A_gram = A' * A;
if variant == 1 # NOTE: most "economical"
    take_away_mat = off_diag_part(P + ρ * A_gram)
elseif variant == 2 # NOTE: least "economical". Equivalent to linearised PDHG.
    take_away_mat = P + ρ * A_gram
elseif variant == 3 # NOTE: intermediate
    take_away_mat = P + off_diag_part(ρ * A_gram)
elseif variant == 4 # NOTE: also intermediate
    take_away_mat = off_diag_part(P) + ρ * A_gram
else
    error("Invalid variant.")
end

MAX_ITER = 1000
PRINT_MOD = 50
RES_NORM = Inf
RESTART_PERIOD = Inf
ACCEL_MEMORY = 19
ACCELERATION = false
LINESEARCH_PERIOD = 20
KRYLOV_OPERATOR_TILDE_A = false

LINESEARCH_ϵ = 0.03

COMPUTE_OPERATOR_EXPLICITLY = false
SPEC_PLOT_PERIOD = 26

# Choose primal step size as a proportion of maximum allowable to keep M1 PSD
Random.seed!(42) # Seed for reproducible power iteration results.
max_τ = 1 / dom_λ_power_method(Matrix(take_away_mat), 30);
τ = 0.90 * max_τ;

println("RUNNING PROTOTYPE VARIANT $variant...")
println("Problem set/name: $problem_set/$problem_name")
println("Restart period: $RESTART_PERIOD")
println("Acceleration: $ACCELERATION")
if ACCELERATION
    println("Acceleration memory: $ACCEL_MEMORY")
end

end

# @profview begin # For built-in profiler view in VS Code.
@time begin # for single run and timing

# Initialise workspace (initial iterates are set to zero by default).
ws = Workspace(problem, variant, τ, ρ)
ws.cache[:A_gram] = A_gram

# Call the solver.
results = optimise!(ws,
MAX_ITER,
PRINT_MOD,
restart_period = RESTART_PERIOD,
residual_norm = RES_NORM,
acceleration = ACCELERATION,
acceleration_memory = ACCEL_MEMORY,
linesearch_period = LINESEARCH_PERIOD,
linesearch_ϵ = LINESEARCH_ϵ,
krylov_operator_tilde_A = KRYLOV_OPERATOR_TILDE_A,
x_sol = x_ref, s_sol = s_ref, y_sol = y_ref,
explicit_affine_operator = COMPUTE_OPERATOR_EXPLICITLY,
spectrum_plot_period = SPEC_PLOT_PERIOD)

end;


################################################################################
################################# PLOT STUFF ###################################
################################################################################

begin

LINEWIDTH = 2.5
VERT_LINEWIDTH = 1.5
ALPHA = 0.9

title_beginning = "Problem: $problem_set $problem_name.$newline_char Variant $variant $newline_char"
title_end = "$newline_char Restart period = $RESTART_PERIOD.$newline_char Acceleration: $ACCELERATION (memory = period = $ACCEL_MEMORY).$newline_char Linesearch period = $LINESEARCH_PERIOD."
if KRYLOV_OPERATOR_TILDE_A
    krylov_operator_str = "$newline_char Krylov operator is A"
else
    krylov_operator_str = "$newline_char Krylov operator is B = A – I"
end

constraint_lines = constraint_changes(results.enforced_set_flags)

primal_obj_plot = plot(0:MAX_ITER + 1, results.primal_obj_vals, linewidth = LINEWIDTH, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="$title_beginning Objective $krylov_operator_str $title_end")
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(primal_obj_plot)

dual_obj_plot = plot(0:MAX_ITER + 1, results.dual_obj_vals, linewidth = LINEWIDTH, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Dual Objective Value", title="$title_beginning Dual objective $krylov_operator_str $title_end")
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(dual_obj_plot)

gap_plot = plot(0:MAX_ITER + 1, results.primal_obj_vals - results.dual_obj_vals, linewidth = LINEWIDTH, label="Prototype Dual Objective", xlabel="Iteration", ylabel="Duality Gap", title="$title_beginning Duality Gap $krylov_operator_str $title_end")
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(gap_plot)

pres_plot = plot(0:MAX_ITER + 1, results.pri_res_norms, linewidth = LINEWIDTH, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="$title_beginning Primal Residual Norm $krylov_operator_str $title_end", yaxis=:log)
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(pres_plot)

dres_plot = plot(0:MAX_ITER + 1, results.dual_res_norms, linewidth = LINEWIDTH, label="Prototype Dual Residual", xlabel="Iteration", ylabel="Dual Residual", title="$title_beginning Dual Residual Norm $krylov_operator_str $title_end", yaxis=:log)
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(dres_plot)

# display(plot(results.x_dist_to_sol / sqrt(ws.p.n), label="Prototype x Distance", xlabel="Iteration", ylabel="Distance to Solution", title="$title_beginning 'Normalised' x Distance to Solution $title_end", yaxis=:log))

# display(plot(results.s_dist_to_sol / sqrt(ws.p.m), label="Prototype s Distance", xlabel="Iteration", ylabel="Distance to Solution", title="$title_beginning 'Normalised' s Distance to Solution $title_end", yaxis=:log))

# display(plot(results.y_dist_to_sol / sqrt(ws.p.m), label="Prototype y Distance", xlabel="Iteration", ylabel="Distance to Solution", title="$title_beginning 'Normalised' y Distance to Solution $title_end", yaxis=:log))

# Plot characteristic seminorm to optimiser.
seminorm_plot = plot(0:MAX_ITER + 1, results.xy_semidist, linewidth = LINEWIDTH, label="Prototype Seminorm Distance (Theory)", xlabel="Iteration", ylabel="Distance to Solution", title="$title_beginning Seminorm Distance to Solution (Theory) $krylov_operator_str $title_end", yaxis=:log)
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(seminorm_plot)

xv_dist_to_sol = sqrt.(results.x_dist_to_sol .^ 2 .+ results.v_dist_to_sol .^ 2)
xv_dist_plot = plot(0:MAX_ITER + 1, xv_dist_to_sol, linewidth = LINEWIDTH, label="Prototype Concatenated Distance", xlabel="Iteration", ylabel="Distance to Solution", title="$title_beginning (x, v) Distance to Solution $krylov_operator_str $title_end", yaxis=:log)
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(xv_dist_plot)

# enforced_constraints_plot(enforced_set_flags, 10)

# Plot norms of xv steps from data in results. Display immediately.
xv_step_norms_plot = plot(0:MAX_ITER, results.xv_step_norms, linewidth = LINEWIDTH, label="(x, v) Step l2 Norm", xlabel="Iteration", ylabel="Step Norm", title="$title_beginning (x, v) l2 Step Norm $krylov_operator_str $title_end", yaxis=:log)
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(xv_step_norms_plot)

sing_vals_ratio_plot = plot(results.update_mat_iters, results.update_mat_singval_ratios, linewidth = LINEWIDTH, label="Prototype Update Matrix", xlabel="Iteration", ylabel="First Two Singular Values' Ratio", title="$title_beginning Update Matrix Singular Value Ratio $krylov_operator_str $title_end", yaxis=:log,
marker = :circle)
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dash, ALPHA, :green),
    label = "Active set changes"
)
display(sing_vals_ratio_plot)

# Create the base plot with rank data
update_ranks_plot = plot(
    results.update_mat_iters,
    results.update_mat_ranks,
    label = "Prototype Update Matrix",
    xlabel = "Iteration",
    ylabel = "Rank",
    title = "$title_beginning Update Matrix Rank $krylov_operator_str $title_end",
    linewidth = LINEWIDTH,
    # marker = :circle,
    xticks = 0:100:MAX_ITER,
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines, line = (:dash, ALPHA, :green), label = "Active set changes")

# plot!(results.update_mat_iters, results.update_mat_singval_ratios, label="Prototype Update Matrix", xlabel="Iteration", ylabel="First Two Singular Values' Ratio", yaxis=:log)

# Add vertical lines for accelerated steps
# vline! adds vertical lines at specified x-coordinates
vline!(results.acc_step_iters, 
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
display(update_ranks_plot)

# Plot consecutive cosines of update vectors.
xv_update_cosines_plot = plot(1:MAX_ITER, results.xv_update_cosines, linewidth = LINEWIDTH, label="Prototype Update Cosine", xlabel="Iteration", ylabel="Cosine of Consecutive Updates", title="$title_beginning Consecutive Update Cosines $krylov_operator_str $title_end")
vline!(results.acc_step_iters,
    line = (:dash, ALPHA, :red, VERT_LINEWIDTH),  # Use dashed red lines
    label = "Accelerated Steps"
)
vline!(results.linesearch_iters,
    line = (:dash, ALPHA, :maroon, VERT_LINEWIDTH),
    label = "Line Search Steps"
)
vline!(constraint_lines,
    line = (:dashdot, ALPHA, :green),
    label = "Active set changes"
)
display(xv_update_cosines_plot)

end




################################################################################
# STEP ANGLE STUFF
################################################################################

# begin

# EXP_SMOOTHING_PARAMETER = 0.95

# plot(exp_moving_average(x_step_angles, EXP_SMOOTHING_PARAMETER), label="x Step Angle", xlabel="Iteration", ylabel="Angle (radians)", title="Variant $variant: Step Angles.$newline_char Restart period = $RESTART_PERIOD")
# plot!(exp_moving_average(s_step_angles, EXP_SMOOTHING_PARAMETER), label="s Step Angle")
# plot!(exp_moving_average(y_step_angles, EXP_SMOOTHING_PARAMETER), label="y Step Angle")
# plot!(exp_moving_average(concat_step_angles, EXP_SMOOTHING_PARAMETER), label="Concatenated Step Angle")
# display(plot!(exp_moving_average(normalised_concat_step_angles, EXP_SMOOTHING_PARAMETER), label="NORMALISED Concatenated Step Angle"))


# # Running sums of angles

# plot(cumsum(x_step_angles), label="x Step Angle", xlabel="Iteration", ylabel="Cumulative Angle (radians)", title="Variant $variant: Cumulative Step Angles.$newline_char Restart period = $RESTART_PERIOD")

# plot!(cumsum(s_step_angles), label="s Step")

# plot!(cumsum(y_step_angles), label="y Step")

# plot!(cumsum(concat_step_angles), label="Concatenated Step")

# display(plot!(cumsum(normalised_concat_step_angles), label="NORMALISED Concatenated Step"))
# end

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