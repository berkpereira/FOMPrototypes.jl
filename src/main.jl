include("QPProblem.jl")
include("utils.jl")
include("problem_data.jl")
include("algorithm.jl")
include("printing.jl")

using Printf
using Plots

############################## FIND PROBLEMS ###################################

# suitable_problems = filter_clarabel_problems("sslsq", min_m=100, min_n=100, max_n=500)

############## CHOOSE PROBLEM SET AND PROBLEM NAME #############################

# problem_set = "maros";
# problem_name = "HUESTIS";

problem_set = "sslsq";
problem_name = "HB_ash219_lasso" # pretty good solution from prototype
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
optimize!(model);

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

max_iters = 200;  # Maximum number of iterations (adjust as needed)
print_modulo = 50;  # Print every print_modulo iterations




take_away = take_away_matrix(variant)

# Choose primal step size as 90% of maximum allowable while to keep M1 PSD
τ = 0.9 / maximum(eigvals(Matrix(take_away)));
M1 = (1 / τ) * Matrix{Float64}(I, n, n) - take_away;

# Create a problem instance
problem = QPProblem(data.P, data.c, data.A, data.b, M1, data.K, τ, ρ);

# Initialize sequences x, s, and y
x = zeros(n);
s = zeros(m);
y = zeros(m);

# Initialise metrics storage
obj_vals = Float64[];
primal_residuals = Float64[];
dual_residuals = Float64[];


# Main iteration loop
println("Running prototype variant $variant...")
for k in 0:max_iters
    # Append to metrics
    curr_obj = 0.5 * dot(x, P * x) + dot(c, x)
    curr_primal_res = norm(A * x + s - b)
    curr_dual_res = norm(P * x + A' * y + c)
    curr_duality_gap = dot(x, P * x) + dot(c, x) + dot(b, y)
    push!(obj_vals, curr_obj)
    push!(primal_residuals, curr_primal_res)
    push!(dual_residuals, curr_dual_res)
    
    # Print each iteration result with formatted output on a single line
    # NOTE: it seems that SCS uses switched sign in the dual variable compared
    # to me, which is fair because it is the multiplier of equality
    # constraints. (hence the sign switch in the error below)
    if k % print_modulo == 0
        @printf("Iter %4.1d | Obj: %12.5e (err: %12.5e)| x err: %12.5e | s err: %12.5e | y err: %12.5e | Primal res: %12.5e | Dual res: %12.5e | Duality gap: %12.5e\n", k, curr_obj, curr_obj - obj_scs, norm(x - x_scs), norm(s - s_scs), norm(y - (-y_scs)), curr_primal_res, curr_dual_res, curr_duality_gap)
    end

    # Use method iteration
    iterate!(problem, x, s, y)
end
;

# Plot objective values
plot(0:max_iters, obj_vals, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="Variant $variant: Objective")

# Plot primal residuals
plot(0:max_iters, primal_residuals, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm")