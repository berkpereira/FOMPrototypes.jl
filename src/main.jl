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
problem_name = "HB_ash219_lasso"
# also decent from sslsq: "HB_ash85_huber"
# "HB_ash85_lasso"

# Step size parameters chosen by the user
ρ = 1.0; # (dual)

# Choose algorithm based on M1 construction
# int in {1, 2, 3, 4}
variant = 4

############################# PROTOTYPE SOLUTION ###############################

data = load_clarabel_benchmark_prob_data(problem_set, problem_name);
P, c, A, b, m, n, K = data.P, data.c, data.A, data.b, data.m, data.n, data.K;

if variant == 1 # NOTE: most "economical"
    take_away = off_diag_part(P + ρ * A' * A)
elseif variant == 2 # NOTE: least "economical"
    take_away = P + ρ * A' * A
elseif variant == 3 # NOTE: intermediate
    take_away = P + off_diag_part(ρ * A' * A)
elseif variant == 4 # NOTE: also intermediate
    take_away = off_diag_part(P) + ρ * A' * A
else
    error("Invalid variant.")
end

# Choose primal step size as 90% of maximum allowable while to keep M1 PSD
τ = 0.9 / maximum(eigvals(Matrix(take_away)));
M1 = (1 / τ) * Matrix{Float64}(I, n, n) - take_away;
println("Dual step size ρ: $ρ")
println("Primal step size τ: $τ")

# Create a problem instance
problem = QPProblem(data.P, data.c, data.A, data.b, M1, data.K, τ, ρ);

# Initialize sequences x, s, and y
x = zeros(n);
s = zeros(m);
y = zeros(m);

# Initialise metrics storage
obj_vals = Float64[];
residuals = Float64[];

# Main iteration loop
max_iters = 100;  # Maximum number of iterations (adjust as needed)
for k in 0:max_iters
    # Print each iteration result with formatted output on a single line
    # if k % 50 == 0
    #     @printf("Iteration %4.1d: x = [%s]  s = [%s]  y = [%s]\n",
    #         k,
    #         join(map(xi -> @sprintf("%12.5e", xi), x), ", "),
    #         join(map(zi -> @sprintf("%12.5e", zi), s), ", "),
    #         join(map(yi -> @sprintf("%12.5e", yi), y), ", "))
    # end

    # Append to metrics
    push!(obj_vals, 0.5 * dot(x, P * x) + dot(c, x))
    push!(residuals, norm(A * x - b))

    # Use method iteration
    iterate!(problem, x, s, y)
end
;
# Print final solution on a single line
# println("\nPrototype FOM results:")
# @printf("x = [%s]  s = [%s]  y = [%s]\n",
#     join(map(xi -> @sprintf("%12.5e", xi), x), ", "),
#     join(map(zi -> @sprintf("%12.5e", zi), s), ", "),
#     join(map(yi -> @sprintf("%12.5e", yi), y), ", "))

# Plot objective values
plot(0:max_iters, obj_vals, label="Prototype Objective", xlabel="Iteration", ylabel="Objective Value", title="Variant $variant: Objective")

# Plot primal residuals
plot(0:max_iters, residuals, label="Prototype Residual", xlabel="Iteration", ylabel="Primal Residual", title="Variant $variant: Primal Residual Norm")

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

# Constrain s to the product of cones K
start_idx = 1;
for cone in K
    end_idx = start_idx + cone.dim - 1;
    if cone isa Clarabel.NonnegativeConeT
        # Nonnegative cone constraint for this block of s
        @constraint(model, s_scs[start_idx:end_idx] in MOI.Nonnegatives(cone.dim));
    elseif cone isa Clarabel.ZeroConeT
        # Zero cone constraint for this block of s
        @constraint(model, s_scs[start_idx:end_idx] in MOI.Zeros(cone.dim));
    else
        error("Unsupported cone type in K: $cone")
    end
    start_idx = end_idx + 1;
end

# Add the quadratic objective: (1/2)x'Px + c'x
@objective(model, Min, 0.5 * dot(x_scs, P * x_scs) + dot(c, x_scs));

# Solve the optimization problem
optimize!(model);

# Extract solutions
x_scs = value.(x_scs);
s_scs = value.(s_scs);
y_scs = dual.(con);  # Dual variables (Lagrange multipliers)
;
# Print results for SCS
# print_results("SCS", x_scs, s_scs, y_scs, max_iters)
