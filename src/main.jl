include("QPProblem.jl")
include("utils.jl")
include("problem_data.jl")
include("algorithm.jl")
include("printing.jl")

using Printf
using SCS


problem_set = "maros";
problem_name = "HS21";

# Step size parameters chosen by the user
ρ = 1.0 # (dual)

# Choose algorithm based on M1 construction
# int in {1, 2, 3, 4}
variant = 1;

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
τ = 1 / maximum(eigvals(Matrix(take_away)))
M1 = (1 / τ) * Matrix{Float64}(I, n, n) - take_away
println("Dual step size ρ: $ρ")
println("Primal step size τ: $τ")

# Create a problem instance
problem = QPProblem(data.P, data.c, data.A, data.b, M1, data.K, τ, ρ)

# Initialize sequences x, s, and y
x = zeros(n)
s = zeros(m)
y = zeros(m)

# Main iteration loop
max_iters = 500  # Maximum number of iterations (adjust as needed)
for k in 0:max_iters
    # Print each iteration result with formatted output on a single line
    if k % 50 == 0
        @printf("Iteration %4.1d: x = [%s]  s = [%s]  y = [%s]\n",
            k,
            join(map(xi -> @sprintf("%12.5e", xi), x), ", "),
            join(map(zi -> @sprintf("%12.5e", zi), s), ", "),
            join(map(yi -> @sprintf("%12.5e", yi), y), ", "))
    end
    
    iterate!(problem, x, s, y)
end

# Print final solution on a single line
println("\nPrototype FOM results:")
@printf("x = [%s]  s = [%s]  y = [%s]\n",
    join(map(xi -> @sprintf("%12.5e", xi), x), ", "),
    join(map(zi -> @sprintf("%12.5e", zi), s), ", "),
    join(map(yi -> @sprintf("%12.5e", yi), y), ", "))

############################### SCS SOLUTION ###################################

println("\nRunning SCS...")

# Define the SCS optimizer
model = Model(SCS.Optimizer)
set_silent(model)
# Define primal variables x and s
@variable(model, x[1:n])    # Primal variables in R^n
@variable(model, s[1:m])    # Slack variables in R^m

# Add constraints for Ax + s = b
@constraint(model, con, A * x + s .== b)

# Constrain s to the product of cones K
start_idx = 1
for cone in K
    end_idx = start_idx + cone.dim - 1
    if cone isa Clarabel.NonnegativeConeT
        # Nonnegative cone constraint for this block of s
        @constraint(model, s[start_idx:end_idx] in MOI.Nonnegatives(cone.dim))
    elseif cone isa Clarabel.ZeroConeT
        # Zero cone constraint for this block of s
        @constraint(model, s[start_idx:end_idx] in MOI.Zeros(cone.dim))
    else
        error("Unsupported cone type in K: $cone")
    end
    start_idx = end_idx + 1
end

# Add the quadratic objective: (1/2)x'Px + c'x
@objective(model, Min, 0.5 * dot(x, P * x) + dot(c, x))

# Solve the optimization problem
optimize!(model)

# Extract solutions
x_scs = value.(x)
s_scs = value.(s)
y_scs = dual.(con)  # Dual variables (Lagrange multipliers)

# Print results for SCS
print_results("SCS", x_scs, s_scs, y_scs, max_iters)
