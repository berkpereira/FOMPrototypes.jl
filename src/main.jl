include("QPProblem.jl")
include("utils.jl")
include("problem_data.jl")
include("algorithm.jl")

using Printf


problem_set = "maros";
problem_name = "HS21";

# Step size parameters chosen by the user
ρ = 1.0 # (dual)

# Choose algorithm based on M1 construction
# int in {1, 2, 3, 4}
variant = 1;

################################################################################

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

"""
# OSQP, SCS solution, essentially test difficulty of problem
using OSQP

# Convert problem data to sparse type
P_osqp = sparse(P)
A_osqp = sparse(A)
# Create OSQP object
prob = OSQP.Model()
# Set up workspace and change alpha parameter
OSQP.setup!(prob; P=P_osqp, q=c, A=A_osqp, u=b, alpha=1, verbose=false)
# Solve problem
results = OSQP.solve!(prob);

# Print OSQP results
println("OSQP results:")
println("x = [", join(map(v -> @sprintf("%12.5e", v), results.x), ", "), "]")
println("y = [", join(map(v -> @sprintf("%12.5e", v), results.y), ", "), "]")
""";