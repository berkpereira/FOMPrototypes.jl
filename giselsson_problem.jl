using Random, LinearAlgebra, Clarabel, JLD2

# Set random seed for reproducibility
Random.seed!(42)

# Generate C matrix
n = 1000
C = randn(n, n)
row_scales = 0.1 .+ rand(n)
C = C .* row_scales

# Compute P = C'C 
P = C'C

# Generate d and compute c = -C'd
d = randn(n)
c = -C'd

# Generate A and b
A = Matrix{Float64}(I, n, n)
b = zeros(n)

# Create slack variable cone
K = Vector{Clarabel.SupportedCone}([Clarabel.NonnegativeConeT(n)])

# Package data and save
problem_data = (
   P = P,
   c = c,
   A = A,
   b = b,
   m = n,
   n = n,
   K = K
)

save("synthetic_problem_data/giselsson_problem.jld2", "data", problem_data)