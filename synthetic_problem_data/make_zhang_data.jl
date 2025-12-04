"""
Generate SOCP data as in GLOBALLY CONVERGENT TYPE-I ANDERSON ACCELERATION FOR
NONSMOOTH FIXED-POINT ITERATIONS by Zhang, O'Donoghue, and Boyd (2020).
"""

using SparseArrays, Random, LinearAlgebra
using JLD2
include("../src/alg/cones.jl")

Random.seed!(42)

# sizes (paper example)
m = 3000
@assert m % 3 == 0 && m % 2 == 0 "m must be multiple of 6"
n = 5000
density = 0.1               # sparsity of the random block (paper used ~0.1)
half = fld(n,2)

# random sparse Gaussian block and identity block -> horizontal concat
A1 = sprandn(m, half, density)          # sparse Gaussian block
A2 = sparse(I, m, n - half)                 # identity-like block (sparse)
A = hcat(A1, A2)                        # SparseMatrixCSC

# optionally add tiny sparse noise (keeps A sparse)
A += sprandn(m, n, 1e-4)                # very tiny sparse noise

K = Vector{Clarabel.SupportedCone}(undef, 0)

nonneg_cone_dim = 900
zero_cone_dim = 300

@assert (m - (nonneg_cone_dim + zero_cone_dim)) % 3 == 0

push!(K, Clarabel.NonnegativeConeT(nonneg_cone_dim))
push!(K, Clarabel.ZeroConeT(zero_cone_dim))
soc_cone_no = (m - (nonneg_cone_dim + zero_cone_dim)) ÷ 3

for i in 1:soc_cone_no
    push!(K, Clarabel.SecondOrderConeT(3))
end

@assert sum(map(c -> c.dim, K)) == m

# make z*, project to cones
zstar = randn(m)
sstar = deepcopy(zstar)

# example: single SOC of size m (or split into blocks as you like)
project_to_K!(sstar, K)

ystar = sstar .- zstar


# random primal x*
xstar = randn(n)

b = A * xstar + sstar
c = - A' * ystar

P = spzeros(n, n)

# to save
repo_root = normpath(joinpath(@__DIR__, ".."))
file = "synthetic_problem_data/zhang_socp_problem.jld2"
@save file P c A b K m n