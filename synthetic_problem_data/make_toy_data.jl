using Clarabel
using LinearAlgebra
using SparseArrays
using JLD2

# dims
n = 3
m = 3

# must make these
# P => Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
# c => Vector{Float64}
# A => SparseMatrixCSC{Float64, Int64}
# b => Vector{Float64}
# K => Array{Clarabel.SupportedCone}((1,))

# P is identity
P = sparse([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) # note no Symmetric wrapper
c = 0 * ones(Float64, n) # given identity Hessian, optimiser is x = -c

# make a box
# bigM = 1e6
# A = sparse([1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0])
# b = [bigM; bigM; bigM; bigM] # huge box
A = sparse([-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 -1.0])
b = [-1.0; -1.0; -1.0]
K = Vector{Clarabel.SupportedCone}([Clarabel.NonnegativeConeT(m)])

# to save
repo_root = normpath(joinpath(@__DIR__, ".."))
file = "synthetic_problem_data/toy_problem.jld2"
@save file P c A b K m n