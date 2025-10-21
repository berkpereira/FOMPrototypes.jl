"""
Our prototype's "starting point" is AD-PMM as described in Shefi and Teboulle 2014. We set M_2 = 0, then worry about the choice of M_1.

Through different choices of M_1, we may obtain ADMM, PDHG, linearised PDHG,
or other similarly minded first-order methods. One of our main motivations,
in this solver, is to choose M_1 so that the resulting x update has a
closed-form solution which can be obtained without expensive matrix inversions.
Eg we may allow for a diagonal (or perhaps tridiagonal, etc.) system to be
solved, but general systems such as required by (exact) ADMM or PDHG are
disqualified.

Now, we initially proposed differing choices of M_1, which lead to different
methods with a diagonal system inversion to give the x update.
For instance, M_1 = 1/τ I - (ρ A^T A + P) gives linearised PDHG. Replacing
-ρ A^T A and/or -P with just the off-diagonal entries of each of these matrices
gives different variants, which still retain the diagonal structure that we
allow for.

These are the most obvious options to obtain a diagonal update system, but
what if we could make a better choice? We are inspired by the insights of
the paper Liu et al. 2021 (Acceleration of Primal-Dual Methods by
Preconditioning and Simple Subproblem Procedures), which uses the typical
primal-dual gap bounds on PDHG-like methods, in their "characteristic" norm,
along with a Schur complement argument, to say that, in our notation,
the optimal M_1 choice is given by 0 (ie giving ADMM). Further, this being out
of the question, the optimal choice within our constraints on M_1 is minimise
its spectral radius.

We may be interested in some heuristic method to do this, but in order to upper
bound the benefit to doing any such thing, we can simply find the optimal choice
for some of our benchmark problems by solving an SDP, and seeing the
effect on the spectral radius of M_1. This is the purpose of the code here.
"""

using JuMP
using LinearAlgebra
using SCS
using Clarabel
using SparseArrays

# Note: the input argument max_τ corresponds to the critical value such that the
# PSD condition M_1 = (1 / max_τ) I - S ≥ 0 is active.
# Determining this maximum step size is part of the usual procedure prior to
# running PDHG or our own, closely related prototype method.
# The solution to the SDP solved here is therefore the DIFFERENCE in diagonal
# between the diagonal entries of (1 / max_τ) I and the optimal diagonal D.
function solve_spectral_radius_sdp(S::AbstractMatrix{Float64}, max_τ::Float64;
    solver=:Clarabel, time_limit::Float64)
    n = size(S, 1)
    
    # Create model with chosen solver
    if solver === :SCS
        model = Model(SCS.Optimizer)
        # set_optimizer_attribute(model, "eps_abs", 1e-12)
        # set_optimizer_attribute(model, "eps_rel", 1e-12)
        set_optimizer_attribute(model, "time_limit_secs", time_limit)
    elseif solver === :Clarabel
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "time_limit", time_limit)
    else
        error("Unsupported solver")
    end
    # set_silent(model)

    # KrylovVariables
    @variable(model, λ)  # Objective variable
    @variable(model, d[1:n])  # Diagonal entries of D

    # Create the diagonal matrix D
    D = spdiagm(d)
    
    # Constraints
    # λI - ( (1 / max_τ) I + S + D) ≥ 0
    @constraint(model, λ * sparse(I, n, n) - ( (1 / max_τ) * sparse(I, n, n) + S + D) in PSDCone())
    
    # (1 / max_τ) I + S + D ≥ 0
    @constraint(model, (1 / max_τ) * sparse(I, n, n) + S + D in PSDCone())
    
    # Objective: minimize λ
    @objective(model, Min, λ)
    
    # Solve
    println("Model setup done. ABOUT TO SOLVE.")
    optimize!(model)
    println("Optimisation done.")
    
    # Return results
    status = termination_status(model)
    if status == OPTIMAL
        opt_λ = value(λ)
        opt_D = Diagonal(value.(d))
        opt_obj = objective_value(model)
        return (
            status=status,
            λ=opt_λ,
            D=opt_D,
            objective=opt_obj
        )
    else
        return (
            status=status,
            λ=nothing,
            D=nothing,
            objective=nothing
        )
    end
end



# # take_away_mat and max_τ here would usually come from the prototype setup
# # stage of src/main.jl.
# S = - take_away_mat

# # Solve SDP
# result = solve_spectral_radius_sdp(S, max_τ, solver = :Clarabel)

# if result.status == OPTIMAL
#     spec_S = eigvals(Matrix(S))
#     println("Without solving SDP and using VARIANT $variant, the diff between max and min eigenvalues of S is:")
#     println(maximum(spec_S) - minimum(spec_S))
#     println("--------------------------")

#     println("Optimal solution found:")
#     println("Objective: minimum spectral radius (λ): ")
#     println("$(result.λ)")
#     # println("Optimal diagonal matrix D:\n", result.D)
    
#     # Verify the result
#     final_matrix = ((1 / max_τ) * sparse(I, n, n)) + S + result.D;
#     # println("\nVerification:")
#     # println("Eigenvalues of S + D: ", eigvals(final_matrix))
#     # println("Is result PSD? ", isposdef(final_matrix))
# else
#     println("Problem could not be solved to optimality. Status: ", result.status)
# end

# ################################# PLOT THINGS ##################################

# begin
#     # We plot a histogram of the entries in results.D.diag.
#     # We also mark, on the histogram, the horizontal axis value (1 / max_τ)
#     # which is the value of the diagonal entries of D in our original method.
#     # Plot histogram of the diagonal entries of D
#     p = histogram(result.D.diag, label="Diagonal entries of D", xlabel="Value", ylabel="Frequency", title="Histogram of Diagonal Entries of D")

#     display(p)
# end