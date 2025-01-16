using LinearAlgebra
include("utils.jl")
include("acceleration.jl")
include("types.jl")

"""
From a given current iterate (x, v)_k, the main steps to the approach in
Giselsson, Falt, and Boyd 2016 are as follows. This is done very naively
here.
- Compute the would-be vanilla iterate.
- Consider the ray from the current iterate to the would-be vanilla one.
- Backtrack from α_max. The merit function is the norm of the fixed-point
residual at each candidate point. This requires a careful treatment of
the enforced constraints at each candidate, which itself determines its
local dynamics and therefore the fixed-point residual.
- If the trial α goes under 1, we just return the vanilla iterate.

NOTE: for the concurrent Arnoldi process, it is probably fine to just
use the current affine dynamics and the computation of the vanilla iterate
as the Arnoldi step for whatever iteration we are currently on.

This function returns true if the line search was successful, and false
if we end up simply taking the vanilla step instead.
"""
function fixed_point_linesearch!(ws::Workspace, α_max::Float64,
    β::Float64, ϵ::Float64, krylov_operator_tilde_A::Bool)
    @assert α_max > 1.0

    # Compute (would-be) vanilla iterate from the current (x, v)_k and
    # Arnoldi "q vector", which we access from ws.vars.x_v_q.
    # We take the opportunity to also take an Arnoldi process step here,
    # hence the assignment into the "q vector", ws.vars.x_v_q[:, 2].
    if krylov_operator_tilde_A
        vanilla_iterate_x_v_q = tilde_A_prod(ws, ws.enforced_constraints, ws.vars.x_v_q) + [ws.tilde_b zeros(ws.p.m + ws.p.n)]
    else
        vanilla_iterate_x_v_q = tilde_A_prod(ws, ws.enforced_constraints, ws.vars.x_v_q) + [ws.tilde_b (-ws.vars.x_v_q[:, 2])]
    end

    # We can assign to the Arnoldi vector in ws straight away.
    ws.vars.x_v_q[:, 2] .= vanilla_iterate_x_v_q[:, 2]

    # Compute the current iterate's fixed-point residual.
    # We may need to "translate" to (x, s, y) space for the line search.
    # In production code this will be handled much more gracefully...
    curr_iterate_s = project_to_K(ws.vars.x_v_q[ws.p.n+1:end, 1], ws.p.K)
    curr_iterate_y = ws.ρ * (curr_iterate_s - ws.vars.x_v_q[ws.p.n+1:end, 1])
    curr_iterate_xsy = [ws.vars.x_v_q[1:ws.p.n, 1]; curr_iterate_s; curr_iterate_y]

    vanilla_iterate_s = project_to_K(vanilla_iterate_x_v_q[ws.p.n+1:end, 1], ws.p.K)
    vanilla_iterate_y = ws.ρ * (vanilla_iterate_s - vanilla_iterate_x_v_q[ws.p.n+1:end, 1])
    vanilla_iterate_xsy = [vanilla_iterate_x_v_q[1:ws.p.n, 1]; vanilla_iterate_s; vanilla_iterate_y]
    
    # curr_fp_residual = vanilla_iterate_x_v_q[:, 1] - ws.vars.x_v_q[:, 1]
    curr_fp_residual = vanilla_iterate_xsy - curr_iterate_xsy

    # To determine the FP residual at the vanilla iterate, must first determine
    # the local affine dynamics of the method.
    # Recall that s = proj_K(v).
    
    # vanilla_iterate_s = project_to_K(ws.vars.x_v_q[ws.p.n+1:end, 1], ws.p.K)
    vanilla_enforced_constraints = enforced_contraints_bitvec(ws, vanilla_iterate_x_v_q[ws.p.n+1:end, 1], vanilla_iterate_s)

    # Compute T(vanilla iterate), where T is the FOM operator.
    # TODO: include UPDATE OF tilde_b to reflect the dynamics at the vanilla
    # iterate! In the special case of a QP it is invariant, but NOT in general!
    vanilla_lookahead = tilde_A_prod(ws, vanilla_enforced_constraints, vanilla_iterate_x_v_q[:, 1]) + ws.tilde_b

    # Again translate to (x, s, y) space.
    vanilla_lookahead_s = project_to_K(vanilla_lookahead[ws.p.n+1:end], ws.p.K)
    vanilla_lookahead_y = ws.ρ * (vanilla_lookahead_s - vanilla_lookahead[ws.p.n+1:end])
    vanilla_lookahead_xsy = [vanilla_lookahead[1:ws.p.n]; vanilla_lookahead_s; vanilla_lookahead_y]

    # Compute fixed-point residual NORM at the vanilla iterate.
    # vanilla_fp_merit = norm(vanilla_lookahead - vanilla_iterate_x_v_q[:, 1])
    vanilla_fp_merit = norm(vanilla_lookahead_xsy - vanilla_iterate_xsy)

    # Now move on to the line search proper.
    # Note that the search direction is simply the current fp residual.
    α = α_max
    # candidate = similar(ws.vars.x_v_q[:, 1]) # pre-allocate.
    candidate = similar(curr_iterate_xsy)
    
    candidate_s = similar(vanilla_iterate_s)
    candidate_enforced_constraints = similar(vanilla_enforced_constraints)
    candidate_lookahead = similar(vanilla_lookahead)
    
    candidate_fp_merit = 0.0
    while α > 1.0 # Backtrack until we hit the nominal step length.
        
        # candidate .= ws.vars.x_v_q[:, 1] + α * curr_fp_residual
        
        candidate .= curr_iterate_xsy + α * curr_fp_residual

        # Recover v.
        candidate_v = candidate[ws.p.n+1:ws.p.n+ws.p.m] - candidate[ws.p.n+ws.p.m+1:end] / ws.ρ

        # Merit function is norm of fp residual, so we repeat a lot of the
        # steps we went through in the code above.
        # candidate_s .= project_to_K(candidate[ws.p.n+1:end], ws.p.K)
        # candidate_enforced_constraints .= enforced_contraints_bitvec(ws, candidate[ws.p.n+1:end], candidate_s)
        candidate_enforced_constraints .= enforced_contraints_bitvec(ws, candidate_v, candidate[ws.p.n+1:ws.p.n+ws.p.m])
        candidate_lookahead .= tilde_A_prod(ws, candidate_enforced_constraints, [candidate[1:ws.p.n]; candidate_v]) + ws.tilde_b
        
        # candidate_fp_merit = norm(candidate_lookahead - candidate)
        
        # Translate to (x, s, y) space AGAIN.
        candidate_lookahead_s = project_to_K(candidate_lookahead[ws.p.n+1:end], ws.p.K)
        candidate_lookahead_y = ws.ρ * (candidate_lookahead_s - candidate_lookahead[ws.p.n+1:end])
        candidate_lookahead_xsy = [candidate_lookahead[1:ws.p.n]; candidate_lookahead_s; candidate_lookahead_y]
        candidate_fp_merit = norm(candidate_lookahead_xsy - candidate)


        # Acceptance criterion is based on reduction of the merit function
        # as compared to the vanilla iterate, by factor (1 - ϵ).
        if candidate_fp_merit <= (1 - ϵ) * vanilla_fp_merit
            println("Successful line search with α = $α.")
            println("Successful Candidate merit: $candidate_fp_merit")
            println("Vanilla merit: $vanilla_fp_merit")
            
            # Translate back to (x, v) space AGAIN...
            # ws.vars.x_v_q[:, 1] .= candidate
            ws.vars.x_v_q[:, 1] .= [candidate[1:ws.p.n]; candidate_v]
            return true
        end

        # Backtrack by factor β.
        α *= β
    end
    println("Failed linesearch. Last at α = $(α / β), candidate/vanilla residual: $(candidate_fp_merit / vanilla_fp_merit)")
    
    # If we reach here, we just assign the vanilla iterate.
    ws.vars.x_v_q[:, 1] .= vanilla_iterate_x_v_q[:, 1]
    return false
end