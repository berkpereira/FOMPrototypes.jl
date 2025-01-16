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

    # I'm still experimenting with proof of concept, so I implement two
    # versions of this function in parallel.
    # One of them interprets lines in (x, v) space, and the other 
    # interprets lines in (x, s, y) space.
    # I think the latter is correct for the Giselsson et al method, but still
    # good to check practical performance, since I can.
    
    # CHOOSE:
    LINES = :xv # in {:xv, :xsy}

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
    if LINES == :xv
        curr_fp_residual = vanilla_iterate_x_v_q[:, 1] - ws.vars.x_v_q[:, 1]

        # To comute fixed-point residual at the vanilla iterate, first
        # have to determine the local (x, v) dynamics at the vanilla point.
        vanilla_iterate_s = project_to_K(vanilla_iterate_x_v_q[ws.p.n+1:end, 1], ws.p.K)
        vanilla_enforced_constraints = enforced_contraints_bitvec(ws, vanilla_iterate_x_v_q[ws.p.n+1:end, 1], vanilla_iterate_s)

        # Compute T(vanilla iterate), where T is the FOM operator.
        # TODO: include UPDATE OF tilde_b to reflect the dynamics at the vanilla
        # iterate! In the special case of a QP it is invariant, but NOT in general!
        vanilla_lookahead = tilde_A_prod(ws, vanilla_enforced_constraints, vanilla_iterate_x_v_q[:, 1]) + ws.tilde_b

        # Now compute fixed-point residual at vanilla iterate.
        vanilla_fp_merit = norm(vanilla_lookahead - vanilla_iterate_x_v_q[:, 1])

        candidate = similar(vanilla_lookahead)
        candidate_s = similar(vanilla_iterate_s)
        candidate_lookahead = similar(vanilla_lookahead)
        candidate_enforced_constraints = similar(vanilla_enforced_constraints)
    elseif LINES == :xsy
        curr_iterate_s = project_to_K(ws.vars.x_v_q[ws.p.n+1:end, 1], ws.p.K)
        curr_iterate_y = ws.ρ * (curr_iterate_s - ws.vars.x_v_q[ws.p.n+1:end, 1])
        curr_iterate_xsy = [ws.vars.x_v_q[1:ws.p.n, 1]; curr_iterate_s; curr_iterate_y]

        vanilla_iterate_s = project_to_K(vanilla_iterate_x_v_q[ws.p.n+1:end, 1], ws.p.K)
        vanilla_iterate_y = ws.ρ * (vanilla_iterate_s - vanilla_iterate_x_v_q[ws.p.n+1:end, 1])
        vanilla_iterate_xsy = [vanilla_iterate_x_v_q[1:ws.p.n, 1]; vanilla_iterate_s; vanilla_iterate_y]
        
        curr_fp_residual = vanilla_iterate_xsy - curr_iterate_xsy

        # Now need fixed-point residual AT the vanilla point.
        vanilla_lookahead_x = iter_x(vanilla_iterate_xsy[1:ws.p.n], ws.cache[:W_inv], ws.p.P, ws.p.c, ws.p.A, vanilla_iterate_y, curr_iterate_y)
        vanilla_lookahead_s = iter_s(vanilla_iterate_s, ws.p.A, vanilla_lookahead_x, ws.p.b, vanilla_iterate_y, ws.p.K, ws.ρ)
        vanilla_lookahead_y = iter_y(vanilla_iterate_y, ws.p.A, vanilla_lookahead_x, vanilla_lookahead_s, ws.p.b, ws.ρ)
        
        vanilla_lookahead = [vanilla_lookahead_x; vanilla_lookahead_s; vanilla_lookahead_y]

        # Now compute fixed-point residual at vanilla iterate.
        vanilla_fp_merit = norm(vanilla_lookahead - vanilla_iterate_xsy)

        candidate = similar(vanilla_lookahead)
        candidate_lookahead = similar(vanilla_lookahead)
    end

    # Now move on to the actual line search.
    # Note that the search direction is simply the current fp residual.
    α = α_max
    
    candidate_fp_merit = 0.0
    while α > 1.0 # Backtrack until we hit the nominal step length.
        if LINES == :xv
            candidate .= ws.vars.x_v_q[:, 1] + α * curr_fp_residual
            
            # First determine the local method dynamics AT the candidate.
            candidate_s .= project_to_K(candidate[ws.p.n+1:end], ws.p.K)
            candidate_enforced_constraints .= enforced_contraints_bitvec(ws, candidate[ws.p.n+1:end], candidate_s)
            
            # Compute lookahead iterate FROM the candidate.
            # TODO: include UPDATE OF tilde_b to reflect the dynamics at the candidate point. This is ONLY constant in QP special case!
            candidate_lookahead .= tilde_A_prod(ws, candidate_enforced_constraints, candidate) + ws.tilde_b

            # Hence compute fixed-point residual norm at candidate.
            candidate_fp_merit = norm(candidate_lookahead - candidate)

            if candidate_fp_merit <= (1 - ϵ) * vanilla_fp_merit
                println("Successful line search with α = $α.")
                println("Successful Candidate merit: $candidate_fp_merit")
                println("Vanilla merit: $vanilla_fp_merit")
                
                # Assign to the workspace variable.
                ws.vars.x_v_q[:, 1] .= candidate
                return true
            end

            # Backtrack by factor β.
            α *= β
        elseif LINES == :xsy
            candidate .= curr_iterate_xsy + α * curr_fp_residual

            # Compute lookahead iterate FROM the candidate.
            candidate_lookahead[1:ws.p.n] .= iter_x(candidate[1:ws.p.n], ws.cache[:W_inv], ws.p.P, ws.p.c, ws.p.A, candidate[ws.p.n+ws.p.m+1:end], candidate[ws.p.n+ws.p.m+1:end] - ws.ρ * (ws.p.A * candidate[1:ws.p.n] + candidate[ws.p.n+1:ws.p.n+ws.p.m] - ws.p.b))
            candidate_lookahead[ws.p.n+1:ws.p.n+ws.p.m] .= iter_s(candidate[ws.p.n+1:ws.p.n+ws.p.m], ws.p.A, candidate_lookahead[1:ws.p.n], ws.p.b, candidate[ws.p.n+ws.p.m+1:end], ws.p.K, ws.ρ)
            candidate_lookahead[ws.p.n+ws.p.m+1:end] .= iter_y(candidate[ws.p.n+ws.p.m+1:end], ws.p.A, candidate_lookahead[1:ws.p.n], candidate_lookahead[ws.p.n+1:ws.p.n+ws.p.m], ws.p.b, ws.ρ)

            # Hence compute fixed-point residual norm at candidate.
            candidate_fp_merit = norm(candidate_lookahead - candidate)

            if candidate_fp_merit <= (1 - ϵ) * vanilla_fp_merit
                println("Successful line search with α = $α.")
                println("Successful Candidate merit: $candidate_fp_merit")
                println("Vanilla merit: $vanilla_fp_merit")
                
                candidate_v = candidate[ws.p.n+1:ws.p.m+ws.p.n] - candidate[ws.p.m+ws.p.n+1:end] / ws.ρ

                # Assign to the workspace variable.
                ws.vars.x_v_q[:, 1] .= [candidate[1:ws.p.n]; candidate_v]
                return true
            end

            # Backtrack by factor β.
            α *= β
        end
    end
    
    println("Failed linesearch. Last at α = $(α / β), candidate/vanilla residual: $(candidate_fp_merit / vanilla_fp_merit)")

    # If we reach here, we just assign the vanilla iterate.
    ws.vars.x_v_q[:, 1] .= vanilla_iterate_x_v_q[:, 1]
    return false
end