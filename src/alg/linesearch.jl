using LinearAlgebra

"""
Adaptive fixed-point linesearch with step extension.

Attempts a single extended step beyond the vanilla FOM iterate; accepts if the
safeguard check passes (fixed-point metric at candidate <= safeguard_factor *
fixed-point metric at vanilla).

On success:
- Updates state_vanilla in-place to the extended candidate
- Increments consecutive_successes
- Grows global_step_size by linesearch_step_growth (capped at linesearch_max_step)

On failure:
- Leaves state_vanilla unchanged
- Resets global_step_size to linesearch_initial_step (aggressive reset)
- Resets consecutive_successes to 0

Arguments:
- ws: workspace (any type with linesearch_state field)
- config: solver config with linesearch parameters
- record: iteration record (for rolling cosine check)
- state_prev: the iterate BEFORE the vanilla FOM step
- state_vanilla: the vanilla FOM output (modified in-place if successful)
- scratch_candidate: preallocated buffer for candidate (length m+n)

Returns: true if linesearch succeeded, false otherwise
"""
function adaptive_linesearch!(
    ws::AbstractWorkspace,
    config::SolverConfig,
    record::AbstractRecord,
    state_prev::AbstractVector{Float64},
    state_vanilla::AbstractVector{Float64},
    scratch_candidate::AbstractVector{Float64},
)::Bool
    # Early exit if disabled or NullRecord
    if !config.linesearch_enabled || record isa NullRecord
        return false
    end

    # Check trigger: rolling cosine avg > threshold
    rolling_avg = get_rolling_cosine_avg(record, config.linesearch_cosine_window)
    if rolling_avg < config.linesearch_cosine_threshold
        return false
    end

    α = ws.linesearch_state.global_step_size

    # Compute extended candidate: state_prev + α * (state_vanilla - state_prev)
    @. scratch_candidate = state_prev + α * (state_vanilla - state_prev)

    # Compute FOM(vanilla) and its fp metric
    compute_fom_and_fp!(ws, state_vanilla,
                        ws.scratch.extra.state_lookahead,
                        ws.scratch.extra.fp_res)
    fp_metric_vanilla = compute_fp_metric!(ws, ws.scratch.extra.fp_res)

    # Compute FOM(candidate) and its fp metric
    compute_fom_and_fp!(ws, scratch_candidate,
                        ws.scratch.extra.state_lookahead,
                        ws.scratch.extra.fp_res)
    fp_metric_candidate = compute_fp_metric!(ws, ws.scratch.extra.fp_res)

    # Safeguard check: same factor as acceleration
    safeguard_ratio = fp_metric_candidate / fp_metric_vanilla
    if fp_metric_candidate <= config.safeguard_factor * fp_metric_vanilla
        # SUCCESS: update state_vanilla in-place to candidate
        state_vanilla .= scratch_candidate

        println("✅ Linesearch success at iter $(ws.k[]), α = $(α), safeguard ratio = $(safeguard_ratio)")

        # Grow step size on consecutive successes
        ws.linesearch_state.consecutive_successes += 1
        ws.linesearch_state.global_step_size = min(
            ws.linesearch_state.global_step_size * config.linesearch_step_growth,
            config.linesearch_max_step
        )
        return true
    else
        println("❌ Linesearch failure at iter $(ws.k[]), α = $(α), safeguard ratio = $(safeguard_ratio)")
        
        # FAILURE: aggressive reset
        ws.linesearch_state.global_step_size = max(
            config.linesearch_initial_step,
            ws.linesearch_state.global_step_size / sqrt(config.linesearch_step_growth)
        )
        ws.linesearch_state.global_step_size = config.linesearch_initial_step
        ws.linesearch_state.consecutive_successes = 0
        return false
    end
end
