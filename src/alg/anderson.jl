using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Clarabel
import COSMOAccelerators

function anderson_step!(
    ws::AndersonWorkspace,
    ws_diag::Union{DiagnosticsWorkspace, Nothing},
    args::Dict{String, Any},
    record::AbstractRecord,
    full_diagnostics::Bool,
    timer::TimerOutput,
    )
    # Anderson acceleration attempt
    if ws.composition_counter[] == args["anderson-interval"]
        # ws.vars.state might be overwritten, so we take note of it here
        # this is for the sole purpose of checking the norm of the
        # step just below in this code branch
        ws.scratch.extra.state_pre_overwrite .= ws.vars.state

        # attempt acceleration step. if successful (ie no numerical
        # problems), this overwites ws.vars.state with accelerated step
        @timeit timer "anderson accel" COSMOAccelerators.accelerate!(ws.vars.state, ws.vars.state_prev, ws.accelerator, 0)
        
        ws.scratch.extra.accelerated_point .= ws.vars.state # accelerated point if no numerics blowups
        ws.vars.state .= ws.scratch.extra.state_pre_overwrite # retake the iterate from which we are attempting acceleration right now

        # ws.accelerator.success only indicates there was no
        # numerical blowup in the COSMOAccelerators.accelerate!
        # internals
        if ws.accelerator.success
            # at this point:
            # ws.vars.state contains the iterate at which we stopped
            # when we began to pull the accelerate! levers;
            # ws.scratch.extra.accelerated_point contains the candidate
            # acceleration point, which is legitimately different
            # from ws.vars.state in this branch
            @timeit timer "fixed-point safeguard" @views ws.control_flags.accepted_accel = accel_fp_safeguard!(ws, ws_diag, ws.vars.state, ws.scratch.extra.accelerated_point, args["safeguard-factor"], record, full_diagnostics)

            ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
            
            if ws.control_flags.accepted_accel
                println("Anderson success.")
                ws.k_eff[] += 1 # increment effective iter counter (ie excluding unsuccessful acc attempts)
                ws.res.residual_check_count[] += 1

                # assign actually
                ws.vars.state .= ws.scratch.extra.accelerated_point
            else
                println("Anderson rejected by safeguard.")
            end
        end

        # note: uses ws.scratch.extra.state_pre_overwrite if acceleration
        # was proper successful
        push_update_to_record!(ws, record, false)

        # set up the first x to be composed anderon-interval times
        # before being fed to accelerator's update! method
        ws.vars.state_into_accelerator .= ws.vars.state # this is T^0(ws.vars.state)
        ws.composition_counter[] = 0
    else # vanilla iteration
        ws.k_eff[] += 1
        ws.res.residual_check_count[] += 1
        ws.k_vanilla[] += 1
        ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
        
        # copy older iterate before iterating
        ws.vars.state_prev .= ws.vars.state

        # we compute new iterate: if we just tried acceleration,
        # we use the recycled variable stored during the fixed-point
        # safeguarding step
        if ws.control_flags.recycle_next
            ws.vars.state .= ws.scratch.extra.state_recycled
        else
            onecol_method_operator!(ws, Val{ws.method.variant}(), ws.vars.state, ws.scratch.extra.swap_vec, true, true)
            # swap contents of ws.vars.state and ws.scratch.extra.swap_vec
            custom_swap!(ws.vars.state, ws.scratch.extra.swap_vec, ws.scratch.base.temp_mn_vec1)
        end

        # just applied onecol operator, so we increment
        # composition counter
        ws.composition_counter[] += 1

        if ws.composition_counter[] == args["anderson-interval"]
            @timeit timer "anderson update" COSMOAccelerators.update!(ws.accelerator, ws.vars.state, ws.vars.state_into_accelerator, 0)
        end

        push_update_to_record!(ws, record, true)

        # cannot recycle anything at next iteration
        ws.control_flags.recycle_next = false
    end
end