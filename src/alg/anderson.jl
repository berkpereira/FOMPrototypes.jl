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
        # ws.vars.xy might be overwritten, so we take note of it here
        # this is for the sole purpose of checking the norm of the
        # step just below in this code branch
        ws.scratch.xy_pre_overwrite .= ws.vars.xy

        # attempt acceleration step. if successful (ie no numerical
        # problems), this overwites ws.vars.xy with accelerated step
        @timeit timer "anderson accel" COSMOAccelerators.accelerate!(ws.vars.xy, ws.vars.xy_prev, ws.accelerator, 0)
        
        ws.scratch.accelerated_point .= ws.vars.xy # accelerated point if no numerics blowups
        ws.vars.xy .= ws.scratch.xy_pre_overwrite # retake the iterate from which we are attempting acceleration right now

        # ws.accelerator.success only indicates there was no
        # numerical blowup in the COSMOAccelerators.accelerate!
        # internals
        if ws.accelerator.success
            # at this point:
            # ws.vars.xy contains the iterate at which we stopped
            # when we began to pull the accelerate! levers;
            # ws.scratch.accelerated_point contains the candidate
            # acceleration point, which is legitimately different
            # from ws.vars.xy
            @timeit timer "fixed-point safeguard" @views ws.control_flags.accepted_accel = accel_fp_safeguard!(ws, ws_diag, ws.vars.xy, ws.scratch.accelerated_point, args["safeguard-factor"], record, full_diagnostics)

            ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application

            if ws.control_flags.accepted_accel
                ws.k_eff[] += 1 # increment effective iter counter (ie excluding unsuccessful acc attempts)

                if !args["run-fast"]
                    push_update_to_record!(ws, record, false)
                end

                # assign actually
                ws.vars.xy .= ws.scratch.accelerated_point
            elseif !args["run-fast"] # prevent recording zero update norm
                push_update_to_record!(ws, record, false)
            end


            # note that this flag serves for us to use recycled
            # work from the fixed-point safeguard step when
            # the next iteration comes around
            ws.control_flags.just_tried_accel = true
        elseif !args["run-fast"] # prevent recording zero update norm
            push_update_to_record!(ws, record, false)
        end

        # set up the first x to be composed anderon-interval times
        # before being fed to accelerator's update! method
        ws.vars.xy_into_accelerator .= ws.vars.xy # this is T^0(ws.vars.xy)
        ws.composition_counter[] = 0
    else # standard iteration with Anderson acceleration
        ws.k_eff[] += 1
        ws.k_vanilla[] += 1
        ws.k_operator[] += 1 # note: applies even when using recycled iterate from safeguard, since in safeguarding step only counted 1 operator application
        
        # copy older iterate before iterating
        ws.vars.xy_prev .= ws.vars.xy

        # we compute new iterate: if we just tried acceleration,
        # we use the recycled variable stored during the fixed-point
        # safeguarding step
        if ws.control_flags.just_tried_accel
            ws.vars.xy .= ws.scratch.xy_recycled
        else
            onecol_method_operator!(ws, ws.vars.xy, ws.scratch.swap_vec, true)
            # swap contents of ws.vars.xy and ws.scratch.swap_vec
            custom_swap!(ws.vars.xy, ws.scratch.swap_vec, ws.scratch.temp_mn_vec1)
        end

        # just applied onecol operator, so we increment
        # composition counter
        ws.composition_counter[] += 1

        if ws.composition_counter[] == args["anderson-interval"]
            @timeit timer "anderson update" COSMOAccelerators.update!(ws.accelerator, ws.vars.xy, ws.vars.xy_into_accelerator, 0)
        end

        if !args["run-fast"]
            push_update_to_record!(ws, record, true)
        end

        # cannot recycle anything at next iteration
        ws.control_flags.just_tried_accel = false
    end
end