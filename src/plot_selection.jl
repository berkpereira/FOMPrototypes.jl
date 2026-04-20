# Plot selection: presets and per-plot guards for plot_results()

const ALL_PLOTS = [
    :primal_obj, :dual_obj, :duality_gap,
    :primal_res, :dual_res,
    :state_dist, :state_chardist,
    :step_l2, :step_char,
    :update_angles, :soc_angles,
    :proj_diffs, :enforced_constraints,
    :active_set_deviation, :unseen_deviations,
    :preproj_late_flippers, :scaled_preproj_late_flippers,
    :flip_prediction_quality, :flip_prediction_quality_rate,
    :fp_metric,
]

const PLOT_GROUPS = Dict{Symbol, Vector{Symbol}}(
    :objectives   => [:primal_obj, :dual_obj, :duality_gap],
    :residuals    => [:primal_res, :dual_res],
    :distances    => [:state_dist, :state_chardist],
    :steps        => [:step_l2, :step_char],
    :angles       => [:update_angles, :soc_angles],
    :active_set   => [:proj_diffs, :enforced_constraints,
                      :active_set_deviation, :unseen_deviations,
                      :preproj_late_flippers, :scaled_preproj_late_flippers,
                      :flip_prediction_quality, :flip_prediction_quality_rate],
    :acceleration => [:fp_metric],
)

const PLOT_PRESETS = Dict{Symbol, Vector{Symbol}}(
    :none     => Symbol[],
    :minimal  => [:residuals],
    :standard => [:objectives, :residuals, :steps, :acceleration],
    :full     => collect(keys(PLOT_GROUPS)),
)

"""
    resolve_plot_set(plot_set::Symbol) -> Set{Symbol}

Expand a preset name, group name, or individual plot name into the
set of plot symbols to render.
"""
function resolve_plot_set(plot_set::Symbol)
    if haskey(PLOT_PRESETS, plot_set)
        groups = PLOT_PRESETS[plot_set]
        return Set(vcat([PLOT_GROUPS[g] for g in groups]...))
    end
    if haskey(PLOT_GROUPS, plot_set)
        return Set(PLOT_GROUPS[plot_set])
    end
    if plot_set in ALL_PLOTS
        return Set([plot_set])
    end
    @warn "Unknown plot_set :$plot_set, falling back to :full"
    return resolve_plot_set(:full)
end

should_plot(active::Set{Symbol}, name::Symbol) = name in active
