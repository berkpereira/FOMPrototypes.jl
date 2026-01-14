import FOMPrototypes
using Infiltrator
using JLD2

const ITER_COUNT = 200;
const SAVE_MATRIX = false  # Set to true when you want to save
const MATRIX_TAG = "optimal"  # Set to "optimal" or "non-optimal"

args = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    # "problem-set" => "sslsq",
    # "problem-name" => "NYPA_Maragal_2_huber",

    # "problem-set"  => "sslsq",
    # "problem-name" => "HB_ash292_huber",

    "problem-set"  => "mpc",
    "problem-name" => "pendulum_1",

    # "problem-set"  => "opf_socp",
    # "problem-name" => "case3_lmbd",
    
    # "problem-set"  => "opf_socp",
    # "problem-name" => "case89_pegase__sad",

    # "problem-set" => "synthetic",
    # "problem-name" => "zhang_socp", # in {toy, giselsson, zhang_socp}
    
    #####################
    # Acceleration Settings
    #####################

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-10,

    "accel-memory" => 15,
    "acceleration" => :none, # in {:none, :krylov, :anderson, :randomized}
    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 0.80, # factor for fixed-point residual safeguard check

    # Krylov-specific
    "krylov-tries-per-mem"  => 2,
    "krylov-operator"       => :B, # in {:tilde_A, :B}

    # Anderson-specific (defaults: reg = :none, with :restarted and :QR2)
    "anderson-interval"     => 10,
    "anderson-broyden-type" => :QR2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :restarted, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    # Randomized-specific
    "randomized-regularization" => 1e-8, # λ for G = V'V + λI (Tikhonov regularization)
    "randomized-operator" => :tilde_A, # in {:tilde_A, :B} - use L-I or L operator

    "rho"   => 100.0,
    "rho-update-period" => Inf,
    "theta" => 1.0,
    
    # "restart-period"    => Inf,
    # "linesearch-period" => Inf,
    # "linesearch-eps"    => 0.001,

    "max-iter"           => ITER_COUNT, # ONLY relevant with no acceleration!
    "max-k-operator"     => ITER_COUNT, # ONLY relevant with Anderson/Krylov/Randomized
    "print-mod"          => 100,
    "print-res-rel"      => true, # print relative (or absolute) residuals
    "show-vlines"        => true,
    "run-fast"           => false,
    "global-timeout"     => Inf, # seconds, including set-up time
    "loop-timeout"       => Inf, # seconds, loop excluding set-up time
);

config = FOMPrototypes.SolverConfig(args);

# run everything with a single call:
# ws, ws_diag, results, to, x_ref, y_ref = FOMPrototypes.main(args);

# get problem data:
problem = FOMPrototypes.fetch_data(config.problem_set, config.problem_name);

# call reference solver:
# model_ref, state_ref, obj_ref = FOMPrototypes.solve_reference(problem, config.problem_set, config.problem_name, config);

# call my solver:
ws, ws_diag, results, to = FOMPrototypes.run_prototype(
    problem,
    config.problem_set,
    config.problem_name,
    config,
    full_diagnostics = true,
    spec_plot_period = 50
    );

# save diagnostic matrices if requested:
if SAVE_MATRIX
    save_diagnostic_matrix(ws_diag, config.problem_set, config.problem_name, MATRIX_TAG, config.variant, config.rho, config)
end

# plot results if applicable:
if !config.run_fast
    FOMPrototypes.plot_results(
        ws,
        results,
        config.problem_set,
        config.problem_name,
        config,
        :gr)
end

function save_diagnostic_matrix(ws_diag, problem_set, problem_name, tag, variant, rho, config)
    if ws_diag === nothing
        @warn "Cannot save matrix: ws_diag is nothing (full_diagnostics was false)"
        return
    end

    # Validate that acceleration is disabled (FP residuals only tracked for vanilla method)
    if config.acceleration != :none
        error("Cannot save FP residual history with acceleration enabled. " *
              "FP residuals are only tracked for vanilla (:none) acceleration. " *
              "Current acceleration: $(config.acceleration)")
    end

    # Create output directory
    output_dir = joinpath(@__DIR__, "experimental", "saved_matrices")
    mkpath(output_dir)

    # Format rho for filename (e.g., 0.1 -> rho0p1, 1.0 -> rho1p0)
    rho_str = replace(string(rho), "." => "p")

    # Create filename: problemset_problemname_variant_rhoXpY_tag.jld2
    filename = "$(problem_set)_$(problem_name)_$(variant)_rho$(rho_str)_$(tag).jld2"
    filepath = joinpath(output_dir, filename)

    # Extract FP residuals history in correct chronological order
    # Handle circular buffer wraparound
    col_idx = ws_diag.fp_history_col[]
    history_size = size(ws_diag.fp_residuals_history, 2)

    # Determine how many columns were actually filled
    # col_idx points to the NEXT column to write, so col_idx-1 columns have been written
    # (unless we've wrapped around)
    if col_idx == 1
        # Either no iterations or exactly wrapped back to 1
        # Check if the last column has non-zero data to distinguish
        if all(ws_diag.fp_residuals_history[:, history_size] .== 0.0)
            # No data at all
            fp_residuals_history = zeros(eltype(ws_diag.fp_residuals_history), size(ws_diag.fp_residuals_history, 1), 0)
        else
            # Fully wrapped, take all columns in order [1:end]
            fp_residuals_history = ws_diag.fp_residuals_history
        end
    elseif col_idx <= history_size
        # Check if we've wrapped by looking at the column after current write position
        if col_idx < history_size && all(ws_diag.fp_residuals_history[:, col_idx] .== 0.0)
            # Haven't wrapped yet, take first (col_idx-1) columns
            fp_residuals_history = ws_diag.fp_residuals_history[:, 1:col_idx-1]
        else
            # Wrapped around: oldest data is at col_idx, newest at col_idx-1
            # Order: [col_idx:end, 1:col_idx-1]
            fp_residuals_history = hcat(
                ws_diag.fp_residuals_history[:, col_idx:end],
                ws_diag.fp_residuals_history[:, 1:col_idx-1]
            )
        end
    end

    # Save matrices, FP history, and metadata
    @save filepath tilde_A=ws_diag.tilde_A tilde_b=ws_diag.tilde_b W_inv_mat=ws_diag.W_inv_mat fp_residuals_history problem_set problem_name tag variant rho

    @info "Matrix saved to: $filepath"
    @info "  tilde_A size: $(size(ws_diag.tilde_A))"
    @info "  tilde_b size: $(size(ws_diag.tilde_b))"
    @info "  W_inv_mat size: $(size(ws_diag.W_inv_mat))"
    @info "  fp_residuals_history size: $(size(fp_residuals_history))"
end
;