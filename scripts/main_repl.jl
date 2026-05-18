import FOMPrototypes
using Infiltrator
using JLD2

include("script_utils.jl")  # For save_diagnostic_matrix

const ITER_COUNT = 300
const SAVE_MATRIX = false  # Set to true when you want to save
const MATRIX_TAG = "optimal"  # Set to "optimal" or "non-optimal"

args = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    # "problem-set" => "sslsq",
    # "problem-name" => "NYPA_Maragal_3_huber",

    # "problem-set"  => "sslsq",
    # "problem-name" => "HB_ash292_huber",

    "problem-set"  => "mpc",
    "problem-name" => "nonlinearChain_3",

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
    "rel-kkt-tol"  => 1e-9,

    "accel-memory" => 15,
    "acceleration" => :krylov, # in {:none, :krylov, :anderson, :randomized}
    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 0.9, # factor for fixed-point residual safeguard check

    # Krylov-specific
    "krylov-tries-per-mem"  => 1,
    "krylov-operator"       => :B, # in {:tilde_A, :B}
    "krylov-zero-init"      => false, # if true, initialise Krylov basis with random unit vector instead of warm-started FP residual

    # Anderson-specific (defaults: reg = :none, with :restarted and :QR2)
    "anderson-interval"     => 10,
    "anderson-broyden-type" => :QR2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :restarted, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    # Randomized-specific
    "randomized-regularization" => 1e-8, # λ for G = V'V + λI (Tikhonov regularization)
    "randomized-operator" => :tilde_A, # in {:tilde_A, :B} - use L-I or L operator

    "rho"   => 0.1,
    "rho-update-period" => 100,
    "theta" => 1.0,

    # "restart-period"    => Inf,
    # "linesearch-period" => Inf,
    # "linesearch-eps"    => 0.001,

    "max-iter"           => ITER_COUNT, # ONLY relevant with no acceleration!
    "max-k-operator"     => ITER_COUNT, # ONLY relevant with Anderson/Krylov/Randomized
    "print-mod"          => 100,
    "print-res-rel"      => true, # print relative (or absolute) residuals
    "plot-set"           => :full, # in {:none, :minimal, :standard, :full} or a group/plot name
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
    spec_plot_period = 50,
    # break_at = [60, 120],   # drop into Infiltrator at these iterations to inspect ws, ws_diag live
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
        :plotlyjs)
end

if config.acceleration == :krylov && config.krylov_zero_init
    println("⚠️⚠️⚠️⚠️⚠️  WARNING: Krylov GMRES is currently initialised with zeros (random unit vector) instead of the warm-started fixed-point residual — results are expected to be degraded!  ⚠️⚠️⚠️⚠️⚠️")
end
;