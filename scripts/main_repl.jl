import FOMPrototypes
using Infiltrator

const ITER_COUNT = 20_000;

args = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    # "problem-set" => "sslsq",
    # "problem-name" => "NYPA_Maragal_3_lasso",

    "problem-set"  => "socp",
    "problem-name" => "options_pricing_K_20",

    # "problem-set"  => "opf_socp",
    # "problem-name" => "case60_c",

    # "problem-set" => "synthetic",
    # "problem-name" => "zhang_socp", # in {toy, giselsson, zhang_socp}
    
    #####################

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-6,

    "accel-memory" => 15,
    "acceleration" => :krylov, # in {:none, :krylov, :anderson}
    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 0.99, # factor for fixed-point residual safeguard check in accelerated methods

    "krylov-tries-per-mem"  => 2,
    "krylov-operator"       => :tilde_A, # in {:tilde_A, :B}
    
    # note defaults are reg = :none, with :restarted and :QR2
    "anderson-interval"     => 10,
    "anderson-broyden-type" => :QR2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :restarted, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    "rho"   => 0.1,
    "rho-update-period" => 25,
    "theta" => 1.0,
    
    # "restart-period"    => Inf,
    # "linesearch-period" => Inf,
    # "linesearch-eps"    => 0.001,

    "max-iter"           => ITER_COUNT, # ONLY relevant with no acceleration!
    "max-k-operator"     => ITER_COUNT, # ONLY relevant with Anderson/Krylov
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
;
