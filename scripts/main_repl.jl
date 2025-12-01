import FOMPrototypes
using Infiltrator

const ITER_COUNT = 210;

args = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    # "problem-set" => "sslsq",
    # "problem-name" => "NYPA_Maragal_3_lasso",

    "problem-set"  => "socp",
    "problem-name" => "options_pricing_K_50",

    # "problem-set"  => "opf_socp",
    # "problem-name" => "case60_c",
    
    #####################

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-5,

    "accel-memory" => 15,
    "acceleration" => :anderson, # in {:none, :krylov, :anderson}
    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 0.99, # factor for fixed-point residual safeguard check in accelerated methods

    "krylov-tries-per-mem"  => 3,
    "krylov-operator"       => :B, # in {:tilde_A, :B}
    
    # note defaults are reg = :none, with :restarted and :QR2
    "anderson-interval"     => 10,
    "anderson-broyden-type" => Symbol(1), # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    "rho"   => 1.0,
    "theta" => 1.0,
    
    # "restart-period"    => Inf,
    # "linesearch-period" => Inf,
    # "linesearch-eps"    => 0.001,

    "max-iter"           => ITER_COUNT, # ONLY relevant with no acceleration!
    "max-k-operator"     => ITER_COUNT, # ONLY relevant with Anderson/Krylov
    "print-mod"          => 1000,
    "print-res-rel"      => true, # print relative (or absolute) residuals
    "show-vlines"        => true,
    "run-fast"           => true,
    "global-timeout"     => Inf, # seconds, including set-up time
    "loop-timeout"       => Inf, # seconds, loop excluding set-up time
);

config = FOMPrototypes.SolverConfig(args);

# run everything with a single call:
# ws, ws_diag, results, to, x_ref, y_ref = FOMPrototypes.main(args);

# get problem data:
problem = FOMPrototypes.fetch_data(config.problem_set, config.problem_name);

# call reference solver:
model_ref, state_ref, obj_ref = FOMPrototypes.solve_reference(problem, config.problem_set, config.problem_name, config);

# call my solver:
ws, ws_diag, results, to = FOMPrototypes.run_prototype(
    problem,
    config.problem_set,
    config.problem_name,
    config,
    full_diagnostics = false,
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
        :plotlyjs)
end
;
