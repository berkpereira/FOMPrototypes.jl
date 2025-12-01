import FOMPrototypes
using Infiltrator

const ITER_COUNT = 300_000

args = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    # "problem-set" => "sslsq",
    # "problem-name" => "NYPA_Maragal_3_lasso",

    "problem-set"  => "socp",
    "problem-name" => "options_pricing_K_50",

    # "problem-set"  => "opf_socp",
    # "problem-name" => "case60_c",
    
    #####################

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-3,

    "accel-memory" => 15,
    "acceleration" => :krylov, # in {:none, :krylov, :anderson}
    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 0.99, # factor for fixed-point residual safeguard check in accelerated methods

    "krylov-tries-per-mem"  => 3,
    "krylov-operator"       => :tilde_A, # in {:tilde_A, :B}
    
    # note defaults are reg = :none, with :restarted and :QR2
    "anderson-interval"     => 10,
    "anderson-broyden-type" => Symbol(1), # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    "rho"   => 100.0,
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

# run everything with a single call:
# ws, ws_diag, results, to, x_ref, y_ref = FOMPrototypes.main(args);

# get problem data:
problem = FOMPrototypes.fetch_data(args["problem-set"], args["problem-name"]);

# call reference solver:
# model_ref, state_ref, obj_ref = FOMPrototypes.solve_reference(problem, args["problem-set"], args["problem-name"], args);

# call my solver:
ws, ws_diag, results, to = FOMPrototypes.run_prototype(
    problem,
    args["problem-set"],
    args["problem-name"],
    args,
    full_diagnostics = false,
    spec_plot_period = 50
    );

# plot results if applicable:
if !args["run-fast"]
    FOMPrototypes.plot_results(
        ws,
        results,
        args["problem-set"],
        args["problem-name"],
        args,
        :plotlyjs)
end
;