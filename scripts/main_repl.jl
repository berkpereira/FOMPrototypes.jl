import FOMPrototypes
using Infiltrator

args = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    # "problem-set"  => "maros",
    # "problem-name" => "AUG3DQP",

    # "problem-set" => "sslsq",
    # "problem-name" => "NYPA_Maragal_3_lasso",

    # "problem-set"  => "socp",
    # "problem-name" => "options_pricing_K_30",

    "problem-set"  => "opf_socp",
    "problem-name" => "case240_pserc",

    # this can break when estimation of max_τ goes wrong (negative! even)
    # "problem-set"  => "mpc",
    # "problem-name" => "springMass_4",

    # NB this LP gives bad Arnoldi breakdowns (rank deficient
    # LLS system, breaks forward solve) when using ADMM, Krylov,
    # mem = 40, tries_per_mem = 3, euclidean safeguard, krylov operator = :tilde_A
    
    #####################

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-3,

    "accel-memory" => 15,
    "acceleration" => :krylov, # in {:none, :krylov, :anderson}
    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 1.0, # factor for fixed-point residual safeguard check in accelerated methods

    "krylov-tries-per-mem"  => 3,
    "krylov-operator"       => :tilde_A, # in {:tilde_A, :B}
    
    # note defaults are reg = :none, with :restarted and :QR2
    "anderson-interval"     => 10,
    "anderson-broyden-type" => Symbol(1), # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    "rho"   => 0.1,
    "theta" => 1.0,
    
    # "restart-period"    => Inf,
    # "linesearch-period" => Inf,
    # "linesearch-eps"    => 0.001,

    "max-iter"           => 1000, # ONLY relevant with no acceleration!
    "max-k-operator"     => 1000, # ONLY relevant with Anderson/Krylov
    "print-mod"          => 100,
    "print-res-rel"      => true, # print relative (or absolute) residuals
    "show-vlines"        => true,
    "run-fast"           => false,
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
    FOMPrototypes.plot_results(ws, results, args["problem-set"], args["problem-name"], args, :gr)
end
;