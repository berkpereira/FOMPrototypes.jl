# using Revise
import FOMPrototypes

args = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :PDHG, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    # "problem-set"  => "sslsq",
    # "problem-name" => "HB_abb313_lasso", # with PDHG, Krylov accelerated gets stuck even though unaccelerated works well...?
    # "problem-name" => "HB_ash219_lasso", # well with old Krylov
    # "problem-name" => "NYPA_Maragal_5_lasso",
    # "problem-set"  => "maros",
    # "problem-name" => "STADAT3",

    "problem-set"  => "toy",
    "problem-name" => "toy",
    
    "res-norm"     => Inf,
    "max-iter"     => 100,
    "rel-kkt-tol"  => 1e-12,

    "acceleration" => :krylov,
    "accel-memory" => 5,
    "safeguard-norm" => :euclid, # in {:euclid, :char, :none}
    # "safeguard-factor" => 1.0, # NOT YET IN USE, factor for fixed-point residual safeguard check in accelerated methods

    "krylov-tries-per-mem"  => 1,
    "krylov-operator"       => :tilde_A, # in {:tilde_A, :B}
    
    # note defaults are reg = :none, with :restarted and :QR2
    "anderson-interval"     => 1,
    "anderson-broyden-type" => :normal2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :restarted, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}

    "rho"   => 1.0,
    "theta" => 1.0,
    
    # "restart-period"    => Inf,
    # "linesearch-period" => Inf,
    # "linesearch-eps"    => 0.001,

    "print-mod"          => 1000,
    "print-res-rel"      => true, # print relative (or absolute) residuals/duality gaps
    "show-vlines"        => true,
    "run-fast"           => true,
    "global-timeout"     => 2.0, # seconds, including set-up time
    "loop-timeout"       => Inf, # seconds, loop excluding set-up time
);

# run everything with a single call:
# ws, results, to, x_ref, y_ref = FOMPrototypes.main(args);

# get problem data:
problem = FOMPrototypes.fetch_data(args["problem-set"], args["problem-name"]);

# call reference solver:
model_ref, x_ref, s_ref, y_ref, obj_ref = FOMPrototypes.solve_reference(problem, args["problem-set"], args["problem-name"], args);

# call my solver:
ws, results, to = FOMPrototypes.run_prototype(problem, args["problem-set"], args["problem-name"], args);

# plot results if applicable:
if !args["run-fast"]
    FOMPrototypes.plot_results(results, args["problem-set"], args["problem-name"], args, :gr);
end
;