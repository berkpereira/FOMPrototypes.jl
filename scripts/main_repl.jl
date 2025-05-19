using Revise
import FOMPrototypes

config = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :PDHG, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    "problem-set"  => "sslsq",
    "problem-name" => "HB_abb313_lasso", # with PDHG, Krylov accelerated gets stuck even though unaccelerated works well...?
    # "problem-name" => "NYPA_Maragal_7_huber",
    
    "res-norm"     => Inf,
    "max-iter"     => 50000,
    "rel-kkt-tol"  => 1e-12,
    
    "acceleration"    => :none,
    "accel-memory"    => 50,
    "krylov-operator" => :tilde_A,
    
    "anderson-broyden-type" => :normal2, # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}
    "anderson-period"       => 2,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,

    "print-mod"          => 3000,
    "print-res-rel"      => true, # print relative (or absolute) residuals/duality gaps
    "show-vlines"        => true,
    "run-fast"           => false,
    "global-timeout"     => Inf, # seconds, including set-up time
    "loop-timeout"       => Inf, # seconds, loop excluding set-up time
);

# run everything with a single call:
ws, results, to, x_ref, y_ref = FOMPrototypes.main(config);

# for just reference solver:
# ws_ref, results_ref, x_ref, y_ref = FOMPrototypes.solve_reference(problem, config["problem-set"], config["problem-name"], config);

# for just prototype:
# problem = FOMPrototypes.fetch_data(config["problem-set"], config["problem-name"]);
# ws, results, to = FOMPrototypes.run_prototype(problem, config["problem-set"], config["problem-name"], config);
;