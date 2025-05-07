using Revise
import FOMPrototypes

config = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}
    "problem-set"  => "sslsq",
    "problem-name" => "NYPA_Maragal_5_lasso",
    
    "res-norm"     => Inf,
    "max-iter"     => 1000,
    "rel-kkt-tol"  => 1e-10,
    
    "acceleration"    => :none,
    "accel-memory"    => 200,
    "krylov-operator" => :tilde_A,
    
    "anderson-broyden-type" => Symbol(1), # in {Symbol(1), :normal2, :QR2}
    "anderson-mem-type"     => :rolling, # in {:rolling, :restarted}
    "anderson-reg"          => :none, # in {:none, :tikonov, :frobenius}
    "anderson-period"       => 2,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,

    "print-mod"          => 50,
    "residuals-relative" => true,
    "show-vlines"        => true,
    "run-fast"           => false,
);

# run everything with a single call:
ws, results, x_ref, y_ref = FOMPrototypes.main(config);

# for just reference solver:
# ws_ref, results_ref, x_ref, y_ref = FOMPrototypes.solve_reference(problem, config);

# for just prototype:
# problem = FOMPrototypes.fetch_data(config);
# ws, results = FOMPrototypes.run_prototype(problem, config);
;