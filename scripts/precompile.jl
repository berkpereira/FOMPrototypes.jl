using FOMPrototypes

# call main on dummy configs for precompilation

config_admm_none = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :ADMM,
    "problem-set"  => "sslsq",
    "problem-name" => "HB_ash219_lasso", # small ish
    "res-norm"     => Inf,
    "max-iter"     => 10,
    
    "acceleration"    => :none,
    "accel-memory"    => 19,
    "krylov-operator" => :tilde_A,
    "anderson-interval" => 10,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,

    "print-mod" => 10,
    "show-vlines" => true,
    
    "run-fast"  => false,
)

config_admm_krylov = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :ADMM,
    "problem-set"  => "sslsq",
    "problem-name" => "HB_ash219_lasso", # small ish
    "res-norm"     => Inf,
    "max-iter"     => 10,
    
    "acceleration"    => :krylov,
    "accel-memory"    => 19,
    "krylov-operator" => :tilde_A,
    "anderson-interval" => 10,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,

    "print-mod" => 10,
    "show-vlines" => true,
    
    "run-fast"  => false,
)

config_admm_anderson = Dict(
    "ref-solver"   => :SCS,
    "variant"      => :ADMM,
    "problem-set"  => "sslsq",
    "problem-name" => "HB_ash219_lasso", # small ish
    "res-norm"     => Inf,
    "max-iter"     => 10,
    
    "acceleration"    => :anderson,
    "accel-memory"    => 19,
    "krylov-operator" => :tilde_A,
    "anderson-interval" => 10,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,

    "print-mod" => 10,
    "show-vlines" => true,
    
    "run-fast"  => false,
)

config_1_none = Dict(
    "ref-solver"   => :SCS,
    "variant"      => Symbol("1"),
    "problem-set"  => "sslsq",
    "problem-name" => "HB_ash219_lasso", # small ish
    "res-norm"     => Inf,
    "max-iter"     => 10,
    
    "acceleration"    => :none,
    "accel-memory"    => 19,
    "krylov-operator" => :tilde_A,
    "anderson-interval" => 10,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,
    "print-mod" => 10,
    "show-vlines" => true,
    "run-fast"  => false,
)

config_1_krylov = Dict(
    "ref-solver"   => :SCS,
    "variant"      => Symbol("1"),
    "problem-set"  => "sslsq",
    "problem-name" => "HB_ash219_lasso", # small ish
    "res-norm"     => Inf,
    "max-iter"     => 10,
    
    "acceleration"    => :krylov,
    "accel-memory"    => 19,
    "krylov-operator" => :tilde_A,
    "anderson-interval" => 10,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,
    "print-mod" => 10,
    "show-vlines" => true,
    "run-fast"  => false,
)

config_1_anderson = Dict(
    "ref-solver"   => :SCS,
    "variant"      => Symbol("1"),
    "problem-set"  => "sslsq",
    "problem-name" => "HB_ash219_lasso", # small ish
    "res-norm"     => Inf,
    "max-iter"     => 10,
    
    "acceleration"    => :anderson,
    "accel-memory"    => 19,
    "krylov-operator" => :tilde_A,
    "anderson-interval" => 10,

    "rho"   => 1.0,
    "theta" => 1.0,
    
    "restart-period"    => Inf,
    "linesearch-period" => Inf,
    "linesearch-eps"    => 0.001,
    "print-mod" => 10,
    "show-vlines" => true,
    "run-fast"  => false,
)

# run solvers:
ws_admm_none, results_admm_none, x_ref_admm_none, y_ref_admm_none = FOMPrototypes.main(config_admm_none);
ws_admm_krylov, results_admm_krylov, x_ref_admm_krylov, y_ref_admm_krylov = FOMPrototypes.main(config_admm_krylov);
ws_admm_anderson, results_admm_anderson, x_ref_admm_anderson, y_ref_admm_anderson = FOMPrototypes.main(config_admm_anderson);
ws_1_none, results_1_none, x_ref_1_none, y_ref_1_none = FOMPrototypes.main(config_1_none);
ws_1_krylov, results_1_krylov, x_ref_1_krylov, y_ref_1_krylov = FOMPrototypes.main(config_1_krylov);
ws_1_anderson, results_1_anderson, x_ref_1_anderson, y_ref_1_anderson = FOMPrototypes.main(config_1_anderson);