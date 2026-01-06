import FOMPrototypes

const ITER_COUNT = 20_000;

# Example configuration for randomized subspace acceleration
args = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :ADMM, # in {:PDHG, :ADMM, Symbol(1), Symbol(2), Symbol(3), Symbol(4)}

    "problem-set" => "sslsq",
    "problem-name" => "NYPA_Maragal_3_lasso",

    #####################
    # Randomized Subspace Acceleration Settings
    #####################

    "acceleration" => :randomized, # in {:none, :krylov, :anderson, :randomized}
    "accel-memory" => 15, # subspace dimension (s) for randomized acceleration

    "safeguard-norm" => :char, # in {:euclid, :char, :none}
    "safeguard-factor" => 1.0, # factor for fixed-point residual safeguard check

    "randomized-regularization" => 1e-8, # λ for G = V'V + λI (Tikhonov regularization)
    "randomized-operator" => :tilde_A, # in {:tilde_A, :B} - use L-I or L operator

    #####################
    # General Settings
    #####################

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-3,

    "rho"   => 0.1,
    "rho-update-period" => Inf,
    "theta" => 1.0,

    "max-iter"           => ITER_COUNT, # ONLY relevant with no acceleration!
    "max-k-operator"     => ITER_COUNT, # ONLY relevant with Anderson/Krylov/Randomized
    "print-mod"          => 100,
    "print-res-rel"      => true, # print relative (or absolute) residuals
    "show-vlines"        => true,
    "run-fast"           => true,
    "global-timeout"     => Inf, # seconds, including set-up time
    "loop-timeout"       => Inf, # seconds, loop excluding set-up time
);

config = FOMPrototypes.SolverConfig(args);

# Get problem data:
problem = FOMPrototypes.fetch_data(config.problem_set, config.problem_name);

# Run the solver with randomized acceleration:
ws, ws_diag, results, to = FOMPrototypes.run_prototype(
    problem,
    config.problem_set,
    config.problem_name,
    config,
    full_diagnostics = false,
    spec_plot_period = 50
);

# Plot results if applicable:
if !config.run_fast
    FOMPrototypes.plot_results(
        ws,
        results,
        config.problem_set,
        config.problem_name,
        config,
        :gr)
end
