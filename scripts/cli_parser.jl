# CLI argument parsing - moved from FOMPrototypes module
# This file should be included by scripts that need CLI parsing

using ArgParse

##################################
# Parsing command line arguments #
##################################

function parse_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--ref-solver"
        help = "Reference solver to use: SCS or Clarabel."
        arg_type = Symbol
        required = true

        "--variant", "-v"
        help = "Variant to use: ADMM, PDHG, 1, 2, 3, or 4."
        arg_type = Symbol
        required = true

        "--problem-set"
        help = "Problem identifier to run"
        arg_type = String
        required = true

        "--problem-name"
        help = "Name of the problem to run"
        arg_type = String
        required = true

        "--res-norm"
        help = "Residual p-norm to use for various solver purposes."
        arg_type = Float64
        default = Inf

        "--max-iter"
        help = "Maximum number of solver iterations"
        arg_type = Int
        default = 1000

        "--max-k-operator"
        help = "Maximum number of operator applications (relevant for accelerated methods)"
        arg_type = Int
        default = 1000

        "--print-mod"
        help = "How many iterations between printing info."
        arg_type = Int
        default = 50

        "--rho"
        help = "PrePDHG ρ step size"
        arg_type = Float64
        default = 1.0

        "--theta"
        help = "PrePDHG θ parameter"
        arg_type = Float64
        default = 1.0

        "--acceleration", "-a"
        help = "Acceleration type: none, anderson, or krylov."
        arg_type = Symbol
        default = :none

        "--accel-memory"
        help = "Memory size for acceleration methods."
        arg_type = Int
        default = 20

        "--krylov-operator"
        help = "Krylov operator type: tilde_A or B."
        arg_type = Symbol
        default = :tilde_A

        "--krylov-tries-per-mem"
        help = "How many acceleration attempts per Krylov memory fill-up."
        arg_type = Int
        default = 3

        "--safeguard-norm"
        help = "Norm used in acceleration safeguard: euclid, char, or none."
        arg_type = Symbol
        default = :char

        "--safeguard-factor"
        help = "Factor for fixed-point residual safeguard check in accelerated methods."
        arg_type = Float64
        default = 0.99

        "--anderson-interval"
        help = "Anderson acceleration is applied to the operator obtained from composing the optimiser operator THIS many times."
        arg_type = Int
        default = 10

        "--anderson-broyden-type"
        help = "Which type of Broyden update to use: 1, normal2, or QR2."
        arg_type = Symbol
        default = :normal2

        "--anderson-mem-type"
        help = "Memory type for Anderson acceleration: rolling or restarted."
        arg_type = Symbol
        default = :rolling

        "--anderson-reg"
        help = "Regulariser for Anderson least-squares problem: none, Tikonov, or Frobenius."
        arg_type = Symbol
        default = :none

        "--rel-kkt-tol"
        help = "Relative KKT tolerance for stopping criterion."
        arg_type = Float64
        default = 1e-6

        "--residual-period"
        help = "How many iterations between residual metric refreshes."
        arg_type = Int
        default = 25

        "--rho-update-period"
        help = "How many iterations between adaptive ρ updates (set to Inf to disable)."
        arg_type = Real
        default = Inf

        "--run-fast"
        help = "Run fast mode (no plotting, less data recording during run)."
        arg_type = Bool
        default = true

        "--print-res-rel"
        help = "Use relative metrics when printing iter info."
        arg_type = Bool
        default = true

        "--show-vlines"
        help = "Show relevant vertical dashed lines in plots."
        arg_type = Bool
        default = false

        "--global-timeout"
        help = "Global timeout for the solver (in seconds)."
        arg_type = Float64
        default = 60.0

        "--loop-timeout"
        help = "Timeout for iterative loop (in seconds)."
        arg_type = Float64
        default = 30.0

        ### ignoring these at the moment... ###

        "--restart-period"
        help = "Restart period for the solver."
        arg_type = Real
        default = Inf

        "--linesearch-period"
        help = "Period for performing line search."
        arg_type = Real
        default = Inf

        "--linesearch-eps"
        help = "Epsilon parameter for line search."
        arg_type = Float64
        default = 0.001
    end

    return FOMPrototypes.SolverConfig(parse_args(s))
end
