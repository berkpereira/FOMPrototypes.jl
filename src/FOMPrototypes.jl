module FOMPrototypes

# Package imports - core dependencies used across the module
using Revise
using ArgParse
using Infiltrator
using TimerOutputs
using LinearMaps
using Profile
using BenchmarkTools
using SparseArrays
using SCS, COSMO
using Random
using JLD2

# Core types and config
include("core/core.jl")

# Linear ops and algorithms (interleaved due to dependencies)
include("linops/custom_nla.jl")
include("linops/types_utils.jl")
include("alg/cones.jl")           # depends on types_utils
include("linops/alg_utils.jl")    # depends on cones
include("linops/residuals.jl")

# Algorithm implementations
include("alg/record.jl")
include("alg/vanilla.jl")
include("alg/krylov.jl")
include("alg/anderson.jl")
include("alg/randomized.jl")
include("alg/linesearch.jl")
include("alg/safeguard.jl")

# Diagnostics
include("diagnostics/diagnostics.jl")

# Solver loop
include("solver.jl")

# Initialization utilities
include("initialization.jl")

# Results plotting
include("plotting_results.jl")

# High-level API
include("api.jl")

# Exports
export SolverConfig,
       # Core API
       main,
       run_prototype,
       solve_reference,
       fetch_data,
       choose_problem,
       # Plotting/Initialization
       plot_results,
       initialise_misc,
       # Types (commonly needed by scripts)
       ProblemData,
       AbstractWorkspace,
       VanillaWorkspace,
       KrylovWorkspace,
       AndersonWorkspace,
       RandomizedWorkspace

end # module FOMPrototypes
