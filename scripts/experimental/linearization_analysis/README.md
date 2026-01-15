# linearization_analysis

This directory contains scripts for analyzing linearization (Jacobian) matrices of first-order methods, along with saved `.jld2` matrix data files.

## Scripts

### `linear_solver_comparison.jl`

Compares iterative linear system solvers on the system `(tilde_A - I)x = -tilde_b`:

- GMRES (full and restarted variants)
- Randomized subspace methods (various subspace dimensions, with/without residual augmentation)
- Produces convergence plots and performance summary tables

**Standalone**: Does not require `fp_residuals_history` in the data file.

### `spectral_residual_analysis.jl`

Performs spectral and residual trajectory analysis:

- Eigenvalue spectrum visualization on the complex plane
- Near-unit eigenmode identification and projection tracking
- Phase evolution for complex conjugate pairs (oscillatory behavior detection)
- SVD-based low-rank subspace approximation
- Eigenbasis decomposition showing mode energies

**Graceful degradation**: Uses `fp_residuals_history` when available; otherwise performs only eigenvalue spectrum analysis.

### `utils.jl`

Shared utilities for both analysis scripts:

- Configuration constants (tolerances, iteration limits, plot settings)
- File I/O and data loading/validation
- Matrix property display helpers
- Plotting setup

## Usage

1. Edit the script to select the matrix file by setting these constants:

```julia
const PROBLEM_SET = "mpc"
const PROBLEM_NAME = "pendulum_1"
const VARIANT = :ADMM
const RHO = 100.0
const TAG = "non-optimal"
```

2. Run scripts independently:

```bash
julia linear_solver_comparison.jl
julia spectral_residual_analysis.jl
```

## Matrix Data Files

Each `.jld2` file follows the naming convention:
`{problemset}_{problemname}_{variant}_rho{value}_{tag}.jld2`

Where:
- `{variant}` is the solver variant (e.g., ADMM, PDHG)
- `rho{value}` is the rho parameter with decimal point replaced by 'p' (e.g., rho0p1 for ρ=0.1)
- `{tag}` is either `optimal` or `non-optimal` (chosen manually in `main_repl.jl`) to denote whether it came from a solved problem (optimal active set) or not.

## Notes on specific matrices

### `sslsq/NYPA_Maragal_1_lasso`

At optimality this is an example of a matrix which does ***not*** have any unit eigenvalues, so associated linear systems are always well-determined.

### `sslsq/NYPA_Maragal_1_huber`

At optimality this is an example of a matrix which ***does*** have a unit eigenvalue, so the associated linear systems matrix is singular!

### `mpc/pendulum_1`

Out of optimality, this was saved just following a seemingly quite spirally trajectory!

No unit eigenvalues (only very close), and null space of dimension almost half of dimension of `tilde_A`.
