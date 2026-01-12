This directory stores (in `.jld2` files) linearisation (ie operator Jacobian) matrices (`tilde_A`) obtained from problems I use for benchmarking and trying out methods (plus some metadata).

Each `.jld2` file follows the naming convention:
`{problemset}_{problemname}_{variant}_rho{value}_{tag}.jld2`

Where:
- `{variant}` is the solver variant (e.g., ADMM, PDHG)
- `rho{value}` is the rho parameter with decimal point replaced by 'p' (e.g., rho0p1 for ρ=0.1)
- `{tag}` is either `optimal` or `non-optimal` (chosen manually in `main_repl.jl`) to denote whether it came from a solved problem (optimal active set) or not.