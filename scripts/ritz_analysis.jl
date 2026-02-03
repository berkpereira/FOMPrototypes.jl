"""
Ritz Value and Vector Analysis for Krylov Subspace Acceleration

Visualizes the convergence of Ritz values (eigenvalues of the Hessenberg matrix)
to the true spectrum of the linearized operator, and how well the Krylov subspace
captures the near-unit eigenspace (the "slow" modes that limit convergence).

Prerequisites:
  - Run main_repl.jl with `full_diagnostics = true` to populate `ws` and `ws_diag`
  - Requires ws.krylov_basis, ws.givens_count[], and ws_diag.H_unmod, ws_diag.tilde_A

Usage (from REPL after running main_repl.jl):
  include("scripts/ritz_analysis.jl")
"""

using Plots, LinearAlgebra

# =============================================================================
# Configuration
# =============================================================================
const UNIT_TOL = 0.05       # threshold for "near-unit" eigenvalues: |λ - 1| < UNIT_TOL
const FPS = 2              # frames per second for animations
const XLIMS = (-0.1, 1.1)  # complex plane x-axis limits
const YLIMS = (-0.6, 0.6)  # complex plane y-axis limits

# =============================================================================
# 1. Ritz Value Frame Export (for Beamer presentations)
# =============================================================================
"""
Export Ritz value convergence as individual PDF frames for Beamer overlays.

Creates one PDF per frame in `output_dir`. Use `step` to skip frames (e.g., step=2
exports every other frame). Prints Beamer usage snippet after export.

Example:
    export_ritz_frames(ws, ws_diag, output_dir="export_frames", step=1)
"""
function export_ritz_frames(ws, ws_diag;
        output_dir="ritz_frames", xlims=XLIMS, ylims=YLIMS, step=1)
    mkpath(output_dir)
    λ_true = eigvals(Matrix(ws_diag.tilde_A))

    frames = 1:step:ws.givens_count[]
    for (i, k) in enumerate(frames)
        λ_ritz = eigvals(Matrix(ws_diag.H_unmod[1:k, 1:k] + I))
        n_true_1 = count(x -> abs(x - 1) < UNIT_TOL, λ_true)
        n_ritz_1 = count(x -> abs(x - 1) < UNIT_TOL, λ_ritz)

        # Unit circle overlay (radius 1/2 centered at 0.5 + 0i)
        θ_circle = range(0, 2π, length=100)
        circle_x = 0.5 .+ 0.5 .* cos.(θ_circle)
        circle_y = 0.5 .* sin.(θ_circle)

        p = plot(circle_x, circle_y,
            linecolor=:lightgray, linestyle=:dash, linewidth=1.5, label="",
            xlims=xlims, ylims=ylims, aspect_ratio=:equal,
            xlabel="Re(λ)", ylabel="Im(λ)",
            title="m=$(ws.p.m), n=$(ws.p.n), k=$k, ≈unit: $n_ritz_1/$n_true_1")

        scatter!(p, real.(λ_true), imag.(λ_true),
            marker=:x, markersize=6, markerstrokewidth=2, color=:gray, alpha=0.5,
            label="True spectrum")
        scatter!(p, real.(λ_ritz), imag.(λ_ritz),
            marker=:circle, markersize=8, markerstrokewidth=2,
            color=:dodgerblue, markerstrokecolor=:navy,
            label="Ritz values")

        savefig(p, joinpath(output_dir, "ritz_$(lpad(i, 3, '0')).pdf"))
    end

    n = length(frames)
    @info "Exported $n frames to $output_dir/"
    println("""
    Beamer usage (add to the .tex file):

    \\begin{frame}{Ritz Value Convergence}
        \\foreach \\n in {1,...,$n} {%
            \\only<\\n>{\\includegraphics[width=0.8\\textwidth]{$output_dir/ritz_\\ifnum\\n<10 00\\n\\else\\ifnum\\n<100 0\\n\\else\\n\\fi\\fi}}%
        }
    \\end{frame}
    """)
end

# =============================================================================
# 2. Ritz Value Convergence Animation (GIF)
# =============================================================================
"""
Animate Ritz values (eigenvalues of H_k + I) converging to the true spectrum
of tilde_A as the Krylov subspace dimension k increases.

The true spectrum is shown as gray X markers; Ritz values as blue circles.
"""
function animate_ritz_values(ws, ws_diag; fps=FPS, xlims=XLIMS, ylims=YLIMS)
    # Compute true spectrum once (B = tilde_A - I, so tilde_A spectrum = B spectrum + 1)
    λ_true = eigvals(Matrix(ws_diag.tilde_A))

    # Unit circle overlay (radius 1/2 centered at 0.5 + 0i)
    θ_circle = range(0, 2π, length=100)
    circle_x = 0.5 .+ 0.5 .* cos.(θ_circle)
    circle_y = 0.5 .* sin.(θ_circle)

    n_true_1 = count(x -> abs(x - 1) < UNIT_TOL, λ_true)

    anim = @animate for k in 1:ws.givens_count[]
        λ_ritz = eigvals(Matrix(ws_diag.H_unmod[1:k, 1:k] + I))
        n_ritz_1 = count(x -> abs(x - 1) < UNIT_TOL, λ_ritz)

        # Plot circle first as background
        p = plot(circle_x, circle_y,
            linecolor=:lightgray, linestyle=:dash, linewidth=1.5, label="",
            xlims=xlims, ylims=ylims, aspect_ratio=:equal,
            xlabel="Re(λ)", ylabel="Im(λ)",
            title="m=$(ws.p.m), n=$(ws.p.n), k=$k, ≈1: $n_ritz_1/$n_true_1")

        # Plot true spectrum
        scatter!(p, real.(λ_true), imag.(λ_true),
            marker=:x, markersize=6, markerstrokewidth=2, color=:gray, alpha=0.5,
            label="True spectrum")

        # Overlay Ritz values with stronger markers
        scatter!(p, real.(λ_ritz), imag.(λ_ritz),
            marker=:circle, markersize=8, markerstrokewidth=2,
            color=:dodgerblue, markerstrokecolor=:navy,
            label="Ritz values")
    end

    gif(anim, fps=fps)
end

# =============================================================================
# 2. Near-Unit Eigenspace Capture
# =============================================================================
"""
Compute the true eigensystem and identify near-unit eigenvalues/eigenvectors.
Returns (λ_true, V_true, near_unit_indices, V_near_unit, λ_near_unit).
"""
function compute_near_unit_eigenspace(ws_diag; unit_tol=UNIT_TOL)
    F_true = eigen(Matrix(ws_diag.tilde_A))
    λ_true, V_true = F_true.values, F_true.vectors

    near_unit_mask = abs.(λ_true .- 1.0) .< unit_tol
    near_unit_indices = findall(near_unit_mask)
    V_near_unit = V_true[:, near_unit_indices]
    λ_near_unit = λ_true[near_unit_indices]

    @info "Found $(length(near_unit_indices)) near-unit eigenvalues (|λ-1| < $unit_tol)"

    return (λ_true=λ_true, V_true=V_true,
            near_unit_indices=near_unit_indices,
            V_near_unit=V_near_unit, λ_near_unit=λ_near_unit)
end

"""
Plot how well the Krylov subspace captures the near-unit eigenspace as k grows.

For each k, computes the mean squared projection norm of near-unit eigenvectors
onto the Krylov subspace. Value of 1.0 means full capture.
"""
function plot_near_unit_capture(ws, ws_diag; unit_tol=UNIT_TOL)
    eig_data = compute_near_unit_eigenspace(ws_diag; unit_tol=unit_tol)
    V_near_unit = eig_data.V_near_unit

    if isempty(eig_data.near_unit_indices)
        @warn "No near-unit eigenvalues found with tolerance $unit_tol"
        return nothing
    end

    subspace_capture = Float64[]
    for k in 1:ws.givens_count[]
        Q_k = ws.krylov_basis[:, 1:k]

        total_capture = 0.0
        for j in 1:size(V_near_unit, 2)
            v = V_near_unit[:, j]
            proj = Q_k * (Q_k' * v)
            total_capture += norm(proj)^2 / norm(v)^2
        end
        push!(subspace_capture, total_capture / size(V_near_unit, 2))
    end

    plt = plot(1:ws.givens_count[], subspace_capture,
        xlabel="k (Krylov dimension)",
        ylabel="mean capture (||Π v||² / ||v||²)",
        title="Krylov subspace capture of near-unit eigenspace\n($(length(eig_data.near_unit_indices)) eigenvectors with |λ-1| < $unit_tol)",
        marker=:circle, legend=false, ylims=(0, 1.1),
        linewidth=2)
    hline!(plt, [1.0], linestyle=:dash, color=:gray)

    display(plt)
    return plt
end

# =============================================================================
# Run analysis (requires ws and ws_diag to be defined)
# =============================================================================
if @isdefined(ws) && @isdefined(ws_diag)
    @info "Running Ritz analysis..."

    @info "Generating Ritz value convergence animation..."
    animate_ritz_values(ws, ws_diag)

    # @info "Plotting near-unit eigenspace capture..."
    # plot_near_unit_capture(ws, ws_diag)
else
    @warn "ws and ws_diag not defined. Run main_repl.jl with full_diagnostics=true first."
end
