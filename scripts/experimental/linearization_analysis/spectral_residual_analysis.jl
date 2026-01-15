"""
Spectral and Residual Analysis for Linearization Matrices

This script analyzes the spectral properties of tilde_A matrices
and the fixed-point residual trajectories from optimization iterations.

Analysis sections:
1. Eigenvalue Trajectory Correlation
   - Eigenvalue spectrum on complex plane
   - Projections onto near-unit eigenmodes
   - Phase evolution for complex conjugate pairs

2. Low-Rank Subspace Approximation (SVD)
   - Singular value spectrum
   - Cumulative variance explained
   - Principal component heatmaps

3. Eigenbasis Decomposition
   - Coefficient energy distribution
   - Dominant eigenmode identification
   - Coefficient magnitude heatmaps

Graceful degradation: If fp_residuals_history is not available,
only the eigenvalue spectrum analysis is performed.
"""

using JLD2
using LinearAlgebra
using SparseArrays
using Statistics
using Plots

# Include shared utilities
include("utils.jl")

# Set up plotting
setup_plotting()

# =============================================================================
# Configuration
# =============================================================================
# Import defaults from utils.jl (can override locally if needed)
const UNIT_TOL = DEFAULT_UNIT_TOL
const COMPLEX_TOL = DEFAULT_COMPLEX_TOL
const EIGEN_MAX_SIZE = DEFAULT_EIGEN_MAX_SIZE
const EIGEN_WARN_THRESHOLD = DEFAULT_EIGEN_WARN_THRESHOLD

# =============================================================================
# File Selection
# =============================================================================
# Select which file to load by specifying its components
const MATRICES_DIR = @__DIR__
const PROBLEM_SET = "mpc"
const PROBLEM_NAME = "pendulum_1"
const VARIANT = :ADMM  # e.g., :ADMM, :PDHG
const RHO = 100.0       # e.g., 0.1, 1.0
const TAG = "non-optimal"  # "optimal" or "non-optimal"

# =============================================================================
# Load Data
# =============================================================================
filename, filepath = construct_filepath(MATRICES_DIR, PROBLEM_SET, PROBLEM_NAME, VARIANT, RHO, TAG)
@info "Loading matrix data from: $filename"

data = load_matrix_data(filepath)
display_matrix_info(data)

# Extract the matrices
tilde_A = data["tilde_A"]
tilde_b = data["tilde_b"]
fp_residuals_history = get(data, "fp_residuals_history", nothing)

display_matrix_properties(tilde_A)

# =============================================================================
# Spectral Analysis Utilities
# =============================================================================

"""
Compute eigendecomposition with size checks and error handling.
Returns named tuple (eigenvalues, eigenvectors) or nothing if computation fails/skipped.
"""
function safe_eigen(A; max_size=EIGEN_MAX_SIZE, warn_threshold=EIGEN_WARN_THRESHOLD)
    n = size(A, 1)

    if n > max_size
        @warn "Matrix too large for full eigendecomposition (size=$n > $max_size), skipping spectral analysis"
        return nothing
    end

    if n > warn_threshold
        @info "Computing eigendecomposition for matrix of size $n (this may take a while)..."
    end

    try
        A_dense = Matrix(A)
        λ, V = eigen(A_dense)
        return (eigenvalues=λ, eigenvectors=V)
    catch e
        @warn "Eigendecomposition failed" exception=e
        return nothing
    end
end

"""
Classify eigenvalues into categories: near-unit, complex, etc.
Returns named tuple with masks and distances.
"""
function classify_eigenvalues(λ; unit_tol=UNIT_TOL, complex_tol=COMPLEX_TOL)
    distances = abs.(λ .- 1.0)
    near_unit_mask = distances .< unit_tol
    complex_mask = abs.(imag.(λ)) .> complex_tol

    return (
        near_unit_mask=near_unit_mask,
        complex_mask=complex_mask,
        distances=distances,
    )
end

"""
Compute projection energies of residuals onto an eigenvector subspace.

Note: For non-symmetric matrices, eigenvectors are generally NOT orthonormal.
This function computes the squared magnitudes of the coefficients when expressing
the residual in the eigenbasis (via V \\ r), not orthogonal projections.

The "energy" here represents the contribution of each eigenspace to the residual
when decomposed in the eigenbasis. For a proper energy partition that sums to
||r||², one would need orthogonal projections, but that loses the connection
to the eigenspaces of tilde_A.
"""
function compute_subspace_energies(residuals, eigenvectors, mask)
    num_residuals = size(residuals, 2)
    energies = zeros(num_residuals)

    # Compute coefficients in full eigenbasis (V * coeffs ≈ residual)
    # This is more numerically stable than computing for subspaces separately
    V_full = eigenvectors

    for i in 1:num_residuals
        # Solve V * c = r for coefficients c (in least-squares sense)
        coeffs_full = V_full \ residuals[:, i]
        # Sum squared magnitudes for the masked subset
        energies[i] = sum(abs2.(coeffs_full[mask]))
    end

    return energies
end

"""
Plot eigenvalue spectrum on complex plane with reference circle.

The reference circle has radius 0.5 centered at (0.5, 0), which is the
critical boundary for convergence of averaged iterates. Eigenvalues inside
this circle contribute to convergent behavior.
"""
function plot_eigenvalue_spectrum(λ; unit_tol=UNIT_TOL, title_suffix="")
    plt = scatter(
        real.(λ), imag.(λ),
        xlabel="Real part",
        ylabel="Imaginary part",
        title="Eigenvalue Spectrum of tilde_A" * title_suffix,
        legend=:topright,
        aspect_ratio=:equal,
        markersize=4,
        alpha=0.6,
        label="Eigenvalues",
        size=(600, 600)
    )

    # Add reference circle: radius 0.5 centered at (0.5, 0)
    # This is the critical boundary for convergence analysis
    θ = range(0, 2π, length=200)
    plot!(plt, 0.5 .+ 0.5 .* cos.(θ), 0.5 .* sin.(θ),
          linestyle=:solid, color=:red, label="Reference circle", linewidth=2)

    # Add marker at center of reference circle
    scatter!(plt, [0.5], [0.0], markershape=:circle, color=:red,
             markersize=4, label="")

    # Highlight near-unit eigenvalues (near λ=1)
    near_unit = abs.(λ .- 1.0) .< unit_tol
    if any(near_unit)
        scatter!(plt, real.(λ[near_unit]), imag.(λ[near_unit]),
                markersize=8, color=:orange, markershape=:star5,
                label="Near-unit (|λ-1| < $unit_tol)")
    end

    return plt
end

# =============================================================================
# Eigenvalue Spectrum Analysis (always runs)
# =============================================================================
@info "=== Eigenvalue Spectrum Analysis ==="

eigen_result = safe_eigen(tilde_A)

if isnothing(eigen_result)
    @warn "Skipping all spectral analysis (eigendecomposition unavailable)"
else
    λ = eigen_result.eigenvalues
    V = eigen_result.eigenvectors

    # Classify eigenvalues
    eigen_class = classify_eigenvalues(λ)
    near_unit_indices = findall(eigen_class.near_unit_mask)
    complex_indices = findall(eigen_class.complex_mask)

    @info "Eigenvalue classification:"
    @info "  Total eigenvalues: $(length(λ))"
    @info "  Near-unit (|λ-1| < $UNIT_TOL): $(length(near_unit_indices))"
    @info "  Complex (|Im(λ)| > $COMPLEX_TOL): $(length(complex_indices))"

    # Plot eigenvalue spectrum on complex plane
    plt_spectrum = plot_eigenvalue_spectrum(λ; unit_tol=UNIT_TOL,
        title_suffix=" ($PROBLEM_NAME, $VARIANT, ρ=$RHO)")
    display(plt_spectrum)
end

# =============================================================================
# Fixed-Point Residual Analysis (requires fp_residuals_history)
# =============================================================================
if isnothing(fp_residuals_history)
    @info "Skipping fixed-point residual analysis (fp_residuals_history not available in this file)"
else
    num_residuals = size(fp_residuals_history, 2)

    # Check for zero columns (incomplete history)
    nonzero_cols = [i for i in 1:num_residuals if norm(fp_residuals_history[:, i]) > 1e-14]
    if length(nonzero_cols) < 2
        @info "Skipping fixed-point residual analysis (fewer than 2 non-zero residuals)"
    else
        # Use only non-zero columns
        fp_residuals = fp_residuals_history[:, nonzero_cols]
        num_residuals = size(fp_residuals, 2)

        @info "=== Fixed-Point Residual Analysis ==="
        @info "  Number of residual vectors: $num_residuals"
        @info "  Residual dimension: $(size(fp_residuals, 1))"

        # =========================================================================
        # Section 1: Eigenvalue Trajectory Correlation
        # =========================================================================
        @info "--- Section 1: Eigenvalue Trajectory Correlation ---"

        if isnothing(eigen_result)
            @warn "Skipping spectral analysis sections (eigendecomposition unavailable)"
        else
            λ = eigen_result.eigenvalues
            V = eigen_result.eigenvectors
            eigen_class = classify_eigenvalues(λ)
            near_unit_indices = findall(eigen_class.near_unit_mask)
            complex_indices = findall(eigen_class.complex_mask)

            # Analyze projections onto near-unit eigenmodes
            if !isempty(near_unit_indices)
                k = min(5, length(near_unit_indices))
                # Sort by distance to 1 (closest first)
                sorted_near_unit = near_unit_indices[sortperm(eigen_class.distances[near_unit_indices])]

                # Compute projection magnitudes for top-k near-unit modes
                projs_near_unit = zeros(k, num_residuals)
                for i in 1:num_residuals
                    for (j, idx) in enumerate(sorted_near_unit[1:k])
                        projs_near_unit[j, i] = abs(V[:, idx]' * fp_residuals[:, i])
                    end
                end

                # Plot: Time series of projections onto near-unit eigenvectors
                plt_near_unit = plot(
                    title="Projection onto Near-Unit Eigenmodes ($PROBLEM_NAME)",
                    xlabel="Residual Iteration",
                    ylabel="Projection Magnitude",
                    legend=:topright,
                    linewidth=2,
                    size=(800, 500)
                )
                for j in 1:k
                    idx = sorted_near_unit[j]
                    label_str = "λ=$(round(λ[idx], digits=4)), |λ-1|=$(round(eigen_class.distances[idx], digits=5))"
                    plot!(plt_near_unit, 1:num_residuals, projs_near_unit[j, :],
                          label=label_str, marker=:circle, markersize=3)
                end
                display(plt_near_unit)
            else
                @info "  No near-unit eigenvalues found, skipping near-unit projection plot"
            end

            # Detect helical/spiral patterns in complex eigenspaces
            if !isempty(complex_indices)
                # Find complex conjugate pairs (only process positive imaginary part)
                complex_pairs = Tuple{Int,Int}[]
                processed = Set{Int}()
                for idx in complex_indices
                    if idx in processed
                        continue
                    end
                    if imag(λ[idx]) > 0
                        # Find conjugate
                        conj_idx = findfirst(i -> abs(λ[i] - conj(λ[idx])) < 1e-10, eachindex(λ))
                        if !isnothing(conj_idx)
                            push!(complex_pairs, (idx, conj_idx))
                            push!(processed, idx)
                            push!(processed, conj_idx)
                        end
                    end
                end

                if !isempty(complex_pairs)
                    # Sort pairs by proximity to the point λ=1 (slow-converging modes)
                    pair_distances = [eigen_class.distances[p[1]] for p in complex_pairs]
                    sorted_pairs = complex_pairs[sortperm(pair_distances)]

                    @info "  Found $(length(complex_pairs)) complex conjugate pairs total"

                    # Plot phase angle evolution for the top few pairs closest to λ=1
                    # These are the most interesting because they correspond to slow oscillatory modes
                    # The phase angle tracks the "rotation" component of the dynamics
                    num_pairs_to_plot = min(3, length(sorted_pairs))

                    plt_phase = plot(
                        title="Phase Evolution for Top $num_pairs_to_plot Complex Pairs Near λ=1 ($PROBLEM_NAME)",
                        xlabel="Residual Iteration",
                        ylabel="Phase Angle (degrees, unwrapped)",
                        legend=:topright,
                        linewidth=2,
                        size=(800, 500)
                    )

                    for p_idx in 1:num_pairs_to_plot
                        pos_idx = sorted_pairs[p_idx][1]
                        dist_to_one = eigen_class.distances[pos_idx]
                        phases = [angle(V[:, pos_idx]' * fp_residuals[:, i]) for i in 1:num_residuals]
                        # Unwrap phases for cleaner visualization (avoid jumps at ±π boundary)
                        for i in 2:length(phases)
                            while phases[i] - phases[i-1] > π
                                phases[i] -= 2π
                            end
                            while phases[i] - phases[i-1] < -π
                                phases[i] += 2π
                            end
                        end
                        label_str = "λ=$(round(λ[pos_idx], digits=3)), |λ-1|=$(round(dist_to_one, digits=3))"

                        # convert to degrees
                        phases .*= 180 / π
                        plot!(plt_phase, 1:num_residuals, phases,
                              label=label_str, marker=:circle, markersize=3)
                    end
                    display(plt_phase)

                    @info "  Plotted phase evolution for $num_pairs_to_plot pairs closest to λ=1"
                    @info "  (Linear phase drift indicates oscillatory convergence behavior)"
                else
                    @info "  Complex eigenvalues found but no conjugate pairs identified"
                end
            else
                @info "  No complex eigenvalues found, skipping phase angle analysis"
            end
        end

        # =========================================================================
        # Section 2: Low-Rank Subspace Approximation
        # =========================================================================
        @info "--- Section 2: Low-Rank Subspace Approximation ---"

        # Compute SVD of residual history
        U, σ, Vt = svd(fp_residuals)

        @info "SVD of fp_residuals_history:"
        @info "  Residual space dimension: $(size(fp_residuals, 1))"
        @info "  Number of residuals: $num_residuals"
        @info "  Singular values computed: $(length(σ))"

        # Explained variance analysis
        total_var = sum(σ.^2)
        explained_var = σ.^2 ./ total_var
        cumulative_var = cumsum(explained_var)

        # Find effective ranks at different thresholds
        rank_95 = findfirst(cumulative_var .>= 0.95)
        rank_99 = findfirst(cumulative_var .>= 0.99)
        rank_ratio = findfirst((σ ./ σ[1]) .< 1e-6)

        @info "Effective subspace dimensionality:"
        @info "  95% variance: rank $(isnothing(rank_95) ? ">$(num_residuals)" : rank_95) / $num_residuals"
        @info "  99% variance: rank $(isnothing(rank_99) ? ">$(num_residuals)" : rank_99) / $num_residuals"
        @info "  σ_i/σ_1 < 1e-6: rank $(isnothing(rank_ratio) ? ">$(num_residuals)" : rank_ratio-1) / $num_residuals"

        @info "Leading singular values:"
        for i in 1:min(5, length(σ))
            @info "  σ[$i] = $(σ[i]) ($(round(100*explained_var[i], digits=2))% variance)"
        end

        # Plot 1: Singular value spectrum
        plt_sv = scatter(
            1:length(σ), σ,
            xlabel="Singular Value Index",
            ylabel="Singular Value (log scale)",
            title="Singular Value Spectrum of Residual History ($PROBLEM_NAME)",
            yscale=:log10,
            legend=:topright,
            markersize=6,
            label="Singular values",
            size=(800, 500)
        )
        # Add threshold lines
        if !isnothing(rank_95)
            vline!(plt_sv, [rank_95], linestyle=:dash, color=:green, label="95% variance", linewidth=2)
        end
        if !isnothing(rank_99)
            vline!(plt_sv, [rank_99], linestyle=:dash, color=:orange, label="99% variance", linewidth=2)
        end
        display(plt_sv)

        # Plot 2: Cumulative variance explained
        plt_cumvar = plot(
            1:length(σ), cumulative_var,
            xlabel="Number of Components",
            ylabel="Cumulative Variance Explained",
            title="Cumulative Variance Explained ($PROBLEM_NAME)",
            legend=:bottomright,
            linewidth=2,
            marker=:circle,
            markersize=4,
            label="Cumulative variance",
            size=(800, 500)
        )
        hline!(plt_cumvar, [0.95], linestyle=:dash, color=:green, label="95% threshold", linewidth=2)
        hline!(plt_cumvar, [0.99], linestyle=:dash, color=:orange, label="99% threshold", linewidth=2)
        display(plt_cumvar)

        # Plot 3: Heatmap of leading principal components (if dimensions allow)
        num_components_to_show = min(5, length(σ))
        if size(U, 1) <= 200  # Only show heatmap for reasonably sized problems
            plt_pc = heatmap(
                U[:, 1:num_components_to_show]',
                xlabel="Component Index in State Vector",
                ylabel="Principal Component",
                title="Leading Principal Components ($PROBLEM_NAME)",
                color=:viridis,
                size=(800, 400)
            )
            display(plt_pc)
        else
            @info "  Skipping principal component heatmap (dimension $(size(U,1)) > 200)"
        end

        # =========================================================================
        # Section 3: Eigenbasis Decomposition Analysis
        # =========================================================================
        if !isnothing(eigen_result)
            @info "--- Section 3: Eigenbasis Decomposition Analysis ---"

            λ = eigen_result.eigenvalues
            V = eigen_result.eigenvectors
            eigen_class = classify_eigenvalues(λ)

            # Compute coefficients in the eigenbasis: V * coeffs = fp_residuals
            # For each residual r, we solve V * c = r to get the representation
            # r = sum_i c_i * v_i where v_i are eigenvectors
            # Note: V' * r gives inner products, NOT coefficients (unless V is unitary)
            coeffs = V \ fp_residuals  # Size: n × num_residuals

            # Compute "energy" = |c_i|² for each mode, summed over all residuals
            mode_energies = vec(sum(abs2.(coeffs), dims=2))

            # Compute total coefficient energy per residual (for normalization)
            total_coeff_energy = vec(sum(abs2.(coeffs), dims=1))

            # Energy distribution: near-unit vs far-from-unit eigenspaces
            if any(eigen_class.near_unit_mask)
                energy_near_unit = vec(sum(abs2.(coeffs[eigen_class.near_unit_mask, :]), dims=1))
            else
                energy_near_unit = zeros(num_residuals)
            end
            energy_far = vec(sum(abs2.(coeffs[.!eigen_class.near_unit_mask, :]), dims=1))

            # Normalize by total coefficient energy (should now sum to 1)
            energy_near_unit_norm = energy_near_unit ./ total_coeff_energy
            energy_far_norm = energy_far ./ total_coeff_energy

            @info "Energy distribution across eigenspaces:"
            @info "  Mean fraction in near-unit eigenspace: $(round(mean(energy_near_unit_norm), digits=4))"
            @info "  Mean fraction in far-from-unit eigenspace: $(round(mean(energy_far_norm), digits=4))"
            @info "  (These should sum to ~1.0: $(round(mean(energy_near_unit_norm .+ energy_far_norm), digits=4)))"

            # Plot 1: Energy distribution evolution
            plt_energy = plot(
                title="Coefficient Energy Distribution Across Eigenspaces ($PROBLEM_NAME)",
                xlabel="Residual Iteration",
                ylabel="Fraction of Total Coefficient Energy",
                legend=:right,
                linewidth=2,
                size=(800, 500)
            )
            plot!(plt_energy, 1:num_residuals, energy_near_unit_norm,
                  label="Near-unit eigenspace (|λ-1| < $(UNIT_TOL))", color=:red, marker=:circle, markersize=3)
            plot!(plt_energy, 1:num_residuals, energy_far_norm,
                  label="Far-from-unit eigenspace", color=:blue, marker=:square, markersize=3)
            hline!(plt_energy, [0.5], linestyle=:dash, color=:gray, label="")
            display(plt_energy)

            # Identify dominant eigenmodes
            sorted_mode_indices = sortperm(mode_energies, rev=true)

            @info "Top 10 dominant eigenmodes (by total coefficient energy):"
            for i in 1:min(10, length(sorted_mode_indices))
                idx = sorted_mode_indices[i]
                pct = 100 * mode_energies[idx] / sum(mode_energies)
                @info "  Mode $idx: λ=$(round(λ[idx], digits=4)), energy=$(round(mode_energies[idx], sigdigits=4)) ($(round(pct, digits=2))%)"
            end

            # Plot 2: Dominant eigenmode contributions (bar chart with log scale for visibility)
            num_modes_to_plot = min(15, length(sorted_mode_indices))
            top_indices = sorted_mode_indices[1:num_modes_to_plot]
            top_energies = mode_energies[top_indices]
            # Convert to percentage for better interpretability
            top_pcts = 100 .* top_energies ./ sum(mode_energies)

            plt_modes = bar(
                1:num_modes_to_plot, top_pcts,
                xlabel="Rank (by energy)",
                ylabel="Percentage of Total Coefficient Energy (%)",
                title="Dominant Eigenmode Contributions ($PROBLEM_NAME)",
                legend=false,
                size=(800, 500),
                color=:steelblue,
                yscale=:log10  # Log scale to see small contributions
            )
            # Add eigenvalue annotations for top modes
            for i in 1:min(5, num_modes_to_plot)
                if top_pcts[i] > 0
                    annotate!(plt_modes, i, top_pcts[i],
                             text("λ=$(round(λ[top_indices[i]], digits=2))", 8, :bottom))
                end
            end
            display(plt_modes)

            # Plot 3: Heatmap of coefficient magnitudes for top modes (log scale for visibility)
            num_modes_heatmap = min(20, length(sorted_mode_indices))
            coeffs_top = abs.(coeffs[sorted_mode_indices[1:num_modes_heatmap], :])

            # Apply log transform for visibility (add small epsilon to avoid log(0))
            coeffs_top_log = log10.(coeffs_top .+ 1e-16)

            plt_coeffs = heatmap(
                coeffs_top_log,
                xlabel="Residual Iteration",
                ylabel="Eigenmode Rank (by energy)",
                title="Log₁₀ Coefficient Magnitudes for Dominant Eigenmodes ($PROBLEM_NAME)",
                color=:viridis,
                size=(800, 500),
                colorbar_title="log₁₀|coefficient|"
            )
            display(plt_coeffs)

        else
            @info "--- Section 3: Skipped (eigendecomposition unavailable) ---"
        end

        @info "=== Fixed-Point Residual Analysis Complete ==="
    end
end
