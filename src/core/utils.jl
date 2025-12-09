using SparseArrays
using JuMP
using LinearAlgebra, LinearMaps
using Clarabel
using BenchmarkTools
using Plots
using TimerOutputs
using SCS
using Statistics
using FFTW
using Printf

"""
Plots the spectrum of matrix A in the complex plane.
Adds some reference elements relevant to our spectral analysis.
"""
function plot_spectrum(A::AbstractMatrix{Float64}, k::Union{Int, Nothing} = nothing)
    eig_decomp = eigen(Matrix(A))
    
    # Compute the spectral radius.
    spectral_radius = maximum(abs.(eig_decomp.values))
    
    # Build the title string.
    title_str = "Spectrum, k = $k"
    title_str *= ", spectral radius = " * @sprintf("%0.5f", spectral_radius)
    max_imag = maximum(imag.(eig_decomp.values))
    title_str *= ", max imag: " * @sprintf("%0.5f", max_imag)
    # Define tolerances for "small" distances
    tol0 = 1e-4  # tolerance for eigenvalues near 0 
    tol1 = 1e-4  # tolerance for eigenvalues near 1

    # Count eigenvalues near 0 and near 1
    num_near_zero = count(x -> abs(x) < tol0, eig_decomp.values)
    num_near_one  = count(x -> abs(x - 1) < tol1, eig_decomp.values)

    # Append the counts to the title string
    title_str *= ", near 0: $(num_near_zero)"
    title_str *= ", near 1: $(num_near_one)"
    
    p = scatter(
        real(eig_decomp.values), imag(eig_decomp.values),
        xlabel="Re", ylabel="Im",
        title=title_str,
        legend=false, aspect_ratio=:equal, marker=:x)

    # add a single red marker at (0.5, 0.0)
    scatter!(p, [0.5], [0.0], markershape=:circle, color=:red)

    # add a solid red circle (outline) of radius 0.5 centered at (0.5, 0.0)
    θ = range(0, 2π, length=100)
    rx = 0.5 .+ 0.5 * cos.(θ)
    ry = 0.0 .+ 0.5 * sin.(θ)
    plot!(p, rx, ry, linecolor=:red, linestyle=:solid, label="")

    display(p)

    # NB: the function returns the eigendecomposition.
    return eig_decomp
end

"""
Given the input vector, this function computes the inner product of the 
normalised vector with the columns of the eigenvectors input. It plots these
on the y axis versus the magnitude of the phase of the corresponding
eigenvalues on the x axis.
"""
function plot_eigenvec_alignment_vs_phase(vector::AbstractVector{Float64}, eigenvalues::AbstractVector{T}, eigenvectors::AbstractMatrix{T}, k::Union{Int, Nothing} = nothing) where {T <: Number}
    # Normalise the input vector.
    normalised_vec = vector / norm(vector)
    
    # Compute the inner product of the normalised vector with the eigenvectors.
    # NB: complex eigenvectors introduce complications here...
    inner_products = vec(abs.(adjoint(normalised_vec) * eigenvectors))
    
    # NB: the modulo pi/2 operation maps all phases to the first quadrant.
    # Some zero eigvals may sometimes have their phase computed as pi
    # due to numerical approx erorrs. This fixes that.
    threshold = 1e-14
    # We also set approx 0 eigvals to 0 exactly.
    eigenvalues[abs.(eigenvalues) .< threshold] .= 0.0
    phases = abs.(angle.(eigenvalues)) .% (π / 2)
    
    # Plot the inner products against the phases.
    display(scatter(phases, inner_products,
        xlabel="Phase of Eigenvalues", ylabel="Abs Inner Product",
        title="Alignment of Input Vector with Eigenvectors, k = $k",
        legend=false, marker=:x, msw = 2))
end

################################################################################
############################# MISC HELPER FUNCTIONS ############################
################################################################################
function insert_update_into_matrix!(matrix, update, current_col_ref)
    # Insert the new update into the current column.
    matrix[:, current_col_ref[]] = update

    # Update the column index in a circular fashion.
    current_col_ref[] = current_col_ref[] % size(matrix, 2) + 1
end

"""
This function swaps, in-place, the contents of vectors x and y.
It requires a temp working vector.
"""
function custom_swap!(x::AbstractVector{Float64}, y::AbstractVector{Float64}, temp::AbstractVector{Float64})
    copy!(temp, x)
    copy!(x, y)
    copy!(y, temp)
end

"""
Counts number of input matrix's normalised singular values (ie divided by
largest one) larger than given small threshold tol.
"""
function effective_rank(A::AbstractMatrix{Float64}, tol::Float64 = 1e-8)
    # Compute NORMALISED singular values of A.
    sing_vals_normalised = svd(A).S
    sing_vals_normalised ./= sing_vals_normalised[1]
    
    # Effective rank = number of normalised singular values larger than
    # given small threshold tol.
    # NOTE extra output argument, ratio of two largest singular values.
    return count(x -> x > tol, sing_vals_normalised), sing_vals_normalised[1] / sing_vals_normalised[2]
end

################################################################################
########################## SIGNAL PROCESSING FUNCTIONS #########################
################################################################################

# Simple moving average filter
function moving_average(data::Vector{Float64}, window_size::Int)
    n = length(data)
    padding = floor(Int, window_size / 2)  # Symmetric padding
    padded_data = vcat(zeros(padding), data, zeros(padding))  # Pad with zeros
    
    filtered_data = zeros(Float64, n)
    for i in 1:n
        filtered_data[i] = mean(padded_data[i:i + window_size - 1])
    end
    return filtered_data
end;

function cumulative_average(data::Vector{Float64})
    n = length(data)
    filtered_data = zeros(Float64, n)
    filtered_data[1] = data[1]
    for i in 2:n
        filtered_data[i] = ((i - 1) * filtered_data[i-1] + data[i]) / i
    end
    return filtered_data
end;

function exp_moving_average(data::Vector{Float64}, alpha::Float64)
    n = length(data)
    filtered_data = zeros(Float64, n)
    filtered_data[1] = data[1]
    for i in 2:n
        filtered_data[i] = alpha * filtered_data[i-1] + (1 - alpha) * data[i]
    end
    return filtered_data
end;


function extract_dominant_frequencies(data::Vector{Float64}, num_frequencies::Int, sampling_rate::Float64 = 1.0)
    # Compute the FFT of the signal
    fft_result = fft(data)
    
    # Compute the magnitude of the FFT
    fft_magnitude = abs.(fft_result)
    
    # Compute the corresponding frequencies
    n = length(data)  # Number of samples
    frequencies = (0:n-1) .* (sampling_rate / n)  # Frequency bins

    # Ignore the DC component (frequency = 0)
    fft_magnitude[1] = 0.0  # Set the first element to 0 if DC is not relevant

    # Find the indices of the top frequencies
    top_indices = partialsortperm(fft_magnitude, rev=true, 1:num_frequencies)

    # Extract the top frequencies and their magnitudes
    top_frequencies = frequencies[top_indices]
    top_magnitudes = fft_magnitude[top_indices]

    return top_frequencies, top_magnitudes
end

# # Function to plot the equality of consecutive entries in a vector of Booleans.
# function plot_equal_segments(v_proj_flags::Vector{Vector{Bool}})
#     # Compute whether each entry is equal to the previous one
#     is_equal = [v_proj_flags[i] == v_proj_flags[i - 1] for i in 2:length(v_proj_flags)]
    
#     # Generate x-axis indices (1 to length of `is_equal`)
#     x_vals = 2:length(v_proj_flags)

#     # Create the plot
#     plot(
#         x_vals, is_equal,
#         seriestype = :scatter,
#         markershape = :circle,
#         markercolor = :blue,
#         xlabel = "Index",
#         ylabel = "Equality with Previous",
#         title = "Equality of Consecutive Enforced Sets",
#         legend = false
#     )
# end;

# Helper functions to get total iterations
_get_total_iters(data::Matrix) = size(data, 2)
_get_total_iters(data::Vector) = length(data)

# Helper functions to check if iteration is valid
_is_valid_iter(data::Matrix, i::Int) = i <= size(data, 2)
_is_valid_iter(data::Vector, i::Int) = i <= length(data)

# Helper functions to check if data changed between iterations
_data_changed(data::Matrix, i::Int) = any(data[:, i] .!= data[:, i - 1])
_data_changed(data::Vector{Vector{T}}, i::Int) where T = data[i] != data[i - 1]

function constraint_changes(
    nn_flags::Union{Vector{Vector{Bool}}, Matrix{Bool}},
    soc_states::Union{Vector{Vector{SOCAction}}, Matrix{SOCAction}},
    )
    total_iters = max(_get_total_iters(nn_flags), _get_total_iters(soc_states))
    if total_iters < 2
        return Int[]
    end

    changes = Int[]
    for i in 2:total_iters
        nn_changed = _is_valid_iter(nn_flags, i) && _is_valid_iter(nn_flags, i - 1) && _data_changed(nn_flags, i)
        soc_changed = _is_valid_iter(soc_states, i) && _is_valid_iter(soc_states, i - 1) && _data_changed(soc_states, i)
        if nn_changed || soc_changed
            push!(changes, i)
        end
    end
    return changes
end
