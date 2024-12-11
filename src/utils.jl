using SparseArrays
using LinearAlgebra
using JuMP
using SCS
using Statistics
using Clarabel
using FFTW

function primal_obj_val(P::AbstractMatrix{Float64}, c::AbstractVector{Float64}, x::AbstractVector{Float64})
    return 0.5 * dot(x, P * x) + dot(c, x)
end

function dual_obj_val(P::AbstractMatrix{Float64}, b::AbstractVector{Float64},
    x::AbstractVector{Float64}, y::AbstractVector{Float64})
    return -0.5 * dot(x, P * x) - dot(b, y)
end

function duality_gap(primal_val::Float64, dual_val::Float64)
    return primal_val - dual_val
end

# Extracts the diagonal of a square matrix (works with sparse/dense matrices)
function diag_part(A::AbstractMatrix{Float64})
    # if A isa SparseMatrixCSC
    #     return spdiagm(0 => diag(A))  # Create a sparse diagonal matrix
    # else
    #     return Diagonal(diag(A))     # Create a dense diagonal matrix
    # end
    # NOTE: we make the vector of the diagonal have a dense type, since none
    # of the entries are zero (we will usually invert this matrix).
    return Diagonal(Vector(diag(A)))
end;

# Returns a matrix with zero diagonal and other entries the same as A
function off_diag_part(A::AbstractMatrix{Float64})
    if A isa SparseMatrixCSC
        return A - spdiagm(0 => diag(A))  # Subtract the diagonal in sparse form
    else
        return A - Diagonal(diag(A))      # Subtract the diagonal in dense form
    end
end

"""
This function estimates the dominant eigenvalue of a matrix using the power
method.
"""
function dom_λ_power_method(A::AbstractMatrix{Float64}, max_iter::Integer = 100)
    n = size(A, 1)
    x = randn(n)
    x /= norm(x)
    
    for k in 1:max_iter
        x = A * x
        x /= norm(x)
    end
    
    # Rayleigh quotient estimate (x has unit l2 norm at this point).
    return dot(x, A * x)
end

"""
This function normalises vectors while controlling for their dimension, which
may be useful when comparing vectors of different dimensions.
In this case we use the l2 norm, with a presumed scaling with sqrt(dim(v)).
NOTE: if input is zero vector, output is also zero vector!
"""
function dim_adjusted_vec_normalise(v::AbstractVector{Float64})
    if all(==(0.0), v)
        return v
    else
        return sqrt(size(v)[1]) * v / norm(v, 2)
    end
end

function take_away_matrix(variant_no::Integer, A_gram::AbstractMatrix{Float64})
    # NOTE: A_gram = A' * A
    if variant_no == 1 # NOTE: most "economical"
        return off_diag_part(P + ρ * A_gram)
    elseif variant_no == 2 # NOTE: least "economical"
        return P + ρ * A_gram
    elseif variant_no == 3 # NOTE: intermediate
        return P + off_diag_part(ρ * A_gram)
    elseif variant_no == 4 # NOTE: also intermediate
        return off_diag_part(P) + ρ * A_gram
    else
        error("Invalid variant.")
    end
end;

# This diagonal matrix premultiplies the x step in our method.
# TODO: consider operating on upper triangular parts only, since all of the
# matrices involved are symmetric.
# This is often in my/Paul's notes referred to
# as $W^{-1} = (M_1 + P + ρ A^T A)^{-1}$.
function pre_x_step_matrix(variant_no::Integer, P::AbstractMatrix,
    A_gram::SparseMatrixCSC, τ::Float64, ρ::Float64, n::Integer)
    if variant_no == 1
        pre_matrix = I(n) / τ + diag_part(P + ρ * A_gram)
    elseif variant_no == 2
        pre_matrix = I(n) / τ
    elseif variant_no == 3
        pre_matrix = I(n) / τ + diag_part(ρ * A_gram)
    elseif variant_no == 4
        pre_matrix = I(n) / τ + diag_part(P)
    else
        error("Invalid variant: $variant_no.")
    end
        
    # Invert.
    return Diagonal(1.0 ./ pre_matrix.diag)
end

# This function implements the projection of s block-wise onto the cones in K.
# This is the NON-mutating version of the function.
function project_to_K(s::AbstractVector{Float64}, K::Vector{Clarabel.SupportedCone})
    projected_s = copy(s)
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1

        # Project portion of s depending on the cone type
        if cone isa Clarabel.NonnegativeConeT
            @views projected_s[start_idx:end_idx] = max.(s[start_idx:end_idx], 0)
        elseif cone isa Clarabel.ZeroConeT
            @views projected_s[start_idx:end_idx] = zeros(Float64, cone.dim)
        else
            error("Unsupported cone type: $typeof(cone)")
        end

        start_idx = end_idx + 1
    end
    
    return projected_s
end

# This version of project_to_K! mutates the input vector s in place.
function project_to_K!(s::AbstractVector{Float64}, K::Vector{Clarabel.SupportedCone})
    s .= project_to_K(s, K)
    return s
end

function add_cone_constraints_scs!(model::JuMP.Model, s::JuMP.Containers.Array, K::Vector{Clarabel.SupportedCone})
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1
        if cone isa Clarabel.NonnegativeConeT
            # Add Nonnegative cone constraint
            @constraint(model, s[start_idx:end_idx] in MOI.Nonnegatives(cone.dim))
        elseif cone isa Clarabel.ZeroConeT
            # Add Zero cone constraint
            @constraint(model, s[start_idx:end_idx] in MOI.Zeros(cone.dim))
        else
            error("Unsupported cone type in K: $cone")
        end
        start_idx = end_idx + 1
    end
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

# Function to plot the equality of consecutive entries in a vector of Booleans.
function plot_equal_segments(v_proj_flags::Vector{Vector{Bool}})
    # Compute whether each entry is equal to the previous one
    is_equal = [v_proj_flags[i] == v_proj_flags[i - 1] for i in 2:length(v_proj_flags)]
    
    # Generate x-axis indices (1 to length of `is_equal`)
    x_vals = 2:length(v_proj_flags)

    # Create the plot
    plot(
        x_vals, is_equal,
        seriestype = :scatter,
        markershape = :circle,
        markercolor = :blue,
        xlabel = "Index",
        ylabel = "Equality with Previous",
        title = "Equality of Consecutive Enforced Sets",
        legend = false
    )
end;