using SparseArrays
using LinearAlgebra
import LinearAlgebra:givensAlgorithm
using JuMP
using Plots
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

function add_cone_constraints!(model::JuMP.Model, s::JuMP.Containers.Array, K::Vector{Clarabel.SupportedCone})
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

function plot_spectrum(A::AbstractMatrix{Float64})
    spectrum = eigvals(Matrix(A))
    
    # Extract real and imaginary parts of the eigenvalues
    real_parts = real(spectrum)
    imag_parts = imag(spectrum)
    
    # Plot the spectrum in the complex plane
    display(scatter(real_parts, imag_parts,
        xlabel="Re", ylabel="Im",
        title="Spectrum of tilde_A",
        legend=false, aspect_ratio=:equal, marker=:circle))
end

"""
Initialises an UpperHessenberg view of a n by (n - 1) matrix filled with zeros.
"""
function init_upper_hessenberg(n::Int)
    # TODO: consider whether to use sparse or dense represenation (see lines
    # below).
    # H = spzeros(n + 1, n)
    H = zeros(n, n - 1)
    return UpperHessenberg(H)
end

"""
Perform one Modified Gram-Schmidt orthogonalisation step for Krylov methods.
RETURNS the result of the orthogonalisation of the input new vector (usable
for fields of mutable structs).

Inputs:
- V::Matrix{T}: Existing tall matrix of orthonormal vectors (pre-allocated, may contain zero columns).
- v_new::Vector{T}: New vector to orthonormalize.
- H::Matrix{T}: Upper Hessenberg matrix (to be updated in-place).

Assumptions:
- V is pre-allocated (with zeros); the leftmost zero column determines where to place the new orthonormal vector.
- H has been pre-allocated with the correct size (e.g. mem+1 by mem).
- Orthogonalisation uses modified Gram-Schmidt (numerically stable).
"""
function arnoldi_step!(V::AbstractMatrix{T},
    v_new::AbstractVector{T},
    H::AbstractMatrix{T}) where T <: AbstractFloat
    # Determine the current column of interest.
    k = findfirst(col -> all(iszero, view(V, :, col)), 1:size(V, 2))
    if isnothing(k)
        k = size(V, 2) - 1
    else
        k -= 1
    end

    # Loop through previous basis vectors to orthogonalise v_new.
    for j in 1:k
        # Compute the projection coefficient.
        H[j, k] = dot(V[:, j], v_new)

        # Subtract the projection from v_new.
        v_new -= H[j, k] .* V[:, j]
    end

    # Compute the norm of the orthogonalised vector.
    H[k + 1, k] = norm(v_new)

    # Check for breakdown
    if H[k + 1, k] == 0.0
        error("(Happy?) breakdown: orthogonalized vector has zero norm.")
    end

    # Normalize v_new to make it unit length
    v_new /= H[k + 1, k]

    # Place the orthonormalized vector into the basis matrix V
    V[:, k + 1] .= v_new

    return v_new
end

"""
    krylov_least_squares!(H, rhs_res)

Solve the least-squares problem `min_y ||H * y + rhs_res||^2`
arising in Krylov subspace methods, where `rhs_res` is assumed to be pre-transformed by `Q`.

Inputs:
- `H`: Upper Hessenberg matrix of size (k+1) × k (modified in-place).
- `rhs_res`: Vector of size (k+1) (modified in-place).

Outputs:
- `y`: Solution vector of size k.
"""
function krylov_least_squares!(H::AbstractMatrix{T}, rhs_res::AbstractVector{T}) where T
    k = size(H, 2)  # Number of columns in H

    # println("Smallest values of H square part: ", svd(H).S[end-10:end])

    # Check dimensions
    # @assert size(H, 1) == k + 1 "H must have dimensions (k+1) × k"
    # @assert length(rhs_res) == k + 1 "rhs_res must have length k+1"

    # Givens rotation application to reduce H to upper triangular form
    for j in 1:k
        # Generate Givens rotation for element (j, j) and (j+1, j)
        c, s, r = givensAlgorithm(H[j, j], H[j+1, j])

        # Apply Givens rotation to the j-th and (j+1)-th rows of H
        for i in j:k
            temp = c * H[j, i] + s * H[j+1, i]
            H[j+1, i] = -s * H[j, i] + c * H[j+1, i]
            H[j, i] = temp
        end

        # Apply Givens rotation to the RHS vector
        temp = c * rhs_res[j] + s * rhs_res[j+1]
        rhs_res[j+1] = -s * rhs_res[j] + c * rhs_res[j+1]
        rhs_res[j] = temp
    end

    # The Givens rotations reduce the problem to a k by k linear system.
    # Now solve the upper triangular system H[1:k, 1:k] * y = -rhs_res[1:k]
    return H[1:k, 1:k] \ (- rhs_res[1:k])
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

function constraint_changes(v_proj_flags::Vector{Vector{Bool}})
    is_equal = [v_proj_flags[i] == v_proj_flags[i - 1] for i in 2:length(v_proj_flags)]
    return (2:length(v_proj_flags))[.!is_equal]
 end