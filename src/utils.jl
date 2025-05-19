using SparseArrays
using BenchmarkTools
using LinearAlgebra, LinearMaps
import LinearAlgebra:givensAlgorithm
using JuMP
using Plots
using TimerOutputs
using SCS
using Statistics
using Clarabel
using FFTW

"""
This function multiplies in-place a matrix by another matrix with two columns.
To make it fast, it exploits complex vectors, storing the two columns of the
second matrix as a single complex vector, and calling mul! for this product.
It then decomposes the result back into the two target columns.

temp_n_vec and temp_m_vec are workspace complex to store intermediate results.
temp_n_vec is n-vector.
temp_m_vec is m-vector.
"""
@inline function two_col_mul!(Y::AbstractMatrix{Float64},
    A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    temp_n_vec::Vector{ComplexF64},
    temp_m_vec::Vector{ComplexF64})
    
    # Dimensions:
    # A is m×n, B is n×2, Y must be m×2
    # temp_n_vec is n-vector, temp_m_vec is m-vector

    # pack the two real columns of B into one complex vector
    @views temp_n_vec .= complex.(B[:, 1], B[:, 2])

    # compute temp_m_vec = A * temp_n_vec in-place, avoiding extra allocations
    mul!(temp_m_vec, A, temp_n_vec)

    # unpack the complex result into the two columns of Y
    @views Y[:, 1] .= real.(temp_m_vec)
    @views Y[:, 2] .= imag.(temp_m_vec)

    return nothing
end

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
This function efficiently computes R(P) * x, where R(P) denotes the matrix
obtained from P by setting the diagonal to zero. Done in place.
Assumes ws has a field dP storing diag(P), and ws.p.P storing Hessian P.
"""
function mul_P_nodiag!(in_vec::AbstractVector{Float64},
    out_vec::AbstractVector{Float64},
    ws::AbstractWorkspace,)
    # compute P * x
    mul!(out_vec, ws.p.P, in_vec)

    # subtract ws.dP * x
    @inbounds for i in eachindex(out_vec)
        out_vec[i] -= ws.dP[i] * in_vec[i]
    end
    return nothing
end

"""
    build_takeaway_op(variant, P, A, A_gram, ρ)

Constructs a LinearMap for one of the following operators (assuming P is n×n):

  PDHG. ρ AᵀA               - Augmented Lagrangian penalty is difference between PDHG and ADMM
  1. R(P + ρ AᵀA)          — Off-diagonal part of P + ρ AᵀA
  2. P + ρ AᵀA             — The full matrix
  3. P + R(ρ AᵀA)          — P plus the off-diagonal part of ρ AᵀA
  4. R(P) + ρ AᵀA          — Off-diagonal part of P plus ρ AᵀA

The operator R(B) is defined as B with its diagonal set to zero.

This linear operators is required, in each variant, in order to determine
the maximum step size τ associated with the algorithm for standard convergence
guarantees (see AD-PMM in Shefi and Teboulle 2014 paper). Namely,
the theoretical (1 / τ_max) given by the reciprocal of the largest eigenvalue
of the operator returned by this function.
"""
function build_takeaway_op(variant::Symbol, P::Symmetric{Float64},
    A::SparseMatrixCSC{Float64, Int64}, A_gram::LinearMap{Float64},
    ρ::Float64)
    n = size(P, 1)  # assume P is square
    
    op = nothing  # will hold our operator function
    
    # TODO use mul_P_nodiag! wherever applicable here, instead of the 
    # current R(P) * x operation implementation. Will require temp/working
    # vectors

    if variant == :PDHG
        # PDHG operator
        op = x -> ρ * (A_gram * x)
    elseif variant == Symbol(1)
        # R(P + ρ AᵀA) = (P + ρ AᵀA) - diag(P + ρ AᵀA)
        
        # precompute diagonals
        dP = Vector(diag(P))
        dA = vec(sum(abs2, A; dims=1))
        
        op = x -> begin
            y = P * x + ρ * (A_gram * x)
            y .-= (dP + ρ * dA) .* x
            y
        end
    elseif variant == Symbol(2)
        # P + ρ AᵀA (full matrix)
        # TODO: consider reducing mem allocations with in-place computations here?
        op = x -> P * x + ρ * (A_gram * x)
    elseif variant == Symbol(3)
        # P + R(ρ AᵀA) = P + [ρ AᵀA - diag(ρ AᵀA)]

        # precompute diagonals
        dP = Vector(diag(P))
        dA = vec(sum(abs2, A; dims=1))

        op = x -> begin
            y = P * x + ρ * (A_gram * x)
            y .-= ρ * (dA .* x)
            y
        end
    elseif variant == Symbol(4)
        # R(P) + ρ AᵀA = [P - diag(P)] + ρ AᵀA

        # precompute diagonals
        dP = Vector(diag(P))

        op = x -> begin
            y = P * x + ρ * (A_gram * x)
            y .-= dP .* x
            y
        end
    else
        error("Variant not applicable: choose 1, 2, 3, or 4.")
    end
    
    # In our use cases the resulting operator is symmetric.
    return LinearMap(op, n, n; issymmetric=true, ishermitian=true)
end

"""
This function computes M1 matrix-vector products efficiently and in place.
"""
function M1_op!(x::AbstractVector{Float64}, ws::KrylovWorkspace,
    variant::Symbol,
    A_gram::LinearMap{Float64},
    temp_n_vec1::Vector{Float64}, temp_n_vec2::Vector{Float64})

    if variant == :PDHG # M1 = 1/τ * I
        x ./= ws.τ
    elseif variant == :ADMM # M1 = ρ * A' * A
        x .*= A_gram
        x .*= ws.ρ
    elseif variant == Symbol(1) # M1 = 1/τ * I - R(P) + ρ * D(A' * A)
        temp_n_vec1 .= x
        temp_n_vec2 .= x
        
        temp_n_vec1 ./= ws.τ
        x .*= ws.ρ * ws.dA
        x .+= temp_n_vec1 # now stores (1/τ * I + ρ * D(A' * A)) * x
        
        temp_n_vec1 .*= ws.τ # recover original input vector again
        mul_P_nodiag!(temp_n_vec1, temp_n_vec2, ws) # temp_n_vec2 = R(P) * original input
        x .+= temp_n_vec2
    elseif variant == Symbol(2) # M1 = 1/τ * I - P
        temp_n_vec1 .= x
        temp_n_vec1 ./= ws.τ
        x .*= ws.p.P
        x .*= -1.0
        x .+= temp_n_vec1
    elseif variant == Symbol(3) # M1 = 1/τ * I - P + ρ * D(A' * A)
        temp_n_vec1 .= x
        temp_n_vec2 .= x

        temp_n_vec1 ./= ws.τ
        temp_n_vec2 .*= ws.ρ * ws.dA

        x .*= ws.p.P
        x .*= -1.0

        x .+= temp_n_vec1
        x .+= temp_n_vec2
    elseif variant == Symbol(4) # M1 = 1/τ * I - R(P)
        temp_n_vec1 .= x
        temp_n_vec2 .= x

        temp_n_vec1 ./= ws.τ
        
        mul_P_nodiag!(temp_n_vec2, x, ws)
        x .*= -1.0

        x .+= temp_n_vec1
    else
        error("Variant not applicable: choose PDHG, ADMM, 1, 2, 3, or 4.")
    end

    return nothing
end


"""
This function estimates the dominant eigenvalue of a matrix using the power
method.
"""
function dom_λ_power_method(A::Union{LinearMap{Float64}, AbstractMatrix{Float64}},
    max_iter::Integer = 50)
    n = size(A, 1)
    x = randn(n)
    temp = zeros(n)
    
    x ./= norm(x)
    
    for k in 1:max_iter
        mul!(temp, A, x)
        x .= temp
        x ./= norm(temp)
    end
    
    # Rayleigh quotient estimate dot(x, A * x)
    # note that x has unit l2 norm at this point
    mul!(temp, A, x)
    return dot(x, temp)
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

# This diagonal matrix premultiplies the x step in our method.
# TODO: consider operating on upper triangular parts only, since all of the
# matrices involved are symmetric.
# This is often in my/Paul's notes referred to
# as $W^{-1} = (M_1 + P + ρ A^T A)^{-1}$.
function W_operator(variant::Symbol, P::Symmetric, A::AbstractMatrix, A_gram::LinearMap, τ::Union{Float64, Nothing}, ρ::Float64)
    n = size(A_gram, 1)
    
    ################## NON-DIAGONAL  pre-gradient operators ####################
    
    if variant == :PDHG
        pre_operator = Symmetric(sparse(I(n)) / τ + P)
    elseif variant == :ADMM
        # note: I think in this case I am forced to form the 
        # matrix P + ρ * A' * A explicitly, in order to then compute
        # its Cholesky factors for inverse-vec products
        pre_operator = Symmetric(P + ρ * A' * A)
    
    ################ DIAGONAL pre-gradient operators ################

    elseif variant == Symbol(1)
        dP = Vector(diag(P))
        dA = vec(sum(abs2, A; dims=1))
        pre_operator = Diagonal(ones(n) / τ + dP + ρ * dA)
    elseif variant == Symbol(2)
        pre_operator = Diagonal(ones(n) / τ)
    elseif variant == Symbol(3)
        dA = vec(sum(abs2, A; dims=1))
        pre_operator = Diagonal(ones(n) / τ + ρ * dA)
    elseif variant == Symbol(4)
        dP = Vector(diag(P))
        pre_operator = Diagonal(ones(n) / τ + dP)
    else
        error("Invalid variant: $variant.")
    end

    return pre_operator
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

# This version of project_to_K! changes the input vector s in place.
function project_to_K!(s::AbstractVector{Float64}, K::Vector{Clarabel.SupportedCone})
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1

        # Project portion of s depending on the cone type
        if cone isa Clarabel.NonnegativeConeT
            s[start_idx:end_idx] .= max.(s[start_idx:end_idx], 0)
        elseif cone isa Clarabel.ZeroConeT
            s[start_idx:end_idx] .= zeros(Float64, cone.dim)
        else
            error("Unsupported cone type: $typeof(cone)")
        end

        start_idx = end_idx + 1
    end
    
    return
end

"""
Given a cone K, this projects the variable y into the DUAL cone of K.
"""
function project_to_dual_K(y::AbstractVector{Float64}, K::Vector{Clarabel.SupportedCone})
    projected_y = copy(y)
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1

        # Project portion of y depending on each cone's DUAL
        if cone isa Clarabel.NonnegativeConeT
            @views projected_y[start_idx:end_idx] = max.(y[start_idx:end_idx], 0)
        elseif cone isa Clarabel.ZeroConeT
            nothing # dual cone is the whole vector space
        else
            error("Unsupported cone type: $typeof(cone)")
        end

        start_idx = end_idx + 1
    end
    
    return projected_y
end

"""
Given a cone K, this projects the variable y into the DUAL cone of K.
"""
function project_to_dual_K!(y::AbstractVector{Float64}, K::Vector{Clarabel.SupportedCone})
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1

        # Project portion of y depending on each cone's DUAL
        if cone isa Clarabel.NonnegativeConeT
            # nonnegative orthant is self-dual
            @views y[start_idx:end_idx] .= max.(y[start_idx:end_idx], 0)
        elseif cone isa Clarabel.ZeroConeT
            nothing # dual cone is the whole space
        else
            error("Unsupported cone type: $typeof(cone)")
        end

        start_idx = end_idx + 1
    end
    
    return nothing
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

function plot_spectrum(A::AbstractMatrix{Float64}, k::Union{Int, Nothing} = nothing)
    eig_decomp = eigen(Matrix(A))
    
    # Plot the spectrum in the complex plane
    display(scatter(real(eig_decomp.values), imag(eig_decomp.values),
        xlabel="Re", ylabel="Im",
        title="Spectrum of tilde_A, k = $k",
        legend=false, aspect_ratio=:equal, marker=:x))

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

function constraint_changes(v_proj_flags::Vector{Vector{Bool}})
    is_equal = [v_proj_flags[i] == v_proj_flags[i - 1] for i in 2:length(v_proj_flags)]
    return (2:length(v_proj_flags))[.!is_equal]
 end
