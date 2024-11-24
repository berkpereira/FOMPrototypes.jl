using SparseArrays
using LinearAlgebra
using JuMP
using SCS
using Statistics
using Clarabel

# Extracts the diagonal of a square matrix (works with sparse/dense matrices)
function diag_part(A::AbstractMatrix{Float64})
    if A isa SparseMatrixCSC
        return spdiagm(0 => diag(A))  # Create a sparse diagonal matrix
    else
        return Diagonal(diag(A))     # Create a dense diagonal matrix
    end
end;

# Returns a matrix with zero diagonal and other entries the same as A
function off_diag_part(A::AbstractMatrix{Float64})
    if A isa SparseMatrixCSC
        return A - spdiagm(0 => diag(A))  # Subtract the diagonal in sparse form
    else
        return A - Diagonal(diag(A))      # Subtract the diagonal in dense form
    end
end;

function take_away_matrix(variant_no::Integer)
    if variant_no == 1 # NOTE: most "economical"
        return off_diag_part(P + ρ * A' * A)
    elseif variant_no == 2 # NOTE: least "economical"
        return P + ρ * A' * A
    elseif variant_no == 3 # NOTE: intermediate
        return P + off_diag_part(ρ * A' * A)
    elseif variant_no == 4 # NOTE: also intermediate
        return off_diag_part(P) + ρ * A' * A
    else
        error("Invalid variant.")
    end
end;

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
end;

# Simple moving average filter
function moving_average(data::Vector{Float64}, window_size::Int)
    n = length(data)
    padding = floor(Int, window_size / 2)  # Half window size for symmetric padding
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

function exponential_moving_average(data::Vector{Float64}, alpha::Float64)
    n = length(data)
    filtered_data = zeros(Float64, n)
    filtered_data[1] = data[1]
    for i in 2:n
        filtered_data[i] = alpha * filtered_data[i-1] + (1 - alpha) * data[i]
    end
    return filtered_data
end;
;