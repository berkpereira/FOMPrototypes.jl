using SparseArrays
using LinearAlgebra

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
;