using Clarabel

"""
    QPProblem   # type name
    P::AbstractMatrix{Float64}  # PSD matrix in R^{n x n}
    c::Vector{Float64}  # Vector in R^n
    A::AbstractMatrix{Float64}  # Matrix in R^{m x n}
    b::Vector{Float64}  # Vector in R^m
    K::Vector{Clarabel.SupportedCone}
"""

# Define the QP problem data
struct QPProblem
    P::AbstractMatrix{Float64}  # PSD matrix in R^{n x n}
    c::Vector{Float64}  # Vector in R^n
    A::AbstractMatrix{Float64}  # Matrix in R^{m x n}
    b::Vector{Float64}  # Vector in R^m
    K::Vector{Clarabel.SupportedCone}
    
    
    # Constructor with a check for P being PSD
    function QPProblem(P::AbstractMatrix{Float64},
        c::Vector{Float64},
        A::AbstractMatrix{Float64},
        b::Vector{Float64},
        K::Vector{Clarabel.SupportedCone})
        # Check if P is symmetric (special handling for sparse matrices)
        if !(issymmetric(P) || (P isa SparseMatrixCSC && isequal(P, P')))
            error("Matrix P must be symmetric.")
        end
        
        # Check positive semidefiniteness of P
        # if minimum(eigvals(Matrix(P))) < 0
        #     error("Matrix P must be positive semidefinite (PSD).")
        # end

        # Check if K (cones) is a valid vector of Clarabel.SupportedCone
        if !(K isa Vector{Clarabel.SupportedCone})
            error("K must be a vector of Clarabel.SupportedCone.")
        end
        
        # Create the instance of the struct
        new(P, c, A, b, K) # what to store in struct, to be accessed later
    end
end