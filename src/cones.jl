using SparseArrays
using JuMP
using LinearAlgebra, LinearMaps
using Clarabel

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