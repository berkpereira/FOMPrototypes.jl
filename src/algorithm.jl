function iterate!(problem::QPProblem, x::Vector{Float64}, s::Vector{Float64}, y::Vector{Float64})
    # Unpack problem data for easier reference
    P, c, A, b, M1, K, τ, ρ = problem.P, problem.c, problem.A, problem.b, problem.M1, problem.K, problem.τ, problem.ρ

    ### Step 1: Update x ###
    # Kept in general form here; may or may not be a diagonal system, depending on choice of M1
    x_new = x - (M1 + P + ρ * A' * A) \ (P * x + c + A' * (y + ρ * (A * x + s - b)))

    ### Step 2: Update s ###
    # At this moment allows for cone K = K_1 \times \ldots \K_l,
    # with each cone in the product either nonnegative orthant or zero cone.
    s_new = zeros(Float64, size(A, 1))
    start_idx = 1
    for cone in K
        end_idx = start_idx + cone.dim - 1
        s_slice = b[start_idx:end_idx] - A[start_idx:end_idx, :] * x_new - y[start_idx:end_idx] / ρ

        # Project portion of s depending on the cone type
        if cone isa Clarabel.NonnegativeConeT
            s_new[start_idx:end_idx] = max.(s_slice, 0)
        elseif cone isa Clarabel.ZeroConeT
            s_new[start_idx:end_idx] = zeros(Float64, cone.dim)
        else
            error("Unsupported cone type: $typeof(cone)")
        end
        
        start_idx = end_idx + 1
    end

    ### Step 3: Update y ###
    y_new = y + ρ * (A * x_new + s_new - b)

    # Update the sequences
    x .= x_new
    s .= s_new
    y .= y_new
end;
;