function almost_thermal(
    ρ_th::Matrix{ComplexF64},
    α::Float64,
    ξ::Float64
)
    N = size(ρ_th, 1)

    # Hermitian perturbation
    A = zeros(ComplexF64, N, N)
    for i in 1:N, j in i+1:N
        w = exp(-abs(j - i) / ξ)
        v = randn() + 1im * randn()
        A[i,j] = w * v
        A[j,i] = conj(A[i,j])
    end

    # Small unitary rotation
    U = exp(1im * α * A)

    # Rotate thermal state
    ρ = U * ρ_th * U'
    return ρ
end
