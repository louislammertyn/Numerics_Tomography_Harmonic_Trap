function x_matrix_ho(L::Int)
    m = zeros(ComplexF64, L, L)
    for n in 0:(L-2)  # n = 0, 1, ..., L-2
        m[n+1, n+2] = sqrt((n+1)/2)  # |n> → |n+1> (add 1 because Julia indices start at 1)
        m[n+2, n+1] = sqrt((n+1)/2)  # |n+1> → |n>
    end
    return m
end
function p_matrix_ho(L::Int)
    m =  zeros(ComplexF64, L, L)
    for n in 0:(L-2)  # n = 0, 1, ..., L-2
        m[n+1, n+2] = sqrt((n+1))  # |n> → |n+1> (add 1 because Julia indices start at 1)
        m[n+2, n+1] = -sqrt((n+1))  # |n+1> → |n>
    end
    return -1im/sqrt(2) .*m
end

function a_matrix_ho(L::Int)
    m = zeros(ComplexF64, L, L)
    for n in 1:L-1
        m[n,n+1] = sqrt(n)
    end
    return m
end

function ad_matrix_ho(L::Int)
    return a_matrix_ho(L::Int)'
end

# J(i → j)
function local_currents(ρ::Matrix{ComplexF64}, H::Matrix{ComplexF64})
    J = similar(ρ)
    for i in axes(J,1), j in axes(J,2)
        J[i,j] = -2* imag(H[i,j] * ρ[j,i])
    end
    return J
end

function center_of_mass_evolution(sol, p)
    l_coms = Float64[]

    for s in eachindex(sol.t)
        state = sol.u[s]
        l_com = real(tr(state * Diagonal(0:(p.L-1))))
        push!(l_coms, l_com)
    end
    return hcat(sol.t, l_coms)
end

function  λ_com(ρ, p)

    return real(tr(ρ* Diagonal(0:(p.L-1))))
end


