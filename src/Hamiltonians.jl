function HO_matrix(L::Int, ω::Float64=1.) 
    
    D = zeros(ComplexF64, L, L)
    for i in 1:L
        D[i, i] = (i - 1) * ω + 0.5
    end
    return D
end


function RWA_FL_Hamiltonian(Vd::Matrix{ComplexF64}, p, freqs_phis_amps::Vector{Tuple{Int, Float64, Float64}}, drive_type=:smooth)
    if drive_type == :smooth
        ho = HO_matrix(p.L, 1 - p.ω_d)
        hops = zeros(ComplexF64, size(Vd))
        for i in axes(hops,1)
            for (f, ϕ, k) in freqs_phis_amps
                (i+f > p.L) && continue
                hops[i,i+f] = k * Vd[i,i+f] *exp(-1im * ϕ)/ (2im)
                hops[i+f, i] = conj( hops[i,i+f])
            end
        end
        return ho + hops
    elseif drive_type == :step
        ho = HO_matrix(p.L, 1 - p.ω_d)
        hops = zeros(ComplexF64, size(Vd))
        for i in axes(hops,1)
            for (f, ϕ, k) in freqs_phis_amps
                (i+f > p.L) && continue
                hops[i,i+f] = k * Vd[i,i+f] *exp(-1im * ϕ) * (-2im)/π
                hops[i+f, i] = conj( hops[i,i+f])
            end
        end
        return ho + hops
    end
end

function V_AOD_FL(drive, p)
    ts = LinRange(0, 2π /p.ω_d, 5000)
    V_ds_exp = Vector{Matrix{ComplexF64}}(undef, length(ts))

    for (it,t) in enumerate(ts) 
        V_d_exp = Matrix{ComplexF64}(undef, p.L, p.L)
        V_d = drive(t)
        for i in axes(V_d_exp,1), j in axes(V_d_exp, 2)
            V_d_exp[i,j] = exp(-1im *(j-i) * t*p.ω_d) * V_d[i,j]
        end
        V_ds_exp[it] = V_d_exp
    end

    V_fl = zero(V_ds_exp[1])
    for k in 1:length(ts)-1
        dt = ts[k+1] - ts[k]
        V_fl .+= (V_ds_exp[k] .+ V_ds_exp[k+1]) .* (dt / 2)
    end

    return V_fl ./ (2π /p.ω_d)
end

function drive_elements(xs::AbstractVector{<:Real}, f::Function, L::Int)
    λmax = L - 1
    fx = f.(xs)
    H = [hermite.(λ, xs) for λ in 0:λmax]
    V = zeros(ComplexF64, L, L)

    for λ1 in 0:λmax
        for λ2 in λ1:λmax 
            val = trapz(xs, H[λ1+1] .* H[λ2+1] .* fx)
            V[λ1+1, λ2+1] = val
            V[λ2+1, λ1+1] = val
        end
    end
    return V
end

function drive_elements(xs::AbstractVector{<:Real}, fxs::AbstractVector{<:Real}, L::Int)
    λmax = L - 1
    fx = fxs
    H = [hermite.(λ, xs) for λ in 0:λmax]
    V = zeros(ComplexF64, L, L)

    for λ1 in 0:λmax
        for λ2 in λ1:λmax 
            val = trapz(xs, H[λ1+1] .* H[λ2+1] .* fx)
            V[λ1+1, λ2+1] = val
            V[λ2+1, λ1+1] = val
        end
    end
    return V
end

function get_only_m_site_hopping(M::Matrix, m::Int)
    V = zeros(ComplexF64, size(M))
    for i in 1:size(V)[1]-m
        V[i, i+m] = M[i, i+1]
        V[i+m, i] = M[i+1,i]
    end
    return V 
end


function drive_Coherent_FL(L::Int, Δ=0.0)
    H = zeros(ComplexF64, L, L)
    for i in 1:L-1
        n = i  # minimum of site i and i+1 is i
        J = sqrt(n * (L - (n - 1)))
        H[i, i+1] = -J
        H[i+1, i] = -J   # J is real so no conjugation needed
    end
    onsite = HO_matrix(L, Δ)
    @assert ishermitian(Hermitian(onsite + H))
    return onsite, H
end


function uniform_Ham(L::Int)
    H = zeros(ComplexF64, L, L)
    for i in 1:L-1
        H[i, i+1] = -1.0
        H[i+1, i] = -1.0
    end
    return HO_matrix(L), H
end


