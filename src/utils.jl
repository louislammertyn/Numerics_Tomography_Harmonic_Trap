using Trapz, ProgressMeter
############################################
# EFFICIENT HERMITE FUNCTIONS
############################################
function hermite(n, x)
    iszero(n) && return (π)^(-0.25) * exp(-x^2 / 2)
    isone(n) && return sqrt(2) * x * π^(-0.25) * exp(-x^2 / 2)

    hi, hj = hermite(1, x), hermite(0, x)

    for k in 2:n
        hj = sqrt(2 / k) * x * hi - sqrt((k - 1) / k) * hj
        hi, hj = hj, hi
    end

    return hi
end
############################################


############################################
# PRECOMPUTE INTERACTION TERMS
############################################
function interaction_integral(λs, xs, hs)
    first(trapz(xs, prod(hs[λs, :], dims=1)))
end

function double_hermite_integral(h1, h2, xs)
    first(trapz(xs, h1[:] .* h2[:]))
end

function interaction_matrix(λmax, xs, hs; fl=false)
    U = zeros(Float64, λmax + 1, λmax + 1, λmax + 1, λmax + 1)

    @showprogress for λ₁ in 1:λmax+1, λ₂ in λ₁:λmax+1, λ₃ in λ₂:λmax+1, λ₄ in λ₃:λmax+1
        (fl && (λ₁ +  λ₂ != λ₃ + λ₄)) && continue
        if iseven(λ₁ + λ₂ + λ₃ + λ₄)
            U[λ₁, λ₂, λ₃, λ₄] = interaction_integral([λ₁, λ₂, λ₃, λ₄], xs, hs)
        end

        for perm in permutations([λ₁, λ₂, λ₃, λ₄])
            U[perm...] = U[λ₁, λ₂, λ₃, λ₄]
        end
    end

    return U
end

function hopping_integral(λ1::Int64, λ2::Int64)
    xs = range(start=0., stop=20., length= 2000)
    h1 = hermite.(λ1, xs)
    h2 = hermite.(λ2, xs)
    

    return double_hermite_integral(h1, h2, xs)
end


############################################
function thermal_ρ_matrix(H::Matrix{ComplexF64}, β::Float64)
    @assert H == H' "H is not Hermitian"

    # Diagonalize
    es, vs = eigen(Hermitian(H))

    # Exponentiate eigenvalues
    exp_es = Diagonal(exp.(-β .* es))

    # Partition function
    Z = tr(exp_es)

    # Density matrix
    ρ = vs * exp_es * vs' / Z

    # Checks
    @assert isapprox(ρ, ρ'; atol=1e-9) "ρ is not Hermitian"
    @assert all(real.(eigvals(ρ)) .>= -1e-12) "ρ has negative eigenvalues"
    @assert isapprox(tr(ρ), 1.0; atol=1e-9) "ρ does not have unit trace"

    return ρ
end

function lambda_com(ρ_l::Matrix{ComplexF64})
    L = size(ρ_l)[1]
    ls = 0:(L-1)
    return sum([ρ_l[i+1,i+1] * i for i in ls])
end





function density_x(ρ::Matrix{ComplexF64}, hs::Matrix{Float64})
    return hs * ρ * hs'
end


function density_x_matrix(ρ::Matrix{ComplexF64}; grid=2000, xlims=(-15,15))
    N = size(ρ, 1)
    xs = LinRange(xlims[1], xlims[2], grid)

    # Precompute Hermite functions
    hs = zeros(Float64, grid, N)
    for n in 1:N
        hs[:, n] = hermite.(n-1, xs)
    end

    # Compute x-space density
    ρ_x = density_x(ρ, hs)  # assume you have a density_x function

    return xs, ρ_x
end

function density_λ_matrix(ρ_x::Matrix{ComplexF64}, xs::AbstractVector, N::Int)
    grid = length(xs)
    dx = xs[2] - xs[1]

    # Precompute Hermite functions
    hs = zeros(Float64, grid, N)
    for n in 1:N
        hs[:, n] = hermite.(n-1, xs)
    end

    # Integrate to compute Fock-space density matrix
    ρ_fock = zeros(ComplexF64, N, N)
    for n in 1:N, m in 1:N
        # Use midpoint / trapezoidal integration over x and x'
        ρ_fock[n, m] = sum(hs[:, n] .* (ρ_x * hs[:, m])) * dx^2
    end

    return ρ_fock
end

function density_x_diag(ρ::Matrix{ComplexF64}, xs::AbstractVector, hs::AbstractMatrix)
    # diag_rho_x[i] = sum_{n,m} ρ[n,m] * H_n(x_i) * H_m(x_i)
    diag_rho_x = zeros(Float64, length(xs))
    
    # Loop over x-grid
    for i in 1:length(xs)
        diag_rho_x[i] = real(hs[i, :]'* ρ * hs[i, :])  # hs[i,:] is row vector
    end

    return diag_rho_x
end

function Displacement_op(dx::Float64,dp::Float64, L)
    α = (dx + 1im*dp) / sqrt(2)
    a = a_matrix_ho(L)
    ad = ad_matrix_ho(L)
    D = exp(-abs2(α)/2) * exp(α * ad) * exp(-1*conj(α) * a) 
    return D
end

function translate_density_x(
    ρ::Matrix{ComplexF64},
    xs::AbstractVector,
    dx::Float64,
    dp::Float64
)
    Nx = length(xs)
    Δx = xs[2] - xs[1]
    L  = Nx * Δx
    

    # --- snap dx to nearest integer grid shift ---
    n = round(Int, dx / Δx)          # integer shift
    dx_eff = n * Δx                  # effective dx

    println("the displacement error is $(dx_eff- dx)")
    # --- snap dp to nearest momentum grid value ---
    Δp = 2π / L
    m = round(Int, dp / Δp)          # integer momentum index
    dp_eff = m * Δp                  # effective dp

    # --- exact integer translation in x (periodic) ---
    ρ_old = copy(ρ)
    ρ_new = copy(ρ)
    for i in 1:Nx, j in 1:Nx
        i_from = mod1(i - n, Nx)
        j_from = mod1(j - n, Nx)
        ρ_new[i,j] = ρ_old[i_from, j_from]
    end

    # --- exact momentum kick ---
    phase = exp.(1im * dp_eff .* xs)
    ρ_new .*= phase * phase'

    return ρ_new
end

function plot_x_density(ρ::Matrix{ComplexF64}; grid=2000)
    hs = zeros(Float64, grid, size(ρ, 1))
    xs = LinRange(-15, 15, grid)
    for i in 1:size(ρ,1)
        hs[:, i] = hermite.(i-1, xs)
    end
    ρ_x = abs.(density_x(ρ, hs))
    
    xs_s = ["$(round(i, digits=1))" for i in xs]
    
    return ρ_x, heatmap(xs_s, xs_s, ρ_x)
end





function compare_energies(ω::Float64, T::Float64)
    ħ = 1.054571817e-34   # Js
    kB = 1.380649e-23     # J/K
    return (ħ * ω) / (kB * T)
end

function gaussian_diag_matrix(n::Int; center::Float64=n/2, sigma::Float64=1.0)
    indices = 1:n
    diag_elements = exp.(- (indices .- center).^2 ./ (2*sigma^2))
    
    # Normalize so that trace = 1
    diag_elements ./= sum(diag_elements)
    
    return Matrix(ComplexF64.(Diagonal(diag_elements)))
end

function plot_density_matrix_errors(rhos, times)
    Nt = length(rhos)
    herm_err = zeros(Nt)
    trace_err = zeros(Nt)
    pos_err = zeros(Nt)

    for i in 1:Nt
        ρ = rhos[i]
        herm_err[i] = norm(ρ - ρ')
        trace_err[i] = abs(tr(ρ) - 1)
        pos_err[i] = minimum(eigvals(Hermitian(ρ)))  # negative → violation
        
    end

    plot(times, herm_err, label="Hermitian error", lw=2)
    plot!(times, trace_err, label="Trace error", lw=2, linestyle=:dashdot)
    plot!(times, pos_err, label="Minimum eigval", lw=2)
    xlabel!("Time")
    ylabel!("Error / violation")
    title!("Density matrix consistency over time")
    
end

