iref_to_k(iref) = iref * 2.28e7 / 440

# Quantize times to hardware resolution
quantize_times(t_holds, Δt_res) = round.(t_holds ./ Δt_res) .* Δt_res

# Generate "ideal" uniform sampling over one estimated period 
function t_holds_ideal(ω, N)
    return (2π/ω) .* (collect(0:N-1) ./ N)
end



function Kernel(λ, λ_, t, hs_scaled)
    θ = atan(t)
    b= sqrt(1+t^2)
    phase = exp(-1im *(λ - λ_)* θ)
    h1 = hs_scaled[:, λ+1]
    h2 = hs_scaled[:, λ_+1]
    return (phase / b) .* h1 .* h2 
end

# return n(x*b(t), t_tof, t_hold)
function get_n_tof(ρ_l::Matrix{ComplexF64}, t_holds::Vector{Float64}, t_tof::Float64, xs::AbstractArray, hs; ω=1.)

    densities = Array{Array}(undef, length(t_holds))
    kernels = Array{AbstractArray}(undef, size(ρ_l))
    L = size(ρ_l)[1]

    hs_scaled = scaled_hs(xs, sqrt(1+t_tof^2), L)

    for i in 1:L, j in 1:L
            kernels[i,j] =  Kernel(i-1, j-1, t_tof, hs_scaled)
    end

    Threads.@threads for i in eachindex(t_holds)
        t_h = t_holds[i] 
        density = zeros(ComplexF64, length(xs))
        for i in axes(ρ_l, 1), j in axes(ρ_l, 2)
            density .+=  kernels[i,j] .* ρ_l[i,j] .* exp(-1im * ω * t_h * (i-j))
        end
        densities[i] = real.(density )
    end

    return densities
    
end


function scaled_hs(xs, b, L)
    M = length(xs)
    H = Matrix{Float64}(undef, M, L)
    for n in 0:L-1
        @inbounds for k in 1:M
            H[k, n+1] = hermite(n, xs[k] / b)
        end
    end
    return H
end

function get_n_tof_optimized(ρ_l::Matrix{ComplexF64}, t_holds::Vector{Float64}, t_tof::Float64, xs::AbstractArray; ω=1.0)
    """
    Compute time-of-flight density profiles for a set of hold times.

    The computation is accelerated by grouping density-matrix elements
    according to their diagonal offset r = i-j. The spatial contraction is
    performed once and stored in C[r,k], while the dependence on hold time
    is incorporated afterward through simple phase factors exp(-im*r*ω*t).
    This avoids recomputing the full density-matrix contraction for every
    hold time.
    """
    
    L = size(ρ_l, 1)
    M = length(xs)
    
    # 1. Precompute loop-invariant constants
    b = sqrt(1 + t_tof^2)
    θ = atan(t_tof)
    
    # 2. Compute the Hermite grid directly into a single matrix
    H = scaled_hs(xs, b, L)
    
    # 3. Pre-absorb the θ phase shift into a modified density matrix
    ρ_tilde = Matrix{ComplexF64}(undef, L, L)
    for j in 1:L
        for i in 1:L
            @inbounds ρ_tilde[i, j] = ρ_l[i, j] * cis(-(i - j) * θ)
        end
    end
    
    # 4. Precompute contract matrix C (L x M). 
    # This reduces the L x L loop down to a single index r = i - j
    C = zeros(ComplexF64, L, M)
    for r in 0:L-1
        for i in (r+1):L
            j = i - r
            val = ρ_tilde[i, j]
            @simd for k in 1:M
                @inbounds C[r+1, k] += H[k, i] * H[k, j] * val
            end
        end
    end
    
    # 5. Preallocate the structure for the results
    densities = [Vector{Float64}(undef, M) for _ in 1:length(t_holds)]
    
    # 6. Multi-threaded evaluation over holding times
    Threads.@threads for t_idx in eachindex(t_holds)
        t_h = t_holds[t_idx]
        dest = densities[t_idx]
        
        # Precompute time-dependent phases for this specific time step
        phases = Vector{ComplexF64}(undef, L-1)
        for r in 1:L-1
            @inbounds phases[r] = cis(-r * ω * t_h)
        end
        
        # Stream through the spatial grid sequentially (highly cache-friendly layout)
        @simd for k in 1:M
            @inbounds res = real(C[1, k]) # r = 0 (diagonal) term
            for r in 1:L-1
                @inbounds res += 2.0 * real(C[r+1, k] * phases[r])
            end
            @inbounds dest[k] = res / b
        end
    end
    
    return densities
end


    


function convolve_gaussian(density, xs, σ)
    dx = xs[2] - xs[1]

    # kernel (centered at x=0)
    G = exp.(-xs.^2 ./ (2σ^2)) 
    G ./= sum(G) * dx

    # align with FFT indexing
    G_fft = fft(fftshift(G))

    # convolution
    return real(ifft(fft(density) .* G_fft)) .*dx
end


function bin_pixels(density, xs, Δx_pix)
    dx = xs[2] - xs[1]

    # number of grid points per pixel
    nbin = round(Int, Δx_pix / dx)

    Npix = div(length(xs), nbin)

    density_pix = zeros(Float64, Npix)
    xs_pix = zeros(Float64, Npix)

    for i in 1:Npix
        idx_start = (i-1)*nbin + 1
        idx_end   = i*nbin

        density_pix[i] = sum(density[idx_start:idx_end]) / (idx_end - idx_start)
        xs_pix[i] = sum(xs[idx_start:idx_end]) / (idx_end - idx_start)
    end

    return density_pix, xs_pix
end

function bin_after_convolve(density, xs, σ, Δx)
    return bin_pixels(convolve_gaussian(density,xs,σ) , xs, Δx)
end

function add_OD_noise(densities_meas, σ=0.01, μ=0.08)
    noisy_densities = similar(densities_meas)

    for i in eachindex(densities_meas)
        density = densities_meas[i]
        n = norm(density)

        mask = abs.(density ./ maximum(density)) .> 0.01

        #mask = convolve_gaussian(mask, 1:length(density),1)   # optional: broadens the noisy region

        noise = ( σ .* randn(size(density))) .* n

        noisy_densities[i] = density .+ noise .* mask
    end

    return noisy_densities
end


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