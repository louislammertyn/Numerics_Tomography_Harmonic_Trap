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

function get_n_tof_optimized(ρ_l::Matrix{ComplexF64}, t_holds::Vector{Float64}, t_tof::Float64, xs::AbstractArray, hs; ω=1.0)
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