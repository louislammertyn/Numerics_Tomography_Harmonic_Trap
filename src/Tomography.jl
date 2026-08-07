


begin ### HERMITE
	# to generate hermite functions or polynomials
	function hermite_recurrence(f, nmax, x, h0)
	    f(0, h0)
	    nmax == 0 && return
	
	    h1 = sqrt(2) * x * h0
	    f(1, h1)
	
	    hm2, hm1 = h0, h1
	    for k in 2:nmax
	        h = sqrt(2 / k) * x * hm1 - sqrt((k - 1) / k) * hm2
	        f(k, h)
	        hm2, hm1 = hm1, h
	    end
	end

	# product of hermite polynomials w/ normalization factors of hermite functions
	function hermite_function_poly_prod(x, ns::Tuple)
	    res = 1.0
	    hermite_recurrence(maximum(ns), x, π^(-0.25)) do k, h
	        for n in ns
	            n == k && (res *= h)
	        end
	    end
	    return res
	end

	
	# integral of the product of n hermite functions
	function hermite_prod_integral(ns::Tuple; quad_rule=gausshermite(ceil(Int, (sum(ns) + 1) / 2)))
	    isodd(sum(ns)) && return 0.0
	    xs, ws = quad_rule
	    s = sqrt(2 / length(ns))
	    res = 0.0
	    for i in eachindex(ws)
	        res += ws[i] * hermite_function_poly_prod(s * xs[i], ns)
	    end
	    return s * res
	end
end


begin ## TOMOGRAPHY

	
	function build_kernel_tensor(λmax::Int, xs::Vector{Float64}, b::Float64, j::Int; convolve=false, σ=10.)
		Nx = length(xs)
		Kx = j==0 ? Array{Float64}(undef, Nx, λmax+1 - j) : Array{ComplexF64}(undef, Nx, λmax+1 - j)

		if convolve 
			xs_ = LinRange(minimum(xs), maximum(xs), length(xs)*20)
			Δx  = xs[2] - xs[1]
			for n in 0:λmax-j
				phase = j==0 ? 1. : exp(1im * j * π/2)
				Kx[:, n+1] .= phase * bin_after_convolve(hermite.(n, xs_ ./ b) .* hermite.(n+j, xs_ ./ b), xs_, σ, Δx)[1]
			end

			return Kx
		
		else
			for n in 0:λmax-j
				Kx[:, n+1] = exp(-1im * j * π/2)* hermite.(n, xs ./ b) .* hermite.(n+j, xs ./ b)
			end
			return Kx
		end

		
	end

	function build_kernel_tensor(λmax::Int, xs::Vector{Float64}, b::Float64; convolve=false, σ=10.)
		Nx = length(xs)
		Kx = Array{ComplexF64}(undef, Nx, λmax+1, λmax+1) 

		if convolve 
			xs_ = LinRange(minimum(xs), maximum(xs), length(xs)*20)
			xs_b = xs_ ./ b
			Δx  = xs[2] - xs[1]
			for n in 0:λmax, m in n:λmax
				Kx[:, n+1, m+1] .= exp(-1im * (m-n) * π/2)* bin_after_convolve(hermite.(n, xs_b) .* hermite.(m, xs_b ), xs_, σ, Δx)[1]
				Kx[:, m+1, n+1] = conj.(Kx[:, n+1, m+1])
			end

			return Kx
		
		else
			xs_b = xs ./ b
			for n in 0:λmax, m in n:λmax
				Kx[:, n+1, m+1] = exp(-1im * (m-n) * π/2)* hermite.(n, xs_b) .* hermite.(m, xs_b)
				Kx[:, m+1, n+1] = conj.(Kx[:, n+1, m+1])
			end
			return Kx
		end

		
	end
	
	# Build K_h(x,l,m)= K1(x,l, l+j) K2(x, m, m + k)
	function build_higher_kernel_tensor(λmax::Int, xs::Vector{Float64}, b::Float64, j::Int, k::Int; convolve=false, σ=10.)

		Nx = length(xs)
		K_h = Array{ComplexF64}(undef, Nx, Nx, λmax+1 - j, λmax+1 - k) 

		if convolve 
			xs_ = collect(LinRange(minimum(xs), maximum(xs), length(xs)*20))
			Δx  = xs[2] - xs[1]
			K1 = build_kernel_tensor(λmax, xs_, b, j )
			K2 = build_kernel_tensor(λmax, xs_, b, k )

			
			K1c = [bin_after_convolve(K1[:,n+1], xs_, σ, Δx)[1]
				for n in 0:λmax-j]

			K2c = [bin_after_convolve(K2[:,m+1], xs_, σ, Δx)[1]
				for m in 0:λmax-k]

			for n in 0:λmax-j, m in 0:λmax-k
				K_h[:,:,n+1,m+1] = K1c[n+1] * transpose(K2c[m+1])
			end

			return K_h
		
		else
			K1 = build_kernel_tensor(λmax, xs, b, j )
			K2 = build_kernel_tensor(λmax, xs, b, k )
			for n in 0:λmax-j, m in 0:λmax-k
				K_h[:, :, n+1, m+1] = K1[:, n+1] * Transpose(K2[:,m+1])
			end
			return K_h
		end
	end


		

	function Kernel_svd(K::Array)
		if length(size(K)) == 3 "Not the right kernel for this function, need K(x,n,m)"

			nx, n, m = size(K)
			A = reshape(K, nx, n*m)

			U, S, V = svd(A)

			r = length(S)
			V = reshape(V, n, m, r)

			
		else
			U, S, V = svd(K)

		end
		return U,S,V
	end

	function Higher_Kernel_svd(K::Array)
		if length(size(K)) == 4 "Not the right kernel for this function, need K(x,n,m)"

			nx, ny, n, m = size(K)
			A = reshape(K, nx*ny, n*m)

			U, S, V = svd(A)

			r = length(S)
			U = reshape(U[:,1:r], nx, ny, r)
			V = reshape(V[:,1:r], n, m, r)

			
		else
			U, S, V = svd(K)

		end
		return U,S,V
	end
	
	function svd_stats(S; tol=1e-3)

		r = count(s -> s > tol*S[1], S)

		κ = S[1]/S[end]
		κeff = S[1]/S[r]

		return (
			rank = r,
			cond = κ,
			cond_eff = κeff,
			retained_fraction = r/length(S)
		)
	end



	

	
	function plot_singular_values(S)

		# --- plot singular values ---
		p1 = plot(S ./ S[1], yscale=:log10,
				xlabel="mode index",
				ylabel="singular value",
				title="Singular value spectrum",
				legend=false)

		display(p1)

	end

	function svd_rank(S, tol::Float64)
		σmax = maximum(S)
		return sum(S .> tol*σmax)
	end
	
	function compute_resolvability(U, S, K; tol=1e-6)
		Nx, Nn = size(K)

		# determine rank
		σmax = maximum(S)
		r = sum(S .> tol*σmax)
		println(r)

		U_r = U[:, 1:r]

		resolvability = zeros(Nn)

		for j in 1:Nn
			col = K[:, j]
			proj = U_r' * col
			resolved_part = U_r * proj

			resolvability[j] = norm(resolved_part)^2 / norm(col)^2
		end

		return resolvability
	end






	function build_F(densities, t_holds, j::Int)
		nx, nt = length(densities[1]), length(t_holds)
 		F = zeros(ComplexF64, nx)
	    for i in 1:nt
	        F .+= densities[i] .* exp(-1im * t_holds[i] * j)
	    end
	    F ./= nt
	
		return F 
	end

	function tikhonov_lstsq(K, n, α; norm=false)  
		F = svd(K) 
		f = F.S ./ (F.S.^2 .+ α^2)   
		ρ = F.V * (f .* (F.U' * n)) 
		if norm==false
			norm = sum(ρ)
			ρ ./= norm
		else
			ρ ./= norm
		end
		return ρ, norm
	end

	function run_tomography(
		ρ::Matrix,
		lmax::Int64,
		t_tof::Float64,
		Δt::Float64;
		diags = [0],
		α = 1e-2,
		σ_ker = 12.0,
		σ_meas = 12.0,
		noise_level=0.,
		N_holds = 15,
		ω = 1.0,
		quantize=true
	)
		"""
		This function runs a full tomography experiment on an initial state given by density matrix ρ.
		It includes:
		1: free trap evolution
		2: free space evolution
		3: conversion to real space densities
		4: imaging modeling with convolution before binning and adding optional gaussian noise
		5: applying tomography for diagonals asked
		"""

		xs = collect(LinRange(-853, 911, 170))
		b = sqrt(1 + t_tof^2)

		# Always reconstruct diagonal 0 first for normalization
		diags = unique(vcat(0, diags))

		Ks = Dict(
			j => build_kernel_tensor(
				lmax,
				xs,
				b,
				j;
				convolve=true,
				σ=σ_ker,
			)
			for j in diags
		)

		t_ideal =
			N_holds < 23 ?
			t_holds_ideal(1., N_holds) :
			t_holds_ideal(0.5, N_holds)

		t_quant =
			quantize ?
			quantize_times(t_ideal, Δt) :
			t_ideal

		xs_ = LinRange(minimum(xs), maximum(xs), length(xs) * 20)
		Δx = xs[2] - xs[1]

		densities_tof = get_n_tof_optimized(
			ρ,
			t_quant,
			t_tof,
			xs_;
			ω=ω,
		)

		densities_non_pix = add_OD_noise(
			[
				convolve_gaussian(
					densities_tof[i],
					xs_,
					σ_meas,
				)
				for i in 1:N_holds
			],
			noise_level,
		)

		densities_meas = [
			bin_pixels(
				densities_non_pix[i],
				xs_,
				Δx,
			)[1]
			for i in 1:N_holds
		]

		ρ_tomo = zeros(ComplexF64, size(ρ))

		# -------------------------
		# requested diagonals
		# -------------------------
		n=1.

		for (id_j, j) in enumerate(diags)

			F_x_j = build_F(densities_meas, t_quant, j)

			if j == 0
				ρ_tomo[diagind(ρ_tomo, j)], n =
					tikhonov_lstsq(Ks[j], F_x_j, α)
					
			else
				ρ_tomo[diagind(ρ_tomo, j)] =
					tikhonov_lstsq(
						Ks[j],
						F_x_j,
						α;
						norm = n
					)[1]
			end
		end

	

		return ρ_tomo
	end

	function run_tomography(
			ρs_sim::Vector{Matrix{ComplexF64}},
			lmax::Int64,
			t_tof::Float64,
			Δt::Float64;
			kwargs...
		)

			ρs_sim_tomo = similar(ρs_sim)

			for (id_t, ρ) in enumerate(ρs_sim)
				ρs_sim_tomo[id_t] = run_tomography(
					ρ,
					lmax,
					t_tof,
					Δt;
					kwargs...
				)
			end

			return ρs_sim_tomo
		end

	
			


	# expects xs in units of lh, and ts in units 1/ω
	# densities[t_holds][xs] is an Array of Arrays
	# returns [rho(n, n+k) for n in [0, λmax-1]]
	function reconstruct_density_matrix_slice(xs, t_tof, densities, t_holds, k=0; λmax=100)
	    θ = atan(t_tof)
	    b = sqrt(1 + t_tof^2)
	    nx = length(xs)
	    nt = length(t_holds)
	
	    # compute the time-filtered signal F(x, k)
	    F = zeros(ComplexF64, nx)
	    for i in 1:nt
	        F .+= densities[i] .* exp(-1im * t_holds[i] * k)
	    end
	    F ./= nt

		# projection of F onto Hermite basis
	    # f[m] = ∫ dx F(x, k) ψ_m(x/b)
		Φ = hcat([hermite.(n, xs ./ b) for n in 0:λmax-1]...)'
	    #Φ = hcat([hermite_function_basis(λmax, x / b) for x in xs]...)
	    f = [trapz(xs, F .* Φ[m, :]) for m in 1:λmax]

		# projection of kernel onto Hermite basis
	    # M[m, λ] = ∫ dx K(x, λ, λ+k, t) ψ_m(x/b)
	    M = zeros(ComplexF64, λmax, λmax)
	    quad_rule = gausshermite(ceil(Int, (3 * λmax + k + 1) / 2))
	
	    # exp(-im * (λi - λj) * θ)
	    kernel_phase = exp(1im * k * θ)
	
	    for λ_idx in 1:λmax
	        λ = λ_idx - 1
	        for m_idx in 1:λmax
	            m = m_idx - 1
	            # triple overlap integral: ∫ du ψ_λ(u) ψ_{λ+k}(u) ψ_m(u)
			    # jacobian dx = b*du cancels the 1/b in the kernel
	            overlap = hermite_prod_integral((λ, λ + k, m); quad_rule=quad_rule)
	            M[m_idx, λ_idx] = kernel_phase * overlap
	        end
	    end
	
	    return pinv(M) * f
	end


end