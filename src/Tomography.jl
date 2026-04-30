


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
	        F .+= densities[i] .* exp(1im * t_holds[i] * k)
	    end
	    F ./= nt

		# projection of F onto Hermite basis
	    # f[m] = ∫ dx F(x, k) ψ_m(x/b)
	    Φ = hcat([hermite_function_basis(λmax, x / b) for x in xs]...)
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