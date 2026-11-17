###############################################################################
#  Tomography.jl
#
#  State tomography of the single-particle density matrix ρ from TOF density
#  profiles n(x, t_hold) recorded after a variable hold time in the bare trap.
#
#  Conventions (same as the forward model `get_n_tof_optimized`):
#    * ρ[n+1, n+j+1] (j ≥ 0) is element ρ_{n, n+j} on the j-th upper diagonal,
#      i.e. `diagind(ρ, j)`.
#    * It enters n(x, t) with time factor  e^{+i j ω t}  and TOF phase e^{+i j θ},
#      θ = atan(ω t_tof). (The paper uses the transposed convention, ρ ↔ ρᵀ.)
#    * Positions are in units of the harmonic length, times in units of 1/ω.
#
#  Pipeline (see paper, "State tomography"):
#    1. `tomography_weights` / `build_F`: project the hold-time series onto
#       harmonic j with weights w_j(t_n) satisfying Σ_n w_j(t_n) e^{i j' ω t_n} = δ_{jj'}
#       for |j'| ≤ δλ_R. For equidistant t_n over one period this is the DFT.
#    2. `build_kernel_tensor`: kernel K(x, λ, λ+j) for that diagonal.
#    3. `tikhonov_lstsq`: regularized inversion, one diagonal at a time.
###############################################################################

const _OVERSAMPLE = 20   # fine-grid oversampling factor used before pixel binning


# =============================================================================
# Hermite helpers
# =============================================================================

"""
    hermite_recurrence(f, nmax, x, h0)

Run the normalized Hermite-function recurrence up to order `nmax` at `x`,
calling `f(k, h_k)` for every order k (h0 is the order-0 value).
"""
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

"""
    hermite_function_poly_prod(x, ns::Tuple)

Product of the Hermite polynomials of orders `ns` at `x`, including the
normalization factors of the Hermite functions (Gaussian weight excluded).
"""
function hermite_function_poly_prod(x, ns::Tuple)
    res = 1.0
    hermite_recurrence(maximum(ns), x, π^(-0.25)) do k, h
        for n in ns
            n == k && (res *= h)
        end
    end
    return res
end

"""
    hermite_prod_integral(ns::Tuple; quad_rule)

∫ dx Π_{n ∈ ns} ψ_n(x), computed exactly with Gauss–Hermite quadrature.
"""
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


# =============================================================================
# Kernels  K(x, λ, λ')  ∝  e^{+i(λ'-λ)θ} ψ_λ(x/b) ψ_λ'(x/b)
# =============================================================================

# TOF phase of the j-th upper diagonal. θ = π/2 is the far-field limit; pass
# θ = atan(t_tof) for the exact value used in the forward model.
kernel_phase(j; θ=π/2) = cis(j * θ)

# Grid on which the Hermite functions are evaluated: the pixel grid itself, or
# an oversampled grid when the imaging blur + pixel binning is modelled.
function _hermite_grid(xs, b, L; convolve)
    grid = convolve ? collect(LinRange(minimum(xs), maximum(xs), _OVERSAMPLE * length(xs))) : xs
    return grid, scaled_hs(grid, b, L)
end

# Real profile ψ_n(x/b) ψ_m(x/b), optionally blurred (σ) and binned to the pixels of xs.
function _pair_profile(H, grid, n, m; convolve, σ, Δx)
    p = H[:, n+1] .* H[:, m+1]
    return convolve ? bin_after_convolve(p, grid, σ, Δx)[1] : p
end

# Nx × (λmax+1-j) matrix of real profiles for the j-th diagonal.
function _diag_profiles(λmax, xs, b, j; convolve=false, σ=10.0)
    0 ≤ j ≤ λmax || throw(ArgumentError("diagonal j=$j outside 0:$λmax"))
    grid, H = _hermite_grid(xs, b, λmax + 1; convolve)
    Δx = xs[2] - xs[1]
    P = Matrix{Float64}(undef, length(xs), λmax + 1 - j)
    for n in 0:λmax-j
        P[:, n+1] = _pair_profile(H, grid, n, n + j; convolve, σ, Δx)
    end
    return P
end

"""
    build_kernel_tensor(λmax, xs, b, j; convolve=false, σ=10., θ=π/2)

Kernel for the j-th diagonal: matrix `K[x, n+1] = e^{ijθ} ψ_n(x/b) ψ_{n+j}(x/b)`,
n = 0…λmax-j. Real for j = 0. With `convolve=true` each profile is blurred by a
Gaussian of width σ and averaged over the camera pixels of `xs`.
"""
function build_kernel_tensor(λmax::Int, xs::AbstractVector{<:Real}, b::Real, j::Int;
                             convolve=false, σ=10.0, θ=π/2)
    P = _diag_profiles(λmax, xs, b, j; convolve, σ)
    return j == 0 ? P : kernel_phase(j; θ) .* P
end

"""
    build_kernel_tensor(λmax, xs, b; convolve=false, σ=10., θ=π/2)

Full kernel `K[x, n+1, m+1]` for all (n, m), Hermitian in (n, m).
"""
function build_kernel_tensor(λmax::Int, xs::AbstractVector{<:Real}, b::Real;
                             convolve=false, σ=10.0, θ=π/2)
    grid, H = _hermite_grid(xs, b, λmax + 1; convolve)
    Δx = xs[2] - xs[1]
    K = Array{ComplexF64}(undef, length(xs), λmax + 1, λmax + 1)
    for n in 0:λmax, m in n:λmax
        K[:, n+1, m+1] = kernel_phase(m - n; θ) .* _pair_profile(H, grid, n, m; convolve, σ, Δx)
        K[:, m+1, n+1] = conj.(K[:, n+1, m+1])
    end
    return K
end

"""
    build_higher_kernel_tensor(λmax, xs, b, j, k; convolve=false, σ=10., θ=π/2)

Two-point kernel `K_h[x, y, l+1, m+1] = K_j(x, l) K_k(y, m)` built from the
diagonal kernels of diagonals j and k.
"""
function build_higher_kernel_tensor(λmax::Int, xs::AbstractVector{<:Real}, b::Real, j::Int, k::Int;
                                    convolve=false, σ=10.0, θ=π/2)
    K1 = build_kernel_tensor(λmax, xs, b, j; convolve, σ, θ)
    K2 = build_kernel_tensor(λmax, xs, b, k; convolve, σ, θ)
    Nx = length(xs)
    K_h = Array{ComplexF64}(undef, Nx, Nx, size(K1, 2), size(K2, 2))
    for n in axes(K1, 2), m in axes(K2, 2)
        K_h[:, :, n, m] = K1[:, n] * transpose(K2[:, m])
    end
    return K_h
end


# =============================================================================
# Kernel analysis (SVD, rank, resolvability)
# =============================================================================

"""
    Kernel_svd(K)

SVD of a kernel. For a 3-tensor `K[x, n, m]` it is flattened to (x, n·m) and
`V` is returned reshaped as `V[n, m, mode]`; a matrix is decomposed directly.
"""
function Kernel_svd(K::AbstractArray)
    ndims(K) == 3 || return svd(K)
    nx, n, m = size(K)
    U, S, V = svd(reshape(K, nx, n * m))
    return U, S, reshape(V, n, m, length(S))
end

"""
    Higher_Kernel_svd(K)

SVD of a 4-tensor `K[x, y, n, m]` flattened to (x·y, n·m); returns
`U[x, y, mode]`, `S`, `V[n, m, mode]`. A matrix is decomposed directly.
"""
function Higher_Kernel_svd(K::AbstractArray)
    ndims(K) == 4 || return svd(K)
    nx, ny, n, m = size(K)
    U, S, V = svd(reshape(K, nx * ny, n * m))
    r = length(S)
    return reshape(U[:, 1:r], nx, ny, r), S, reshape(V[:, 1:r], n, m, r)
end

"Number of singular values above `tol · max(S)`."
svd_rank(S, tol::Float64) = count(>(tol * maximum(S)), S)

"Rank, condition numbers and retained fraction of a singular-value spectrum."
function svd_stats(S; tol=1e-3)
    r = count(s -> s > tol * S[1], S)
    return (
        rank = r,
        cond = S[1] / S[end],
        cond_eff = S[1] / S[r],
        retained_fraction = r / length(S),
    )
end

"Plot the normalized singular-value spectrum on a log scale."
function plot_singular_values(S)
    p = plot(S ./ S[1]; yscale=:log10, xlabel="mode index", ylabel="singular value",
             title="Singular value spectrum", legend=false)
    display(p)
end

"""
    compute_resolvability(U, S, K; tol=1e-6, verbose=true)

Fraction of each kernel column `K[:, j]` lying in the span of the singular
vectors with σ > tol·σ_max (1 = fully resolvable).
"""
function compute_resolvability(U, S, K; tol=1e-6, verbose=true)
    r = svd_rank(S, tol)
    verbose && println(r)
    U_r = U[:, 1:r]
    return [norm(U_r * (U_r' * col))^2 / norm(col)^2 for col in eachcol(K)]
end


# =============================================================================
# Harmonic projection over hold times
# =============================================================================

"""
    tomography_weights(t_holds, δλ_R; ω=1.0) -> W

Projection weights for isolating the hold-time harmonics |j| ≤ δλ_R.

With `A[n, j] = e^{i j ω t_n}` (N × (2δλ_R+1)), `W = pinv(A)` satisfies
`W * A = I`, i.e. Σ_n W[j, n] e^{i j' ω t_n} = δ_{jj'}. Row `δλ_R + 1 + j` holds
w_j(t_n). Requires N ≥ 2δλ_R + 1; for N > 2δλ_R + 1 it is the least-squares
solution. For equidistant t_n over one period it reduces to the DFT, w_j = e^{-ijωt_n}/N.
"""
function tomography_weights(t_holds, δλ_R::Int; ω=1.0)
    N = length(t_holds)
    N ≥ 2δλ_R + 1 || throw(ArgumentError("need N ≥ 2δλ_R+1 hold times (N=$N, δλ_R=$δλ_R)"))
    A = [cis(j * ω * t) for t in t_holds, j in -δλ_R:δλ_R]
    return pinv(A)
end

_default_δλ_R(t_holds) = (length(t_holds) - 1) ÷ 2

"""
    build_F(densities, t_holds, j; ω=1.0, δλ_R=(N-1)÷2)

Projected density ñ_j(x) = Σ_n w_j(t_n) n(x, t_n), which contains only the j-th
diagonal of ρ: ñ_j(x) = Σ_λ K(x, λ, λ+j) ρ_{λ, λ+j}.

`δλ_R` is the assumed coherence range (harmonics |j'| > δλ_R are taken to be
negligible). The default uses the largest range the N hold times allow.
"""
function build_F(densities, t_holds, j::Int; ω=1.0, δλ_R::Int=_default_δλ_R(t_holds))
    abs(j) ≤ δλ_R || throw(ArgumentError("|j|=$(abs(j)) exceeds δλ_R=$δλ_R"))
    w = tomography_weights(t_holds, δλ_R; ω)[δλ_R + 1 + j, :]
    F = zeros(ComplexF64, length(densities[1]))
    for (wn, n) in zip(w, densities)
        F .+= wn .* n
    end
    return F
end

"""
    harmonic_residual(densities, t_holds; ω=1.0, δλ_R=(N-1)÷2)

Relative residual ‖n − n_fit‖/‖n‖ of the harmonic fit with |j| ≤ δλ_R. Only
informative for N > 2δλ_R + 1; a large value signals coherences beyond δλ_R.
"""
function harmonic_residual(densities, t_holds; ω=1.0, δλ_R::Int=_default_δλ_R(t_holds))
    D = permutedims(reduce(hcat, densities))           # N × Nx
    A = [cis(j * ω * t) for t in t_holds, j in -δλ_R:δλ_R]
    return norm(D - A * (pinv(A) * D)) / norm(D)
end


# =============================================================================
# Inversion
# =============================================================================

"""
    tikhonov_lstsq(K, n, α; normalization=nothing) -> (ρ, normalization)

Tikhonov-regularized least squares, ρ = argmin ‖Kρ − n‖² + α²‖ρ‖².
The result is divided by `normalization`; if not given, `sum(ρ)` is used (the
trace when solving for the main diagonal) and returned for reuse on the other diagonals.
"""
function tikhonov_lstsq(K, n, α; normalization=nothing)
    F = svd(K)
    ρ = F.V * ((F.S ./ (F.S .^ 2 .+ α^2)) .* (F.U' * n))
    normalization = something(normalization, sum(ρ))
    return ρ ./ normalization, normalization
end


# =============================================================================
# Full simulated tomography experiment
# =============================================================================

"""
    run_tomography(ρ, lmax, t_tof, Δt; kwargs...) -> (ρ_tomo, densities_meas)

Simulate a full tomography experiment on the state `ρ` and reconstruct it:
1. hold-time evolution in the trap (true frequency `ω`),
2. TOF expansion and conversion to real-space densities,
3. imaging: Gaussian blur (`σ_meas`), pixel binning, optional OD noise (`noise_level`),
4. harmonic projection (`build_F`) and Tikhonov inversion for each diagonal in `diags`.

Diagonal 0 is always reconstructed first and its trace normalizes the other diagonals.

Keywords
- `t_holds`: hold times; default `N_holds` equidistant times over 1 (N<23) or 2 periods,
  quantized to `Δt` if `quantize`.
- `diags=[0]`, `α=1e-2` (Tikhonov), `σ_ker=12.` (blur assumed in the kernel), `σ_meas=12.`
- `ω=1.0`: true trap frequency in the simulation. `ω_rec=1.0`: frequency assumed in the
  reconstruction.
- `δλ_R`: assumed coherence range for the harmonic projection (default (N-1)÷2).
- `θ=π/2`: TOF phase used in the kernel (exact value: `atan(t_tof)`).
- `xs`: camera pixel positions.
"""
function run_tomography(
    ρ::Matrix,
    lmax::Int64,
    t_tof::Float64,
    Δt::Float64;
    t_holds = nothing,
    diags = [0],
    α = 1e-2,
    σ_ker = 12.0,
    σ_meas = 12.0,
    noise_level = 0.0,
    N_holds = 15,
    ω = 1.0,
    ω_rec = 1.0,
    δλ_R = nothing,
    θ = π/2,
    quantize = true,
    xs = collect(LinRange(-853, 911, 170)),
)
    b = sqrt(1 + t_tof^2)
    diags = unique(vcat(0, diags))   # diagonal 0 first, for normalization

    # --- hold times ---
    if isnothing(t_holds)
        t_ideal = t_holds_ideal(N_holds < 23 ? 1.0 : 0.5, N_holds)
        t_meas = quantize ? quantize_times(t_ideal, Δt) : t_ideal
    else
        t_meas = t_holds
    end
    R = something(δλ_R, _default_δλ_R(t_meas))

    # --- simulated measurement ---
    xs_fine = LinRange(minimum(xs), maximum(xs), _OVERSAMPLE * length(xs))
    Δx = xs[2] - xs[1]
    densities_tof = get_n_tof_optimized(ρ, t_meas, t_tof, xs_fine; ω=ω)
    densities_binned = [bin_pixels(convolve_gaussian(n, xs_fine, σ_meas), xs_fine, Δx)[1]
                        for n in densities_tof]
    densities_meas = add_OD_noise(densities_binned, noise_level)

    # --- reconstruction, diagonal by diagonal ---
    ρ_tomo = zeros(ComplexF64, size(ρ))
    normalization = nothing
    for j in diags
        K = build_kernel_tensor(lmax, xs, b, j; convolve=true, σ=σ_ker, θ=θ)
        F = build_F(densities_meas, t_meas, j; ω=ω_rec, δλ_R=R)
        ρ_tomo[diagind(ρ_tomo, j)], normalization = tikhonov_lstsq(K, F, α; normalization)
    end

    return ρ_tomo, densities_meas
end

"Run `run_tomography` on every state of a series; returns the reconstructed states."
function run_tomography(ρs_sim::Vector{Matrix{ComplexF64}}, lmax::Int64, t_tof::Float64, Δt::Float64; kwargs...)
    return [first(run_tomography(ρ, lmax, t_tof, Δt; kwargs...)) for ρ in ρs_sim]
end


# =============================================================================
# Analytic reconstruction via Hermite projection (no pixel model)
# =============================================================================

"""
    reconstruct_density_matrix_slice(xs, t_tof, densities, t_holds, k=0; λmax=100, δλ_R=(N-1)÷2)

Reconstruct the k-th diagonal [ρ_{n, n+k} for n = 0…λmax-1] by projecting the
harmonic ñ_k(x) onto ψ_m(x/b) and inverting the triple-overlap matrix
M[m, λ] = e^{ikθ} ∫ du ψ_λ ψ_{λ+k} ψ_m (the Jacobian dx = b du cancels the 1/b
of the kernel). Expects `xs` in harmonic lengths and `t_holds` in units of 1/ω.
"""
function reconstruct_density_matrix_slice(xs, t_tof, densities, t_holds, k=0;
                                          λmax=100, δλ_R::Int=_default_δλ_R(t_holds))
    θ = atan(t_tof)
    b = sqrt(1 + t_tof^2)

    F = build_F(densities, t_holds, k; δλ_R)

    # f[m] = ∫ dx F(x) ψ_m(x/b)
    f = [trapz(xs, F .* hermite.(m, xs ./ b)) for m in 0:λmax-1]

    quad_rule = gausshermite(ceil(Int, (3λmax + k + 1) / 2))
    phase = cis(k * θ)
    M = [phase * hermite_prod_integral((λ, λ + k, m); quad_rule) for m in 0:λmax-1, λ in 0:λmax-1]

    return pinv(M) * f
end
