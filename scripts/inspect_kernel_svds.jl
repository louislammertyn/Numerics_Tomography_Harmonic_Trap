using Revise 
using ExpSynthDim
using Plots, ColorSchemes
using LinearAlgebra

begin
    for lmax in 40:60
        
        xs = collect(LinRange(-830, 1000, 100) ./ 33)
        K = build_kernel_tensor(lmax, xs, 1; convolve=true, σ=0.01)
        U, S, V = svd(K)

        plot_singular_values(S)

        R = compute_resolvability(U,S, K; tol=1e-2)
    end

end


function plot_singular_values_comparison(lmax, xs, j; sigmas=[0.5, 0.1, 0.05, 0.01, 0.001])

    # --- baseline (no convolution) ---
    K0 = build_kernel_tensor(lmax, xs, j; convolve=false)
    S0 = svdvals(K0)
    S0[S0 .< 1e-19] .= 1e-19 
    colors = [get(ColorSchemes.viridis, t) for t in range(0, 1; length=length(sigmas))]
    p = plot(yscale=:log10,
             xlabel="Index",
             ylabel="Singular value",
             title="Singular values vs σ (λmax = $lmax, pixel= 5 μm)",
             legend=:topright,
             linewidth=8,
             dpi=200,
             framestyle=:box
            )
            



    plot!(p, S0, label="no convolution", linestyle=:dot, color=:black, linewidth=2)
   

    # --- convolved cases ---
    for (i,σ) in enumerate(sigmas)
        K = build_kernel_tensor(lmax, xs, j; convolve=true, σ=σ)
        S = svdvals(K)
        S[S .< 1e-19] .= 1e-19 


        plot!(p, S, label="σ = $(round(σ * 0.514;digits=1)) μm", linewidth=3, color=colors[i])
    end

    return p
end


xs = collect(LinRange(-830, 1000, 200) ./ 33)

p = plot_singular_values_comparison(200, xs, 3;
    sigmas=[10, 5, 2, 1, 0.5, 0.1])

display(p)

savefig(p, "svd_values_kernel.png")
