using Revise
using ExpSynthDim
using LinearAlgebra, Plots, ColorSchemes
using Printf, LaTeXStrings

# --------------------------------------------------
# Effective-rank diagnostics
# --------------------------------------------------
begin
    function effective_rank(S; α=1e-2)
        smax = maximum(S)
        count(s -> s > α*smax, S)
    end

    function sector_parameter_count(lmax, j)
        if j == 0
            return lmax + 1
        else
            # complex coefficients -> 2 real parameters
            return lmax + 1 - abs(j)
        end
    end

    function full_parameter_count(lmax)
        return (lmax + 1)^2
    end

    function kernel_summary(S, Nparams; α=1e-2)
        reff = effective_rank(S; α=α)

        (
            reff = reff,
            Nparams = Nparams,
            η = reff / Nparams
        )
    end
end

# --------------------------------------------------
# Main appendix table
# --------------------------------------------------

lmax  = 60
σconv = 12.0
α     = 1e-2

xs = collect(LinRange(-853., 911.,170))
b  = 33.2
begin
println()
println("====================================================")
println("APPENDIX CONDITIONING ANALYSIS")
println("====================================================")
println("lmax     = $lmax")
println("Nx       = $(length(xs))")
println("σconv    = $σconv")
println("α/smax   = $α")
println("====================================================")
println()

# --------------------------------------------------
# Full reconstruction
# --------------------------------------------------

Kfull = build_kernel_tensor(
    lmax,
    xs,
    b;
    convolve=true,
    σ=σconv
)

_, Sfull, _ = Kernel_svd(Kfull)

full_stats = kernel_summary(
    Sfull,
    full_parameter_count(lmax);
    α=α
)

println("FULL TOMOGRAPHY")
println("----------------------------------------------")
println("Nparams = ", full_stats.Nparams)
println("reff    = ", full_stats.reff)
println("η       = ", round(full_stats.η,digits=4))
println()

# --------------------------------------------------
# Sector-resolved reconstruction
# --------------------------------------------------

Rsec = 0

println("SECTOR RESOLVED")
println("----------------------------------------------")

for j in 0:lmax
    
    Kj = build_kernel_tensor(
        lmax,
        xs,
        b,
        j;
        convolve=true,
        σ=σconv
    )

    _, Sj, _ = Kernel_svd(Kj)

    st = kernel_summary(
        Sj,
        sector_parameter_count(lmax,j);
        α=α
    )

    Rsec += st.reff
    if j ∈ [0,1,2,3,30, 50, 60]
    @printf(
        "δλ=%2d   Nparams=%4d   reff=%4d   η=%6.3f\n",
        j,
        st.Nparams,
        st.reff,
        st.η
    )
    end
end

println()
println("------------------------------------------------")
println("SUMMARY")
println("------------------------------------------------")
println("r_eff(full)      = ", full_stats.reff)
println("R_sec            = ", Rsec)
println("Gain factor      = ", round(Rsec/full_stats.reff,digits=2))
println("η(R_sec)          = ", round(Rsec / full_stats.Nparams,digits=4))
println("η(full)          = ", round(full_stats.η,digits=4))
println("------------------------------------------------")
end


begin 



# --------------------------------------------------
# Parameters
# --------------------------------------------------

σconv = 12.0
α     = 1e-2

xs = collect(LinRange(-853., 911., 170))
b  = 33.2

lmax_list = 1:4:41

ηfull_vals = Float64[]
ηRsec_vals = Float64[]

# --------------------------------------------------
# Scan over lmax
# --------------------------------------------------

for lmax in lmax_list

    # ---------- full tomography ----------

    Kfull = build_kernel_tensor(
        lmax,
        xs,
        b;
        convolve=true,
        σ=σconv
    )

    _, Sfull, _ = Kernel_svd(Kfull)

    reff_full = effective_rank(Sfull; α=α)
    Nfull     = full_parameter_count(lmax)

    ηfull = reff_full / Nfull

    # ---------- sector tomography ----------

    Rsec = 0

    for j in 0:lmax

        Kj = build_kernel_tensor(
            lmax,
            xs,
            b,
            j;
            convolve=true,
            σ=σconv
        )

        _, Sj, _ = Kernel_svd(Kj)

        Rsec += effective_rank(Sj; α=α)

    end

    ηRsec = Rsec / Nfull

    push!(ηfull_vals, ηfull)
    push!(ηRsec_vals, ηRsec)

    println(
        "lmax=$lmax   ηfull=$(round(ηfull,digits=4))   ",
        "ηRsec=$(round(ηRsec,digits=4))"
    )
end

using Plots
gr()

default(
    linewidth = 2.5,
    framestyle = :box,
    dpi = 600,
    guidefontsize = 16,
    tickfontsize = 13,
    legendfontsize = 13,
    fontfamily = "Computer Modern",
)

p = plot(
    lmax_list,
    ηfull_vals;
    color = :black,
    linestyle = :solid,
    xlabel = L"\lambda_{\max}",
    ylabel = L"\eta",
    label = L"\eta_{\rm full}",
    legend = :topright,
    size = (500, 350),
    margin = 4Plots.mm,
)

plot!(
    p,
    lmax_list,
    ηRsec_vals;
    color = RGB(0.8,0.1,0.1),
    linestyle = :dash,
    label = L"\eta_{\rm sec}",
)
display(p)
savefig(p, "eta_vs_lmax.png")
end

begin

lmax = 60
b    = 33.2

jvals = [0, 10]

# vary σ/b
σ_over_b = [0.10, 0.20, 0.30, 0.40]

# vary Δx/b
Δx_over_b = [0.10, 0.20, 0.30, 0.40]

xmin, xmax = -853., 911.
colors = get(ColorSchemes.viridis, range(0.1,0.9,length=4))
pσ = plot(
    layout=(1,2),
    size=(900,350),
    dpi=600,
    framestyle=:box,
    yscale=:log10,
    xlabel=L"s_n",
    ylabel=L"s_n/s_1",
    legend=:bottomleft,
    left_margin = 4Plots.mm,
    bottom_margin = 4Plots.mm,
    palette = palette([:purple, :blue], 4)
)

for (panel,j) in enumerate(jvals)

    for rσ in σ_over_b

        σ = rσ*b

        # dense sampling so only σ changes
        xs = collect(LinRange(xmin,xmax,500))

        Kj = build_kernel_tensor(
            lmax,
            xs,
            b,
            j;
            convolve=true,
            σ=σ
        )

        _, Sj, _ = Kernel_svd(Kj)

        plot!(
            pσ[panel],
            Sj ./ Sj[1];
            lw=2.5,
            label=L"\sigma/b=%$(round(rσ,digits=2)) \; l_{h.o.}"
        )
    end

    title!(pσ[panel], L"\delta\lambda=%$j")
end

savefig(pσ,"svd_sigma_over_b.png")
display(pσ)

end

begin

lmax = 60
b    = 33.2

jvals = [0,10]

Δx_over_b = [0.10,0.20,0.30,0.40]

xextent = 911.0 + 853.0

pΔx = plot(
    layout=(1,2),
    size=(900,350),
    dpi=600,
    framestyle=:box,
    yscale=:log10,
    xlabel=L"s_n",
    ylabel=L"s_n/s_1",
    legend=:bottomleft,
    left_margin = 4Plots.mm,
    bottom_margin = 4Plots.mm,
    palette = palette([:orange, :green], 4)
)

σ = 12.0

for (panel,j) in enumerate(jvals)

    for rΔ in Δx_over_b

        Δx = rΔ*b

        Nx = Int(round(xextent/Δx))
        Nx = max(Nx,80)

        xs = collect(LinRange(-853.,911.,Nx))

        Kj = build_kernel_tensor(
            lmax,
            xs,
            b,
            j;
            convolve=true,
            σ=σ
        )

        _, Sj, _ = Kernel_svd(Kj)

        plot!(
            pΔx[panel],
            Sj ./ Sj[1];
            lw=2.5,
            label=L"\Delta x/b=%$(round(rΔ,digits=2)) \; l_{h.o.}"
        )
    end

    title!(pΔx[panel], L"\delta\lambda=%$j")
end

savefig(pΔx,"svd_deltax_over_b.png")
display(pΔx)

end
