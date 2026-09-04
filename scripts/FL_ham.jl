using Revise
using LinearAlgebra
using LaTeXStrings
using JLD2, NPZ
using ExpSynthDim
using Plots

AOD_single_drive = "data/pattern_single_drive.npy"
frames_sd = npzread(AOD_single_drive)
timesteps_per_drive_sd = size(frames_sd)[1]
for i in 1:57
    display(plot(frames_sd[i,:]))
end

plot([(frames_sd[3,:] .- 1.5), -(frames_sd[56,:].-1.5)])

AOD_2drive = "data/pattern_2drive.npy"
frames_2d = npzread(AOD_2drive)
timesteps_per_drive_2d = size(frames_2d)[1] 


k=2
begin
    p = make_params(;κ = iref_to_k(k*1e-5), L = 50)
    xs_μm =  LinRange(-10, 10, 5000)
    xs = xs_μm .*1e-6 ./p.lₕ 
    
    ps = LinRange(-20, 20, 10000)

    hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

    t_tof = 0.012 * p.ω
    Δt = 0.1e-3 * p.ω
    AOD_res = 38.596 *1e-6 * p.ω

    drive_width = 20

    harmonics_phases_amps = [(1, 0., 1.)]

end

# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L ) - 0.5 * I(p.L)
    step(x) = theta(x,0,drive_width)

    const V_sd = [drive_elements(xs,   p.κ .*  frames_sd[nt, :], p.L) for nt in 1:timesteps_per_drive_sd]
    drive_sd(t) =  V_sd[Int(mod(div(t, AOD_res)+1, timesteps_per_drive_sd)+1)]

    const V_2d =    [drive_elements(xs, p.κ .* frames_2d[nt, :], p.L) for nt in 1:timesteps_per_drive_2d]
    drive_2d(t) = V_2d[Int(mod(div(t, AOD_res)+1, timesteps_per_drive_2d)+1)]

    V_d_ideal = p.κ .* drive_elements(xs, step, p.L)

    H_fl_ideal = RWA_FL_Hamiltonian(V_d_ideal, p, harmonics_phases_amps, :smooth)
    H_fl_ideal -= I(p.L) *0.5

    original_wd = copy(p.ω_d)
   
    V_fl_sd = V_AOD_FL(drive_sd, p)
    V_fl_sd -= I(p.L) * (sum(diag(V_fl_sd,0)) / p.L)
    H_fl_sd = V_fl_sd  + HO_matrix(p.L, p.Δ/p.ω) - 0.5*I(p.L)

    p = make_params(;κ = iref_to_k(2e-5), L = 50, Δ=0.)
    H_fl_ideal_2d = RWA_FL_Hamiltonian(V_d_ideal, p, [(1, 0., 1.), (3, 0.,3. )], :smooth)
    H_fl_ideal_2d -= I(p.L) *0.5
    
    V_fl_2d = V_AOD_FL(drive_2d, p)
    V_fl_2d -= I(p.L) * (sum(diag(V_fl_2d,0)) / p.L)
    H_fl_2d = V_fl_2d 

    p = make_params(;κ = iref_to_k(2e-5), L = 50)

end
diag(H_fl_ideal_2d,0)
heatmap(abs.(abs.(H_fl_ideal  ) - abs.(H_fl_sd)))
heatmap(abs.(abs.(H_fl_ideal_2d  ) - abs.(H_fl_2d)))
heatmap(abs.(angle.(H_fl_ideal  ) - angle.( H_fl_sd)))

timesteps_per_drive_sd * AOD_res
2π/ p.ω_d

begin 
    T_d = (2π/p.ω_d)
    U_fl_ideal = exp(-1im * T_d * H_fl_ideal)
    U_fl_AOD_ideal = exp(-1im * T_d * (H_fl_sd))
    #U_fl = Floquet_operator_num_int([H_ho, V_d_ideal], [t -> 1., t -> sin(t)], 2π/p.ω_d)
    U_fl_AOD = Floquet_operator_seq(V_sd .+ [(H_ho - I(p.L)*0.5) for i in 1:length(V_sd)],T_d)

    U_fl_AOD_2d = Floquet_operator_seq(V_2d .+ [(H_ho - I(p.L)*0.5) for i in 1:length(V_2d)],2π)
    U_fl_ideal_2d = exp(-1im * T_d * (H_fl_ideal_2d))
    U_fl_AOD_ideal_2d = exp(-1im * T_d * (H_fl_2d))
    #U_fl_2d = Floquet_operator_num_int([H_ho, V_d_ideal, V_d_ideal], [t -> 1., t -> sin(t), t -> 3. * sin(3*t)], 2π)
end

heatmap(abs.(abs.(U_fl_AOD ) - abs.(U_fl_ideal)))
heatmap(abs.(angle.(U_fl_AOD) - angle.(U_fl_ideal)))
heatmap((real.(U_fl_AOD) - real.(U_fl_ideal)) ./ maximum(real.(U_fl_ideal)))
heatmap(imag.(U_fl_AOD) - imag.(U_fl_ideal) ./ maximum(imag.(U_fl_ideal)))

heatmap(abs.(angle.(U_fl_2d) - angle.(U_fl_ideal_2d)))
heatmap(abs.(abs.(U_fl_2d) - abs.(U_fl_ideal_2d)))
H_fl_2d
ns = 0:50
mean(x) = sum(x)/length(x)

ρ_diag = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/initial_state_diag.npy")
ρ0 = zeros(ComplexF64, p.L, p.L)
[ρ0[i,i] = ρ_diag[i] for i in 1:p.L ]

######################################
### Plot 1: evolution full vs ideal (fl+proj)
# This should give 2 plots, where one gives the mean of the absolute difference per diagonal.
# The other plot should give this mean divided by the max abs value of that diagonal as to show the relative scale
######################################

Nj = 6
Nn = length(ns)

data = zeros(Float64, Nj, 2, Nn)

for (idx, j) in enumerate(0:5)

    Δρs = [
        diag(
            abs.(U_fl_AOD^n * ρ0 * (U_fl_AOD')^n -
                 U_fl_ideal^n * ρ0 * (U_fl_ideal')^n),
            j
        )
        for n in ns
    ]

    ρs_ideal = [
        diag(
            abs.(U_fl_ideal^n * ρ0 * (U_fl_ideal')^n),
            j
        )
        for n in ns
    ]

    mean_abs_err = [mean(Δρ) for Δρ in Δρs]

    typical_rel_err = [
        mean(Δρ) / maximum(ρ)
        for (Δρ, ρ) in zip(Δρs, ρs_ideal)
    ]

    data[idx, 1, :] .= mean_abs_err
    data[idx, 2, :] .= typical_rel_err
end

npzwrite("data/diag_errors_iref$k.npy", data)



######################################
### Plot 2: Elements of ideal vs aod zeroth

# Here I expect a plot for each drive where the real and imaginary parts of the first n diagonals are plotted and compared for ideal and sd and ideal_2d and 2d
######################################

npzwrite("data/ideal_vs_aod_zeroth_iref$k.npy", Dict( "H_fl_ideal" => H_fl_ideal, "H_fl_sd" => H_fl_sd, "H_fl_ideal_2d" => H_fl_ideal_2d," H_fl_2d" => H_fl_2d))

######################################
### Plot 3: similarity of ideal vs ideal full
######################################
# Here there should be a comparison of the unitaries based on it's eigenvalues or eigenvectors for the sd and 2d to show the similarities, or maybe an exact comparison of matrixelements
npzwrite("data/ideal_vs_ideal_full_iref$k.npy", Dict("U_fl_ideal" => U_fl_ideal,"U_fl" => U_fl,"U_fl_ideal_2d" => U_fl_ideal_2d," U_fl_2d" => U_fl_2d))


######################################
### Plot 4: evolution full vs ideal AOD (fl+proj)
# should show how each diagonal compares
######################################
begin
    Nj = 6
    Nn = length(ns)

    data = zeros(ComplexF64,  3, Nn+1, p.L, p.L)
    
    for n in ns 
        data[1,n+1,:,:] .= U_fl_ideal^n * ρ0 * (U_fl_ideal')^n
        data[2,n+1,:,:] .= U_fl_AOD_ideal^n * ρ0 * (U_fl_AOD_ideal')^n
        data[3,n+1,:,:] .= U_fl_AOD^n * ρ0 * (U_fl_AOD')^n
        
    end

    npzwrite("data/ideal_vs_AOD_sd_iref$k.npy", data)

end

begin
    Nj = 6
    Nn = length(ns)

    data = zeros(ComplexF64,  3, Nn+1, p.L, p.L)
    
    for n in ns 
        data[1,n+1,:,:] .= U_fl_ideal_2d^n * ρ0 * (U_fl_ideal_2d')^n
        data[2,n+1,:,:] .= U_fl_AOD_ideal_2d^n * ρ0 * (U_fl_AOD_ideal_2d')^n
        data[3,n+1,:,:] .= U_fl_AOD_2d^n * ρ0 * (U_fl_AOD_2d')^n
        
    end

    npzwrite("data/ideal_vs_AOD_2d_iref$k.npy", data)

end





##### OLD STUFF, NOT IMPORTANT AND DONT PLOT! ####











errs_abs = [
    (abs.(
        U_fl_AOD^n * ρ0 * (U_fl_AOD')^n) .-
    abs.(    U_fl_ideal^n * ρ0 * (U_fl_ideal')^n
         )) ./ 0.1
    for n in ns
]


anim = @animate for (n, E) in enumerate(errs_abs)

    heatmap(
        abs.(E .+ 1e-12),
        c = :viridis,
        clims=(0,0.2),
        xlabel = "j",
        ylabel = "i",
        title = "relative error, n = $(n-1)",
        colorbar_title = "err"
    )

end

gif(anim, "relative_error.gif", fps = 10)


ρ_diag_exact = [
    real.(diag(U_fl_ideal^n * ρ0 * (U_fl_ideal')^n, 0))
    for n in ns
]

ρ_diag_AOD = [
    real.(diag(U_fl_AOD^n * ρ0 * (U_fl_AOD')^n, 0))
    for n in ns
]
N = 18
ρ___ = U_fl_AOD^N * ρ0 * (U_fl_AOD')^N
npzwrite("data/systematic_error_state.npy", ρ___)
display(heatmap(abs.(ρ___)))

anim = @animate for (idx,n) in enumerate(ns)

    plot(
        1:p.L,
        ρ_diag_exact[idx],
        lw=3,
        label="Ideal",
        xlabel="site",
        ylabel="population",
        ylim=(-maximum(ρ_diag), maximum(ρ_diag)),
        title="Diagonal populations, n = $n"
    )

    plot!(
        1:p.L,
        ρ_diag_AOD[idx],
        lw=3,
        ls=:dash,
        label="Ideal_AOD"
    )

end

gif(anim, "rho_diag_evolution_.gif", fps=5)

begin
using Plots

pl_ = plot(
    xlabel = "Number of driving periods ",
    ylabel = "Mean relative error",
    title = "Relative error per density-matrix diagonal",
    lw = 3,
    legend = :topleft,
    framestyle = :box,
    grid = true,
    dpi = 300,
    size = (700,600),
    guidefontsize = 14,
    tickfontsize = 12,
    legendfontsize = 11,
    titlefontsize = 16,
)

for j in 0:5

    Δρs = [
        diag(
            abs.(U_fl_AOD^n * ρ0 * (U_fl_AOD')^n -
                 U_fl_ideal^n * ρ0 * (U_fl_ideal')^n),
            j
        )
        for n in ns
    ]

    ρs_ideal = [
        diag(
            abs.(U_fl_ideal^n * ρ0 * (U_fl_ideal')^n),
            j
        )
        for n in ns
    ]

    typical_rel_err = [
        mean(Δρ) / mean(ρ)
        for (Δρ, ρ) in zip(Δρs, ρs_ideal)
    ]

    plot!(
        pl_,
        ns,
        typical_rel_err,
        lw = 3,
        label = j == 0 ? "Main diagonal" : "Off-diagonal $(j)"
    )
end

display(pl_)
end

begin
using Plots

# Colorblind-safe, perceptually distinct palette (Okabe-Ito)
okabe_ito = [
    RGB(0.90, 0.60, 0.00),  # orange
    RGB(0.35, 0.70, 0.90),  # sky blue
    RGB(0.00, 0.60, 0.50),  # bluish green
    RGB(0.95, 0.90, 0.25),  # yellow (avoid if on white bg with low contrast — swapped below)
    RGB(0.00, 0.45, 0.70),  # blue
    RGB(0.80, 0.40, 0.00),  # vermillion
]
# main diagonal first, then off-diagonals — reorder for good contrast, drop the weak yellow
palette_ = [okabe_ito[5], okabe_ito[1], okabe_ito[3], okabe_ito[6], RGB(0.5,0.3,0.7), RGB(0.3,0.3,0.3)]

common = (
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    guidefontsize = 13,
    tickfontsize = 11,
    legendfontsize = 9,
)

pl_rel = plot(;
    ylabel = "Mean relative error",
    legend = :outertopright,
    common...,
)

pl_abs = plot(;
    xlabel = "Number of driving periods",
    ylabel = "Mean absolute error",
    legend = false,
    common...,
)

for (idx, j) in enumerate(0:5)

    Δρs = [
        diag(
            abs.(U_fl_AOD^n * ρ0 * (U_fl_AOD')^n -
                 U_fl_ideal^n * ρ0 * (U_fl_ideal')^n),
            j
        )
        for n in ns
    ]

    ρs_ideal = [
        diag(abs.(U_fl_ideal^n * ρ0 * (U_fl_ideal')^n), j)
        for n in ns
    ]

    mean_abs_err = [mean(Δρ) for Δρ in Δρs]
    typical_rel_err = [mean(Δρ) / maximum(ρ) for (Δρ, ρ) in zip(Δρs, ρs_ideal)]

    label_ = idx == 1 ? "Main diagonal" : "Off-diagonal $(j)"
    c = palette_[idx]

    plot!(pl_rel, ns, typical_rel_err, lw = 2.5, color = c, label = label_)
    plot!(pl_abs, ns, mean_abs_err,    lw = 2.5, color = c, label = label_)
end

pl_ = plot(
    pl_rel, pl_abs,
    layout = @layout([a; b]),
    size = (650, 780),
    dpi = 300,
    link = :x,
    plot_title = "Mean errors per density-matrix diagonal",
    plot_titlefontsize = 15,
    left_margin = 6Plots.mm,
    right_margin = 10Plots.mm,   # room for outer legend
    top_margin = 4Plots.mm,
    bottom_margin = 4Plots.mm,
)

# panel labels (a)/(b), positioned in axis-relative coordinates
annotate!(pl_[1], (0.03, 0.93), text("(a)", 13, :left, :bold))
annotate!(pl_[2], (0.03, 0.93), text("(b)", 13, :left, :bold))

display(pl_)
#savefig(pl_, "error_per_diagonal.pdf")
end
println("||ρ0||_F = ", norm(ρ0))

H_AOD = log(U_fl_AOD)


ϕ_ideal = sort(angle.(eigvals(U_fl_ideal)))
ϕ_num   = sort(angle.(eigvals(U_fl)))
ϕ_AOD   = sort(angle.(eigvals(U_fl_AOD)))

plot(
    scatter(ϕ_num .- ϕ_ideal,
        xlabel="index",
        ylabel="Δ phase",
        title="Numerical vs ideal"),
    scatter(ϕ_AOD .- ϕ_ideal,
        xlabel="index",
        ylabel="Δ phase",
        title="AOD vs ideal"),
    layout=(1,2),
    size=(900,400)
)

using LinearAlgebra, Plots

Δ_num = abs.(U_fl .- U_fl_ideal) 
Δ_AOD = abs.(U_fl_AOD .- U_fl_ideal) 

E_ideal = eigen(U_fl_ideal)
E_num   = eigen(U_fl)
E_AOD   = eigen(U_fl_AOD)

# Floquet mode overlaps
O_num = abs.(E_ideal.vectors' * E_num.vectors).^2
O_AOD = abs.(E_ideal.vectors' * E_AOD.vectors).^2

plot(
    heatmap(Δ_num,
        title="|U_fl - U_ideal|",
        aspect_ratio=:equal),

    heatmap(Δ_AOD,
        title="|U_AOD - U_ideal|",
        aspect_ratio=:equal),

    heatmap(O_num,
        title="|V₀ᵀV_num|²",
        aspect_ratio=:equal,
        clim=(0,1)),

    heatmap(O_AOD,
        title="|V₀ᵀV_AOD|²",
        aspect_ratio=:equal,
        clim=(0,1)),

    layout=(2,2),
    size=(900,800)
)


using Plots
U_fl
abs.(log(U_fl_ideal) ./ 2π .* -1im)
H_fl_ideal
(-1im / 2π) .*  log(U_fl)
Δ = U_fl_AOD - U_fl_ideal
err = abs.(Δ) -abs.(Diagonal(Δ))

p1 = heatmap(
    err ,
    aspect_ratio = 1,
    color = :viridis,
    title = L"\log_{10}|U_{mn}^{AOD}-U_{mn}^{ideal}|",
    xlabel = L"m",
    ylabel = L"n",
    colorbar_title = L"\log_{10}|ΔU|"
)








# experimental parameters
begin 
p = make_params(;κ=1, t_e_exp=0.08, L = 50, T_exp=150e-9, f_t_symbol=:smooth)
xs = LinRange(-20,20,2000)
f(x) = theta(x,0,20)
p.κ
H_ho = HO_matrix(p.L)
V = p.κ * drive_elements(xs, f, p.L)

H_walls = drive_elements(xs, x->20 * walls(x, 3, 11), p.L)

H = PeriodicFockMatrix([H_ho, V], [triv,p.f_t], 2π)
H_f = Fourier_op(H)
U_T = []
tols =  [ 1e-1, 5e-2 ]
for tol in tols
    U,_ = compute_Floquet(H_f, 0., 2; tol=tol)
    push!(U_T, U)
end
U_T
Id = Matrix(I, size(first(U_T)))
for i in 2:length(U_T)
    println("Norms comparing tolerances $(tols[i-1]) and $(tols[i])")
    println("Unitary Norm: ", norm(U_T[i-1]*U_T[i]' - Id, Inf))
    println("diff norm: " , norm(U_T[i]-U_T[i-1], Inf), "\n")
end
U_fl = U_T[end]
U_fl * U_fl'

es, vs = eigen(U_fl)
q_es = log.(es) / 1im .+ π

#plot(diag(abs.(U_fl), 1))
heatmap(abs.(U_fl))
end

