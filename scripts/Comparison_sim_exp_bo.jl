using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using NPZ 

AOD_file = "data/pattern_single_drive.npy"
frames = npzread(AOD_file)
timesteps_per_drive = size(frames)[1]

begin
    p = make_params(; κ=iref_to_k(2e-5), L=70, T_exp=200e-9, t_e_exp= 0.1,)
    xs_μm =  LinRange(-10, 10, 5000)
    xs = xs_μm .*1e-6 ./p.lₕ 
    
    ps = LinRange(-20, 20, 10000)

    hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

    t_tof = 0.012 * p.ω
    Δt = 0.1e-3 * p.ω
    AOD_res = 38.596 *1e-6 * p.ω

    pixel_width = 5.6e-6 ./ p.lₕ
    σ = 6e-6 ./ p.lₕ

    drive_width = 20

    harmonics_phases_amps = [(1, 0., 1.)]

end

# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,drive_width)
    const V_ds =  [drive_elements(xs,  p.κ .* frames[nt, :], p.L) for nt in 1:timesteps_per_drive]
    drive(t) = V_ds[Int(mod(div(t, AOD_res)+1, timesteps_per_drive)+1)]
    gif_pattern = false 
    if gif_pattern 
        anim = @animate for nt in 1:timesteps_per_drive
            plot(xs_μm, p.κ .* frames[nt, :], ylims=(0,p.κ .* 2.5), framestyle=:box, xlabel="x (μm)", ylabel = "magnitude", label="AOD", linewidth=2, color=:red)
        end

        gif(anim, "animation_AOD_drive.gif", fps=30)
    end
end


begin
    V_d_approx = p.κ .* drive_elements(xs, step, p.L)
    H_fl = RWA_FL_Hamiltonian(V_d_approx, p, harmonics_phases_amps, :smooth)
    V_fl = V_AOD_FL(drive, p)
    H_AOD_fl = HO_matrix(p.L, p.Δ / p.ω) + V_fl
end


begin
    ρ_diag = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/initial_state_diag.npy")
    ρ = zeros(ComplexF64, p.L, p.L)
    [ρ[i,i] = ρ_diag[i] for i in 1:p.L ]
    ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)
end

# run bloch oscillations 
begin 
    
    ops_ts = (t-> H_ho, drive)

    t1  = time()
    sol = Time_Evolution_TDM_VN(copy(ρ), ops_ts, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    ops_ts_fl = (t->H_AOD_fl,)
    
    t1  = time()
    
    sol_fl = Time_Evolution_TDM_VN(copy(ρ), ops_ts_fl, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

end

begin
    # Select the simulation states corresponding to tomography times once

    measurements_exp = npzread(
        "C:/dev/SyntheticDimensionsUtils/data/tomography_results/density_matrices_ls.npz"
    )

    ts    = measurements_exp["t"]
    ρs_exp = measurements_exp["rho"]

    idxs = [
        findmin(x -> abs(x * 1e3 / p.ω - t), sol.t)[2]
        for t in ts
    ]

    ρs_sim = [sol.u[idx] for idx in idxs]
end

_, ds = run_tomography(ρs_sim[3], p.L - 1, t_tof, Δt; diags=[0,1,2], noise_level=0.1)
ds = ds ./ norm.(ds)
plot(ds[1:4])



begin
t_hold = t_holds_ideal(1., 15)

id_rho = 2
j = 10
pl = plot(
    xlabel = "Position",
    ylabel = "Normalized density",
    title = "TOF densities at different hold times",
    linewidth =5,
    legend = :right,
    framestyle = :box,
    gridalpha = 0.2,

    size = (900, 500),
)

densities = []
for i in 1:30
_, ds = run_tomography(
    ρs_sim[id_rho], p.L - 1, t_tof, Δt;
    diags=[0],
    noise_level=0.1,
)

ds = ds ./ norm.(ds)


push!(densities, ds[j])

    plot!(pl, xs, ds[j],lw = 3, label = "", alpha=0.3)
end
_, ds = run_tomography(
    ρs_sim[id_rho], p.L - 1, t_tof, Δt;
    diags=[0],
    noise_level=0.,
)
ds = ds ./ norm.(ds)
plot!(pl,xs, ds[j], lw=2, alpha=1, color=:black)
plot!(pl, xs, sum(densities) / length(densities),lw=2, alpha=1, color=:green, linestyle=:dash)
display(pl)
end



t_holds_ideal(1.,15) ./ p.ω * 1e3
plot(abs.(build_F(ds,t_holds_ideal(1.,15), 4)))



ρs_sim_tomo = run_tomography(ρs_sim, p.L - 1, t_tof, Δt; diags=[0,1,2], noise_level=0.1, α=0.4)


for (idt, t) in enumerate(ts)

    ρ_ex = ρs_sim[idt]
    ρ_sim_tomo = ρs_sim_tomo[idt]
    ρ_exp = ρs_exp[idt, :, :]

    pl = plot(
        layout = (3, 1),
        size = (900, 700),
        legend = :topright,
        title = "Density matrix comparison at t = $(round(t, digits=3))",
    )

    for k in 0:2

        d_ex = abs.(diag(ρ_ex, k))
        d_tomo = abs.(diag(ρ_sim_tomo, k))
        d_exp = abs.(diag(ρ_exp, k))

        plot!(
            pl[k + 1],
            d_ex,
            lw = 3,
            label = "Exact",
            color = :black,
        )

        plot!(
            pl[k + 1],
            d_tomo,
            lw = 2,
            ls = :dot,
            marker = :square,
            label = "Sim Tomo",
            color = :red,
        )

        plot!(
            pl[k + 1],
            d_exp,
            lw = 2,
            ls = :dash,
            marker = :circle,
            label = "Exp Tomo",
            color = :blue,
        )

        ylabel!(pl[k + 1], "Diagonal k = $k")
        xlabel!(pl[k + 1], "Index")
    end

    display(pl)
end


begin

Exp_tomo_data = "C:/dev/Python/SyntheticDimensionsUtils-main/data/tomography_results/density_matrices_ls.npz"
measurements_exp = npzread(Exp_tomo_data)

ts    = measurements_exp["t"]
ρs_exp = measurements_exp["rho"]

Nt = length(ts)
N  = size(ρs_exp, 2)

ρ_exact_all = Array{ComplexF64}(undef, Nt, N, N)

for (idt, t) in enumerate(ts)

    ρ_exact_all[idt, :, :] .= ρs_sim[idt]

end

npzwrite(
    "density_matrices_comparison.npz",
    Dict(
        "t" => ts,
        "rho_sim_ex" => ρ_exact_all,
        "rho_sim_tomo" => _,
        "rho_exp_tomo" => ρs_exp
    )
)
end