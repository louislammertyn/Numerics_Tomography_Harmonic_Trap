using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using NPZ 
using JLD2
using SHA

# here the file name of the AOD pattern
file = "data/pattern_single_drive.npy"
frames = npzread(file)
timesteps_per_drive = size(frames)[1]

begin
    p = make_params(; κ=iref_to_k(3.e-5), L=60, t_e_exp= 0.1,)
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

    harmonics_phases_amplitudes = [(1, 0., 1.)]

end
p.κ
# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,drive_width)
    const V_ds =  [drive_elements(xs,  p.κ .* frames[nt, :], p.L) for nt in 1:timesteps_per_drive]
    drive(t) = V_ds[Int(mod(div(t, AOD_res)+1, timesteps_per_drive)+1)]
    gif_pattern =false
    if gif_pattern 
        anim = @animate for nt in 1:timesteps_per_drive
            plot(xs_μm, p.κ .* frames[nt, :], ylims=(0,p.κ .* 2.5), framestyle=:box, xlabel="x (μm)", ylabel = "magnitude", label="AOD", linewidth=2, color=:red)
        end

        gif(anim, "animation_AOD_drive.gif", fps=30)
    end
end
p.κ
begin
    V_d_approx = p.κ .* drive_elements(xs, step, p.L)
    H_fl = RWA_FL_Hamiltonian(V_d_approx, p, harmonics_phases_amplitudes, :smooth) - Diagonal((0.5) * ones(p.L))
    V_fl = V_AOD_FL(drive, p)
    V_fl -= Diagonal(V_fl)
    force_fl = HO_matrix(p.L, p.Δ / p.ω)- Diagonal(0.5 * ones(p.L))
    H_AOD_fl = force_fl + V_fl
end   


begin
    ρ_diag = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/initial_state.npz")
    ρ = zeros(ComplexF64, p.L, p.L)
    [ρ[i,i] = ρ_diag["initial"][i] for i in 1:p.L ]

    #ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)
end


begin 
fname = "data/sims/sim_$(bytes2hex(sha1(repr(p)))).jld2"

if isfile(fname)

    @load fname sol sol_fl

else

    ops_ts = (t -> H_ho, drive)

    sol = Time_Evolution_TDM_VN(
        copy(ρ), ops_ts, (p.t_i, p.t_e), p.save_ts;
        rtol=1e-10, atol=1e-10
    )

    ops_ts_fl = (t -> H_AOD_fl,)

    sol_fl = Time_Evolution_TDM_VN(
        copy(ρ), ops_ts_fl, (p.t_i, p.t_e), p.save_ts;
        rtol=1e-10, atol=1e-10
    )

    @save fname sol sol_fl

end
end


begin
using NPZ

data = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/tomography_stats.npz")

tdrives = sort(unique([
    parse(Int, m.captures[1])
    for k in keys(data)
    for m in [match(r"^mean_tdrive_(\d+)_j_(\d+)$", k)]
    if !isnothing(m)
]))

tmap = Dict(t => i for (i, t) in enumerate(tdrives))

n = length(data["mean_tdrive_0_j_0"])

ρ_m = zeros(ComplexF64, length(tdrives), n, n)

for key in keys(data)

    m = match(r"^mean_tdrive_(\d+)_j_(\d+)$", key)
    isnothing(m) && continue

    t = parse(Int, m.captures[1])
    j = parse(Int, m.captures[2])

    tidx = tmap[t]

    diagvals = ComplexF64.(data[key])

    for k in 1:length(diagvals)
        ρ_m[tidx, k, k+j] = diagvals[k]
    end

    if j > 0
        for k in 1:length(diagvals)
            ρ_m[tidx, k+j, k] = conj(diagvals[k])
        end
    end
end
ρ_init_diag = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/initial_state.npz")
ρ_init = zeros(ComplexF64, p.L, p.L)
[ρ_init[i,i] = ρ_diag["initial"][i] for i in 1:p.L ]
end

begin
    #measurements = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/exp_density_matrices.npy")

    ts = [0., 11., 19.8, 30.8, 39.6, 50.6, 59.4]
    println(ts)

    idxs = [findmin(abs.((t * p.ω * 1e-3) .- sol.t))[2] for t in ts]
    println(idxs)

    sol_m = sol.u[idxs]

    # ----------------------------
    # Experimental densities
    # ----------------------------
    n = length(ts)

    val_H = zeros(n)
    val_F = zeros(n)
    val_V = zeros(n)
    val_H_ideal = zeros(n)

    for (i, t) in enumerate(ts)
        if iszero(t)
            ρ = ρ_init
        else
            ρ = Transpose(ρ_m[i-1, 1:p.L, 1:p.L])
        end

        val_H[i]       = real(tr(H_AOD_fl * ρ))
        val_F[i]       = real(tr(force_fl * ρ))
        val_V[i]       = real(tr(V_fl * ρ))
        val_H_ideal[i] = real(tr(H_fl * ρ))
    end

    # ----------------------------
    # Exact TD simulation
    # ----------------------------
    val_H_exact = [
        real(tr(H_AOD_fl * ρ))
        for ρ in sol_m
    ]

    val_F_exact = [
        real(tr(force_fl * ρ))
        for ρ in sol_m
    ]

    val_V_exact = [
        real(tr(V_fl * ρ))
        for ρ in sol_m
    ]

    val_H_ideal_exact = [
        real(tr(H_fl * ρ))
        for ρ in sol_m
    ]

    # ----------------------------
    # Colors
    # ----------------------------
    colors = (
        H      = :dodgerblue3,
        F      = :darkorange,
        V      = :forestgreen,
        Hideal = :purple,
    )

    # ----------------------------
    # Plot
    # ----------------------------
    plt = plot(
        framestyle = :box,
        xlabel = "t (ms)",
        ylabel = "Expectation value (ħω)",
        title = "Observables vs time, k=$(round(p.κ; digits=2))",
        legend = :outertopright,
        size = (900, 550),
    )

    # ----------------------------
    # Experiment (markers)
    # ----------------------------
    scatter!(
        plt, ts, val_H;
        label = "⟨H_AOD⟩ (exp)",
        color = colors.H,
        marker = :circle,
        markersize = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
        
    )

    scatter!(
        plt, ts, val_F;
        label = "⟨Detuning⟩ (exp)",
        color = colors.F,
        marker = :square,
        markersize = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
       
    )

    scatter!(
        plt, ts, val_V;
        label = "⟨V_AOD⟩ (exp)",
        color = colors.V,
        marker = :diamond,
        markersize = 8,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
        
    )

    scatter!(
        plt, ts, val_H_ideal;
        label = "⟨H_ideal⟩ (exp)",
        color = colors.Hideal,
        marker = :utriangle,
        markersize = 9,
        markerstrokecolor = :black,
        markerstrokewidth = 1,
        )

    # ----------------------------
    # Exact TD simulation (lines)
    # ----------------------------
    # plot!(
    #     plt, ts, val_H_exact;
    #     label = "⟨H_AOD⟩ (exact)",
    #     color = colors.H,
    #     lw = 3,
    # )

    plot!(
        plt, ts, val_F_exact;
        label = "⟨Detuning⟩ (exact)",
        color = colors.F,
        lw = 3,
    )

    plot!(
        plt, ts, val_V_exact;
        label = "⟨V_AOD⟩ (exact)",
        color = colors.V,
        lw = 3,
    )

    plot!(
        plt, ts, val_H_ideal_exact;
        label = "⟨H_ideal⟩ (exact)",
        color = colors.Hideal,
        lw = 3,
    )

    plt
end
begin
    using NPZ

npzwrite(
    "C:/dev/SyntheticDimensionsUtils/data_for_num/observable_plot_data.npz",
    Dict(
        # x-axis
        "ts" => ts,

        # experiment
        "val_H_exp"       => val_H,
        "val_F_exp"       => val_F,
        "val_V_exp"       => val_V,
        "val_H_ideal_exp" => val_H_ideal,

        # exact theory
        "val_H_exact"       => val_H_exact,
        "val_F_exact"       => val_F_exact,
        "val_V_exact"       => val_V_exact,
        "val_H_ideal_exact" => val_H_ideal_exact,

        # optional metadata
        "kappa" => [p.κ],
    )
)
end

