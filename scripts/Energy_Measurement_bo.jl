using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using NPZ 

# here the file name of the AOD pattern
file = "data/pattern_single_drive.npy"
frames = npzread(file)
timesteps_per_drive = size(frames)[1]
p.κ
begin
    p = make_params(; κ=iref_to_k(2.35e-5), L=60, t_e_exp= 0.1,)
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
    ρ_diag = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/initial_state_diag.npy")
    ρ = zeros(ComplexF64, p.L, p.L)
    [ρ[i,i] = ρ_diag[i] for i in 1:p.L ]

    #ρ = thermal_state(H_ho, p.β)
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
    measurements = npzread("C:/dev/SyntheticDimensionsUtils/data_for_num/exp_density_matrices.npy")

    ts = [11., 19.8, 30.8, 39.6, 50.6, 59.4]
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
        ρ = Transpose(measurements[i, 1:p.L, 1:p.L])

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
    plot!(
        plt, ts, val_H_exact;
        label = "⟨H_AOD⟩ (exact)",
        color = colors.H,
        lw = 3,
    )

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

i = 5
plot([real.(diag(sol_m[i],1)),real.(diag(measurements[i,:,:],1))])
print(tr(sol_m[i]))
print(tr(measurements["rho"][i,:,:]))