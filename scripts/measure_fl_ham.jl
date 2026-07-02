using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using NPZ 

frames = npzread("C:/Users/loulamme/OneDrive - UGent/Documenten/Assistent ELIS/Phd Research/projects/Exp_SynthDim_HarmTrap/code/python_files/patterns.npy")
timesteps_per_drive = size(frames)[1]

begin
    p = make_params(; κ=.7, L=60, T_exp=250e-9, t_e_exp= 0.1,)
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

    harmonics_and_phases = [(1, 0.)]

end

# plot of some drive patterns
begin 
    pl = plot(framestyle=:box, xlabel="x (μm)",ylabel="magnitude")
    for h in 1:60
        lbl =  h ==1 ? "60 Hermite modes" : ""
        plot!(pl, xs_μm, hermite.(h, xs), label=lbl, color=:green)
    end
    plot!(pl, xs_μm,  p.κ .* frames[40,:], color=:blue, linewidth=2, label="AOD drive")
    plot!(pl, xs_μm,  p.κ .* frames[1,:], color=:blue, linewidth=2, label="")
    plot!(pl, xs_μm,  p.κ .* frames[22,:], color=:blue, linewidth=2, label="")
    display(pl)
end

# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,drive_width)
    const V_ds =  [drive_elements(xs,  p.κ .* frames[nt, :], p.L) for nt in 1:timesteps_per_drive]
    drive(t) = V_ds[Int(mod(div(t, AOD_res)+1, timesteps_per_drive)+1)]
    anim = @animate for nt in 1:timesteps_per_drive
        plot(xs_μm, p.κ .* frames[nt, :], ylims=(0,p.κ .* 2.5), framestyle=:box, xlabel="x (μm)", ylabel = "magnitude", label="AOD", linewidth=2, color=:red)
    end

    gif(anim, "animation_AOD_drive.gif", fps=30)
end
begin
    V_d_approx = p.κ .* drive_elements(xs, step, p.L)
    H_fl = RWA_FL_Hamiltonian(V_d_approx, p, harmonics_and_phases, :smooth) - Diagonal((0.5) * ones(p.L))
    
    V_fl = V_AOD_FL(drive, p)
    V_fl -= Diagonal(V_fl)
    H_AOD_fl = HO_matrix(p.L, p.Δ / p.ω) + V_fl - Diagonal(0.5 * ones(p.L))
    force_fl = HO_matrix(p.L, p.Δ / p.ω)- Diagonal(0.5 * ones(p.L))
end
V_fl
force_fl
plot([norm(drive(t)) for t in LinRange(0,2π,1000)])
begin
measurements = npzread("data/density_matrices.npz")
ts = measurements["t"]


# Preallocate
n = length(ts)
val_H = zeros(n)
val_F = zeros(n)
val_V = zeros(n)
val_H_ideal = zeros(n)

# Collect observables
for (i, t) in enumerate(ts)
    ρ = Transpose(measurements["rho"][i, :, :])

    val_H[i] = real(tr(H_AOD_fl * ρ))
    val_F[i] = real(tr(force_fl * ρ))
    val_V[i] = real(tr(V_fl * ρ))
    val_H_ideal[i] = real(tr(H_fl * ρ))
end

# Plot
plot(ts, val_H,
    label = "⟨H_fl⟩",
    marker=:x,
    lw = 2,
    framestyle=:box)

plot!(ts, val_F,
    label = "⟨Detuning⟩",
    marker=:o,
    lw = 2)

plot!(ts, val_V,
    label = "⟨V_AOD⟩",
    marker=:diamond,
    lw = 2)

plot!(ts, val_H_ideal,
    label = "⟨H_fl⟩ (ideal)",
    linestyle = :dash,
    color = :black,
    marker=:utriangle,
    lw = 2)


xlabel!("t (ms)")
ylabel!("Expectation value (ħω)")
title!("Observables vs time")
end

measurements["rho"][3,:,:]