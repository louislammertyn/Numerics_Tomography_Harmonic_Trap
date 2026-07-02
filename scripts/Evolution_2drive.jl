using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using OrdinaryDiffEq
using FoSpDynamics
using NPZ



frames = npzread("C:/Users/loulamme/OneDrive - UGent/Documenten/Assistent ELIS/Phd Research/projects/Exp_SynthDim_HarmTrap/code/data/pattern_2drive.npy")
timesteps_per_drive = size(frames)[1]
# Experimental and physical parameters
begin
    p = make_params(; κ=1., L=60, T_exp=250e-9, t_e_exp= 0.1,)
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

    harmonics_and_phases = [(1, 0., 1.), (3, 0., 3.)]

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
    H_fl = RWA_FL_Hamiltonian(V_d_approx, p, harmonics_and_phases, :smooth)
    V_fl = V_AOD_FL(drive, p)
    V_fl -= Diagonal(V_fl)
    H_AOD_fl = HO_matrix(p.L, p.Δ ./ p.ω) + V_fl
end

heatmap(abs.(H_fl) - abs.(H_AOD_fl))

begin
    ρ = npzread("./data/initial_state.npy")[1:p.L,1:p.L]
    #ρ = thermal_state(H_ho, p.β)
    #ρ = almost_thermal(ρ, 4., 2.)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)
    
end

# run bloch oscillations 

begin 
    
    ops_ts = (t-> H_ho, drive)

    t1  = time()
    sol = Time_Evolution_TDM_VN(copy(ρ), ops_ts, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10, solver=Tsit5())
    t2  = time()
    println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    
    ops = (H_fl,)
    f_ts = (_-> 1,)

    t1  = time()
    
    sol_fl = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10, solver=Tsit5())
    t2  = time()
    println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

end
sol.t / p.ω *1e3
abs.(sol.u[end])

gif_ρ(sol, p; Nt=4000, every=10)