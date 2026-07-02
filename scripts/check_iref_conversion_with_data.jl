using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using Statistics
using OrdinaryDiffEq
using FoSpDynamics
using NPZ 

frames = npzread("C:/Users/loulamme/OneDrive - UGent/Documenten/Assistent ELIS/Phd Research/projects/Exp_SynthDim_HarmTrap/code/python_files/patterns.npy")
timesteps_per_drive = size(frames)[1]



irefs = [
    1e-6,
    2.5e-6,
    5e-6,
    7.5e-6,
    1e-5,
    2e-5,
    3e-5,
    4e-5,
    5e-5
]
p = make_params()
1/p.lₕ
coms34 = Vector{Float64}(undef, length(irefs))
coms69 = Vector{Float64}(undef, length(irefs))

com_evs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, length(irefs))
Threads.@threads for i in eachindex(irefs)
    iref = irefs[i]
    # Experimental and physical parameters
    begin
        k = iref_to_k(iref)
        p = make_params(; κ=k, L=100, T_exp=250e-9, t_e_exp= 0.1,)
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


    # initial state and relevant matrices 
    begin 
        H_ho = HO_matrix(p.L)
        step(x) = theta(x,0,drive_width)
        V_ds =  [drive_elements(xs, p.κ .* frames[nt, :], p.L) for nt in 1:timesteps_per_drive]
        drive(t) =  V_ds[Int(mod(div(t, AOD_res)+1, timesteps_per_drive)+1)]

        V_d_approx = p.κ .* drive_elements(xs, step, p.L)
        H_fl_approx = RWA_FL_Hamiltonian(V_d_approx, p, harmonics_and_phases, :smooth)
        V_fl = V_AOD_FL(drive, p)
        H_fl = HO_matrix(p.L, p.Δ) + V_fl
    end

    begin
        ρ = npzread("./data/initial_state.npy")[1:p.L,1:p.L]
        #ρ = thermal_state(H_ho, p.β)
        #ρ = almost_thermal(ρ, 4., 2.)
        #plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)
    end

    # run bloch oscillations 


    begin 
        
        ops_ts = (t-> H_ho, drive)

        t1  = time()
        sol = Time_Evolution_TDM_VN(copy(ρ), ops_ts, (p.t_i, p.t_e), p.save_ts; rtol=1e-10, atol=1e-10, solver=Tsit5())
        t2  = time()
        println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

        # why does this not work with H_AOD_fl????
        ops = (H_fl,)
        t1  = time()
        
        #sol = Time_Evolution_VN(copy(ρ), H_fl, (p.t_i, p.t_e), p.save_ts; rtol=1e-10, atol=1e-10, solver=Tsit5())
        t2  = time()
        println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    end

    if true
        _, idt34 = findmin(ts -> abs2.(ts.- (34e-3 .* p.ω )), sol.t)
        _, idt69 = findmin(ts -> abs2.(ts.- (69e-3 .* p.ω )), sol.t)


        ρ34 = sol.u[idt34]
        ρ69 = sol.u[idt69]

        com34 = sum([ρ34[i,i] * (i-1) for i in 1:p.L])
        com69 = sum([ρ69[i,i] * (i-1) for i in 1:p.L])

        coms34[i]=  real(com34)
        coms69[i]= real(com69)
    end
    if false
        com_evs[i] = center_of_mass_evolution(sol, p)
    end
end

begin 
    colors = cgrad([:orange, :blue], length(irefs))
    pl = plot(framestyle=:box, grid=false,xlabel="t (ms)", ylabel="Δλ com", dpi=300 )
    for (i,ev) in enumerate(com_evs) 
        plot!(ev[1] ./ p.ω *1e3, ev[2], color = colors[i], label="$(irefs[i])")
    end
    display(pl)
end
num_diff = coms34 .- coms69
num_diff

exp_diff = [
    0.16596952934457398,
   -0.017057254722996795,
    1.6364314757469431,
    1.422611352891396,
    2.5077855912236444,
    6.637451830902053,
    10.56397186084059,
    13.357879165461169,
    13.26773099379951
]

# Standard deviations
exp_diff_std = [
    0.49223435835028584,
    0.44563643498530575,
    0.7094326405543542,
    0.6713724531356371,
    0.4446104216903167,
    1.1245044630813665,
    0.6170630946451269,
    0.859689644420663,
    2.244490790806235
]

begin
pl = plot(title="Bloch oscillation for AOD conversion", framestyle=:box, xlabel="iref ", ylabel="Δλ (at 34ms and 69ms)", dpi=300)
plot!(pl, irefs, num_diff, marker=:x, linewidth=2, label="Numerics", color=:blue)
plot!(pl, irefs, exp_diff, yerror=exp_diff_std, label="exp", marker=:o, markerstrokewidth=0, linewidth=2, linestyle=:dot, color=:red)
display(pl)
end

abs.(num_diff .- exp_diff)