using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using Statistics
using NPZ 


AOD_file="data/pattern_single_drive.npy"
frames = npzread(AOD_file)
timesteps_per_drive = size(frames)[1]


# Experimental and physical parameters
begin
    p = make_params(; κ=iref_to_k(3e-5), L=70, T_exp=250e-9, t_e_exp= 0.1,)
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
    #ρ = npzread("./data/initial_state.npy")[1:p.L,1:p.L]
    ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)
end

H_fl
V_fl
# run bloch oscillations 
begin 
    
    ops_ts = (t-> H_ho, drive)

    t1  = time()
    sol = Time_Evolution_TDM_VN(copy(ρ), ops_ts, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    ops_ts_fl = (t->H_fl,)
    
    t1  = time()
    
    sol_fl = Time_Evolution_TDM_VN(copy(ρ), ops_ts_fl, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

end

function plot_stroboscopic_diag_errors(sol, sol_FL, p; n_diags=5)
    T = 2π / p.ω_d
    
    tsexact = sol.t
    tsfl    = sol_FL.t

    n = norm(sol.u[1])

    nmax = floor(Int, minimum((maximum(tsexact), maximum(tsfl))) / T)

    indsexact = [argmin(abs.(tsexact .- n*T)) for n in 0:nmax]
    indsfl    = [argmin(abs.(tsfl    .- n*T)) for n in 0:nmax]

    pl = plot(
        xlabel="Periods",
        ylabel=L"|\rho_{\rm exact}-\rho_{\rm Floquet}|_F \, / \, |\rho_{\rm exact}|_F",
        title="Stroboscopic error per diagonal",
        framestyle=:box,
        legend=:topleft,
    )

    for k in 0:n_diags-1
        errs = [
            norm(diag(sol.u[ie], k) .- diag(sol_FL.u[ifl], k)) / max(norm(diag(sol_FL.u[ie], k)), 1e-8)
            for (ie, ifl) in zip(indsexact, indsfl)
        ]
        plot!(pl, 0:nmax, errs, lw=2, marker=:circle, label="k=$k")
    end

    return pl
end

plot_stroboscopic_diag_errors(sol, sol_fl, p; n_diags=1)

plot_center_of_mass_λ(sol, sol_fl, p)
gif_ρ_diag(sol,sol_fl,p, 3)
gif_ρ_diff(sol,sol_fl,p; dt = (2π/p.ω_d) )

data_exp = [
    2.2   11.858909037431372  0.7217373168802633;
    13.2  14.340743794888574  1.810665402895463;
    24.2  18.356888596357454  1.3361105414420893;
    35.0  20.003817015854096  0.9574618297611185;
    41.8  18.953989614325458  2.6321437735684627;
    48.4  16.646149972042068  0.862230455726438;
    55.0  14.562037035027348  0.5644882032223157;
    61.6  12.198346534580427  1.877113564436251;
    63.8  11.719942854180898  1.7653411878428773;
    66.0  11.397837224799222  1.3401859284698108;
    68.2  11.9636451046607    0.5268170105861274;
    74.8  12.665404596671486  0.2688325507588324;
    81.4  13.988014061497092  0.8502914972887096;
    88.0  16.429113423284946  1.0061751734222992;
    94.6  17.23315515282696   1.442540122611133
]

begin
    pl = plot_center_of_mass_λ(sol, sol_fl, p)
    plot!(pl, data_exp[:,1], data_exp[:,2], yerror=data_exp[:,3],color=:blue,  label="Experiment", marker=:o, markerstrokewidth=1)
end

data_num_ex = center_of_mass_evolution(sol, p)
data_num_eff = center_of_mass_evolution(sol_fl, p)
data_num_ex[:,1] .*= 1e3 / p.ω
data_num_eff[:,1] .*= 1e3 / p.ω




npzwrite("results_bo_comparison.npz", Dict(
    "num_ex" => data_num_ex,
    "num_eff" => data_num_eff,
    "exp" => data_exp
))


begin
    t_targets =[2.2, 11., 19.8, 30.8, 39.6, 50.6, 59.4, 68.2]
    t_all_ms = sol.t ./ p.ω .* 1e3
    indices = [findmin(abs.(t_all_ms .- t))[2] for t in t_targets]

    ncols = length(indices)

    # create 2 x N grid
    plot_grid = Array{Any}(undef, ncols, 2)

    for (k, i) in enumerate(indices)
        t = sol.t[i]
        ρ__ = sol.u[i]

        dens = real(diag(ρ__))
        off = real(ρ__[diagind(ρ__, 1)])

        Js = local_currents(ρ__, H_AOD_fl)
        curr = real.(Js[diagind(Js, 1)])

        title_str = "t_drive = $(round(t_targets[k], digits=1)) ms"

        # top: density
        plot_grid[ k,1] = bar(dens[1:40],
            title=title_str,
            legend=false,
            color=:blue,
            ylims=(0, 0.08),
            dpi=300)

        # bottom: current
        plot_grid[k,2] = bar(curr[1:40],
            xlabel="State index n",
            legend=false,
            color=:orange,
            ylims=(-0.02,0.02),
            dpi=300)
    end

    plot(plot_grid..., layout=(2, ncols), size=(300*ncols, 500), dpi=500)
    savefig("bo_tomography.png")
end

