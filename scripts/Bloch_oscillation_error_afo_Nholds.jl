using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using NPZ



# set parameters 
# physical 
# experimental params
AOD_file="data/pattern_single_drive.npy"
frames = npzread(AOD_file)
timesteps_per_drive = size(frames)[1]


# Experimental and physical parameters
begin
    k=2
    p = make_params(; κ=iref_to_k(k*1e-5), L=70, T_exp=250e-9, t_e_exp= 0.1,)
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
    ρ = npzread("./data/initial_state.npy")[1:p.L,1:p.L]
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

    ops_ts_fl = (t->H_fl,)
    
    t1  = time()
    
    sol_fl = Time_Evolution_TDM_VN(copy(ρ), ops_ts_fl, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

end

shots = 10
plot(abs.(sol.t .-60))
total_simsteps = length(sol.u)
 ts = [n * 2π/p.ω_d for n in [0,10,20, 30]]
idx = [findmin(abs.(sol.t .- t))[2] for t in ts]
ρ_shots = sol.u[idx]
t_shots = sol.t[idx]

# ── change the storage type ──────────────────────────────────────────────────
n_vals   = [4, 10, 20, 30]          # the n-labels for each shot
N_holds  = 3:30
diags    = [0, 1]

# data_N_hold[id_N] = (err_ideal, err_quant), each a (n_shots × n_diags) matrix
data_N_hold = Array{Tuple{Matrix{Float64}, Matrix{Float64}}}(undef, length(N_holds))

for (id_N_hold, N) in enumerate(N_holds)

    tstart = time()

    t_ideal = N < 23 ?
        t_holds_ideal(1.0, N) :
        t_holds_ideal(0.5, N)

    t_quant = quantize_times(t_ideal, Δt)

    function compute_errors(ρs_tomo)
        err_mat = zeros(length(ρ_shots), length(diags))
        for (i_shot, (ρ_shot, ρ_tomo)) in enumerate(zip(ρ_shots, ρs_tomo))
            for (ik, k) in enumerate(diags)
                measured = ρ_tomo[diagind(ρ_tomo, k)]
                truth    = ρ_shot[diagind(ρ_shot, k)][1:length(measured)]
                nrm = norm(truth)
                err_mat[i_shot, ik] = nrm > 0 ?
                    norm(truth - measured) / nrm :
                    norm(measured)
            end
        end
        return err_mat
    end

    kw = (;
        diags       = diags,
        α           = 1e-3,
        σ_ker       = σ,
        σ_meas      = σ,
        noise_level = 0.0,
        ω           = 1.0,
        N_holds=N
    )

    ρs_ideal = run_tomography(ρ_shots, p.L-1, t_tof, Δt; t_holds = t_ideal, kw...)
    ρs_quant = run_tomography(ρ_shots, p.L-1, t_tof, Δt; t_holds = t_quant, kw...)

    data_N_hold[id_N_hold] = (compute_errors(ρs_ideal), compute_errors(ρs_quant))

    println("Finished N = $N in ", round(time() - tstart; digits=2), " s")
end


# ── one subplot per shot, ideal vs quantized ─────────────────────────────────
begin
    default(lw = 2, ms = 4, markerstrokewidth = 1, framestyle = :box)

    # solid = ideal, dashed = quantized; blue = diag 0, red = diag 1
    colors = [:blue, :red]
    x      = collect(N_holds)

    subplots = map(enumerate(n_vals)) do (i_shot, n)

        p = plot(
            title  = "Shot  $n Td",
            xlabel = "# hold times",
            ylabel = "ϵ_rel",
            legend = :topright,
        )

        for (ik, k) in enumerate(diags)
            errs_ideal = [data_N_hold[id_N][1][i_shot, ik] for id_N in eachindex(N_holds)]
            errs_quant = [data_N_hold[id_N][2][i_shot, ik] for id_N in eachindex(N_holds)]

            plot!(p, x, errs_ideal;
                  marker    = :circle,
                  color     = colors[ik],
                  linestyle = :solid,
                  label     = "diag $k  (ideal)")

            plot!(p, x, errs_quant;
                  marker    = :circle,
                  color     = colors[ik],
                  linestyle = :dot,
                  label     = "diag $k  (quantized)")
        end

        p
    end

    display(plot(
        subplots...,
        layout        = (2, 2),
        size          = (900, 600),
        dpi           = 300,
        bottom_margin = 6Plots.mm,
        left_margin   = 4Plots.mm,
    ))
end