using Revise
using LinearAlgebra, Plots
using ExpSynthDim



# set parameters 
# physical 
# experimental params
begin
    p = make_params(; κ=0.5, L=50, T_exp=200e-9)
    xs = LinRange(-500, 500, 40000)
    ps = LinRange(-500, 500, 40000)

    hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

    t_tof = 0.012 * p.ω
    Δt = 0.1e-3 * p.ω


    pixel_width = 5.6e-6 ./ p.lₕ
    σ = 6e-6 ./ p.lₕ

    drive_width = 20

    harmonics_phases_amplitudes = [(1, 0., 1.)]

end



# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,drive_width)
    V_d = drive_elements(xs, step, p.L)
    H_fl = RWA_FL_Hamiltonian(V_d, p, harmonics_phases_amplitudes)

    ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)

end


# run bloch oscillations 
begin 
    drive(t) = p.κ * sum([k * p.f_t(harm * t + ϕ) for (harm, ϕ, k) in harmonics_phases_amplitudes])
    ops_fs = (t->H_ho, t -> drive(t) * V_d)
   
    t1  = time()
    sol = Time_Evolution_TDM_VN(copy(ρ), ops_fs, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    ops = (H_fl,)
    f_ts = (_-> 1,)

    t1  = time()
    sol = Time_Evolution_TDM_VN(copy(ρ), ops_fs, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10)
    t2  = time()
    println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")


end

total_simsteps = length(sol.u)
idx = round.(Int, range(1, total_simsteps, shots))
ρ_shots = sol.u[idx]
t_shots = sol.t[idx]

N_holds = 3:15
shots = 10
diags = [0, 1]

data_N_hold = Array{Any}(undef, length(N_holds))

for (id_N_hold, N) in enumerate(N_holds)

    tstart = time()

    t_ideal = N < 23 ?
        t_holds_ideal(1.0, N) :
        t_holds_ideal(0.5, N)

    t_quant = quantize_times(t_ideal, Δt)

    # Run tomography on every shot
    ρs_tomo = run_tomography(
        ρ_shots,
        p.L-1,
        t_tof,
        Δt;
        diags = diags,
        α = 1e-2,
        σ_ker = 12.0,
        σ_meas = σ,
        noise_level = 0.0,
        N_holds = N,
        ω = 1.0,
    )

    # Compare reconstruction with ground truth
    shot_errors = Vector{Vector{Float64}}()

    for (ρ_shot, ρ_tomo) in zip(ρ_shots, ρs_tomo)

        diag_errors = Float64[]

        for k in diags

            measured_diag = ρ_tomo[diagind(ρ_tomo, k)]
            true_diag = ρ_shot[diagind(ρ_shot, k)]

            true_diag = true_diag[1:length(measured_diag)]

            norm_true = norm(true_diag)

            err_k = norm_true > 0 ?
                norm(true_diag - measured_diag) / norm_true :
                norm(measured_diag)

            push!(diag_errors, err_k)
        end

        push!(shot_errors, diag_errors)
    end

    mean_error = zeros(length(diags))
    sigma_error = zeros(length(diags))

    for ik in eachindex(diags)

        vals = [e[ik] for e in shot_errors]

        mean_error[ik] = sum(vals) / length(vals)

        sigma_error[ik] =
            sqrt(sum((vals .- mean_error[ik]).^2) / length(vals))
    end

    data_N_hold[id_N_hold] = [mean_error, sigma_error]

    println(
        "Finished N = $N in ",
        round(time() - tstart; digits=2),
        " s"
    )
end


begin
    mean_k0 = [d[1][1] for d in data_N_hold]
    mean_k1 = [d[1][2] for d in data_N_hold]

    err_k0  = [d[2][1] for d in data_N_hold]
    err_k1  = [d[2][2] for d in data_N_hold]

    x = N_holds

    default(
        lw = 2,                 # thicker lines
        ms = 5,                 # marker size
        markerstrokewidth = 1,  # clean marker
        framestyle = :box
    )

    # --- plot k = 0 ---
    p1 = plot(
        x, mean_k0,
        yerror = err_k0,
        marker = :circle,
        color = :blue,
        xlabel = "# hold times",
        ylabel = "ϵᵣₑₗ",
        label = "Diagonal = 0",
    );

    # --- plot k = 1 ---
    p2 = plot(
        x, mean_k1,
        yerror = err_k1,
        marker = :circle,
        color = :red,
        xlabel = "# hold times",
        ylabel = "ϵᵣₑₗ",
        label = "Diagonal = 1",
    );

    display(plot(p1, p2, layout = (1,2), size=(800,300), dpi=300,bottom_margin = 6Plots.mm, left_margin=3Plots.mm))
end

