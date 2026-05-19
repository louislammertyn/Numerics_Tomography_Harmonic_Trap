using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using Statistics
using OrdinaryDiffEq
using FoSpDynamics


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

    harmonics_and_phases = [(1, 0.)]

end



# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,drive_width)
    V_d = p.κ * drive_elements(xs, step, p.L)
    H_fl = RWA_FL_Hamiltonian(V_d, p, harmonics_and_phases)

    ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)

end


# run bloch oscillations 
begin 
    drive(t) = sum([p.f_t(harm * t + ϕ) for (harm, ϕ) in harmonics_and_phases])
    ops = (H_ho, V_d)
    f_ts = (_-> 1, drive)

    t1  = time()
    sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10, solver=Vern7())
    t2  = time()
    println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    ops = (H_fl,)
    f_ts = (_-> 1,)

    t1  = time()
    sol_fl = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                        rtol=1e-10, atol=1e-10, solver=Vern7())
    t2  = time()
    println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")


end
sol.t ./ p.ω

#gif_ρ_diag(sol,sol_fl, p)
# process data 
total_simsteps = length(sol.u)
idx = round.(Int, range(1, total_simsteps, shots))
ρ_shots = sol.u[idx]
t_shots = sol.t[idx]

N =30
ρ_ = ρ_shots[2]

t_ideal = N < 23 ? t_holds_ideal(1., N) : t_holds_ideal(1. / 2, N)
t_quant = quantize_times(t_ideal, Δt)

densities_tof = get_n_tof(ρ_, t_quant, t_tof, xs, hs)
plot(xs, densities_tof)
densities_convolved = [convolve_gaussian(densities_tof[i], xs, σ) for i in 1:N]
plot(xs, densities_convolved)
densities_pixelated = [bin_pixels(densities_convolved[i], xs, pixel_width)[1] for i in 1:N]
xs_pixelated = bin_pixels(densities_convolved[1], xs, pixel_width)[2]
#plot(xs_pixelated, densities_pixelated)

# do tomography on data
k=1
measured_diag = reconstruct_density_matrix_slice(xs, t_tof, densities_tof, t_quant, k; λmax = 100)[1:p.L - k]

plot(xs_pixelated, densities_pixelated)
plot(abs.(measured_diag))


measured_diag 
ρ_[diagind(ρ, k)]
plot((abs.(measured_diag - ρ_[diagind(ρ, k)] )./ abs.(ρ_[diagind(ρ, k)]))[1:end-2])


N_holds = collect(5:1:20)
shots = 10
diags = [0,1]

data_N_hold = Array{AbstractVector}(undef, size(N_holds))

Threads.@threads for id_N_hold in eachindex(N_holds)
    tstart = time()
    N = N_holds[id_N_hold]

    t_ideal = N < 23 ? t_holds_ideal(1., N) : t_holds_ideal(1. / 2, N)
    t_quant = quantize_times(t_ideal, Δt)

    # store one error vector per shot (indexed by k)
    shot_errors = Vector{Vector{Float64}}()

    for (_, ρ_shot) in zip(t_shots, ρ_shots)

        densities_tof = get_n_tof(ρ_shot, t_quant, t_tof, xs, hs)
        densities_convolved = [convolve_gaussian(densities_tof[i], xs, σ) for i in 1:N]
        densities_pixelated = [bin_pixels(densities_convolved[i], xs, pixel_width)[1] for i in 1:N]
        xs_pixelated = bin_pixels(densities_convolved[1], xs, pixel_width)[2]

        xs_ = xs_pixelated
        densities_ = densities_pixelated
        ρ_m = zeros(ComplexF64, size(ρ))

        # collect errors per diagonal for this shot
        diag_errors = Float64[]

        for k in diags
            measured_diag = reconstruct_density_matrix_slice(
                xs_, t_tof, densities_, t_quant, k; λmax=100
            )[1:p.L - k]

            ρ_m[diagind(ρ_m, k)] = measured_diag

            if k > 0 
                ρ_m[diagind(ρ_m, -k)] = conj.(ρ_m[diagind(ρ_m, k)])
            end


            true_diag = ρ_shot[diagind(ρ_shot, k)][1:length(measured_diag)]
           
            norm_true = norm(true_diag)
            err_k = norm_true > 0 ? norm(true_diag - measured_diag) / norm_true : norm(measured_diag) 

            push!(diag_errors, err_k)

            all(abs.(measured_diag) .< 1e-7) && break
        end

        push!(shot_errors, diag_errors)
    end

    mean_error = zeros(Float64, length(diags) )
    sigma_error = zeros(Float64, length(diags))

        
    for (ik, k) in enumerate(diags)
        vals = [e[ik] for e in shot_errors if length(e) ≥ ik]

        μ = sum(vals) / length(vals)
        σ = sqrt(sum((vals .- μ).^2) / length(vals))

        mean_error[ik] = μ
        sigma_error[ik] = σ
    end


    data_N_hold[id_N_hold] = [mean_error, sigma_error]

    println("Finished N = $N in $(round(time()-tstart;digits=2)) s")
end

data_N_hold


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







# plot results