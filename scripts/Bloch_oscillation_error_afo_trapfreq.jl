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
    xs = LinRange(-800, 800, 20000)
    ps = LinRange(-800, 800, 20000)

    hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

    δHz = [-10, -5, -4, -3, -2,-1, 0, 1, 2, 3, 4, 5, 10] 
    ωs = (p.ω .+ δHz .* 2π) ./ p.ω

    t_tof = 0.018 * p.ω
    Δt = 0.1e-3 * p.ω
    N_holds = 16

    diags = [0,1]
    shots = 5



    pixel_width = 5.6e-6 ./ p.lₕ
    σ = 6e-6 ./ p.lₕ

    drive_width = 20

    harmonics_and_phases = [(1, 0.)]

end
ωs
sols = Array{Any}(undef, length(ωs))
Threads.@threads for i in eachindex(ωs)
    ω_true = ωs[i]
    # initial state and relevant matrices 
    begin 
        H_ho = HO_matrix(p.L, ω_true)
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
        sol = Time_Evolution_thermal_ρ_TD_VN(ρ, ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                            rtol=1e-10, atol=1e-10, solver=Vern7())
        t2  = time()
        println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

        ops = (H_fl,)
        f_ts = (_-> 1,)

        t1  = time()
        sol_fl = Time_Evolution_thermal_ρ_TD_VN(ρ, ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                            rtol=1e-10, atol=1e-10, solver=Vern7())
        t2  = time()
        println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")


    end

    sols[i] = sol
end

data_ωs = Array{AbstractArray}(undef, length(ωs))

Threads.@threads for i in eachindex(ωs)
    tstart = time()
    ω_true, sol = ωs[i], sols[i]

    # process data 
    total_simsteps = length(sol.u)
    idx = round.(Int, range(1, round(Int, total_simsteps *3/4), shots))
    ρ_shots = sol.u[idx]

    N = N_holds

    t_ideal = N < 23 ? t_holds_ideal(1., N) : t_holds_ideal(1. / 2, N)
    t_quant = t_ideal #quantize_times(t_ideal, Δt)

    # store one error vector per shot (indexed by k)
    tof_errors = Vector{Vector{Float64}}()

    shot_errors = []
    for ρ_shot in ρ_shots
        t1 = time()
        densities_tof = get_n_tof_optimized(ρ_shot, t_quant, t_tof, xs, hs; ω = ω_true)
        t2 = time()
        densities_convolved = [convolve_gaussian(densities_tof[i], xs, σ) for i in 1:N]
        t3 = time()
        densities_pixelated = [bin_pixels(densities_convolved[i], xs, pixel_width)[1] for i in 1:N]
        t4 = time()
        xs_pixelated = bin_pixels(densities_convolved[1], xs, pixel_width)[2]
        t5 = time()
        
        #println("1:$(t2-t1), 2: $(t3-t2), 3: $(t4-t3), 4: $(t5-t4)")
        
        xs_ = xs
        densities_ = densities_tof
        ρ_m = zeros(ComplexF64, size(ρ_shot))

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

    data_ωs[i] = shot_errors
    

    println("Finished for δHz = $(δHz[i]) in $(round(time()-tstart;digits=2)) s")
end

data = hcat(data_ωs...)
errors = data[3, :]
errors = hcat(errors...)
errors_k0 = errors[1,:]
errors_k1 = errors[2,:]


begin

sol_t = sols[1].t
idx = round.(Int, range(1, round(Int, length(sol_t)*3/4), shots))
tshots = sols[1].t[idx]


default(
    lw = 2,                 # thicker lines
    ms = 5,                 # marker size
    markerstrokewidth = 0,  # clean marker
    framestyle = :box
)
colors = cgrad([:orange, :blue], shots)
# --- plot k = 0 ---
p1 = plot(
    palette=colors,
    marker = :circle,
    color = :blue,
    xlabel = "δω (2π Hz)",
    ylabel = "ϵᵣₑₗ",
    title = "Diagonal = 0",

);

# --- plot k = 1 ---
p2 = plot(
    palette = colors,
    marker = :circle,
    color = :red,
    xlabel = "δω (2π Hz)",
    ylabel = "ϵᵣₑₗ",
    title= "Diagonal = 1",
        legend=false
);

for i in 1:shots
    #i ==1 && continue
    data = hcat(data_ωs...)
    errors = data[i, :]
    errors = hcat(errors...)
    errors_k0 = errors[1,:]
    errors_k1 = errors[2,:]

    plot!(p1, δHz, errors_k0,marker = :circle, color=colors[i], label="t_d =$(round(Int, tshots[i] *1e3 / p.ω)) ms")
    plot!(p2, δHz, errors_k1,marker = :circle, color=colors[i], label="Shot $i")

end 

display(plot(p1, p2, layout = (1,2), size=(800,300), dpi=300,bottom_margin = 6Plots.mm, left_margin=3Plots.mm))
end




