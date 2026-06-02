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

    t_tofs = vcat(0.01* p.ω, [n * 0.006 * p.ω for n in 2:4]...)
    Δt = 0.1e-3 * p.ω
    N_holds = collect(5:1:20)

    diags = [0,1]
    shots = 4


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


#gif_ρ_diag(sol,sol_fl, p)
# process data 
total_simsteps = length(sol.u)
idx = round.(Int, range(1, total_simsteps, 10))[6]
ρ_shot = sol.u[idx]
t_shot = sol.t[idx] ./p.ω


N_holds = 6:22
data_N_hold = Array{AbstractVector}(undef, size(N_holds))

for id_N_hold in eachindex(N_holds)
    tstart = time()
    N = N_holds[id_N_hold]

    t_ideal = N < 23 ? t_holds_ideal(1., N) : t_holds_ideal(1. / 2, N)
    t_quant = quantize_times(t_ideal, Δt)
    
    #println("experimental hold times:\n")
    #[println("$(round(t /p.ω *1e3;digits=1)) ms") for t  in t_quant]

    # store one error vector per shot (indexed by k)
    tof_errors = Vector{Vector{Float64}}()

    for t_tof in t_tofs

        densities_tof = get_n_tof_optimized(ρ_shot, t_quant, t_tof, xs, hs)
        densities_convolved = [convolve_gaussian(densities_tof[i], xs, σ) for i in 1:length(t_quant)]
        densities_pixelated = [bin_pixels(densities_convolved[i], xs, pixel_width)[1] for i in 1:length(t_quant)]
        xs_pixelated = bin_pixels(densities_convolved[1], xs, pixel_width)[2]
        
        xs_ = xs_pixelated
        densities_ = densities_pixelated
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
           
            
            #pl = plot(title="Diagonal $k, tof = $(round(Int, t_tof./p.ω *1e3)) ms", framestyle=:box)
            #plot!(pl, 1:p.L-k, abs.(true_diag), color=:blue,  linewidth=1, marker=:x, markersize=3, label="truth")
            #plot!(pl, 1:p.L-k, abs.(measured_diag), color=:red, linewidth=1, linestyle=:dash, markersize=3, marker=:o, label="tomography")
            #display(pl)
           

            norm_true = norm(true_diag)
            err_k = norm_true > 0 ? norm(true_diag - measured_diag) / norm_true : norm(measured_diag) 

            push!(diag_errors, err_k)

            all(abs.(measured_diag) .< 1e-7) && break
        end

        push!(tof_errors, diag_errors)
    end

    data_N_hold[id_N_hold] = tof_errors

    println("Finished N = $N in $(round(time()-tstart;digits=2)) s")
end



begin

default(
    lw = 2,                 # thicker lines
    ms = 5,                 # marker size
    markerstrokewidth = 0,  # clean marker
    framestyle = :box
)

# --- plot k = 0 ---
p1 = plot(
    ylims=(0,0.1),
    marker = :circle,
    color = :blue,
    xlabel = "# hold times",
    ylabel = "ϵᵣₑₗ",
    title = "Diagonal = 0",
    legend=false
);

# --- plot k = 1 ---
p2 = plot(
    ylims=(0,0.2),
    marker = :circle,
    color = :red,
    xlabel = "# hold times",
    ylabel = "ϵᵣₑₗ",
    title= "Diagonal = 1",
);

for (i,t_tof) in enumerate(t_tofs)
    #i ==1 && continue
    data = hcat(data_N_hold...)
    errors = data[i, :]
    errors = hcat(errors...)
    errors_k0 = errors[1,:]
    errors_k1 = errors[2,:]

    plot!(p1, N_holds, errors_k0,marker = :circle, label="$(round(t_tof *1e3 / p.ω; digits=0)) ms tof")
    plot!(p2, N_holds, errors_k1,marker = :circle, label="$(round(t_tof * 1e3 / p.ω; digits=0)) ms tof")

end 

display(plot(p1, p2, layout = (1,2), size=(800,300), dpi=300,bottom_margin = 6Plots.mm, left_margin=3Plots.mm))
end







# plot results