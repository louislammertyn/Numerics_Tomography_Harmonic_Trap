using Revise 
using LinearAlgebra, Plots
using ExpSynthDim
BLAS.set_num_threads(1)


δωs = 2π .* collect(0:5:20)
ξs = [0.5, 2, 5]
N_holds = 30
samples_initial = 5
diag_id = 0


# parameters
p = make_params(;L = 50, T_exp=200e-9)
xs = LinRange(-60,60,5000)
ps = LinRange(-60, 60,5000)
hs = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

t_tof = t_tof = 0.001 * p.ω
t_holds = [n * (p.ω * 0.0001) for n in 0:N_holds-1]

########## Experiment ##########
all_data = []   

for δω in δωs
    println("Starting run δω = $(δω):")

    
    ω_HO = (p.ω + δω) / p.ω
    H_ho = HO_matrix(p.L, ω_HO)

    data_δω = []   # store all ξ results

    for ξ in ξs
        tstart = time()
        println("------- \n    Starting ξ = $(ξ)")

        # initial states
        ρ = thermal_ρ_matrix(H_ho, p.β)
        initial_states = [almost_thermal(ρ, 0.1, ξ) for _ in 1:samples_initial]

    
        data_ξ = Vector{Any}(undef, length(initial_states))

        Threads.@threads for i in eachindex(initial_states)
            ρ_init = initial_states[i]

            densities = []
            phases_cache = zeros(ComplexF64, length(ps))

            for th in t_holds
                ρ_λ = free_evolve_λ_matrix(ρ_init, th; ω=ω_HO)
                ρ_p = density_p(ρ_λ, hs_p)
                ρ_p_tof = free_evolve_p_matrix(ρ_p, ps, t_tof, phases_cache)
                ρ_x_diag_final = density_x_diagonal_from_p_fft(
                    ρ_p_tof, ps[2]-ps[1], xs
                )

                push!(densities, abs.(ρ_x_diag_final))
            end

            data_ξ[i] = (densities=densities, ρ_init=ρ_init)
        end

        push!(data_δω, data_ξ)

        println("    Time elapsed for ξ = $ξ : $(time() - tstart) s\n")
    end

    push!(all_data, (ω_HO=ω_HO, data=data_δω))
    println("------------------------------------------ \n")
end


######## Processing ##########

data_errors = []
data_spreads = []

for (i_δω, entry) in enumerate(all_data)
    ω_HO = entry.ω_HO
    data_δω = entry.data

    data_errors_ξ = []
    data_spreads_ξ = []

    for (i_ξ, data_ξ) in enumerate(data_δω)
        errors = []
        
        # Optional plotting of differences to get the idea

        for (i,sample) in enumerate(data_ξ)
            densities = sample.densities
            ρ_init = sample.ρ_init

            ρ_measured = reconstruct_density_matrix_slice(
                xs, t_tof, densities, t_holds, diag_id; λmax=p.L-1
            )

            ρ_truth = (ρ_init[diagind(ρ_init, diag_id)])[1:end-1]
            push!(errors, norm(ρ_measured - ρ_truth) / norm(ρ_init))

            if i == 1 
                # ---- plotting ----
                λs = 0:p.L-2
                p1 = plot(λs, abs.(ρ_measured),
                    label="measured", lw=2, color=:green,
                    xlabel="λ", ylabel="|ρ(λ,λ+k)| (k=$diag_id)")

                plot!(p1, λs, abs.(ρ_truth),
                    label="truth", lw=2, ls=:dash, color=:blue)

                ρ_diff = ρ_measured .- ρ_truth
                p2 = plot(λs, abs.(ρ_diff),
                    label="difference", lw=2, color=:red,
                    xlabel="λ", ylabel="error")

                ptot = plot(p1, p2, layout=(2,1), size=(800,600),
                    title="δω=$(div(δωs[i_δω], 2π)), ξ=$(ξs[i_ξ])")

                display(ptot)
            end
        end

        ϵ_av = sum(errors) / length(errors)
        σ_ϵ = sum((errors .- ϵ_av).^2) / length(errors)

        push!(data_errors_ξ, ϵ_av)
        push!(data_spreads_ξ, σ_ϵ)
    end

    push!(data_errors, data_errors_ξ)
    push!(data_spreads, data_spreads_ξ)
end


data_matrix_errors = hcat(data_errors...)
data_matrix_spreads = hcat(data_spreads...)

pl = plot(xlabel="δω (2π Hz)", ylabel="ϵᵣₑₗ", framestyle=:box)
for id_ξ in axes(data_matrix_errors, 1)
    ϵs_ξ = data_matrix_errors[id_ξ, :]
    σs_ξ = data_matrix_spreads[id_ξ, :]

    plot!(pl, δωs ./ 2π, ϵs_ξ, yerrors=σs_ξ, label="ξ = $(ξs[id_ξ])", line=:dash)
end
display(pl)




 #comparison initial cloud and after time of flight
ρx_truth = abs.(diag(density_x(ρ, hs)))
ρx_meas  = densities[1]
p.lₕ * 1e7
plot(
    xs .* (p.lₕ * 1e6),
    [ρx_meas ρx_truth],
    lw = 2,
    ls = [:solid :dash],
    color = [:blue :grey],
    label = ["After TOF" "Initial cloud"],
    xlabel = "Position x (μm)",
    ylabel = "Density ρ(x)",
    title = "Position-Space Density",
    legend = :topright,
    grid = true,
    framestyle = :box,
)


ρ_measured = reconstruct_density_matrix_slice(xs, t_tof, densities, t_holds, 0; λmax=p.L-1)

begin
n = 0:p.L-2
ρ_true = abs.(diag(ρ)[1:end-1])
ρ_meas = abs.(ρ_measured)
ρ_err  = abs.((ρ_meas .- ρ_true) ) 

p1 = plot(
    n, ρ_true,
    lw=1.5, color=:green,
    label="True",
    ylabel="|ρₙₙ|",
    title="Density Matrix Diagonal",
    legend=:topright,
    grid=true,
    dpi=200,
);

plot!(
    n, ρ_meas,
    lw=2.5, ls=:dashdot, color=:blue, alpha=0.7,
    label="Reconstructed"
);

p2 = plot(
    n, ρ_err,
    lw=2, color=:red,
    xlabel="λ",
    ylabel="Absolute error",
    label="",
    grid=true,
    dpi=200
);
end;
plot(p1, p2, layout=(2,1), link=:x, framestyle=:box, dpi=200)

