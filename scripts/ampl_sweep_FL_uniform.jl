using FoSpDynamics
using Plots, ColorSchemes
using LinearAlgebra
using OrdinaryDiffEq
using LaTeXStrings
using JLD2

include("../src/utils.jl")
include("../src/params.jl")
include("../src/Hamiltonians.jl")
include("../src/Processing.jl")
include("../src/Observables.jl")


# experimental parameters
p = make_params(;Δ=-2*π*14.54, t_e_exp=0.08, L = 100, T_exp=150e-9, f_t_symbol=:smooth)

plot(p.f_t.(collect(p.save_ts)))
p.L

### initial state ###
Hx= construct_hamiltonians_uniform(p.L)
ρ = thermal_ρ_matrix(Hx[1], p.β)
Hx[2]
# Convert density matrix to x-space
xs, ρ_x = density_x_matrix(ρ; grid=2000, xlims=(-20,20))
_, pl = plot_x_density(ρ)
display(pl)

# Translate by dx (real shift) and dp (momentum shift)
pl = plot( xlabel="λ mode n", ylabel = "|ρ[n,n]|");
for dx in [ 6.]
#dx = 10.
dp = 0.
ρ_x_translated = translate_density_x(ρ_x, xs, dx, dp)
ρ = density_λ_matrix(ρ_x_translated, xs, p.L)
#heatmap(abs.(ρ))

scatter!(pl, [abs(ρ[i,i]) for i in 1:p.L], label = "dx = $(round(dx * p.lₕ*1e6; digits= 3))μm", markersize=3)
end
display(pl)
sum(diag(ρ))
lambda_com(ρ)


### sims ###
results = Dict()

for k in [0.7]
    p = update(p; κ=k)
    if true 
        # set up Hamiltonians 
        Hx_FL = construct_hamiltonians_FL_uniform(p)
        Hx_FL = Hx_FL[1] + Hx_FL[2]

        ## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
        f_ts_FL = (t-> one(ComplexF64), )
        ops_FL = (Hx_FL,)

        t1 = time()
        sol_FL = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops_FL, f_ts_FL, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
        t2 = time()

        println("Simulation time for FL with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")
    end

    f_ts = (t-> one(ComplexF64), p.f_t)
    ops = Hx

    t1 = time()
    sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
    t2 = time()

    println("Simulation time for Exact with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    results[k] = (sol, sol_FL)
end

jldsave("uniform_displaced_kappasweep_wFL_D=$(p.Δ).jld2"; results, p)

## Plot the solution ##
data = JLD2.load("uniform_kappasweep_wFL_D=$(round(p.Δ/(2*π),2)).jld2")
results = data["results"] 
Maxima = []

for k in [0.5]
    sol = results[k]
    p = update(p; κ=k)
    pl = plot_center_of_mass_wFL(sol, p)
    #pl[1][:title] = "Center of mass motion in p-space for κ=$k ħω"
    display(pl)
    #display(pl3)
    #display(pl4)
    #push!(Maxima, maximum(pl.series_list[end][:y]))
end




plot(sort([collect(keys(results))..., 4.]), Maxima .- 10.3, marker=:o, xlabel="κ (ħω)", ylabel="Δλ_com", legend=false, title="Amplitude of Bloch Oscillation (λspace)", framestyle=:box)

# Gif
data = JLD2.load("kappasweep_wFL_D=91.35751436639117.jld2")
results = data["results"] 
for k in [ .7]
    sol, sol_FL = results[k]
    p = update(p; κ=k)
    total_t = length(sol.t)
    pl = plot_center_of_mass_wFL((sol, sol_FL), p)
    display(pl)
    #display(pl2)
    #display(pl3)
    #display(pl4)
    gif_ρ_diag(sol, sol_FL, p; Nt=div(total_t,1), magfile="uniform_rho_diag_wFL$(k)_D=$(p.Δ).gif",  every=5)
    #gif_ρ(sol, p; Nt=div(total_t,1), magfile="rho_wFL$(k)_D=$(p.Δ).gif",  every=5)
    #gif_x(sol, p; Nt=div(total_t,1), filename="rho_diag_$(k).gif", every=5)
end
results
 
