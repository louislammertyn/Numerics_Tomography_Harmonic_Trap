using QuantumFockDynamics
using Plots, ColorSchemes
using LinearAlgebra
using OrdinaryDiffEq
using LaTeXStrings
using JLD2

include("../src/utils.jl")
include("../src/params.jl")
include("../src/Hamiltonians.jl")
include("../src/Processing.jl")

# set up parameters
p = make_params(;t_e_exp=0.1, L = 70)

# set up Hamiltonians 
Hx = construct_hamiltonians(p.L)

# initial state
ρ = thermal_ρ_matrix(Hx[1], p.β)

results = Dict()
for k in LinRange(0.1, 1, 10)
    p = update(p; κ=k)
    ## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
    f_ts = (t-> one(ComplexF64), p.f_t)
    ops = (Hx[1], Hx[2])

    t1 = time()
    sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
    t2 = time()

    println("Simulation time for $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    results[k] = sol
end
p = update(p; κ=4)
## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
f_ts = (t-> one(ComplexF64), p.f_t)
ops = (Hx[1], Hx[2])

t1 = time()
sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
t2 = time()

println("Simulation time for $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")
keys(results)
results[4.] = sol
## Plot the solution ##
Maxima = []
for k in [0.5, 1.]
    sol = results[k]
    p = update(p; κ=k)
    pl, pl2, pl3, pl4 = plot_center_of_mass(sol, p)
    pl4[1][:title] = "Center of mass motion in p-space for κ=$k ħω"
    display(pl4)
    #display(pl3)
    #display(pl4)
    push!(Maxima, maximum(pl.series_list[end][:y]))
end



plot(sort([collect(keys(results))..., 4.]), Maxima .- 10.3, marker=:o, xlabel="κ (ħω)", ylabel="Δλ_com", legend=false, title="Amplitude of Bloch Oscillation (λspace)", framestyle=:box)

# Gif
data = JLD2.load("kappasweep.jld2")
results = data["results"] 
for k in [ 0.5, 1.0, 4.0]s
    sol = results[k]
    p = update(p; κ=k)
    total_t = length(sol.t)
    #gif_ρ(sol, p; Nt=div(total_t,1), magfile="rho_mag_$(k).gif", phasefile="rho_phase_$(k).gif", every=10)
    gif_x(sol, p; Nt=div(total_t,1), filename="rho_x_$(k).gif", every=5)
end
