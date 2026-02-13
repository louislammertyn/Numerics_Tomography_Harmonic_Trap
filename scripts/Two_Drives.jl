using FoSpDynamics
using Plots, ColorSchemes, ProgressMeter
using LinearAlgebra
using OrdinaryDiffEq
using LaTeXStrings
using JLD2

include("../src/utils.jl")
include("../src/Observables.jl")
include("../src/params.jl")
include("../src/Hamiltonians.jl")
include("../src/Processing.jl")

## T=0
Es_ϕ = Float64[]
number = 200
fluxes = LinRange(0,2*π, number)

m = 3
@showprogress for flux in fluxes
p = make_params(;Δ=0., L=60, ϕ=0., κ = 2.)

# Floquet Hamiltonian 
Hs = construct_hamiltonians_FL_2drives(p, flux, m, p.κ * 1)
Hs_exact = construct_hamiltonians(p.L)

H_fl = sum(collect(Hs))
@assert H_fl == Hermitian(H_fl)

# find energy 
es, vs = eigen(Hermitian(H_fl))
push!(Es_ϕ, minimum(es))
end
Es_ϕ
# calculate current from energy
I_phi = [(Es_ϕ[i+1] - Es_ϕ[i]) / (fluxes[2]-fluxes[1]) for i in 1:(number-1)]
plot(fluxes[1:end-1] ./ (2*π), I_phi, legend=false, ylabel="I(ϕ)=∂E/∂ϕ", xlabel="ϕ/2π")





## T > 0
# experimental parameters
p = make_params(;Δ=0., t_e_exp=0.08, L = 100, T_exp=120e-9, f_t_symbol=:smooth)
# double drive params 
κ2 = 10* p.κ 
flux =0.8 * 2*π
m = 3

### initial state ###
Hx= construct_hamiltonians(p.L)
ρ = thermal_ρ_matrix(Hx[1], p.β)

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

scatter!(pl, [abs(ρ[i,i]) for i in 1:p.L], label = "dx = $(div(dx,2))μm", markersize=3)
end
display(pl)
sum(diag(ρ))


### sims ###
results = Dict()

for k in [1.]
    p = update(p; κ=k)
    if true
        # set up Hamiltonians 
        Hx_FL = construct_hamiltonians_FL_2drives(p, flux, m, κ2)
        Hx_FL = sum(collect(Hx_FL))

        ## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
        f_ts_FL = (t-> one(ComplexF64), )
        ops_FL = (Hx_FL,)

        t1 = time()
        sol_FL = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops_FL, f_ts_FL, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
        t2 = time()

        println("Simulation time for FL with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")
    end

    double_drive(t) = p.f_t(t) + κ2 * cos(m*p.ω_d * t + flux)
    f_ts = (t-> one(ComplexF64), double_drive)
    ops = Hx

    t1 = time()
    sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
    t2 = time()

    println("Simulation time for Exact with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

    results[k] = (sol, sol_FL)
end

for k in [ 1.]
    sol, sol_FL = results[k]
    p = update(p; κ=k)
    total_t = length(sol.t)
    pl, pl2, pl3, pl4 = plot_center_of_mass(sol, p)
    #display(pl)
    #display(pl2)
    #display(pl3)
    #display(pl4)
    gif_ρ_diag(sol, sol_FL, p; Nt=div(total_t,1), magfile="rho_diag_wFL_doubleDr_flux=$(flux)_k=$(k)_D=$(p.Δ).gif",  every=5)
    #gif_ρ(sol, p; Nt=div(total_t,1), magfile="rho_wFL_doubleDr$(k)_D=$(p.Δ).gif",  every=5)
    #gif_x(sol, p; Nt=div(total_t,1), filename="rho_diag_$(k).gif", every=5)
end