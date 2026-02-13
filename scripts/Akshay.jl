## Should be all packages you need
using FoSpDynamics
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using LaTeXStrings
using JLD2

# load the function files
include("../src/utils.jl")
include("../src/params.jl")
include("../src/Hamiltonians.jl")
include("../src/Time_Evolutions.jl")
include("../src/Processing.jl")
include("../src/Observables.jl")


# give experimental parameters in SI units (see params.jl file for details) 
# it then gives the params such that ħ = ω = m = 1
# see also it's printed output
p = make_params(;Δ=0., t_e_exp=0.08, L = 100, T_exp=150e-9, f_t_symbol=:smooth)

# parameters can be updated with the update(p; ...) function

# Driven Hamiltonian H = H_harmonic + f(t) * H_drive where this gives the H_i without f(t) (the drive comes in the time evolution)
H_harm , H_drive= construct_hamiltonians_uniform(p.L)

# Initial state
ρ = thermal_ρ_matrix(H_harm, p.β)

# initial state in x-space
xs, ρ_x = density_x_matrix(ρ; grid=2000, xlims=(-20,20))
_, pl = plot_x_density(ρ)
display(pl)

## Here you can displace the cloud if you want
# Translate by dx (real shift) and dp (momentum shift)

# the loop below is to compare different shifts but looping over 1 value will give you the displacement you entered

pl = plot( xlabel="λ mode n", ylabel = "|ρ[n,n]|");
for dx in [ 6.]
dp = 0.
ρ_x_translated = translate_density_x(ρ_x, xs, dx, dp)
ρ = density_λ_matrix(ρ_x_translated, xs, p.L)
#heatmap(abs.(ρ))

scatter!(pl, [abs(ρ[i,i]) for i in 1:p.L], label = "dx = $(round(dx * p.lₕ*1e6; digits= 3))μm", markersize=3)
end
display(pl)

# some checks
sum(diag(ρ))
lambda_com(ρ)


### sims ###
# results are saved as DiffEq.jl ODESolution type in a dictionary which I pick here to map a drive strength to a solution
results = Dict()


# loop for sweeping several driving strenghts
# You can pick a function from Time_Evolution.jl to run (here are given the Floquet approximate sim, exact Time dependent sim or both)
# please go to the Time_Evolution.jl file and change the Hamiltonians to the appropriate ones (here i use the Θ(x)*cos(ωt))
for k in [.1]
    p = update(p; κ=k)
    #results[k] = run_Exact(p)
    results[k] = run_Floquet(p)
    #results[k] = run_both(p)
end

jldsave("insert_your_file_name.jld2"; results, p)
results[.1]
## Plot the com solution ##
for k in [.1]
    sol = results[k]
    p = update(p; κ=k)

    # you can also give it two sols where the first will be labeled the exact sol and the second will be labeled fl
    pl = plot_center_of_mass_λ(sol, p)
    display(pl)
end
display(pl)



# Gif
for k in [.1]
    sol_FL = results[k]
    p = update(p; κ=k)

    # you can also give it two sols where the first will be labeled the exact sol and the second will be labeled fl
    gif_ρ_diag(sol_FL, p; Nt=div(total_t,1), magfile="Insert_name.gif",  every=5)
    
end

