using QuantumFockDynamics
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using LaTeXStrings

include("../src/utils.jl")
include("../src/params.jl")
include("../src/Hamiltonians.jl")
include("../src/Processing.jl")

# set up parameters
p = make_params(;κ = 0.4, t_e_exp=0.1)

# set up Hamiltonians 
Hx = construct_hamiltonians(p.L)

# initial state
ρ = thermal_ρ_matrix(Hx[1], p.β)

## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
f_ts = (t-> one(ComplexF64), p.f_t)
ops = (Hx[1], Hx[2])

t1 = time()
sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
t2 = time()

println("Simulation time for $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

make_gif_x(sol, p; Nt=500)
pl, _,_,_ = plot_center_of_mass(sol, p)
display(pl)