using FoSpDynamics, FoSpFloquet 
using Plots, ColorSchemes
using LinearAlgebra
using OrdinaryDiffEq
using LaTeXStrings
using JLD2
using ExpSynthDim

# experimental parameters
begin 
p = make_params(;κ=1, t_e_exp=0.08, L = 50, T_exp=150e-9, f_t_symbol=:smooth)
xs = LinRange(-20,20,2000)
f(x) = theta(x,0,20)
p.κ
H_ho = HO_matrix(p.L)
V = p.κ * drive_elements(xs, f, p.L)

H_walls = drive_elements(xs, x->20 * walls(x, 3, 11), p.L)

H = PeriodicFockMatrix([H_ho, V], [triv,p.f_t], 2π)
H_f = Fourier_op(H)
U_T = []
tols =  [ 1e-1, 5e-2 ]
for tol in tols
    U,_ = compute_Floquet(H_f, 0., 2; tol=tol)
    push!(U_T, U)
end
U_T
Id = Matrix(I, size(first(U_T)))
for i in 2:length(U_T)
    println("Norms comparing tolerances $(tols[i-1]) and $(tols[i])")
    println("Unitary Norm: ", norm(U_T[i-1]*U_T[i]' - Id, Inf))
    println("diff norm: " , norm(U_T[i]-U_T[i-1], Inf), "\n")
end
U_fl = U_T[end]
U_fl * U_fl'

es, vs = eigen(U_fl)
q_es = log.(es) / 1im .+ π

#plot(diag(abs.(U_fl), 1))
heatmap(abs.(U_fl))
end

