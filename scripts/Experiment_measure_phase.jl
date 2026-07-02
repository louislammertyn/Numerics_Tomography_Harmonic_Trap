using Revise 
using ExpSynthDim



phis =  LinRange(0,π, 2)
measured_phis = []
measured_spreads= []

for ϕ in phis
# set parameters 
begin
    p = make_params(;κ=1., Δ=0, t_e_exp=0.05, L = 80, T_exp=250e-9, f_t_symbol=:smooth)
    harmonics_and_phases_and_amplitudes =  [(1, 0., 1.), (3, ϕ, 3.)]
    xs = LinRange(-60,60,8000)
    ps = LinRange(-60, 60,8000)
    hs = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^(n) .* hermite.(n, ps) for n in 0:p.L-1]...)
end

# set relevant Hamiltonians and drives
begin
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,20)
    V_d = p.κ * drive_elements(xs, step, p.L)

    H_fl = RWA_FL_Hamiltonian(V_d, p, harmonics_and_phases)
end

# (Optional) Check exact Floquet operator


# initial state
begin 
    ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)
end

# simulate driven time evolution
begin 
    drive(t) = sum([ampl * sin(harm * t + ϕ) for (harm, ϕ, ampl) in harmonics_and_phases_and_amplitudes])
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

begin 
ρ_sol = sol.u[101]

pl = plot(
    xlabel = "λ",
    ylabel = "|ρ(λ, λ + δλ)|",
    title = "Magnitude of Density Matrix off-diagonals",
    legend = :topright,
    lw = 2
)

for k in 1:3
    d = diag(ρ_sol, k)
    plot!(
        pl,
        1:length(d),
        real.(d),
        label = "δλ = $k",
        marker="o",
        markersize=2.5,
        markerstrokewidth=0.1
    )
end

display(pl)
end



idx = [1 + n*100 for n in 0:15]
pl_angle_ev = plot(linestyle=:dot, framestyle=:box,  xlabel="nT", ylabel="⟨Λ⟩ₙ", title="Gauge invariant measure Λ for ϕ=$(round(ϕ;digits=2))")

angles_n = []
ns = 5:2:20
for n in ns
    angles=[]
    for ρ in sol.u[idx]
        push!(angles, (angle((ρ[n,n+1]*ρ[n+1,n+2]*ρ[n+2,n+3])/ (ρ[n,n+3])  )))
    end
    plot!(pl_angle_ev, angles, label="λ=$n")
    push!(angles_n, angles[2])
    
end
display(pl_angle_ev)

mean_angle  =  sum(angles_n) / length(ns)
spread_angle = sum((angles_n .-mean_angle).^2) / length(ns)
push!(measured_phis, mean_angle)
push!(measured_spreads, spread_angle)
end


pl = scatter(phis, abs.(measured_phis), yerror=measured_spreads,xlabel="Input ϕᵢ", ylabel="Measured ϕₒ", marker=:x, color=:red, framestyle=:box, label="Measured");
plot!(pl, phis, phis, color=:blue, label="ϕᵢ = ϕₘ");
display(pl)


# run time evolution in trap + tof 
n_measurements = 1
measurement_idx = Int.(collect(LinRange(1,length(sol.u), n_measurements)))
sol_m= sol[measurement_idx]

t_holds = LinRange(0., 1, 2)
tof = 0.012 * p.ω


for (i,ρ) in enumerate(sol_m.u )
    ρs_hold = []
    for th in t_holds 
        ρ_λ = free_evolve_λ_matrix(ρ, th)
        ρ_p = density_p(ρ_λ, hs_p)
        ρ_p_tof = free_evolve_p_matrix(ρ_p, ps, tof)
        ρ_x_tof = density_x_matrix_from_p(ρ_p_tof, xs, ps)
        #display(plot(xs, abs.(diag(ρ_x_tof))))

    end
    push!(measurements, ρs_hold)
end

# runt time evolution in tof and get x diagonal