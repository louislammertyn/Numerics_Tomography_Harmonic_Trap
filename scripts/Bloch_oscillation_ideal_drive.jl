using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using Statistics
using OrdinaryDiffEq
using FoSpDynamics

# Experimental and physical parameters
begin
    p = make_params(; κ=0.4, L=50, T_exp=250e-9, t_e_exp= 0.1,)
    xs = LinRange(-50, 50, 2000)
    ps = LinRange(-50, 50, 2000)

    hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

    t_tof = 0.012 * p.ω
    Δt = 0.1e-3 * p.ω


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
    H_fl = RWA_FL_Hamiltonian(V_d, p, harmonics_and_phases, :smooth)

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

t_targets = [9.8, 19.7, 29.5, 39.3, 49.1, 59.0, 68.8, 78.6]
t_all_ms = sol.t ./ p.ω .* 1e3
indices = [findmin(abs.(t_all_ms .- t))[2] for t in t_targets]

ncols = length(indices)

# create 2 x N grid
plot_grid = Array{Any}(undef,  ncols, 2)

for (k, i) in enumerate(indices)
    ρ__ = sol.u[i]

    dens = real(diag(ρ__))
    off_diag = ρ__[diagind(ρ__, 1)]
    curr = imag(off_diag)

    title_str = "t_drive = $(round(t_targets[k], digits=1)) ms"

    # top: density
    plot_grid[k, 1] = bar(dens,
        title=title_str,
        legend=false,
        color=:blue,
        dpi=300)

    # bottom: current
    plot_grid[ k, 2] = bar(curr,
        xlabel="State index n",
        legend=false,
        color=:orange,
        ylims=(-0.04,0.04),
        dpi=300)
end

plot(plot_grid..., layout=(2, ncols), size=(300*ncols, 500), dpi=500)

plot_center_of_mass_λ(sol, sol_fl, p)
p1, p2, p3 = plot_center_of_masses(sol, p)

display(p3)

function check_κ(p, hop, ampl)
    return (p.ħ * p.Δ * ampl) / (4*hop) /(p.ω*p.ħ)
end
check_κ(p, 0.3 *2/π,2)