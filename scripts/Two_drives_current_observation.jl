using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using OrdinaryDiffEq
using FoSpDynamics


# set parameters 
# physical 



# experimental params
begin
    p = make_params(; Δ=0, L=50, T_exp=200e-9)
    xs = LinRange(-800, 800, 20000)
    ps = LinRange(-800, 800, 20000)

    hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
    hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

    t_tof = 0.012 * p.ω
    Δt = 0.1e-3 * p.ω
    N_holds = collect(5:1:20)

    diags = [0,1]
    shots = 4


    pixel_width = 5.6e-6 ./ p.lₕ
    σ = 6e-6 ./ p.lₕ

    drive_width = 20


    κs = [0.1, 0.1]
    
end


# initial state and relevant matrices 
begin 
    H_ho = HO_matrix(p.L)
    step(x) = theta(x,0,drive_width)
    V_d = drive_elements(xs, step, p.L)
    
    ρ = thermal_state(H_ho, p.β)
    plot(0:p.L-1, abs.(diag(ρ)), color=:red,framestyle=:box, xlabel="λ", ylabel="|ρ(λ, λ)|", label="$(p.T_exp * 1e9) nK",marker=:o,seriestype=:sticks)

end

# run 2 drives exp 
begin

    ϕs = LinRange(0,π, 9)

    sols = Vector{Any}(undef, length(ϕs))
    Threads.@threads for i in eachindex(ϕs)

        ϕ = ϕs[i]
        harmonics_and_phases = [(1, 0.), (3, ϕ)]
        begin 
            drive(t) = sum([κs[i] * p.f_t(harm * t + ϕ) for (i, (harm, ϕ)) in enumerate(harmonics_and_phases)])
            ops = (H_ho, V_d)
            f_ts = (_-> 1, drive)

            t1  = time()
            sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                                rtol=1e-10, atol=1e-10, solver=Vern7())
            t2  = time()
            println("Simulation time for exact dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")

            H_fl = RWA_FL_Hamiltonian(V_d, p, harmonics_and_phases)
            ops = (H_fl,)
            f_ts = (_-> 1,)

            t1  = time()
            sol_fl = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i, p.t_e), p.save_ts;
                                                rtol=1e-10, atol=1e-10, solver=Vern7())
            t2  = time()
            println("Simulation time for Floquet dynamics with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")


        end
        sols[i] = sol
    end
end

# get observables
begin 
    loop_n = 19
    nT = div(length(sols[1].t), 100)
    idx_strob = [1 + n*100 for n in 0:nT]
    strob_ts = sols[1].t[idx_strob]
    loop_currents = Matrix{Float64}(undef, (length(ϕs), nT+1))

    sols
    for (i,sol) in enumerate(sols) 
        ϕ = ϕs[i]
        harmonics_and_phases = [(1, 0.), (3, ϕ)]
        H_fl = RWA_FL_Hamiltonian(V_d, p, harmonics_and_phases)

        
        sol_strob = sol.u[idx_strob]
        
        
        for (j,ρ) in enumerate(sol_strob)
            J = local_currents(ρ, H_fl)
            I_Φ = J[loop_n,loop_n+1]+ J[loop_n+1,loop_n+2]+ J[loop_n+2,loop_n+3]- J[loop_n,loop_n+3]
            loop_currents[i, j] = I_Φ
            if j == 2
                println(-angle(-ρ[loop_n,loop_n+1]/ (ρ[loop_n,loop_n+3])), " ≈ $ϕ")
            end
        end 
    end
end 

# plot observables
begin 
    pl = plot(ylabel="Iₙ(Φ)", xlabel="n ⋅ T");
    colors= cgrad([:orange, :blue], length(ϕs));
    for i in axes(loop_currents, 1)
        ϕ=ϕs[i]
        currents = loop_currents[i, :]
        plot!(pl, strob_ts, currents, color=colors[i],label="ϕ=$(round(ϕ/π; digits=2)) π")
    end
    display(pl)
end

# now with tomography
begin 
N_holds = 15
diags = [0,1, 3]
ρ_ϕs = Vector{Any}(undef, length(ϕs))
angles = Vector{Float64}(undef, length(ϕs))
for i in eachindex(sols)
    sol = sols[i]
    ϕ = ϕs[i]

    loop_n = 19
    ρ_T = sol.u[101]

    N = N_holds 
    t_ideal = N < 23 ? t_holds_ideal(1., N) : t_holds_ideal(1. / 2, N)
    t_quant = quantize_times(t_ideal, Δt)

    densities_tof = get_n_tof_optimized(ρ_T, t_quant, t_tof, xs, hs)
    densities_convolved = [convolve_gaussian(densities_tof[i], xs, σ) for i in 1:N]
    densities_pixelated = [bin_pixels(densities_convolved[i], xs, pixel_width)[1] for i in 1:N]
    xs_pixelated = bin_pixels(densities_convolved[1], xs, pixel_width)[2]

    xs_ = xs_pixelated 
    densities_ = densities_pixelated

    ρ_m = zeros(ComplexF64, size(ρ_T))
    for k in diags
        measured_diag = reconstruct_density_matrix_slice(
            xs_, t_tof, densities_, t_quant, k; λmax=100
        )[1:p.L - k]

        ρ_m[diagind(ρ_m, k)] = measured_diag
        
        if k > 0 
            ρ_m[diagind(ρ_m, -k)] = conj.(ρ_m[diagind(ρ_m, k)])
        end
        true_diag = ρ_T[diagind(ρ_T, k)][1:length(measured_diag)]
           
            
        pl = plot(title="Diagonal $k, tof = $(round(Int, t_tof./p.ω *1e3)) ms", framestyle=:box)
        plot!(pl, 1:p.L-k, abs.(true_diag), color=:blue,  linewidth=1, marker=:x, markersize=3, label="truth")
        plot!(pl, 1:p.L-k, abs.(measured_diag), color=:red, linewidth=1, linestyle=:dash, markersize=3, marker=:o, label="tomography")
        display(pl)
        

        norm_true = norm(true_diag)
        err_k = norm_true > 0 ? norm(true_diag - measured_diag) / norm_true : norm(measured_diag) 

        println(" the error for diag $k is $err_k")
    end

    println(-angle(ρ_m[loop_n,loop_n+1]*ρ_m[loop_n+1,loop_n+2]*ρ_m[loop_n+2,loop_n+3]/ (ρ_m[loop_n,loop_n+3])), " ≈ $ϕ") #*ρ_m[loop_n+1,loop_n+2]*ρ_m[loop_n+2,loop_n+3]
    angles[i] = -angle(ρ_m[loop_n,loop_n+1]*ρ_m[loop_n+1,loop_n+2]*ρ_m[loop_n+2,loop_n+3]/ (ρ_m[loop_n,loop_n+3]))
    ρ_ϕs[i] = ρ_m
end
end
begin 
pl = plot(xlabel="Input phase ϕ", ylabel="ϕ = arg(ρ(n,n+1)/ρ(n,n+3))", title="Measuring synthetic flux with tomography")
plot!(pl, ϕs, angles, marker=:x, color=:red, linestyle=:dashdot, label="Tomography")
plot!(pl, ϕs, ϕs, color=:blue, label="truth")
display(pl)
end
ρ_ϕs


