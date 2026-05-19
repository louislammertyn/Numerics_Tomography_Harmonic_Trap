using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using Statistics
BLAS.set_num_threads(1)

##############################
# Controls: which tests to run
##############################
RUN_TEST_A = true   # error vs N (ideal uniform sampling)
RUN_TEST_B = true   # timing quantization (0.1 ms)
RUN_TEST_C = true   # wrong assumed ω in reconstruction
RUN_TEST_D = true   # dependence on state extent / coherence size

##############################
# Experiment settings
##############################
 

##############################
# Parameters / grids
##############################
begin 

p = make_params(; L=50, T_exp=200e-9)

xs = LinRange(-60, 60, 5000)
ps = LinRange(-60, 60, 5000)

hs   = hcat([hermite.(n, xs) for n in 0:p.L-1]...)
hs_p = hcat([(-1im)^n .* hermite.(n, ps) for n in 0:p.L-1]...)

t_tof = 0.001 * p.ω
atan(t_tof)
# Hardware timing resolution:
Δt_res = 0.1e-3 * p.ω  # 0.1 ms resolution

2π/30
π / 20
##############################
# Helpers
##############################
# Uniform coherent block density matrix (your choice; keeps your normalization style)
function make_uniform_density(L, extent)
    ρ = zeros(ComplexF64, L, L)
    @inbounds for j in 1:extent, k in 1:extent
        ρ[j,k] = 1/extent   # keep your original convention
    end
    return ρ
end



##############################
# Core simulation for given times
##############################
function simulate_densities(ρ_init, t_holds; ω_HO, hs_p, ps, xs, t_tof)
    densities = Vector{Vector{Float64}}(undef, length(t_holds))
    phases_cache = zeros(ComplexF64, length(ps))

    @inbounds for (ii, th) in enumerate(t_holds)
        ρ_λ = free_evolve_λ_matrix(ρ_init, th)
        ρ_p = density_p(ρ_λ, hs_p)
        ρ_p_tof = free_evolve_p_matrix(ρ_p, ps, t_tof, phases_cache)
        #ρ_x = density_x(ρ_λ, hs)
        ρ_x_diag_final =density_x_diagonal_from_p_fft(ρ_p_tof, ps[2]-ps[1], xs)
        #ρ_x_diag_final = ρ_x[diagind(ρ_x,0)] #
        densities[ii] = abs.(ρ_x_diag_final)
    end

    return densities
end

##############################
# Reconstruction error (ω mismatch handled by rescaling times used in reconstruction)
##############################


function reconstruction_error(densities, t_holds, ρ_init;
        xs, t_tof, diag_id, λmax, abs_=false)

    λeff = λmax - diag_id

    ρ_measured = reconstruct_density_matrix_slice(
        xs, t_tof, densities, t_holds, diag_id; λmax=100
    )[1:λeff]

    ρ_truth = ρ_init[diagind(ρ_init, diag_id)][1:λeff]

    err = abs_ ? (norm(abs.(ρ_measured) - abs.(ρ_truth))) / norm(abs.(ρ_init)) : norm(ρ_measured - ρ_truth) / norm(ρ_init)
    return err, ρ_measured, ρ_truth
end

end
f(k) = k * (π - 2* atan(t_tof))
f(3)

############################################################
# TEST A: error vs N (ideal uniform sampling, ω correct)
############################################################
if RUN_TEST_A
    println("\n==============================")
    println("TEST A: error vs N (ideal sampling)")
    println("==============================")

    N_holds = collect(10:30)  # samples per period

    #try multiple diagonals
    diag_ids = [0, 1, 2, 5]  
    abs_ = false

    ω_HO = 1.
    data_δω = Vector{Any}(undef, length(N_holds))

    ρ_init = make_uniform_density(p.L, 20)

    Threads.@threads for i in eachindex(N_holds)
        N_hold = N_holds[i]
        tstart = time()

        t_holds = t_holds_ideal(ω_HO, N_hold)

        densities = simulate_densities(ρ_init, t_holds;
            ω_HO=ω_HO, hs_p=hs_p, ps=ps, xs=xs, t_tof=t_tof)

        data_δω[i] = (densities=densities, ρ_init=ρ_init, t_holds=t_holds)

        println("  N=$(N_hold) done in $(round(time()-tstart, digits=2)) s")
    end

    all_data_A = (ω_HO=ω_HO, data=data_δω)
    display(plot(data_δω[1].densities[1]))

    # -------------------------
    # Processing (CHANGED)
    # -------------------------
    ω_HO = all_data_A.ω_HO
    data_δω = all_data_A.data

    # store one error curve per diag_id
    errs_by_diag = Dict{Int, Vector{Float64}}()

    for diag_id in diag_ids
        println("\n--- Processing diag_id = $diag_id ---")

        errs_A = Float64[]

        for (i_N, sample_N) in enumerate(data_δω)
            N_hold = N_holds[i_N]
            densities = sample_N.densities
            ρ_init = sample_N.ρ_init
            t_holds = sample_N.t_holds

            err, ρ_meas, ρ_true = reconstruction_error(densities, t_holds, ρ_init;
                xs=xs, t_tof=t_tof, diag_id=diag_id, λmax= p.L -1, abs_=abs_)

            push!(errs_A, err)

            # Use the actually returned length to avoid off-by-one mismatches
            λs = 0:(length(ρ_meas)-1)

            # Magnitude comparison (kept)
            p_mag = plot(λs, abs.(ρ_meas), label="measured |ρ|", lw=2, color=:green,
                xlabel="λ", ylabel="|ρ(λ,λ+k)| (k=$diag_id)")
            plot!(p_mag, λs, abs.(ρ_true), label="truth |ρ|", lw=2, ls=:dash, color=:blue)

            # Phase comparison (principal value in [-π, π])
            # (If you prefer unwrapped phase, tell me and I'll add unwrap.)
            ϕ_meas = angle.(ρ_meas)
            ϕ_true = angle.(ρ_true)

            p_phase = plot(λs, ϕ_meas, label="measured arg(ρ)", lw=2, color=:green,
                xlabel="λ", ylabel="phase (rad)")
            plot!(p_phase, λs, ϕ_true, label="truth arg(ρ)", lw=2, ls=:dash, color=:blue)

            # Absolute difference (magnitude of complex error)
            p_absdiff = plot(λs, abs.(ρ_meas .- ρ_true), label="abs diff", lw=2, color=:red,
                xlabel="λ", ylabel="Absolute difference")
            

            display(plot(p_mag, p_phase, p_absdiff,
                layout=(3,1), size=(800,900),
                title="diag=$diag_id, N=$(N_hold)"))

        end

        errs_by_diag[diag_id] = errs_A

        plA = plot(N_holds, errs_A, marker=:o,
            xlabel="N holds (samples per period)", ylabel="ϵ_rel",
            title="Error vs N (ideal sampling, ω correct) — diag=$diag_id",
            framestyle=:box, label="")
        display(plA)
    end

    
    plAll = plot(xlabel="N holds (samples per period)", ylabel="ϵ_rel",
        title="Error vs N — multiple diagonals", framestyle=:box)
    for diag_id in diag_ids
        plot!(plAll, N_holds, errs_by_diag[diag_id], marker=:o, label="diag=$diag_id")
    end
    display(plAll)
end



############################################################
# TEST B: timing quantization (0.1 ms) vs ideal
############################################################
if RUN_TEST_B
    println("\n==============================")
    println("TEST B: timing quantization (0.1 ms)")
    println("==============================")

    all_data_B = []  # store per N: ideal + quantized densities

    N_holds = vcat(collect(5:5:20), [22], [25, 30, 35, 40, 45])  # samples per period

    #try multiple diagonals
    diag_ids = [0, 1, 2, 3]  

    ω_HO = 1.
    data_δω = Vector{Any}(undef, length(N_holds))

    Threads.@threads for i in eachindex(N_holds)
        N_hold = N_holds[i]
        tstart = time()

        ρ_init = make_uniform_density(p.L, 20)

        t_ideal = N_hold < 22 ? t_holds_ideal(ω_HO, N_hold) : t_holds_ideal(ω_HO / 2, N_hold)

        t_quant = quantize_times(t_ideal, Δt_res)

        dens_ideal = simulate_densities(ρ_init, t_ideal;
            ω_HO=ω_HO, hs_p=hs_p, ps=ps, xs=xs, t_tof=t_tof)

        dens_quant = simulate_densities(ρ_init, t_quant;
            ω_HO=ω_HO, hs_p=hs_p, ps=ps, xs=xs, t_tof=t_tof)

        data_δω[i] = (
            ρ_init=ρ_init,
            t_ideal=t_ideal, dens_ideal=dens_ideal,
            t_quant=t_quant, dens_quant=dens_quant
        )

        println("  N=$(N_hold) done in $(round(time()-tstart, digits=2)) s")
    end

    all_data_B = (ω_HO=ω_HO, data=data_δω)

    # -------------------------
    # Processing (CHANGED)
    # -------------------------
    # store error curves per diag_id
    errs_by_diag = Dict{Int, NamedTuple{(:ideal, :quant), Tuple{Vector{Float64}, Vector{Float64}}}}()

    for diag_id in diag_ids
        println("\n--- Processing diag_id = $diag_id ---")

        errs_ideal = Float64[]
        errs_quant = Float64[]

        for (i_N, sample_N) in enumerate(data_δω)
            ρ_init = sample_N.ρ_init
            
            e1, _, _ = reconstruction_error(sample_N.dens_ideal, sample_N.t_ideal, ρ_init;
                xs=xs, t_tof=t_tof, diag_id=diag_id, λmax=p.L-1)

            e2, ρ_meas, ρ_true = reconstruction_error(sample_N.dens_quant, sample_N.t_quant, ρ_init;
                xs=xs, t_tof=t_tof, diag_id=diag_id, λmax=p.L-1)

            push!(errs_ideal, e1)
            push!(errs_quant, e2)

            if N_holds[i_N] == 45 
                λs = 0:(length(ρ_meas)-1)

                # Magnitude comparison (kept)
                p_mag = plot(λs, abs.(ρ_meas), label="measured |ρ|", lw=2, color=:green,
                    xlabel="λ", ylabel="|ρ(λ,λ+k)| (k=$diag_id)")
                plot!(p_mag, λs, abs.(ρ_true), label="truth |ρ|", lw=2, ls=:dash, color=:blue)

                # Phase comparison (principal value in [-π, π])
                # (If you prefer unwrapped phase, tell me and I'll add unwrap.)
                ϕ_meas = angle.(ρ_meas)
                ϕ_true = angle.(ρ_true)

                p_phase = plot(λs, ϕ_meas, label="measured arg(ρ)", lw=2, color=:green,
                    xlabel="λ", ylabel="phase (rad)")
                plot!(p_phase, λs, ϕ_true, label="truth arg(ρ)", lw=2, ls=:dash, color=:blue)

                # Absolute difference (magnitude of complex error)
                p_absdiff = plot(λs, abs.(ρ_meas .- ρ_true), label="abs diff", lw=2, color=:red,
                    xlabel="λ", ylabel="Absolute difference")
                

                display(plot(p_mag, p_phase, p_absdiff,
                    layout=(3,1), size=(800,900),
                    title="diag=$diag_id, N=45"))
            end
        end

        errs_by_diag[diag_id] = (ideal=errs_ideal, quant=errs_quant)

        # per-diagonal plot
        plB = plot(N_holds, errs_ideal, marker=:o, label="ideal",
            xlabel="N holds", ylabel="ϵ_rel",
            title="Timing quantization (Δt=0.1ms) vs ideal — diag=$diag_id",
            framestyle=:box)
        plot!(plB, N_holds, errs_quant, marker=:o, label="quantized (0.1ms)")
        display(plB)
    end

    
# plot 1: ideal
plIdeal = plot(xlabel="N holds", ylabel="ϵ_rel",
    title="Ideal sampling — multiple diagonals",
    framestyle=:box)

# plot 2: quantized
plQuant = plot(xlabel="N holds", ylabel="ϵ_rel",
    title="Quantized sampling (Δt=0.1ms) — multiple diagonals",
    framestyle=:box)

for diag_id in diag_ids
    plot!(plIdeal, N_holds, errs_by_diag[diag_id].ideal,
        marker=:o, label="diag=$diag_id")

    plot!(plQuant, N_holds, errs_by_diag[diag_id].quant,
        marker=:o, label="diag=$diag_id")
end
display(plIdeal)
display(plQuant)


end


############################################################
# TEST C: wrong assumed ω in reconstruction (ω mismatch)
############################################################
if RUN_TEST_C
    println("\n==============================")
    println("TEST C: ω mismatch in reconstruction")
    println("==============================")

    # Fix a single N for this test
    N_fixed = 45
    δωs = vcat(2π .* collect(0:5), 2π .* collect(10:5:20))

    #try multiple diagonals
    diag_ids = [0, 1, 2, 3, 5]  
    nD = length(diag_ids)

    # Storage (CHANGED to 2D: (diag, δω))
    errs_ω = Array{Float64}(undef, nD, length(δωs))
    ρ_meas_store = Array{Any}(undef, nD, length(δωs))
    ρ_true_store = Array{Any}(undef, nD, length(δωs))

    Threads.@threads for i in eachindex(δωs)
        δω = δωs[i]
        ω_HO = (p.ω + δω) / p.ω

        ρ_init = make_uniform_density(p.L, 20)
        t_ideal = N_fixed < 22 ? t_holds_ideal(1., N_fixed) : t_holds_ideal(1. / 2, N_fixed)
        t_holds = quantize_times(t_ideal, Δt_res)

        densities = simulate_densities(ρ_init, t_holds;
            ω_HO=ω_HO, hs_p=hs_p, ps=ps, xs=xs, t_tof=t_tof)

        # CHANGED: compute reconstruction for each diag_id
        for (j, diag_id) in enumerate(diag_ids)
            err, ρ_meas, ρ_true = reconstruction_error(
                densities, t_holds, ρ_init;
                xs=xs, t_tof=t_tof, diag_id=diag_id, λmax=p.L-1
            )

            errs_ω[j, i] = err
            ρ_meas_store[j, i] = ρ_meas
            ρ_true_store[j, i] = ρ_true
        end

        println("Simulation ran for δω = $(round(δω / 2π))")
    end

    
    # Plot: error vs δω for each diagonal (separate figures)
    for (j, diag_id) in enumerate(diag_ids)
        plC = plot(
            xlabel="δω / (2π)",
            ylabel="ϵ_rel",
            title="ω mismatch sensitivity (N=$N_fixed, diag=$diag_id)",
            framestyle=:box
        )

        plot!(plC, δωs ./ 2π, errs_ω[j, :],
            marker=:o, label="diag=$diag_id")

        display(plC)
    end
end
begin
    λs = 0:p.L-2

    # Per-δω diagnostic plots (now also loops over diag)
    for i in eachindex(δωs)
        δω = δωs[i]

        for (j, diag_id) in enumerate(diag_ids)
            ρ_meas = ρ_meas_store[j, i]
            ρ_true = ρ_true_store[j, i]
            λs = 0:(p.L -2 - diag_id)

            p1 = plot(λs, abs.(ρ_meas),
                label="measured", lw=2, color=:green,
                xlabel="λ", ylabel="|ρ(λ,λ+k)| (k=$diag_id)"
            )

            plot!(p1, λs, abs.(ρ_true),
                label="truth", lw=2, ls=:dash, color=:blue
            )

            p2 = plot(λs, abs.(ρ_meas .- ρ_true),
                label="difference", lw=2, color=:red,
                xlabel="λ", ylabel="Absolute error"
            )

            ptot = plot(p1, p2, layout=(2,1), size=(800,600),
                title="δω = $(round(δω / 2π)), diag=$diag_id"
            )

            display(ptot)
        end
    end
end


############################################################
# TEST D: dependence on off-diagonal size (state_extent sweep)
############################################################
if RUN_TEST_D
    println("\n==============================")
    println("TEST D: dependence on state extent")
    println("==============================")

    # Use δω=0 for clarity
    δω = 0.0
    ω_HO = (p.ω + δω) / p.ω

    N_holds = collect(5:1:20)   # samples per period
    diag_id = 0
    state_extents = [ 10, 15, 20, 25]


    # For each extent, compute error vs N
    plD = plot(xlabel="N holds", ylabel="ϵ_rel",
        title="Error vs N for different state extent", framestyle=:box)

    for extent in state_extents
        errs_ext = Array{Any}(undef, length(N_holds))

        Threads.@threads for i in eachindex(N_holds)
            N_hold = N_holds[i]
            ρ_init = make_uniform_density(p.L, extent)
            t_holds = t_holds_ideal(ω_HO, N_hold)
            #t_holds = quantize_times(t_holds, Δt_res)

            densities = simulate_densities(ρ_init, t_holds;
                ω_HO=ω_HO, hs_p=hs_p, ps=ps, xs=xs, t_tof=t_tof)

            err, _, _ = reconstruction_error(densities, t_holds, ρ_init;
                xs=xs, t_tof=t_tof, diag_id=diag_id, λmax=p.L-1)

            errs_ext[i] =  err
        end

        plot!(plD, N_holds, errs_ext, marker=:o, label="extent=$extent")
    end

    display(plD)
end
