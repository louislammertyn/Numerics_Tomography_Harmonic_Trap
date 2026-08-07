
using Revise
using LinearAlgebra, Plots
using ExpSynthDim
using Statistics

BLAS.set_num_threads(1)


############################################################
# Controls
############################################################

RUN_TEST_A = true   # error vs N, ideal sampling
RUN_TEST_B = true   # timing quantization
RUN_TEST_C = true   # wrong assumed ω
RUN_TEST_D = true   # state extent / coherence size


############################################################
# Parameters
############################################################

p = make_params(; L=50)

# These are only needed for any diagnostics/plots that use
# the spatial/momentum grids directly.
xs = LinRange(-60, 60, 5000)
ps = LinRange(-60, 60, 5000)

hs = hcat([
    hermite.(n, xs)
    for n in 0:p.L-1
]...)

hs_p = hcat([
    (-1im)^n .* hermite.(n, ps)
    for n in 0:p.L-1
]...)

t_tof = 0.001 * p.ω

# Hardware timing resolution:
Δt_res = 0.1e-3 * p.ω


############################################################
# Helper: initial state
############################################################

function make_uniform_density(L, extent)

    ρ = zeros(ComplexF64, L, L)

    @inbounds for j in 1:extent
        for k in 1:extent
            ρ[j, k] = 1 / extent
        end
    end

    return ρ
end


############################################################
# Helper: calculate reconstruction error
############################################################

function tomography_error(
    ρ_tomo,
    ρ_true;
    diag_id=0,
)

    measured = ρ_tomo[diagind(ρ_tomo, diag_id)]
    truth    = ρ_true[diagind(ρ_true, diag_id)]

    # Only compare where the reconstructed diagonal exists
    n = min(length(measured), length(truth))

    measured = measured[1:n]
    truth    = truth[1:n]

    denom = norm(truth)

    err = denom > 0 ?
        norm(measured - truth) / denom :
        norm(measured)

    return err, measured, truth
end


############################################################
# Helper: run tomography and calculate errors for
# multiple diagonals
############################################################

function run_test_tomography(
    ρ_init,
    N_hold;
    diag_ids,
    t_tof,
    Δt,
    ω_actual=1.0,
   
    quantize=true,
    α=1e-2,
    σ_ker=12.,
    σ_meas=12.,
    noise_level=0.0,
)

    ρ_tomo = run_tomography(
        ρ_init,
        p.L-1,
        t_tof,
        Δt;

        diags=diag_ids,

        α=α,
        σ_ker=σ_ker,
        σ_meas=σ_meas,
        noise_level=noise_level,

        N_holds=N_hold,
        ω=ω_actual,
        quantize = quantize
    )

    errors = Dict{Int, Float64}()
    measurements = Dict{Int, Vector}()
    truths = Dict{Int, Vector}()

    for diag_id in diag_ids

        err, measured, truth = tomography_error(
            ρ_tomo,
            ρ_init;
            diag_id=diag_id,
        )

        errors[diag_id] = err
        measurements[diag_id] = measured
        truths[diag_id] = truth
    end

    return (
        ρ_tomo=ρ_tomo,
        errors=errors,
        measurements=measurements,
        truths=truths,
    )
end


############################################################
# TEST A
#
# Error vs N
# Ideal sampling, correct ω
############################################################

if RUN_TEST_A

    println("\n==============================")
    println("TEST A: error vs N (ideal sampling)")
    println("==============================")

    N_holds = collect(10:30)

    diag_ids = [0, 1, 2, 5]

    ω_actual = 1.0

    ρ_init = make_uniform_density(p.L, 20)

    # One error vector per diagonal
    errs_by_diag = Dict(
        diag_id => zeros(Float64, length(N_holds))
        for diag_id in diag_ids
    )

    Threads.@threads for i in eachindex(N_holds)

        N_hold = N_holds[i]

        tstart = time()

        result = run_test_tomography(
            ρ_init,
            N_hold;

            diag_ids=diag_ids,

            t_tof=t_tof,
            Δt=Δt_res,

            ω_actual=ω_actual,

            # IMPORTANT:
            # Test A is supposed to use ideal times.
            quantize=false,
        )

        for diag_id in diag_ids
            errs_by_diag[diag_id][i] =
                result.errors[diag_id]
        end

        println(
            "  N=$(N_hold) done in ",
            round(time() - tstart; digits=2),
            " s"
        )
    end


    # Plot each diagonal separately

    for diag_id in diag_ids

        plA = plot(
            N_holds,
            errs_by_diag[diag_id],

            marker=:o,

            xlabel="N holds (samples per period)",
            ylabel="ϵ_rel",

            title="Error vs N — diag=$diag_id",

            framestyle=:box,
            label="",
        )

        display(plA)
    end


    # Combined plot

    plAll = plot(
        xlabel="N holds (samples per period)",
        ylabel="ϵ_rel",

        title="Error vs N — multiple diagonals",

        framestyle=:box,
    )

    for diag_id in diag_ids

        plot!(
            plAll,
            N_holds,
            errs_by_diag[diag_id],

            marker=:o,

            label="diag=$diag_id",
        )
    end

    display(plAll)
end


############################################################
# TEST B
#
# Timing quantization
#
# Compare:
#
#   ideal times
#       vs
#   quantized times
############################################################

if RUN_TEST_B

    println("\n==============================")
    println("TEST B: timing quantization")
    println("==============================")

    N_holds = vcat(
        collect(5:5:20),
        [22],
        [25, 30, 35, 40, 45],
    )

    diag_ids = [0, 1, 2, 3]

    ω_actual = 1.0

    ρ_init = make_uniform_density(p.L, 20)

    errs_ideal = Dict(
        diag_id => zeros(Float64, length(N_holds))
        for diag_id in diag_ids
    )

    errs_quant = Dict(
        diag_id => zeros(Float64, length(N_holds))
        for diag_id in diag_ids
    )


    Threads.@threads for i in eachindex(N_holds)

        N_hold = N_holds[i]

        tstart = time()


        ####################################################
        # Ideal timing
        ####################################################

        result_ideal = run_test_tomography(
            ρ_init,
            N_hold;

            diag_ids=diag_ids,

            t_tof=t_tof,
            Δt=Δt_res,

            ω_actual=ω_actual,

            quantize=false,
        )


        ####################################################
        # Quantized timing
        ####################################################

        result_quant = run_test_tomography(
            ρ_init,
            N_hold;

            diag_ids=diag_ids,

            t_tof=t_tof,
            Δt=Δt_res,

            ω_actual=ω_actual,

            quantize=true,
        )


        ####################################################
        # Store errors
        ####################################################

        for diag_id in diag_ids

            errs_ideal[diag_id][i] =
                result_ideal.errors[diag_id]

            errs_quant[diag_id][i] =
                result_quant.errors[diag_id]
        end


        println(
            "  N=$(N_hold) done in ",
            round(time() - tstart; digits=2),
            " s"
        )
    end


    ########################################################
    # Plot
    ########################################################

    for diag_id in diag_ids

        plB = plot(
            N_holds,
            errs_ideal[diag_id],

            marker=:o,
            label="ideal",

            xlabel="N holds",
            ylabel="ϵ_rel",

            title="Timing quantization — diag=$diag_id",

            framestyle=:box,
        )

        plot!(
            plB,
            N_holds,
            errs_quant[diag_id],

            marker=:o,
            label="quantized (0.1 ms)",
        )

        display(plB)
    end


    ########################################################
    # Combined plots
    ########################################################

    plIdeal = plot(
        xlabel="N holds",
        ylabel="ϵ_rel",

        title="Ideal sampling — multiple diagonals",

        framestyle=:box,
    )

    plQuant = plot(
        xlabel="N holds",
        ylabel="ϵ_rel",

        title="Quantized sampling — multiple diagonals",

        framestyle=:box,
    )


    for diag_id in diag_ids

        plot!(
            plIdeal,
            N_holds,
            errs_ideal[diag_id],

            marker=:o,
            label="diag=$diag_id",
        )

        plot!(
            plQuant,
            N_holds,
            errs_quant[diag_id],

            marker=:o,
            label="diag=$diag_id",
        )
    end


    display(plIdeal)
    display(plQuant)
end


############################################################
# TEST C
#
# Wrong assumed ω in reconstruction
############################################################

if RUN_TEST_C

    println("\n==============================")
    println("TEST C: ω mismatch")
    println("==============================")


    N_fixed = 45

    δωs = vcat(
        2π .* collect(0:5),
        2π .* collect(10:5:20),
    )

    diag_ids = [0, 1, 2, 3, 5]

    nD = length(diag_ids)


    ρ_init = make_uniform_density(p.L, 20)


    ########################################################
    # Storage
    ########################################################

    errs_ω = Array{
        Float64
    }(
        undef,
        nD,
        length(δωs),
    )

    ρ_meas_store = Array{
        Any
    }(
        undef,
        nD,
        length(δωs),
    )

    ρ_true_store = Array{
        Any
    }(
        undef,
        nD,
        length(δωs),
    )


    ########################################################
    # Run simulations
    ########################################################

    Threads.@threads for i in eachindex(δωs)

        δω = δωs[i]

        tstart = time()


        ####################################################
        # Actual physical frequency
        ####################################################

        ω_actual = 1.0


        ####################################################
        # Frequency assumed during reconstruction
        ####################################################

        ω_reconstruct =
            (p.ω + δω) / p.ω


        ####################################################
        # Run tomography
        ####################################################

        result = run_test_tomography(
            ρ_init,
            N_fixed;

            diag_ids=diag_ids,

            t_tof=t_tof,
            Δt=Δt_res,

            ω_actual=ω_reconstruct,
            
            quantize=true,
        )


        ####################################################
        # Store results
        ####################################################

        for (j, diag_id) in enumerate(diag_ids)

            errs_ω[j, i] =
                result.errors[diag_id]

            ρ_meas_store[j, i] =
                result.measurements[diag_id]

            ρ_true_store[j, i] =
                result.truths[diag_id]
        end


        println(
            "Simulation ran for δω = ",
            round(δω / 2π),
            " in ",
            round(time() - tstart; digits=2),
            " s"
        )
    end


    ########################################################
    # Error vs δω
    ########################################################

    for (j, diag_id) in enumerate(diag_ids)

        plC = plot(
            δωs ./ 2π,
            errs_ω[j, :],

            marker=:o,

            xlabel="δω / (2π)",
            ylabel="ϵ_rel",

            title="ω mismatch sensitivity " *
                  "(N=$N_fixed, diag=$diag_id)",

            framestyle=:box,
            label="diag=$diag_id",
        )

        display(plC)
    end


    ########################################################
    # Diagnostic plots
    ########################################################

    for i in eachindex(δωs)

        δω = δωs[i]

        for (j, diag_id) in enumerate(diag_ids)

            ρ_meas = ρ_meas_store[j, i]
            ρ_true = ρ_true_store[j, i]

            n = min(
                length(ρ_meas),
                length(ρ_true),
            )

            ρ_meas = ρ_meas[1:n]
            ρ_true = ρ_true[1:n]

            λs = 0:(n-1)


            ################################################
            # Magnitude
            ################################################

            p1 = plot(
                λs,
                abs.(ρ_meas),

                label="measured",
                lw=2,

                xlabel="λ",
                ylabel="|ρ(λ,λ+k)| (k=$diag_id)",
            )

            plot!(
                p1,
                λs,
                abs.(ρ_true),

                label="truth",
                lw=2,
                ls=:dash,
            )


            ################################################
            # Difference
            ################################################

            p2 = plot(
                λs,
                abs.(ρ_meas .- ρ_true),

                label="difference",
                lw=2,

                xlabel="λ",
                ylabel="Absolute error",
            )


            ################################################
            # Combined
            ################################################

            ptot = plot(
                p1,
                p2,

                layout=(2,1),
                size=(800,600),

                title=
                    "δω = $(round(δω / 2π)), " *
                    "diag=$diag_id",
            )

            display(ptot)
        end
    end
end


############################################################
# TEST D
#
# Dependence on state extent
############################################################

if RUN_TEST_D

    println("\n==============================")
    println("TEST D: dependence on state extent")
    println("==============================")


    δω = 0.0

    ω_actual = 1.0
    ω_reconstruct = 1.0

    N_holds = collect(5:20)

    diag_id = 0

    state_extents = [
        10,
        15,
        20,
        25,
    ]


    ########################################################
    # Plot
    ########################################################

    plD = plot(
        xlabel="N holds",
        ylabel="ϵ_rel",

        title="Error vs N for different state extent",

        framestyle=:box,
    )


    ########################################################
    # Sweep extent
    ########################################################

    for extent in state_extents

        errs_ext =
            zeros(Float64, length(N_holds))


        Threads.@threads for i in eachindex(N_holds)

            N_hold = N_holds[i]


            ################################################
            # State
            ################################################

            ρ_init =
                make_uniform_density(
                    p.L,
                    extent,
                )


            ################################################
            # Tomography
            ################################################

            result = run_test_tomography(
                ρ_init,
                N_hold;

                diag_ids=[diag_id],

                t_tof=t_tof,
                Δt=Δt_res,

                ω_actual=ω_reconstruct,

                # Original Test D uses ideal timing
                quantize=false,
            )


            ################################################
            # Error
            ################################################

            errs_ext[i] =
                result.errors[diag_id]
        end


        ####################################################
        # Add curve
        ####################################################

        plot!(
            plD,
            N_holds,
            errs_ext,

            marker=:o,

            label="extent=$extent",
        )
    end


    display(plD)
end

