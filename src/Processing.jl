function check_conservation_of_probability(sol)
    prob = []
    for ρ in sol.u 
        push!(prob, sum(diag(ρ)))
    end
    return prob
end

function plot_center_of_masses(sol, p)

    ### --- Create empty plots --- ###
    pl = plot(
        xlabel = L"$t\; (ms)$",
        ylabel = L"$\langle \lambda  \rangle$",
        title  = "Center of mass motion λ-space",
        legend = true, grid = true, framestyle = :box
    )
    pl2 = plot(
        xlabel = "t (ms)",
        ylabel = "Center of Mass <λ>",
        title  = "Center of mass micro-motion",
        legend = false, grid = true, framestyle = :box
    )
    pl3 = plot(
        xlabel = L"$t\; (ms)$",
        ylabel = L"$\langle x  \rangle \; (\mu m)$",
        title  = "Center of mass motion x-space",
        legend = true, grid = true, framestyle = :box
    )
    pl4 = plot(
        xlabel = L"$t\; (ms)$",
        ylabel = L"\langle p \rangle_{com}\; \left(\frac{m_{Ru}\,\mu m}{ms}\right)",
        title  = "Center of mass motion p-space",
        legend = true, grid = true, framestyle = :box
    )

    # Pre-allocate
    times = [sol.t[s] / p.ω * 1e3 for s in eachindex(sol.t)]
    l_coms = Float64[]
    x_coms = Float64[]
    p_coms = Float64[]

    # Add legend entries for stroboscopic markers
    scatter!(pl,  [NaN], [NaN], color=:red, marker=:x, label="Stroboscopic")
    scatter!(pl3, [NaN], [NaN], color=:red, marker=:x, label="Stroboscopic")
    scatter!(pl4, [NaN], [NaN], color=:red, marker=:x, label="Stroboscopic")

    i = 1

    ### --- Main loop --- ###
    for s in eachindex(sol.t)
        state = sol.u[s]

        l_com = real(tr(state * Diagonal(0:(p.L-1))))
        x_com = real(tr(state * x_matrix_ho(p.L))) * p.lₕ * 1e6
        p_com = real(tr(state * p_matrix_ho(p.L))) * sqrt(p.m*p.ħ*p.ω) * (1e6  / (p.m*1e3))

        push!(l_coms, l_com)
        push!(x_coms, x_com)
        push!(p_coms, p_com)

        # micromotion window for pl2
        NT = 10
        T_d = (2π / (p.ω_d*p.ω))
        if NT*T_d < (sol.t[s]/p.ω) < (NT+1)*T_d
            scatter!(pl2, [times[s]], [l_com], marker=:x, color=:red)
        end

        # stroboscopic points
        if i ≤ length(p.strob_ts) && isapprox(sol.t[s], p.strob_ts[i]; atol=1e-9)
            scatter!(pl,  [times[s]], [l_com], marker=:x, color=:red, label="", markersize=3)
            scatter!(pl3, [times[s]], [x_com], marker=:x, color=:red, label="", markersize=3)
            scatter!(pl4, [times[s]], [p_com], marker=:x, color=:red, label="", markersize=3)
            i += 1
        end
    end

    ### --- Plot the lines --- ###
    plot!(pl,  times, l_coms, color=:black, label="", lw=1)
    plot!(pl3, times, x_coms, color=:blue,  label="", lw=1)
    plot!(pl4, times, p_coms, color=:green, label="", lw=1)

    return pl, pl2, pl3, pl4
end

function plot_center_of_mass_λ(sol, p)

    ### --- Create empty plots --- ###
    pl = plot(
        xlabel = L"$t\; (ms)$",
        ylabel = L"$\langle \lambda  \rangle$",
        title  = "Center of mass motion λ-space for k=$(p.κ) (ħω)",
        legend = true, grid = true, framestyle = :box
    )

    # Pre-allocate
    times = [sol.t[s] / p.ω * 1e3 for s in eachindex(sol.t)]
    l_coms = Float64[]


    # Add legend entries for stroboscopic markers
    scatter!(pl,  [NaN], [NaN], color=:red, marker=:x, label="Stroboscopic")

    i = 1

    ### --- Main loop --- ###
    for s in eachindex(sol.t)
        state = sol.u[s]


        l_com = real(tr(state * Diagonal(0:(p.L-1))))
        push!(l_coms, l_com)


        # stroboscopic points
        if i ≤ length(p.strob_ts) && isapprox(sol.t[s], p.strob_ts[i]; atol=1e-9)
            scatter!(pl,  [times[s]], [l_com], marker=:x, color=:red, label="", markersize=3)
            i += 1
        end
    end

    ### --- Plot the lines --- ###
    plot!(pl,  times, l_coms, color=:grey, label="Exact", lw=1)
    return pl
end

function plot_center_of_mass_λ(sol, sol_fl, p)

    ### --- Create empty plots --- ###
    pl = plot(
        xlabel = L"$t\; (ms)$",
        ylabel = L"$\langle \lambda  \rangle$",
        title  = "Center of mass motion λ-space for k=$(p.κ) (ħω)",
        legend = true, grid = true, framestyle = :box
    )

    # Pre-allocate
    times = [sol.t[s] / p.ω * 1e3 for s in eachindex(sol.t)]
    l_coms = Float64[]
    l_coms_fl = Float64[]

    # Add legend entries for stroboscopic markers
    scatter!(pl,  [NaN], [NaN], color=:red, marker=:x, label="Stroboscopic")

    i = 1

    ### --- Main loop --- ###
    for s in eachindex(sol.t)
        state = sol.u[s]
        state_fl = sol_fl.u[s]

        l_com = real(tr(state * Diagonal(0:(p.L-1))))
        push!(l_coms, l_com)

        l_com_fl = real(tr(state_fl * Diagonal(0:(p.L-1))))
        push!(l_coms_fl, l_com_fl)

        # stroboscopic points
        if i ≤ length(p.strob_ts) && isapprox(sol.t[s], p.strob_ts[i]; atol=1e-9)
            scatter!(pl,  [times[s]], [l_com], marker=:x, color=:red, label="", markersize=3)
            i += 1
        end
    end

    ### --- Plot the lines --- ###
    plot!(pl,  times, l_coms, color=:orange, label="Exact", lw=1)
    plot!(pl, times, l_coms_fl, color=:green, label="FL approximate", style=:dash, lw=1)
    return pl
end


function gif_x(sol, p; Nt=1000, fps=20, filename="xspaceEv.gif", every=10)
    times = sol.t ./ p.ω .* 1e3  # convert to ms

    x_coms = Float64[]  # dynamically grow as needed

    xs = collect(LinRange(-20., 20., 3000) )
    N = size(sol.u[1],1)
    hs = zeros(Float64, 3000, N)
    for n in 1:N
        hs[:, n] = hermite.(n-1, xs)
    end
    xm = x_matrix_ho(p.L)

    plt = plot(xs * p.lₕ * 1e6, xs,
            xlabel = L"x \; (\mu m)",
            ylabel = L"\rho (x,x)",
            #title = 
            legend = true,
            grid = true,
            lw = 2,
            framestyle=:box,
            color=:black,
            dpi=200,
            #label="",
            xlims=(-20.,20.),
            ylims=(-0.01, 0.15)
        )
    state = similar(sol.u[1])
    scatter!([0.], zeros(1), color=:blue, label="x_com")
    anim = @animate for s in 1:every:Nt
        state = sol.u[s]
        plt[1][1][:y] .= density_x_diag(state, xs, hs)
        # Compute observable
        x_com = real(tr(state * xm)) 
        push!(x_coms, x_com)

        # Overlay x_com evolution
        start_idx = max(1, s-20)
        plt[1][2][:x] = x_coms[start_idx:end] * p.lₕ * 1e6
        plt[1][2][:y] = zeros(length(x_coms[start_idx:end]))
        plt[1][:title] = "real space density at t=$(Int(round((sol.t[s] / p.ω) *1e3, digits=0))) ms"
        #plot!( x_coms[start_idx:end], zeros(length(x_coms[start_idx:end])),marker=:x, color=:blue, label="x_com")
    end

    gif(anim, filename, fps=fps)
end

function gif_ρ(sol, p; Nt=1000, fps=20, magfile="rho_mag.gif", phasefile=false, every=5)
    times = sol.t ./ p.ω .* 1e3  # convert to ms

    state = similar(sol.u[1])
    data_mag = similar(state)
    data_phase = similar(state)
    
    # Modulus
    axes = 1:p.L
    hm_mag = heatmap(axes, axes , zeros(p.L, p.L),
        xlabel=L"λ'", ylabel=L"λ",
        color=:viridis,
        clims=(0,maximum(abs.(sol.u[1]))),
        dpi=200
    )

    hm_mag[1][1][:z] = abs.(sol.u[1])

    anim_mag = @animate for s in 1:every:Nt
        state .= sol.u[s]
        data_mag .= abs.(state)
        hm_mag[1][1][:z] .= data_mag
        hm_mag[1][:title] = "|ρ(λ,λ')| at t=$(Int(round(times[s]; digits=0))) ms"
    end

    gif(anim_mag, magfile, fps=fps)

    if phasefile == false 
        return nothing
    else 
        # phase
        hm_phase = heatmap(1:p.L, 1:p.L, zeros(p.L, p.L),
            xlabel=L"λ'", ylabel=L"λ",
            color=:cyclic_wrwbw_40_90_c42_n256_s25,
            clims=(0, 2π),
            dpi=200
        )
        hm_phase[1][1][:z] = angle.( sol.u[1])
        anim_phase = @animate for s in 1:every:Nt
            state .=sol.u[s]
            data_phase .= angle.(state)

            hm_phase[1][1][:z] .= data_phase
            hm_phase[1][:title] = "θ = arg(ρ(λ,λ')) at t=$(Int(round(times[s]; digits=0))) ms"
        end

        gif(anim_phase, phasefile, fps=fps)
    end
end

function gif_ρ_diag(sol, p; Nt=1000, fps=20, magfile="rho_mag.gif", phasefile=false, every=5)
    times = sol.t ./ p.ω .* 1e3  # convert to ms

    state = diag(sol.u[1])
    data_mag = similar(state)
    data_phase = similar(state)
    # Modulus
    axes = 1:p.L
    hm_mag = scatter(axes,  zeros(Float64, length(state)),
        xlabel=L"λ", ylabel=L"|ρ(λ,λ)|",
        color=:orange,
        dpi=200, marker=:+,
        ylims=(0., maximum(abs.(state))),
        legend=false
    )

    anim_mag = @animate for s in 1:every:Nt
        state .= diag(sol.u[s])
        data_mag .= abs.(state)
        hm_mag[1][1][:y] .= data_mag
        hm_mag[1][:title] = "for k=(p.κ) (ħω) Diagonal elements of ρ(λ, λ) at t=$(Int(round(times[s]; digits=0))) ms"
    end

    gif(anim_mag, magfile, fps=fps)

    if phasefile == false 
        return nothing
    else 
        # phase
        hm_phase = scatter(axes,  angle.(state),
        xlabel=L"λ", ylabel=L"arg(ρ(λ,λ))",
        color=:red,
        dpi=200
    )
        
        anim_phase = @animate for s in 1:every:Nt
            state .= diag(sol.u[s])
            data_phase .= angle.(state)

            hm_phase[1][1][:y] .= data_phase
            hm_phase[1][:title] = "θ = arg(ρ(λ,λ)) at t=$(Int(round(times[s]; digits=0))) ms"
        end

        gif(anim_phase, phasefile, fps=fps)
    end
end
function gif_ρ_diag(sol, sol_FL, p; Nt=5000, fps=20, magfile="rho_mag.gif", phasefile=false, every=10)
    times = sol.t ./ p.ω .* 1e3  # convert to ms
    if Nt> length(times)
        Nt= length(times)
    end

    state = diag(sol.u[1])
    data_mag = similar(state)

    # Modulus
    axes = 1:p.L
    hm_mag = scatter(axes,  zeros(Float64, length(state)),
        xlabel=L"λ", ylabel=L"|ρ(λ,λ)|",
        color=:orange,
        dpi=200, marker=:+,
        ylims=(0., maximum(abs.(state))),
        label=L"U_{exact}"
    )
    scatter!(hm_mag, axes,  zeros(Float64, length(state)),
        xlabel=L"λ", ylabel=L"|ρ(λ,λ)|",
        color=:green,
        dpi=200, marker=:x,
        ylims=(0., maximum(abs.(state))), 
        label=L"U_{Floquet}^{(0)}"
    )

    anim_mag = @animate for s in 1:every:Nt
        state .= diag(sol.u[s])
        data_mag .= abs.(state)
        hm_mag[1][1][:y] .= data_mag
        hm_mag[1][:title] = "Diagonal elements of ρ(λ, λ) at t=$(Int(round(times[s]; digits=0))) ms"

        state .= diag(sol_FL.u[s])
        data_mag .= abs.(state)
        hm_mag[1][2][:y] .= data_mag
    end

    gif(anim_mag, magfile, fps=fps)

    return nothing
end

function gif_ρ_diag(
    sol,
    sol_FL,
    p,
    diag_idx=0,
    ;
    show_diff=false,
    Nt=5000,
    fps=20,
    outfile="rho_diag.gif",
    every=10,
)

    times = sol.t ./ p.ω .* 1e3

    Nt = min(Nt, length(times))

    d0 = diag(sol.u[1], diag_idx)

    x = 1:length(d0)

    if show_diff
        ymax = maximum([
            maximum(abs.(diag(sol.u[s], diag_idx)-
                         diag(sol_FL.u[s], diag_idx)))
            for s in 1:every:Nt
        ])

        pl = scatter(
            x, zeros(length(d0)),
            xlabel = "index",
            ylabel = "|Δρ|",
            marker = :circle,
            color = :red,
            ylims = (0, 1.05ymax),
            label = "|ρ_exact - ρ_FL|",
            framestyle = :box,
        )

        anim = @animate for s in 1:every:Nt
            Δ = abs.(diag(sol.u[s], diag_idx) .-
                     diag(sol_FL.u[s], diag_idx))

            pl[1][1][:y] .= Δ
            pl[1][:title] =
                "Diagonal k=$(diag_idx), t=$(round(times[s],digits=1)) ms"
        end

    else

        ymax = maximum([
            maximum(abs.(diag(sol.u[s], diag_idx)))
            for s in 1:every:Nt
        ])

        ymax = max(
            ymax,
            maximum([
                maximum(abs.(diag(sol_FL.u[s], diag_idx)))
                for s in 1:every:Nt
            ])
        )

        pl = scatter(
            x, zeros(length(d0)),
            marker = :circle,
            color = :orange,
            xlabel = "index",
            ylabel = "|ρ|",
            ylims = (0, 1.05ymax),
            label = "Exact",
            framestyle = :box,
        )

        scatter!(
            pl,
            x, zeros(length(d0)),
            marker = :x,
            color = :green,
            label = "Floquet",
        )

        anim = @animate for s in 1:every:Nt

            pl[1][1][:y] .= abs.(diag(sol.u[s], diag_idx))
            pl[1][2][:y] .= abs.(diag(sol_FL.u[s], diag_idx))

            pl[1][:title] =
                "Diagonal k=$(diag_idx), t=$(round(times[s],digits=1)) ms"
        end
    end

    gif(anim, outfile, fps=fps)

    nothing
end

function gif_ρ_diff(sol, sol_FL, p; dt=nothing, every=5, Nt=5000, fps=10,
                     magfile="rho_diff_mag.gif", phasefile=false)

    times = sol.t ./ p.ω .* 1e3
    Nt = min(Nt, length(times))

    frames = Int[]
    if dt !== nothing
        t0, tmax = times[1], times[Nt]
        n_frames = floor(Int, (tmax - t0) / dt) + 1
        frames = Vector{Int}(undef, n_frames)
        @inbounds for n in 0:n_frames-1
            target = t0 + n*dt
            i = searchsortedfirst(view(times, 1:Nt), target)
            if i == 1
                frames[n+1] = 1
            elseif i > Nt
                frames[n+1] = Nt
            else
                frames[n+1] = abs(times[i] - target) < abs(times[i-1] - target) ? i : i - 1
            end
        end
        unique!(frames)
    else
        frames = collect(1:every:Nt)
    end

    state, state_FL, Δ = similar(sol.u[1]), similar(sol_FL.u[1]), similar(sol.u[1])
    data_mag = similar(Δ, Float64)
    data_phase = similar(Δ, Float64)

    zmax = 0.0
    for s in frames
        Δ .= sol.u[s] .- sol_FL.u[s]
        zmax = max(zmax, maximum(abs.(Δ)))
    end

    axes = 1:p.L

    # --- Modulus ---
    hm_mag = heatmap(axes, axes, zeros(p.L, p.L),
        xlabel=L"λ'", ylabel=L"λ",
        color=:viridis,
        clims=(0, zmax),
        dpi=200
    )

    # seed the z-surface with a real matrix BEFORE broadcasting into it
    state .= sol.u[frames[1]]
    state_FL .= sol_FL.u[frames[1]]
    Δ .= state .- state_FL
    hm_mag[1][1][:z] = abs.(Δ)

    anim_mag = @animate for s in frames
        state .= sol.u[s]
        state_FL .= sol_FL.u[s]
        Δ .= state .- state_FL
        data_mag .= abs.(Δ)

        hm_mag[1][1][:z] .= data_mag
        hm_mag[1][:title] = "|ρ_exact - ρ_Floquet| at t=$(round(times[s]; digits=2)) ms"
    end

    gif(anim_mag, magfile, fps=fps)

    if phasefile == false
        return nothing
    else
        hm_phase = heatmap(axes, axes, zeros(p.L, p.L),
            xlabel=L"λ'", ylabel=L"λ",
            color=:cyclic_wrwbw_40_90_c42_n256_s25,
            clims=(0, 2π),
            dpi=200
        )

        # seed here too
        state .= sol.u[frames[1]]
        state_FL .= sol_FL.u[frames[1]]
        Δ .= state .- state_FL
        hm_phase[1][1][:z] = angle.(Δ)

        anim_phase = @animate for s in frames
            state .= sol.u[s]
            state_FL .= sol_FL.u[s]
            Δ .= state .- state_FL
            data_phase .= angle.(Δ)

            hm_phase[1][1][:z] .= data_phase
            hm_phase[1][:title] = "θ = arg(ρ_exact - ρ_Floquet) at t=$(round(times[s]; digits=2)) ms"
        end

        gif(anim_phase, phasefile, fps=fps)
    end
end