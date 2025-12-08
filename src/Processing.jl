function plot_center_of_mass(sol, p)

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

function gif_ρ(sol, p; Nt=1000, fps=20, magfile="rho_mag.gif", phasefile="rho_phase.gif", every=5)
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
