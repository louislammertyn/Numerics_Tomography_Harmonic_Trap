include("./utils.jl")

function tof_integral(x, λ, λ_, tof)
    ps = LinRange(-50, 50, 5000)
    p_hermite_λ = hermite.(λ, ps) .* (1im)^λ
    p_hermite_λ_ =  hermite.(λ_, ps) .* (1im)^λ_ 

    exponential = (1im .* ((ps.* x)  .-  (ps.^2 .* tof ./ 2))) .|> exp

    I = exponential .* p_hermite_λ
    I_ =  conj.(exponential .* p_hermite_λ_)
    return first(trapz(ps, I)) * first(trapz(ps, I_))
end

xs = LinRange(-200, 200, 1000)
pl = plot()
for i in 10:11
    I = tof_integral.(xs, i,i, 12)
    plot!(pl, xs, real.(I))
    plot!(pl, xs, imag.(I))
end
display(pl)