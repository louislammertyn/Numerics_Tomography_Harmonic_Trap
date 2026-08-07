

function Von_Neumann_TDM!(dψ, ψ, (tmp, Ops_t), t)
    N = size(tmp, 1)
    O = reshape(ψ, N, N)
    dρ = reshape(dψ, N, N)
    fill!(dψ, 0.0 + 0.0im)

    for Op_t in Ops_t
        Op = Op_t(t)
        α = -im 

        mul!(tmp, Op, O)          # tmp = H*O
        axpy!(α, tmp, dρ)        # dρ += α * tmp  (no allocation)

        mul!(tmp, O, Op)          # tmp = O*H
        axpy!(-α, tmp, dρ)       # dρ -= α * tmp
    end

    return nothing
end


function Time_Evolution_TDM_VN(init::Matrix{ComplexF64},
                           ops_ts::Tuple{Vararg{<:Function, N}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Tsit5()) where {N, M}
    prob = ODEProblem(Von_Neumann_TDM!, init, tspan, (similar(init), ops_ts))
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end

