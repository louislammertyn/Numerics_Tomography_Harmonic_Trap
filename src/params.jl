## Experimental parameters ##
include("./utils.jl")

struct Params
    # physical constants
    ħ::Float64
    m::Float64
    lₕ::Float64

    # experimental parameters
    ω::Float64
    Δ::Float64
    T_exp::Float64
    t_i_exp::Float64
    t_e_exp::Float64
    ϕ::Float64
    κ::Float64
    L::Int
    f_t::Function

    # numerical parameters in units ħ=ω=m=1
    t_i::Float64
    t_e::Float64
    ω_d::Float64
    Temp::Float64
    β::Float64
    strob_ts::Tuple
    save_ts::Tuple
end

# Neat printing without long vectors
function Base.show(io::IO, p::Params)
    println(io, "──────────────────────────")
    println(io, "      Simulation Params")
    println(io, "──────────────────────────")

    rpadval(n,v) = println(io, rpad(n,12)," : ",v)

    println(io, "• Physical constants")
    rpadval("ħ", p.ħ)
    rpadval("m", p.m)
    rpadval("lₕ", p.lₕ)

    println(io, "\n• Experimental")
    rpadval("ω", "2π $(p.ω / 2π)" )
    rpadval("Δ", "2π $(p.Δ / 2π)")
    rpadval("T_exp", p.T_exp)
    rpadval("t_i_exp", p.t_i_exp)
    rpadval("t_e_exp", p.t_e_exp)
    rpadval("ϕ", p.ϕ)
    rpadval("κ", "$(p.κ) ħω")
    rpadval("# modes", p.L)

    println(io, "\n• Derived numerical in units ħ=m=ω=1")
    rpadval("t_i", p.t_i)
    rpadval("t_e", p.t_e)
    rpadval("ω_d", p.ω_d)
    rpadval("Temp", p.Temp)
    rpadval("β", p.β)

    println(io, "\n• Time functions")
    rpadval("f_t", "Function")
    rpadval("strob_ts", "Tuple[length=$(length(p.strob_ts))]")
    rpadval("save_ts", "Tuple[length=$(length(p.save_ts))]")

    println(io, "──────────────────────────")
end

# Construct a Params object from experimental parameters only
function make_params(; ω=2π*440, Δ=2π*14.54, κ=0.1, ϕ=0, L=80, 
                     T_exp=230e-9, t_i_exp=0.0, t_e_exp=0.1, f_t_symbol = :smooth)

    ħ = 1.054571817e-34
    m = 1.443e-25
    lₕ = sqrt(ħ/(m*ω))

    t_i = ω * t_i_exp
    t_e = ω * t_e_exp
    ω_d = (ω - Δ)/ω

    # Temperature
    E_ratio = compare_energies(ω, T_exp)
    Temp = 1/E_ratio
    β = 1/Temp

    step(x) = x > 0 ? 1. : -1.
    f_t = if f_t_symbol == :step
        t ->  step(sin(ω_d * t + ϕ))
    elseif f_t_symbol == :smooth
        t -> cos(ω_d * t + ϕ)
    else
        error("Unknown f_t_symbol: $f_t_symbol")
    end

    # stroboscopic times
    strob_ts = tuple(collect(t_i:(2π/ω_d):t_e)...)
    save_ts   = tuple(collect(t_i:(2π/ω_d)/100:t_e)...)

    return Params(ħ, m, lₕ, ω, Δ, T_exp, t_i_exp, t_e_exp, ϕ, κ, L, f_t,
                  t_i, t_e, ω_d, Temp, β, strob_ts, save_ts)
end

function update(p::Params; kwargs...)
    # Extract current experimental parameters
    params = Dict(
        :ω => p.ω,
        :Δ => p.Δ,
        :κ => p.κ,
        :ϕ => p.ϕ,
        :L => p.L,
        :T_exp => p.T_exp,
        :t_i_exp => p.t_i_exp,
        :t_e_exp => p.t_e_exp
    )

    # Update with any new kwargs
    for (k,v) in kwargs
        params[k] = v
    end

    # Call make_params with updated experimental parameters
    return make_params(; ω=params[:ω], Δ=params[:Δ], κ=params[:κ], ϕ=params[:ϕ],
                       L=params[:L], T_exp=params[:T_exp],
                       t_i_exp=params[:t_i_exp], t_e_exp=params[:t_e_exp])
end

# Make a copy of a Params object
function Base.copy(p::Params)
    return Params(
        p.ħ, p.m, p.lₕ,
        p.ω, p.Δ, p.T_exp, p.t_i_exp, p.t_e_exp, p.ϕ, p.κ, p.L, p.f_t,
        p.t_i, p.t_e, p.ω_d, p.Temp, p.β, p.strob_ts, p.save_ts
    )
end

# Copy and update selected fields in one step
function copy_with(p::Params; kwargs...)
    # Start from a copy
    newp = copy(p)

    # Update fields if provided
    for (k,v) in kwargs
        if hasfield(Params, k)
            setfield!(newp, k, v)
        else
            error("Params has no field named $k")
        end
    end

    # Recompute dependent fields if necessary
    # For your struct, best is to call `make_params` with updated experimental params
    return update(newp; kwargs...)
end