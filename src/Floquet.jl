
function Floquet_operator_num_int(
    terms::Vector{Matrix{ComplexF64}},
    coeffs::Vector{<:Function},
    T::Float64;
    atol=1e-8,
    dt_initial=0.01,
    dt_min=1e-12,
    dt_max=T/ 200,
)

    t0, tf = 0, T

    dim = size(terms[1], 1)
    U = Matrix{ComplexF64}(I, dim, dim)

    H = zeros(ComplexF64, dim, dim)

    dt = dt_initial
    t = t0

    while t < tf

        dt = min(dt, tf - t)

        # One full step
        fill!(H, 0)
        @inbounds for (M,f) in zip(terms, coeffs)
            α = f(t)
            H .+= α .* M
        end

        U1 = exp(-im * dt * H)

        
        # two half steps
        fill!(H, 0)
        @inbounds for (M,f) in zip(terms, coeffs)
            α = f(t)
            H .+= α .* M
        end
        Uhalf1 = exp(-im * (dt/2) * H)

        fill!(H, 0)
        tm = t + dt/2
        @inbounds for (M,f) in zip(terms, coeffs)
            α = f(tm)
            H .+= α .* M
        end
        Uhalf2 = exp(-im * (dt/2) * H)

        U2 = Uhalf2 * Uhalf1

        # local error estimate
        err = opnorm(U2 - U1)

        if err < atol
            # accept step
            U = U2 * U
            t += dt

            # timestep increase
            if err < atol/10
                dt = min(2dt, dt_max)
            end
        else
            # reject step
            dt /= 2
            if dt < dt_min
                error("Minimum timestep reached.")
            end
        end
    end

    return U
end

function Floquet_operator_seq(
    patterns::Vector{Matrix{ComplexF64}},
    T::Float64
)
    nsteps = length(patterns)
    dt = T/nsteps
    
    dim = size(patterns[1])[1]

    U = Matrix{ComplexF64}(I, dim, dim)
    

    for pattern in patterns
        dU = exp(-1im * dt * pattern)
        U = dU *  U
    end

    return U
end


