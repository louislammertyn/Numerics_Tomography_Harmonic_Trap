# these are in particular for varying 

function run_Exact(p)
    Hx = construct_hamiltonians_Heaviside_x(p.L) 
    f_ts = (t-> one(ComplexF64), p.f_t)
    ops = Hx

    t1 = time()
    sol = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops, f_ts, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
    t2 = time()

    println("Simulation time for Exact with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")
end

function run_Floquet(p)
    
   
    # set up Floquet Hamiltonians 
    Hx_FL = construct_hamiltonians_FL_coherent(p)
    Hx_FL = Hx_FL[1] + Hx_FL[2]


    ## Time evolution where the tuples indicate the time dependent functions and their corresponding operators ## 
    f_ts_FL = (t-> one(ComplexF64), )
    ops_FL = (Hx_FL,)

    t1 = time()
    sol_FL = Time_Evolution_thermal_ρ_TD_VN(copy(ρ), ops_FL, f_ts_FL, (p.t_i,p.t_e), p.save_ts; rtol = 1e-10, atol = 1e-10, solver = Vern7())
    t2 = time()

    println("Simulation time for FL with $(p.L) modes is $(round((t2-t1)/60, digits=5)) mins")


    return sol_FL
end

function run_both(p,k)
    sol = run_Exact(p, k)
    sol_FL = run_Floquet(p,k)
    return sol, sol_FL
end