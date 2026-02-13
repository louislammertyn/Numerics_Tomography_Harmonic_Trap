function construct_hamiltonians_Heaviside_x(L)
    ## initialisation of lattice and hilbert space ##
    geometry = (L,)
    V = U1FockSpace(geometry, 1,1)
    states = all_states_U1_O(V)
    lattice = Lattice(geometry)


    #### defining conditions ####
    function fhopx(sites_tuple)
        s1, s2 = sites_tuple
        J = hopping_integral(s1[1], s2[1])
        return -J
    end

    function fonsitex(sites_tuple)
        s1, s2 = sites_tuple
        o =  1 : 0
        return s1==s2 ? ((s1[1]-1) +0.5) : zero(ComplexF64)
    end

    
    condition1 = (fhopx, )
    condition2 = (fonsitex,)

    ## make the tensors and with them the operators that form the Hamiltonian ##
    t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_hop = fill_nbody_tensor(t1, lattice, condition1)

    t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_onsite = fill_nbody_tensor(t2, lattice, condition2)


    Hx_hop = nbody_Op(V, lattice, tens_hop)
    Hx_onsite = nbody_Op(V, lattice, tens_onsite)

    Hx_hop_m = calculate_matrix_elements( Hx_hop, states)
    Hx_onsite_m = calculate_matrix_elements( Hx_onsite, states)

    
    return (Hx_onsite_m, Hx_hop_m) 


end

begin
    function construct_hamiltonians_FL_Heaviside_x(p, xz=false, aspectratio=1.)
    ## initialisation of lattice and hilbert space ##
    L = p.L
    geometry = (L,)
    V = U1FockSpace(geometry, 1,1)
    lattice = Lattice(geometry)


    #### defining conditions ####
    function fhopx(sites_tuple)
        s1, s2 = sites_tuple
        j = s2[1] - s1[1]
        if abs(j) != 1
            return zero(ComplexF64)
        end
        J = hopping_integral(s1[1], s2[1])
        return -J  * exp(1im * j * p.ϕ) * p.κ / 2
       
    end

    function fonsitex(sites_tuple)
        s1, s2 = sites_tuple
        return s1==s2 ? ((s1[1]-1) * (p.Δ / p.ω) + 0.5) : zero(ComplexF64)
    end

    
    condition1 = (fhopx, )
    condition2 = (fonsitex,)

    ## make the tensors and with them the operators that form the Hamiltonian ##
    t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_hop = fill_nbody_tensor(t1, lattice, condition1)

    t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_onsite = fill_nbody_tensor(t2, lattice, condition2)

    Hx_hop_m = Matrix(tens_hop.tensor)
    Hx_onsite_m = Matrix(tens_onsite.tensor)

    @assert ishermitian(Hermitian(Hx_hop_m+Hx_onsite_m))

    return (Hx_onsite_m, Hx_hop_m) 


end

function construct_hamiltonians_uniform(L)
    ## initialisation of lattice and hilbert space ##
    geometry = (L,)
    V = U1FockSpace(geometry, 1,1)
    states = all_states_U1_O(V)
    lattice = Lattice(geometry)


    #### defining conditions ####
    function fhopx(sites_tuple)
        s1, s2 = sites_tuple
        J = isone(abs(s1[1] - s2[1])) ? one(ComplexF64) : zero(ComplexF64)
        return -J
    end

    function fonsitex(sites_tuple)
        s1, s2 = sites_tuple
        return s1==s2 ? ((s1[1]-1) +0.5) : zero(ComplexF64)
    end

    
    condition1 = (fhopx, )
    condition2 = (fonsitex,)

    ## make the tensors and with them the operators that form the Hamiltonian ##
    t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_hop = fill_nbody_tensor(t1, lattice, condition1)

    t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_onsite = fill_nbody_tensor(t2, lattice, condition2)


    Hx_hop = nbody_Op(V, lattice, tens_hop)
    Hx_onsite = nbody_Op(V, lattice, tens_onsite)

    Hx_hop_m = calculate_matrix_elements( Hx_hop, states)
    Hx_onsite_m = calculate_matrix_elements( Hx_onsite, states)

    return (Hx_onsite_m, Hx_hop_m) 


end



function construct_hamiltonians_FL_uniform(p)
## initialisation of lattice and hilbert space ##
L = p.L
geometry = (L,)
V = U1FockSpace(geometry, 1,1)
lattice = Lattice(geometry)


#### defining conditions ####
function fhopx(sites_tuple)
    s1, s2 = sites_tuple
    j = s2[1] - s1[1]
    if abs(j) != 1
        return zero(ComplexF64)
    end
    J = one(ComplexF64)
    return -J  * exp(1im * j * p.ϕ) * p.κ 
    
end

function fonsitex(sites_tuple)
    s1, s2 = sites_tuple
    return s1==s2 ? ((s1[1]-1) * (p.Δ / p.ω) + 0.5) : zero(ComplexF64)
end


condition1 = (fhopx, )
condition2 = (fonsitex,)

## make the tensors and with them the operators that form the Hamiltonian ##
t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
tens_hop = fill_nbody_tensor(t1, lattice, condition1)

t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
tens_onsite = fill_nbody_tensor(t2, lattice, condition2)

Hx_hop_m = Matrix(tens_hop.tensor)
Hx_onsite_m = Matrix(tens_onsite.tensor)

@assert ishermitian(Hermitian(Hx_hop_m+Hx_onsite_m))


return (Hx_onsite_m, Hx_hop_m) 

end


function construct_hamiltonians_FL_2drives(p, phi, m, drive_strength)
    H_onsite_m, H_hop1_m = construct_hamiltonians_FL(p,xz,aspectratio)

    function fhop2x(sites_tuple) 
        s1, s2 = sites_tuple
        j = s2[1] - s1[1]
        if abs(j) != m
            return zero(ComplexF64)
        end
        J = hopping_integral(s1[1], s2[1])
        return -J  * exp(1im * sign(j) * phi) * drive_strength / 2
    end

    L = p.L
    geometry = (L,)
    V = U1FockSpace(geometry, 1,1)
    lattice = Lattice(geometry)

    condition = (fhop2x,)

    t = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens = fill_nbody_tensor(t, lattice, condition)

    return (H_onsite_m, H_hop1_m, Matrix(tens.tensor))
    
end

function construct_hamiltonians_FL_coherent(p)
     ## initialisation of lattice and hilbert space ##
    L = p.L
    geometry = (L,)
    V = U1FockSpace(geometry, 1,1)
    lattice = Lattice(geometry)


    #### defining conditions ####
    function fhopx(sites_tuple)
        s1, s2 = sites_tuple
        j = s2[1] - s1[1]
        if abs(j) != 1
            return zero(ComplexF64)
        end
        n = minimum([s1[1],s2[1]])
        J = sqrt((n)*(p.L - (n-1)))
        return -J  * p.κ
       
    end

    function fonsitex(sites_tuple)
        s1, s2 = sites_tuple
        return s1==s2 ? ((s1[1]-1) * (p.Δ / p.ω) + 0.5) : zero(ComplexF64)
    end

    
    condition1 = (fhopx, )
    condition2 = (fonsitex,)

    ## make the tensors and with them the operators that form the Hamiltonian ##
    t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_hop = fill_nbody_tensor(t1, lattice, condition1)

    t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
    tens_onsite = fill_nbody_tensor(t2, lattice, condition2)

    Hx_hop_m = Matrix(tens_hop.tensor)
    Hx_onsite_m = Matrix(tens_onsite.tensor)

    @assert ishermitian(Hermitian(Hx_hop_m+Hx_onsite_m))

    return (Hx_onsite_m, Hx_hop_m) 
end

end