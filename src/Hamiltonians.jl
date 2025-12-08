function construct_hamiltonians(L, xz=false, aspectratio=1.)
    ## initialisation of lattice and hilbert space ##
    geometry = (L,)
    V = U1FockSpace(geometry, 1,1)
    states = all_states_U1_O(V)
    lattice = Lattice(geometry)


    #### defining conditions ####
    function fhopx(sites_tuple)
        s1, s2 = sites_tuple
        J = hopping_integral(s1[1], s2[1])
        return J
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

    Hx_hop_m = calculate_matrix_elements_parallel(states, Hx_hop)
    Hx_onsite_m = calculate_matrix_elements_parallel(states, Hx_onsite)

    if xz 
        #### defining conditions ####
        function fonsitez(sites_tuple)
            s1, s2 = sites_tuple
            o =  1 : 0
            return s1==s2 ? ((s1[1]-1) * (aspectratio) +0.5) : zero(ComplexF64)
        end

        condition = (fonsitex,)
        t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
        tens_onsite = fill_nbody_tensor(t2, lattice, condition)


        Hz_hop = nbody_Op(V, lattice, tens_hop)
        Hz_onsite = nbody_Op(V, lattice, tens_onsite)

        Hz_hop_m = calculate_matrix_elements_parallel(states, Hz_hop)
        Hz_onsite_m = calculate_matrix_elements_parallel(states, Hz_onsite)

        return (Hx_onsite_m, Hx_hop_m), ( Hz_onsite_m, Hz_hop_m)

    else
        return (Hx_onsite_m, Hx_hop_m) 
    end

end
    begin
    

end