using QuantumFockDynamics
using Plots
using Interpolations
using LinearAlgebra
using OrdinaryDiffEq

include("./utils.jl")

function compare_energies(ω::Float64, T::Float64)
    ħ = 1.054571817e-34   # Js
    kB = 1.380649e-23     # J/K
    return (ħ * ω) / (kB * T)
end

## Experimental parameters ##
begin
    ħ = 1.054571817*1e-34
    m = 1.443e-25
    ω = 2*π * 440
    Δ = 0
    lₕ = sqrt(ħ/(m*ω))
end;
## set constants ##

begin
NT = 100
t_i, t_e= 0., NT * 2 * π
dt = 2*π / 100
ts = t_i:dt:t_e

E_ratio = compare_energies(ω,100e-9)
Temp = ω / E_ratio
β = 1/ (Temp)
L = 20
end;


## define time dependency and initialise the interpolation functions for the time evolution ##
begin

triv(ts) = ones(ComplexF64,length(ts));
interp_f_t = linear_interpolation(ts, f_t(ts));
interp_trivial(t) = 1.;

## set the times at which to save the simulation steps 
strob_ts = tuple(collect(t_i:(2π/ω_d): t_e)...);
save_ts = tuple(collect(t_i:(2π/ω_d)/(2): t_e)...);
end;

begin
## initialisation of lattice and hilbert space ##
geometry = (L,)
V = U1FockSpace(geometry, 1,1)
states = all_states_U1_O(V)
lattice = Lattice(geometry)

## Define the conditions for the hopping structure of the model ##
function hop_int(s1::Int, s2::Int)
    diff = abs(s1-s2)
    return diff==0 ? 1 : sin(diff*π/2)/(diff*π/2)
end



#### defining conditions ####
function fhop_2body(sites_tuple)
    s1, s2 = sites_tuple
    J = hop_int(s1[1],s2[1])
    return J
end

function fonsite_2body(sites_tuple)
    s1, s2 = sites_tuple
    o =  1 : 0
    return s1==s2 ? ((s1[1]-1)  * ω ) : zero(ComplexF64)
end


condition1 = (fhop_2body, )
condition2 = (fonsite_2body,)

## make the tensors and with them the operators that form the Hamiltonian ##
t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
tens_hop = fill_nbody_tensor(t1, lattice, condition1)

t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
tens_onsite = fill_nbody_tensor(t2, lattice, condition2)


Hop = nbody_Op(V, lattice, tens_hop)
H_onsite = nbody_Op(V, lattice, tens_onsite)
end;

begin
Hop_m = calculate_matrix_elements_parallel(states, Hop)
H_onsite_m = calculate_matrix_elements_parallel(states, H_onsite)

H_0 = Hop_m + H_onsite_m

## Choose an initial state (thermal state) ##
ρ = thermal_ρ_matrix(Hop + H_onsite, β, lattice)
end;