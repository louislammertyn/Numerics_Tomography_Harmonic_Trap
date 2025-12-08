using FockSpace
using Plots, ColorSchemes
using Interpolations
using LinearAlgebra
using OrdinaryDiffEq

## set constants ##
t_i, t_e= 0., 0.2
dt = 1e-3
ts = t_i:dt:t_e
ϕs = LinRange(0, π/2, 2)
colors = get(ColorSchemes.darkrainbow, range(0,1, length=length(ϕs))) |> ColorScheme
ω_d = 101
ω = 100
κ = 10


pl = plot(
    xlabel = "t",           # replace "units" with physical units if any
    ylabel = "Center of Mass ", 
    title = "center of mass motion",
    grid = true,
    framestyle = :box,
    dpi=300
);
strob, nonstrob= true, true

for (i,ϕ) in enumerate(ϕs)


## define time dependency and initialise the interpolation functions for the time evolution ##
f_t(ts) = κ .* cos.(ω_d .*ts.+ϕ);
triv(ts) = ones(ComplexF64,length(ts));
interp_f_t(t) = linear_interpolation(ts, f_t(ts))(t);
interp_trivial(t) = 1.;

## set the times at which to save the simulation steps 
strob_ts = tuple(collect(t_i:(2π/ω): t_e)...);
save_ts = tuple(collect(t_i:(2π/ω)/10: t_e)...);

## initialisation of lattice and hilbert space ##
geometry = (20,)
V = U1FockSpace(geometry, 1,1)
states = all_states_U1_O(V)
lattice = Lattice(geometry)

## Define the conditions for the hopping structure of the model ##
function hop_int(s1::Int, s2::Int)
    diff = abs(s1-s2)
    return diff==0 ? 1 : sin(diff*π/2)/(diff*π/2)
end

hop_int(1,9);
scatter(-10:10, hop_int.(0, -10:10));

#### defining conditions ####
function fhop_2body(sites_tuple)
    s1, s2 = sites_tuple
    J = hop_int(s1[1],s2[1])
    return J
end

function fonsite_2body(sites_tuple)
    s1, s2 = sites_tuple
    o = s1==s2 ? 1 : 0
    return s1[1] * o * ω
end


condition1 = (fhop_2body, )
condition2 = (fonsite_2body,)

## make the tensors and with them the operators that form the Hamiltonian ##
tens_hop = fill_nbody_tensor(V, lattice, 2, condition1)
tens_onsite = fill_nbody_tensor(V, lattice, 2, condition2)

Hop = two_body_Op(V, lattice, tens_hop)
H_onsite = two_body_Op(V, lattice, tens_onsite)

Hop_m = calculate_matrix_elements_parallel(states, Hop)
H_onsite_m = calculate_matrix_elements_parallel(states, H_onsite)

H_0 = Hop_m + H_onsite_m

## Choose an initial state (gs of the non driven model) ##
es, vs = eigen(Hermitian(H_onsite_m))
gs_v = vs[:,1]
gs = create_MFS(gs_v, states)


## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
interps = [interp_trivial, interp_f_t]
ops = [H_onsite_m, Hop_m]

typeof((t_i,t_e))
sol = Time_Evolution_TD(gs_v, (ops, interps), (t_i,t_e), save_ts; rtol = 1e-9, atol = 1e-9, solver = Vern7())


if nonstrob==true
    for s in eachindex(sol.t)   
        state = create_MFS(sol[s], states)
        dens = density_onsite(state, lattice.sites, geometry)
        CoM = center_of_mass(dens)
        lbl = s==1 ? "ϕ=$(i-1) π/$(length(ϕs))" : ""
        scatter!(pl, [sol.t[s]], CoM, marker=:o, color=colors[i], markersize=2, label=lbl)
    end
end
if strob == true
    ## Plot the stroboscopic solution ##
    j=1
    for s in eachindex(sol.t)
        if j< length(strob_ts) && isapprox(sol.t[s], strob_ts[j];atol=1e-3) 
            state = create_MFS(sol[s], states)
            dens = density_onsite(state, lattice.sites, geometry)
            CoM = center_of_mass(dens)
            lbl = s==1 ? "ϕ=$(i-1) π/$(length(ϕs))" : ""
            scatter!(pl, [sol.t[s]], CoM, marker=:x, color=colors[i], markersize=2, label=lbl)
            j+=1
        end

        
    end
end
end;
display(pl)
savefig("blochoscillation.png")
nothing