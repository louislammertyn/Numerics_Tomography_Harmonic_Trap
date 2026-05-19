module ExpSynthDim 

using FoSpCore, FoSpDynamics, FoSpFloquet
using Plots, LinearAlgebra, FFTW, Interpolations
using ProgressMeter, LaTeXStrings
using Trapz, FastGaussQuadrature
using JLD2


include("./Hamiltonians.jl")
include("./Observables.jl")
include("./params.jl")
include("./Processing.jl")
include("./Tomography.jl")
include("./utils.jl")
include("./errors.jl")
include("./Exp_realism.jl")


##############################################################

export HO_matrix,
       RWA_FL_Hamiltonian,
       drive_elements,
       get_only_m_site_hopping,
       drive_Coherent_FL,
       uniform_Ham

###############################################################


export x_matrix_ho,
       p_matrix_ho,
       a_matrix_ho,
       ad_matrix_ho,
       local_currents


###############################################################


export Params,
       make_params,
       update,
       copy_with


###############################################################


export check_conservation_of_probability,
       plot_center_of_masses,
       plot_center_of_mass_λ,
       gif_x,
       gif_ρ,
       gif_ρ_diag


###############################################################



export hermite_prod_integral,
       hermite_function_poly_prod

    
###############################################################
export hermite,
       interaction_matrix,
       step_hopping_integral,
       thermal_state,
       lambda_com,
       density_x,
       density_p,
       free_evolve_p_matrix,
       free_evolve_λ_matrix,
       density_λ_matrix,
       density_λ_matrix_from_p,
       density_x_matrix_from_p!,
       density_x_diagonal_from_p_fft,
       density_x_diag,
       Displacement_op,
       translate_density_x,
       compare_energies,
       gaussian_diag_matrix,
       plot_density_matrix_errors,
       theta,
       walls

###############################################################

export reconstruct_density_matrix_slice, get_n_tof, convolve_gaussian, scaled_hs, bin_pixels

###############################################################

export almost_thermal

###############################################################

export t_holds_ideal, quantize_times, scaled_hs, convolve_gaussian, get_n_tof, get_n_tof_optimized, Kernel, bin_pixels
end