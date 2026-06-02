# -*- coding: utf-8 -*-
"""
2D AOD waveform generator for all files in given file path

@author: seth.caliga

edited last by Clara: 2024/10/03
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
from scipy.interpolate import RectBivariateSpline
import time # Testing runtime
from scipy.signal import find_peaks
import scipy.ndimage as scim
from scipy.optimize import least_squares, brute
import h5py, json, sys
from pathlib import Path
import xarray as xr
# Function type hinting
from typing import Union
from PIL import Image
#from tqdm import tqdm

from scipy.fft import ifft
print("hello")
#############
# Constants #
#############
conv_um_MHz = 3.864 # [µm / MHz]
freq_min = 35 # MHz
freq_max = 70 # MHz
bandwidth = freq_max - freq_min
samplerate = 250e6 # Hz

########################################################################################################################


#######################
# Spot decompositions # 
#######################

# construction of interpolation function from image of a spot
module_dir = Path(__file__).parent
c_spot = np.load(module_dir / 'spot_profile.npy')
c_spot = np.pad(c_spot, 1, constant_values=0) # Pad to ensure values outside are clipped to zero
x_cspot = (np.arange(c_spot.shape[1]) - (c_spot.shape[1]-1)//2) / 3 # center spot to zero and convert to [µm] from [pxl]
c_spot_spline = RectBivariateSpline(x_cspot, x_cspot, c_spot.T, kx=2, ky=2)
spot_func_from_data = lambda x, y: c_spot_spline(x=x, y=y, grid=False)

def gaussian2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset=0., ravel=True):
    """
    2D rotated, elliptical gaussian function
    
    :param tuple xy: tuple of meshgrid of x and y coordinates
    :param float amplitude: Central amplitude
    :param float x0: Center in x direction
    :param float y0: Center in y direction
    :param float sigma_x: Standard deviation in the rotated x direction
    :param float sigma_y: Standard deviation in the rotated y direction
    :param float theta: Rotation angle in radians
    :param float offset: General offset
    :param bool ravel: If True, the output is a raveled version. If False, it is the same as x and y
    """
    x, y = xy
    # if sigma_x < sigma_y:
    #     sigma_x, sigma_y = sigma_y, sigma_x
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    #g *= (1+1/20 * np.cos(np.pi/2*np.sqrt(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))))
    if ravel:
        return g.ravel()
    else:
        return g

def spot_decomposition(xs, ys, target_func, spot_spacing, spot_func, spacing_frac, xfreq, yfreq, one_dim='', buffer=2, max_nfev=300, guess_amps=None, precomputed_K=None, verbose=0, bypass_fitting=False):
    """Least-squares fit of 2D target function with a general spot function. This function prepares artifact mitigation with the incommensurate trick and ensure all is a multiple of 1/tau without imposing that requirement on xfreq or yfreq

    Args:
        xs (array_like): Input x values where the target will be evaluated. (1D)
        ys (array_like): Input y values.
        target_func (callable  or np.ndarray): target function. Expects a meshgrid as first and second argument (target_func(Xs, Ys)) If an array, it expects the result of target_func(Xs, Ys)
        spot_spacing (float): the maximal pitch between spots [µm]
        spot_func (callable): function that gives spot profile with unit amplitude and centered at 0. Expects a meshgrid as first and second argument (spot_func(Xs, Ys))
        spacing_frac (float): relative spacing fraction between x and y
        xfreq (float): the center frequency in the x direction
        yfreq (float): the center frequency in the y direction
        one_dim (str): set to 'x' or 'y' if the pattern is one dimensional in that direction. set to empty str for 2D decomposition
        buffer (int): the number of extra spots outside of the target area to consider
        max_nfev (int): the maximal number of function evaluations of the minimizer. 
        guess_amps (ndarray): Use amplitudes of previous pattern as first guess instead of a simple linear interpolation. 
        precomputed_K (ndarray): Use precomputed spot kernel if only target_func has changed
        verbose (int): verbosity of least_squares fitting
    Returns:
        tuple: (centers_x, centers_y, amplitudes, drive, cost)
    """
    assert one_dim in ['', 'x', 'y'], 'one_dim must be "x" or "y" or ""'
    one_dim_select = {'x':0, 'y':1}

    centers_allowed = np.arange(5*250) * spot_spacing
    # Determine which is closest to the desired start
    spot_start_idx_x = np.searchsorted(centers_allowed, xfreq* conv_um_MHz + xs[0] - buffer*spot_spacing) 
    spot_start_idx_y = np.searchsorted(centers_allowed*spacing_frac, yfreq* conv_um_MHz + ys[0] - buffer*spot_spacing*spacing_frac)
    # return back to coordinates diven by xs, assume origin in xs frame corresponds with the center freq
    spot_start_x = centers_allowed[spot_start_idx_x] - xfreq * conv_um_MHz
    spot_start_y = centers_allowed[spot_start_idx_y] * spacing_frac - yfreq * conv_um_MHz
    #print(spot_start - xs[0])
    centers_x = np.arange(spot_start_x, xs[-1] + buffer*spot_spacing, spot_spacing)
    centers_y = np.arange(spot_start_y, ys[-1] + buffer*spot_spacing*spacing_frac, spot_spacing*spacing_frac)
    
    spot_freqs_x = centers_allowed[spot_start_idx_x:spot_start_idx_x + len(centers_x)] / conv_um_MHz
    spot_freqs_y = centers_allowed[spot_start_idx_y:spot_start_idx_y + len(centers_y)] * spacing_frac / conv_um_MHz
    if one_dim:
        N_ampl = (1-one_dim_select[one_dim]) * len(centers_x) + one_dim_select[one_dim] * len(centers_y)
    else:
        N_ampl = len(centers_x)+len(centers_y)
    Xs, Ys = np.meshgrid(xs, ys)
    if callable(target_func): # is a function
        target = target_func(Xs, Ys)
    elif type(target_func) == np.ndarray:
        target = target_func
    else:
        raise TypeError("parameter 'target_func' has wrong in type in 'spot_decomposition'")
    
    def reconstruction_slow(ampl):
        rec = np.zeros_like(Xs)
        for i, x_i in enumerate(centers_x):
            for j, y_j in enumerate(centers_y):
                rec += ampl[i]*ampl[len(centers_x)+j]*spot_func(Xs-x_i, Ys-y_j)
        return rec
    
    if precomputed_K is None:
        # precompute full kernel for vectorized reconstruction
        dx = Xs[..., None, None] - centers_x[None, None, :, None]
        dy = Ys[..., None, None] - centers_y[None, None, None, :]
        K = spot_func(dx, dy)  # shape (ny, nx, n_cx, n_cy)
    else:
        # Prev kernel can be used if only target_func is different (=or the same evaluated at a different timestep)
        K = precomputed_K

    def reconstruction(ampl, K=K):
        # Code by Copilot, verified that it does the same as the slow one
        if one_dim == 'x':
            ax = ampl[:len(centers_x)]
            ay = np.ones(len(centers_y))
        elif one_dim == 'y':
            ax = np.ones(len(centers_x))
            ay = ampl[:len(centers_y)]
        else:
            ax = ampl[:len(centers_x)]
            ay = ampl[len(centers_x):]
        # sum over x- and y-centers using einsum for speed
        return np.einsum('xyij,i,j->xy', K, ax, ay, optimize=True)

    init_guess = np.zeros(0)
    lb = np.zeros(N_ampl)
    ub = np.ones(N_ampl) * 1.2 * target.max()
    if one_dim != 'y':
        init_guess_x = np.interp(centers_x, xs, np.mean(target, axis=0))
        init_guess = np.append(init_guess, init_guess_x) 
    if one_dim != 'x':
        init_guess_y = np.interp(centers_y, ys, np.mean(target, axis=1))
        init_guess = np.append(init_guess, init_guess_y) 
    if bypass_fitting:
        return spot_freqs_x, spot_freqs_y, np.clip(init_guess, lb, ub), None, 0
    if not one_dim:
        # normalize to compensate for closeness of spots
        init_guess /= np.sqrt(reconstruction(init_guess).max() / target.max())
    
    # inspect initial guess
    # plt.figure(figsize=(4, 4))
    # plt.imshow(reconstruction(init_guess))
    # plt.colorbar()
    # plt.show()

    # Add a penalization on buffer amplitude such that the amplitudes do not go out of hand just for minute differences in pattern accuracy
    buffer_selector_x = np.argwhere((centers_x > xs.max()) | (centers_x < xs.min()))
    buffer_selector_y = np.argwhere((centers_y > ys.max()) | (centers_y < ys.min()))
    buffer_regularization_strength = 1e-2
    def buffer_ampl(a): 
        if not one_dim:
            return buffer_regularization_strength*np.hstack((a[buffer_selector_x], a[len(centers_x) + buffer_selector_y])).ravel()
        if one_dim == 'x':
            return buffer_regularization_strength*a[buffer_selector_x].ravel()
        else:
            return buffer_regularization_strength*a[buffer_selector_y].ravel()
    
    def jacobian(ampl, K=K):
        """
        Compute the derivative of reconstruction wrt the amplitude array ampl.
        
        Returns a Jacobian matrix of shape (n_pixels + n_buffers, n_amplitudes) where n_pixels = Xs.size
        """
        if one_dim == 'x':
            ax = ampl[:len(centers_x)]
            ay = np.ones_like(centers_y)
            # Derivative wrt ax only
            jacobian = np.einsum('xyij,j->xyi', K, ay, optimize=True)
            buffer_dif = np.zeros((2*buffer, ax.size))
            if buffer > 0:
                buffer_dif[:buffer, :buffer] = np.eye(buffer)
                buffer_dif[buffer:, -buffer:] = np.eye(buffer)
        elif one_dim == 'y':
            ax = np.ones_like(centers_x)
            ay = ampl[:len(centers_y)]
            # Derivative wrt ay only
            jacobian = np.einsum('xyij,i->xyj', K, ax, optimize=True)
            buffer_dif = np.zeros((2*buffer, ay.size))
            if buffer > 0:
                buffer_dif[:buffer, :buffer] = np.eye(buffer)
                buffer_dif[buffer:, -buffer:] = np.eye(buffer)
        else:
            ax = ampl[:len(centers_x)]
            ay = ampl[len(centers_x):]
            # Derivative wrt ax
            d_ax = np.einsum('xyij,j->xyi', K, ay, optimize=True)
            # Derivative wrt ay
            d_ay = np.einsum('xyij,i->xyj', K, ax, optimize=True)
            # Concatenate along amplitude dimension
            jacobian = np.concatenate((d_ax, d_ay), 2)
            buffer_dif = np.zeros((4*buffer, ampl.size))
            if buffer > 0:
                buffer_dif[:buffer, :buffer] = np.eye(buffer)
                buffer_dif[buffer:3*buffer, len(centers_x)-buffer:len(centers_x)+buffer] = np.eye(2*buffer)
                buffer_dif[3*buffer:, -buffer:] = np.eye(buffer)
        return np.vstack((jacobian.reshape(-1, jacobian.shape[-1]), buffer_regularization_strength*buffer_dif))
    
    residuals = lambda a: np.hstack(((reconstruction(a) - target).ravel(), buffer_ampl(a)))
    if guess_amps is not None:
        if one_dim == 'x':
            guess_amps = guess_amps[:len(centers_x)]
        elif one_dim == 'y':
            guess_amps = guess_amps[len(centers_x):]
        assert guess_amps.size == init_guess.size, 'Guess_amps has wrong size'
        residuals_interp = residuals(init_guess)
        residuals_guess = residuals(guess_amps)
        if np.sum(residuals_guess**2) < np.sum(residuals_interp**2):
            if verbose: print('Prev Amps used in favor of interpolation estimate')
            init_guess = guess_amps
    init_guess = np.clip(init_guess, lb, ub)

    res = least_squares(residuals, x0=init_guess, bounds=(lb, ub), jac=jacobian, max_nfev=max_nfev, ftol=2e-3, verbose=verbose)
    if one_dim == 'x':
        amps = np.hstack((res.x, np.ones_like(centers_y)))
    elif one_dim == 'y':
        amps = np.hstack((np.ones_like(centers_x), res.x))
    else:
        amps = res.x

    if verbose: 
        print(f'cost = {res.cost:.2e}')
        print(f'RMSE = {np.sqrt(np.mean((reconstruction(amps)-target)**2)):.2e}')
        print(f'no. function evals = {res.nfev}')
    def drive(x, y, kernel=False):
        # Assumes x and y are meshgrids
        dx = x[..., None, None] - centers_x[None, None, :, None]
        dy = y[..., None, None] - centers_y[None, None, None, :]
        K_drive = spot_func(dx, dy)
        if kernel:
            return K_drive
        else:
            return reconstruction(res.x, K_drive)
    return spot_freqs_x, spot_freqs_y, amps, drive, res.cost

def spot_decomposition_opt_spacing(xs, ys, target_func, spot_func, spacing_frac, xfreq, yfreq, buffer=2):
    """Wrapper around spot_decomposition that performs a brute force search for the optimal spot spacing. Beware: this is an expensive function. 

    Args:
        xs (array_like): Input x values where the target will be evaluated. (1D)
        ys (array_like): Input y values.
        target_func (): target function. Expects a meshgrid as first and second argument (target_func(Xs, Ys))
        spot_spacing (float): the maximal pitch between spots [µm]
        spot_func (): function that gives spot profile with unit amplitude and centered at 0. Expects a meshgrid as first and second argument (spot_func(Xs, Ys))
        W_um (float): The minimal distance between artifacts [µm]
        xfreq (float): the center frequency in the x direction
        yfreq (float): the center frequency in the y direction
        buffer (int): the number of extra spots outside of the target area to consider
    Returns:
        tuple: ((centers_x, centers_y, amplitudes, drive, cost), spot_spacing)
    """
    spacing0 = 1.05 * 0.6
    cost = lambda spacing: spot_decomposition(xs, ys, target_func, spacing[0], spot_func, spacing_frac, xfreq, yfreq, buffer=buffer, max_nfev=10)[-1]
    #res = minimize(cost, spacing0, bounds=(spacing0*0.7, spacing0*3), method='')
    #res = basinhopping(cost, spacing0, niter=100, stepsize=w, minimizer_kwargs={'bounds':[(spacing0*0.7, spacing0*3)]})
    #res = differential_evolution(cost, bounds=[(spacing0*0.7, spacing0*3)], popsize=3, maxiter=50)
    #res = shgo(cost, bounds=[(spacing0*0.7, spacing0*3)], options={'maxfev':5})
    x0, fval, grid, Jout = brute(cost, (slice(spacing0*0.7, spacing0*1.3, 0.02),), full_output=True)
    #print(f'nfev={res.nfev}')
    plt.scatter(np.arange(spacing0*0.7, spacing0*1.3, 0.02), Jout, marker='o')
    plt.yscale('log')
    plt.xlabel('Spacing [µm]')
    plt.ylabel('Cost')
    plt.show()
    return spot_decomposition(xs, ys, target_func, x0[0], spot_func, spacing_frac, xfreq, yfreq, buffer=buffer), x0[0]



class Experiment:
    """Experiment class for generating and managing AOD waveforms.
    This class handles the creation of waveforms for controlling AODs. It supports two projection methods: holographic (using Gerchberg-Saxton 
    algorithm and incommensurate trick) and line-scanning techniques. 
    Attributes:
        kwargs_default (dict): Default configuration parameters including:
            - title (str): Experiment identifier
            - dur_tot_ms (float): Total experiment duration in milliseconds
            - spot_spacing (float): Spacing between spots in micrometers
            - W_MHz (float): Frequency spacing for 2D coherent artifact mitigation in MHz
            - xfreq, yfreq (float): Center frequencies for x and y AOD channels in MHz
            - proj_method (str): Projection method ('holographic' or 'linescanning')
            - tight_direction (str): Tight confinement direction for holographic method ('x' or 'y')
            - DeltaN (int): Stepping factor for incommensurate trick (Derived from W_MHz, not set by user)
            - GS_iter (int): Gerchberg-Saxton algorithm iterations
            - GS_parallel (int): Number of parallel GS algorithm runs
            - scan_direction (str): Scanning direction for line-scanning method
            - tdead (float): Dead time between lines in line-scanning [µs]
            - n_frames (int): Number of frames for line-scanning (Derived from W_MHz, not set by user)
            - reps (int): Number of repetitions per frame
            - phase_option (int): Phase encoding option for line-scanning (0=zero phases, 1=random phases)
            - tdead_frame (float): Amount of dead time [µs] between frames (=between timesteps or sub-images in NNSVD)
            - rank (int): Number of sub-images used in NNSVD
            - nnsvd_init (str): Method of initializing the NNSVD algorithm. Choose out of ['svd+', 'svd0', 'random'] see 'non_negative_svd' for more information. 
            - drive_period_ms (float): Period of driving signal in milliseconds. If drive is not periodic, set drive_period_ms=dur_tot_ms
            - t_ramp_ms (float): Ramp time duration in milliseconds
            - ramp_profile (callable): Function defining ramp profile
            - i_ref (float): Reference intensity
            - extra_i (float): Extra intensity factor (gets multiplied with i_ref)
            - buffer (int): Number of extra spots placed outside target region of pattern. Ensure smooth edge
            - spot_func (callable): Function defining spot intensity profile
            - corr_factors_applied (bool): Whether AOD correction factors have been applied (Not set by user)
            - verbose (bool): Enable detailed console output
            - awg_file_name (str): Path to output HDF5 file for AWG instructions
        required_params (set): Set of mandatory parameters ('i_ref', 'dur_tot_ms')
        I_x, I_y (ndarray): 2D arrays of spot intensities over time for x and y channels
        xfreq_start, yfreq_start (float): Starting frequencies for spot arrays
        t_drive (list): List of functions representing projected patterns at each timeframe
        wfx_period, wfy_period (ndarray): Generated waveforms for one period
        ramp_wfx, ramp_wfy (ndarray): Ramped waveforms for smooth intensity ramp
    Methods:
        __init__(**kwargs): Initialize Experiment with custom parameters
        __getitem__(key): Dictionary-style access to experiment attributes
        __setitem__(key, value): Dictionary-style setting of experiment attributes
        print_default(): Display all default parameter values
        table_print(): Generate formatted parameter table for display
        construct_ampl_from_func(xt, yt, target_func, ...): Decompose target spatial-temporal pattern into spot intensities
        insert_custom_pattern(xfreq_start, yfreq_start, I_x, I_y, t_drive): Insert manually-defined patterns
        animate_proj(xt, yt, target_func, ...): Create animation comparing projected vs target patterns
        apply_correction_factors(): Apply AOD frequency-dependent correction factors
        generate_wf_single(t_idx): Generate waveforms for a single time index
        generate_wf_period(): Generate waveforms for one complete driving period
        balance_waveforms(): Scale x and y waveforms to balanced amplitudes and apply reference intensity
        ramp_wf(ramp_profile): Create smoothly ramped waveforms
        save_waveforms(): Save waveforms to HDF5 file for AWG hardware
        save_essentials(path): Save experiment configuration and waveforms for later retrieval
    Raises:
        KeyError: If unrecognized kwargs provided or required parameters missing
        ValueError: If drive_period_ms is smaller than the time resolution
        AssertionError: If pattern outside AOD bandwidth or other validation failures
    """

    kwargs_default = {
        'title':'', 
        'dur_tot_ms':30, # ms
        'spot_spacing':0.8, # µm
        'W_MHz':2.0, # MHz default distance between 2D coherent artifacts
        'xfreq':52.5, 'yfreq':52.5, # center frequencies [MHz]
        'proj_method':'holographic', 
        'fixed_seed':0,
        # Default values for holographic method
        'tight_direction':'x', 'DeltaN':1, 'GS_iter':100, 'GS_parallel':3,
        # default values for linescanning method
        'scan_direction':'x', 'tdead':0, 'n_frames':4, 'reps':2, 'phase_option':1, 
        # options for non-separable patterns
        'tdead_frame':0.0, 'rank':1, 'nnsvd_init':'svd+',
        # timing options
        'drive_period_ms':0, 't_ramp_ms':0, 'ramp_profile':lambda x: x, # If projection not periodic, set drive_period_ms=dur_tot_ms
        'i_ref':1e-7, 'extra_i':1, 
        'buffer':2, 'spot_func':lambda x, y: np.where((np.abs(x)<2) & (np.abs(y)<2), gaussian2d((x, y), 1, 0, 0, 0.567, 0.633, np.deg2rad(49.1), 0, False), 0), 
        'corr_factors_applied':False, 'verbose':False, 'awg_file_name':"output.h5"
    }
    
    required_params = {'i_ref', 'dur_tot_ms'}

    def __init__(self,**kwargs):
        key_list = self.kwargs_default.keys()
        error_list = []
        for k in kwargs.keys():
            if k not in key_list and k not in self.required_params:
                error_list.append(k)
        if len(error_list):
            raise KeyError(f"Some kwargs-variables are not recognized: {error_list}.")
        
        for req_param in self.required_params:
            if req_param not in kwargs.keys():
                raise KeyError(f'Required parameter "{req_param}" was not specified when Experiment class was initialized')
        
        self.__dict__.update( self.kwargs_default )
        self.__dict__.update( kwargs )
        self.print_params = ['title', 'i_ref', 'dur_tot_ms', 'spot_spacing', 'W_MHz', 'drive_period_ms', 'proj_method', 'xfreq', 'yfreq', 'avg_fit_fidelity', 'time_res', 'tdead_frame']
        if self.t_ramp_ms: 
            self.print_params.append('t_ramp_ms')

        if self.proj_method == 'holographic':
            self.DeltaN = int(np.ceil(1 + self.W_MHz * conv_um_MHz / self.spot_spacing))
            if self.W_MHz == 0: # no artifact avoidance
                self.spot_spacing = samplerate * 1e-6 * conv_um_MHz/ round(samplerate * 1e-6 * conv_um_MHz / self.spot_spacing)
                self.spacing_frac = 1
            else:
                self.spot_spacing = samplerate * 1e-6 * conv_um_MHz / (self.DeltaN-1) / round(samplerate * 1e-6 * conv_um_MHz / self.spot_spacing / (self.DeltaN-1))
                self.spacing_frac = (self.DeltaN - 1) / self.DeltaN

            self.time_res = self.DeltaN * conv_um_MHz / self.spot_spacing # µs
            self.print_params += ['DeltaN', 'spacing_frac', 'tight_direction', 'GS_iter', 'GS_parallel']
            if self.verbose: 
                print(f'Spot spacing= {self.spot_spacing:.3f} µm\n' \
                      f'Time resolution= {self.time_res:.2f} µs\n' \
                      f'Spacing fraction= {self.spacing_frac:.3f}')

            if self.tight_direction.lower() == 'x': 
                # All code is written as if spacing_frac < 1 and y is tight direction. This flips this scenario
                self.spot_spacing *= self.spacing_frac
                self.spacing_frac = 1 / self.spacing_frac
        elif self.proj_method == 'linescanning':
            # adjust the spot spacing such that it corresponds to an integer number of samples
            self.spot_spacing = samplerate * 1e-6 * conv_um_MHz / round(samplerate * 1e-6 * conv_um_MHz / self.spot_spacing)
            self.n_frames = max(int(np.ceil(self.W_MHz * conv_um_MHz / self.spot_spacing)), 1)
            self.time_res = self.n_frames * (self.reps * conv_um_MHz / self.spot_spacing + self.tdead) # µs
            self.spacing_frac = 1 
            self.print_params += ['n_frames', 'tdead', 'reps', 'scan_direction', 'phase_option']
        else:
            raise KeyError(f'proj_method "{self.proj_method}" not supported.')
        # Add tdead_frame to time resolution
        self.tdead_frame = int(self.tdead_frame*1e-6*samplerate) / (1e-6*samplerate)
        self.extra_i *= 1 + self.tdead_frame / self.time_res
        self.time_res += self.tdead_frame
        assert self.time_res > 0, 'Non-positive value for time resolution'
        if self.drive_period_ms == 0: # time independent pattern
            self.drive_period_ms = 1e-3*self.rank*self.time_res
        elif self.drive_period_ms <= 1e-3*self.rank*self.time_res:
            raise ValueError(f'Driving period is too small: {self.drive_period_ms*1e3:.1f}µs vs $\\Delta t$={self.time_res:.1f}µs and rank={self.rank}')
        # The current time resolution does not fit nicely inside the drive period. Distribute the extra sample as an elongation of the dead time
        n_extra_samp = round((np.round(self.drive_period_ms*1e3,2) % np.round(self.time_res,2)) * samplerate * 1e-6)
        n_times = self.rank * int(round(self.drive_period_ms*1e3 / (self.rank * self.time_res)))
        tdead_frame_extra = int(n_extra_samp / n_times) / (samplerate*1e-6)
        self.extra_i *= 1 + tdead_frame_extra / self.time_res
        self.tdead_frame += tdead_frame_extra
        self.time_res += tdead_frame_extra

    def __getitem__(self, key):
        if len(key) == 0 or key is None:
            return key
        else:
            return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def print_default(self):
        """Prints the default values of the Experiment class"""
        print('Default values.')
        for key, val in self.kwargs_default.items():
            print(f"{key}: {val}")
    
    def table_print(self):
        """
        Generate the table of printing the relevant parameters.
        
        :param self: Experiment class
        """
        params = self.print_params
        n_rows = (len(params) + 1) // 2
        table_data = []
        for i in range(n_rows):
            row = []
            # First column pair
            row.append(params[i].replace('_', ' '))
            param = self.__dict__[params[i]]
            if type(param) in [float, np.float64]:
                if param > 0.1:
                    row.append(str(np.round(param, 3)))
                elif param > 0.01:
                    row.append(str(np.round(param, 6)))
                else:
                    row.append(str(param))
            else:
                row.append(str(param))

            # Second column pair
            if i + n_rows < len(params):
                row.append(params[i + n_rows].replace('_', ' '))
                param = self.__dict__[params[i + n_rows]]
                if type(param) in [float, np.float64]:
                    if param > 0.1:
                        row.append(str(round(param, 3)))
                    else:
                        row.append(str(param))
                else:
                    row.append(str(param))
            else:
                row.append('')
                row.append('')
            table_data.append(row)
        return table_data

    def construct_ampl_from_func(self, xt, yt, target_func, one_dim='', one_dim_len=0.):
        """
        Construct the spot locations and intensities to match the target function in space and time. 
        
        :param self: Experiment class
        :param array_like xt: target x coordinates [µm]
        :param array_like yt: target y coordinates [µm]
        :param func target_func: separable target function f(X, Y, t), even if the function is one dimensional and/or time independent, it needs to accept 3 arguments where the first 2 are meshgrids
        :param str one_dim: If 'x' or 'y', only the spots in that direction are fit. Only enable this when the pattern is constant in the other direction. Leave as empty string for 2D fitting. 
        :param float one_dim_len: length of pattern in constant direction (other than given to one_dim) [µm]

        :return t_drive: a list of functions that represent the projected pattern at each timeframe
        """
        cx, cy, amps_init, drive_0, cost = spot_decomposition(xt, yt, lambda x, y:target_func(x, y, 0), spot_spacing=self.spot_spacing, spot_func=self.spot_func, spacing_frac=self.spacing_frac, xfreq=self.xfreq, yfreq=self.yfreq, one_dim=one_dim, buffer=self.buffer, max_nfev=2)
        Xt, Yt = np.meshgrid(xt, yt)
        precomputed_K = drive_0(Xt, Yt, True)
        n_times = int(round(1e3*self.drive_period_ms / self.time_res))
        self.I_x = np.zeros((cx.size, n_times))
        self.I_y = np.zeros((cy.size, n_times))
        fit_fidelities = np.zeros(n_times)
        prev_amps = amps_init
        self.t_drive = n_times*[0]
        if False: #self.verbose:
            pass#iterable = tqdm(range(n_times), desc='Construction of spot decomposition')
        else:
            iterable = range(n_times)
        for t_idx in iterable:
            cx, cy, amps, drive, cost = spot_decomposition(xt, yt, lambda x, y:target_func(x, y, t_idx*self.time_res), spot_spacing=self.spot_spacing, spot_func=self.spot_func, 
                                                       spacing_frac=self.spacing_frac, xfreq=self.xfreq, 
                                                       yfreq=self.yfreq, one_dim=one_dim, buffer=self.buffer, guess_amps=prev_amps, precomputed_K=precomputed_K)
            assert cx.size + cy.size == amps.size, 'Wrong number of centers and spots'
            self.I_x[:, t_idx] = amps[:cx.size]
            self.I_y[:, t_idx] = amps[cx.size:]
            self.t_drive[t_idx] = drive
            # TODO what gives scale of fit fidelity? Mean is obvious candidate, but sensitive to arbitrary shift. Ptp is bad when pattern is flat... 
            fit_fidelities[t_idx] = np.sqrt(2*cost / (xt.size * yt.size)) / target_func(Xt, Yt, t_idx*self.time_res).mean()
            prev_amps = amps
        if one_dim == 'y':
            n_spots_const = int(np.ceil(one_dim_len / self.spot_spacing))
            self.I_x = np.ones((n_spots_const, n_times))
            self.xfreq_start = self.xfreq - n_spots_const * 0.5 * self.spot_spacing / conv_um_MHz 
            # Ensure the start is a multiple of the freq spacing
            self.xfreq_start -= self.xfreq_start % np.diff(cx)[0]
            self.yfreq_start = cy[0]
        elif one_dim == 'x':
            n_spots_const = int(np.ceil(one_dim_len / self.spot_spacing / self.spacing_frac))
            self.I_y = np.ones((n_spots_const, n_times))
            self.xfreq_start = cx[0]
            self.yfreq_start = self.yfreq - n_spots_const * 0.5 * self.spot_spacing * self.spacing_frac / conv_um_MHz 
            # Ensure the start is a multiple of the freq spacing
            self.yfreq_start -= self.yfreq_start % np.diff(cy)[0]
        else:
            self.xfreq_start = cx[0]
            self.yfreq_start = cy[0]
        self.avg_fit_fidelity = 100*(1-fit_fidelities.mean())
        if self.verbose:
            print(f'{self.I_x.shape[0]} spots in x. {self.I_y.shape[0]} spots in y\n' \
                  f'Mean spot amplitude x:{self.I_x.mean():.2f} y:{self.I_y.mean():.2f}\n' \
                  f'Period split over {n_times} timesteps\n' \
                  f'Avg image fidelity {self.avg_fit_fidelity:.1f}%')
        if not self.title:
            self.title = target_func.__name__
        return self.t_drive
    
    def construct_ampl_from_img(self, img_path:str, rank:int, pxl_pitch:float, init='svd+', k=0):
        """
        This function decomposes an input image into spatial amplitude patterns using Non-Negative Singular Value Decomposition (NNSVD). 

        Parameters
        ----------
        img_path : str
            Path to the image file to be decomposed. The image is converted to grayscale and normalized. 
        rank : int
            Number of components to extract via NNSVD. If rank == -1, the rank is automatically 
            determined based on pattern fidelity threshold (95%).
        pxl_pitch : float
            Physical pixel pitch of the image in µm. Used to scale the coordinate system for spot decomposition.
        init : str, optional
            Initialization method for NNSVD. Default is 'svd+'. Options: ['svd+', 'random', 'svd0']
        k : int, optional
            Number of 90-degree counter-clockwise rotations to apply to the image. Default is 0.
        Returns
        -------
        list
            A list of length `rank` containing drive time arrays (t_drive) for each decomposed component.
        Notes
        -----
        - The function normalizes the input image to [0, 1] range before decomposition.
        - If verbose mode is enabled, displays the original image and its NNSVD reconstruction.
        - The decomposition result is evaluated on a grid defined by image dimensions and pixel pitch.
        - Spot positions and amplitudes are determined via spot_decomposition for each rank component.
        - Sets instance attributes: I_x, I_y (amplitude patterns), t_drive (drive signals), 
          xfreq_start, yfreq_start (initial frequency positions).
        See Also
        --------
        non_negative_svd : Performs the matrix factorization.
        spot_decomposition : Determines spot positions and amplitudes from a 2D pattern.
        """
        img_arr = np.array(Image.open(img_path).convert('L'))
        img_arr = img_arr / img_arr.max()
        img_arr = np.rot90(img_arr, k)
        
        U, S, Vt = non_negative_svd(img_arr, init+f'rank{rank}')
        s = np.diag(S)
        if rank == -1: # determine rank out of pattern accuracy. CAUTION, this option gives worse performance than initializing with the rank obtained by this procedure. 
            pattern_fidelity = .95
            rank = np.argmin(np.abs(np.cumsum(s)/np.sum(s) - pattern_fidelity))
        if self.verbose: 
            # Compare how well the NNSVD decomposed the image 
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6))
            im0 = ax0.imshow(img_arr, cmap='plasma')
            ax0.set_title('Original image')
            ax1.imshow(U@S@Vt, cmap='plasma')
            ax1.set_title('Reconstruction')
            fig.colorbar(im0, ax=ax0, fraction=0.04, pad=0.04)
            plt.show()
        
        # create coordinates for function evaluation
        oversampling_rate = 1 # Evaluate the projection on more (or fewer) points than the given image TODO is this a magic number?
        xt = pxl_pitch*np.arange(oversampling_rate*img_arr.shape[1])/oversampling_rate
        xt -= xt.mean() # shift image towards center of projection plane
        yt = pxl_pitch*np.arange(oversampling_rate*img_arr.shape[0])/oversampling_rate
        yt -= yt.mean()
        def target_func(target, x, y):
            # shift projection coordinates back to image coordinates
            x = (x-x.min())/pxl_pitch
            y = (y-y.min())/pxl_pitch
            return scim.map_coordinates(target, np.vstack((y.flatten(), x.flatten()))).reshape(x.shape)
        # get centers
        cx, cy, amps_init, drive_0, cost = spot_decomposition(xt, yt, lambda x, y:target_func(s[0]*rank*np.outer(U[:, 0], Vt[0, :]), x, y), spot_spacing=self.spot_spacing, spot_func=self.spot_func, spacing_frac=self.spacing_frac, xfreq=self.xfreq, yfreq=self.yfreq, one_dim='', buffer=self.buffer, bypass_fitting=True)
        self.I_x = np.zeros((cx.size, rank))
        self.I_y = np.zeros((cy.size, rank))
        fit_fidelities = np.zeros(rank)
        self.t_drive = rank*[0]
        for i in range(rank):
            cx, cy, amps, drive, cost = spot_decomposition(xt, yt, lambda x, y:target_func(s[i]*rank*np.outer(U[:, i], Vt[i, :]), x, y), spot_spacing=self.spot_spacing, spot_func=self.spot_func, spacing_frac=self.spacing_frac, xfreq=self.xfreq, yfreq=self.yfreq, one_dim='', buffer=self.buffer, max_nfev=1, bypass_fitting=True)
            assert cx.size + cy.size == amps.size, 'Wrong number of centers and spots'
            self.I_x[:, i] = amps[:cx.size]
            self.I_y[:, i] = amps[cx.size:]
            fit_fidelities[i] = np.sqrt(2*cost / (xt.size * yt.size))
            self.t_drive[i] = drive
            
        self.avg_fit_fidelity = 100*(1-fit_fidelities.mean() / img_arr.mean())
        if self.verbose:
            print(f'{self.I_x.shape[0]} spots in x. {self.I_y.shape[0]} spots in y\n' \
                  f'Mean spot amplitude x:{self.I_x.mean():.2f} y:{self.I_y.mean():.2f}\n' \
                  f'Avg image fidelity {self.avg_fit_fidelity:.1f}%')
        self.xfreq_start = cx[0]
        self.yfreq_start = cy[0]
        return self.t_drive
    
    def construct_ampl_from_nonsep_func(self, xt, yt, target_func):
        """
        Construct the spot locations and intensities to match the non-separable target function in space and time. 
        If function is separable, set rank=1. If function is one dimensional, use 'construct_ampl_from_func'
        
        :param self: Experiment class
        :param array_like xt: target x coordinates [µm]
        :param array_like yt: target y coordinates [µm]
        :param func target_func: Target function f(X, Y, t)
        """
        time_res_long = self.rank * self.time_res
        n_times = int(self.drive_period_ms*1e3 / time_res_long)
        self.xt, self.yt = xt, yt
        self.print_params.append('rank')
        # find the centers
        cx, cy, amps_init, drive_0, cost = spot_decomposition(self.xt, self.yt, lambda x, y:np.ones_like(x), spot_spacing=self.spot_spacing, spot_func=self.spot_func, spacing_frac=self.spacing_frac, xfreq=self.xfreq, yfreq=self.yfreq, one_dim='', buffer=self.buffer, max_nfev=2)
        
        # initialise spot intensity arrays
        self.I_x = np.zeros((cx.size, self.rank*n_times))
        self.I_y = np.zeros((cy.size, self.rank*n_times))
        self.t_drive = self.rank*n_times*['']
        self.xfreq_start = cx[0]
        self.yfreq_start = cy[0]
        self.fit_fidelities = []
        for t_idx in range(n_times):
            # Evaluate function on specified grid and start of time interval
            # TODO Evuluate on time between 2 timestamps? Take average? Integrate? 
            img_arr = target_func(*np.meshgrid(self.xt, self.yt), t_idx*time_res_long)
            # img_arr = (target_func(*np.meshgrid(self.xt, self.yt), t_idx*time_res_long) + \
            #            2*target_func(*np.meshgrid(self.xt, self.yt), (t_idx + 1/2)*time_res_long) + \
            #            target_func(*np.meshgrid(self.xt, self.yt), (t_idx + 1)*time_res_long)) / 4

            U, S, Vt = non_negative_svd(img_arr, self.nnsvd_init+f'rank{self.rank}')
            s = np.diag(S)
            residual_img = np.copy(img_arr)
            for r_idx in range(self.rank):
                cx, cy, amps, drive, cost = spot_decomposition(self.xt, self.yt, s[r_idx]*self.rank*np.outer(U[:, r_idx], Vt[r_idx, :]), spot_spacing=self.spot_spacing, spot_func=self.spot_func, spacing_frac=self.spacing_frac, xfreq=self.xfreq, yfreq=self.yfreq, one_dim='', buffer=self.buffer)
                self.I_x[:, t_idx*self.rank + r_idx] = amps[:cx.size]
                self.I_y[:, t_idx*self.rank + r_idx] = amps[cx.size:]
                self.t_drive[t_idx*self.rank + r_idx] = drive
                # Determine how well the desired image is reconstructed
                residual_img -= drive(*np.meshgrid(self.xt, self.yt)) / self.rank
                
            self.fit_fidelities.append(100*(1-np.sqrt(np.mean(residual_img**2)) / img_arr.mean()))
            if self.verbose:
                pass
                # print(f'fit fidelity: {self.fit_fidelities[t_idx]:.1f}%')
                # data_array = xr.DataArray(np.concatenate((img_arr[None, ...], residual_img[None, ...], drive(*np.meshgrid(self.xt, self.yt))[None, ...]/2)), dims=['time', 'y', 'x'], coords={'time':['img', 'residual', 'last drive'], 'x': self.xt, 'y':self.yt})
                # data_array.groupby('time').first().plot(x='x', y='y', col='time')
                # plt.show()
        self.avg_fit_fidelity = np.mean(self.fit_fidelities)
        if self.verbose:
            print(f'{self.I_x.shape[0]} spots in x. {self.I_y.shape[0]} spots in y\n' \
                  f'Mean spot amplitude x:{self.I_x.mean():.2f} y:{self.I_y.mean():.2f}\n' \
                  f'Period split over {n_times} timesteps\n' \
                  f'Avg image fidelity {self.avg_fit_fidelity:.1f}%')
        if not self.title:
            self.title = target_func.__name__

    def insert_custom_pattern(self, xfreq_start, yfreq_start, I_x, I_y, t_drive=None):
        """Use this function to insert custom patterns, not made by 'construct_ampl_from_func"""
        assert I_x.shape[1] == I_y.shape[1], 'Unequal amount of frames specified in I_x and I_y'
        self.xfreq_start = xfreq_start
        self.yfreq_start = yfreq_start
        self.I_x = I_x
        self.I_y = I_y
        self.drive_period_ms = 1e-3*self.time_res*I_x.shape[1]
        self.print_params.remove('avg_fit_fidelity')
        if t_drive is not None:
            self.t_drive = t_drive
            return
        # construct t_drive when it is not given
        n_times = self.I_x.shape[1]
        self.t_drive = n_times*[0]
        for t_idx in range(n_times):
            centers_x = (xfreq_start - self.xfreq)*conv_um_MHz + self.spot_spacing*np.arange(self.I_x.shape[0])
            centers_y = (yfreq_start - self.yfreq)*conv_um_MHz + self.spot_spacing*self.spacing_frac*np.arange(self.I_y.shape[0])
            def drive(x, y): # assume x and y are meshgrids
                dx = x[..., None, None] - centers_x[None, None, :, None]
                dy = y[..., None, None] - centers_y[None, None, None, :]
                K_drive = self.spot_func(dx, dy)
                return np.einsum('xyij,i,j->xy', K_drive, self.I_x[:, t_idx], self.I_y[:, t_idx])
            self.t_drive[t_idx] = drive
    
    def animate_proj(self, xt, yt, target_func, frames_step=3, coord_x_slice=0, coord_y_slice=0, update_spots=True, plot_residual_bool=False, z=0.):
        """
        Generate an animation to inspect the projected pattern. Also provides the user with cross sections through x and y. A table with the most important parameters are displayed in a table. 
        Returns a matplotlib animation instance. 
        
        :param self: Experiment class
        :param array_like xt: target x coordinates [µm]
        :param array_like yt: target y coordinates [µm]
        :param func target_func: Target function f(X, Y, t)
        :param int frames_step: Animation advances with this number of frames to speed up drawing of animation
        :param float coord_x_slice: value of the y coordinate for the slice through x [µm]
        :param float coord_y_slice: value of the x coordinate for the slice through y [µm]
        :param bool update_spots: Set to False to not draw spots for faster animation drawing
        :param bool plot_residual_bool: Set to true to inspect residual on colormap. False displays the projected image
        :param float z: The height relative to the focal plane. Use to inspect z-dependence of projection. 

        :return ani: Matplotlib Animation instance to be drawn or saved
        """
        range_x = 1.1 * (np.ptp(xt) + 4*self.spot_spacing)
        range_y = 1.1 * (np.ptp(yt) + 4*self.spot_spacing*self.spacing_frac)


        x = np.arange(-range_x/2, range_x/2, self.spot_spacing / 3)
        y = np.arange(-range_y/2, range_y/2, self.spot_spacing * self.spacing_frac / 3)
        
        centers_x = (self.xfreq_start - self.xfreq)*conv_um_MHz + self.spot_spacing*np.arange(self.I_x.shape[0])
        centers_y = (self.yfreq_start - self.yfreq)*conv_um_MHz + self.spot_spacing*self.spacing_frac*np.arange(self.I_y.shape[0])
        # Enforce width >= height by swapping x and y if needed
        swap_xy = range_x < range_y
        if swap_xy:
            xt, yt = yt, xt
            x, y = y, x
            coord_x_slice, coord_y_slice = coord_y_slice, coord_x_slice
            centers_x, centers_y = centers_y, centers_x

        X, Y = np.meshgrid(x, y)

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(9, 5))

        def drive(X, Y, f):
            drv = np.zeros_like(X)
            for r_idx in range(self.rank):
                drv += self.t_drive[f*self.rank+r_idx](Y, X) if swap_xy else self.t_drive[f*self.rank+r_idx](X, Y)
            drv /= self.rank
            sigma_x, sigma_y = 0.6*np.abs(z) / 2.9 / np.diff(x)[0], 0.6*np.abs(z) / 3.66 / np.diff(y)[0]# Currently magic numbers. 0.6[µm]=spot waist (average x and y), 2.9/3.6 [µm] Rayleigh length
            # TODO measurements of Rayleigh length do not seem accurate (data from 202507 found in Ghent BEC Lab Procedures->Calibrations->Projection spot)
            if swap_xy:
                sigma_x, sigma_y = sigma_y, sigma_x
            return scim.gaussian_filter(drv, (sigma_y, sigma_x))

        def target(X, Y, t):
            return target_func(Y, X, t) if swap_xy else target_func(X, Y, t)

        start_drive = drive(X, Y, 0)
        start_target = target(X, Y, 0)

        if plot_residual_bool:
            img = start_target - start_drive
            cmap = 'bwr'
            cbar_label = 'Residual intensity'
            vmax = start_target.max() / 5
            vmin = -vmax
        else:
            img = start_drive
            cmap = 'plasma'
            cbar_label = 'Intensity'
            vmin = 0 
            vmax = 1.1*start_drive.max() # TODO this is in general not the global max

        im0 = ax0.imshow(img, cmap=cmap, extent=(x.min(), x.max(), y.min(), y.max()), vmin=vmin, vmax=vmax, origin='lower')
        fig.colorbar(im0, ax=ax0, label=cbar_label)

        xlabel = 'x [µm]' if not swap_xy else 'y [µm]'
        ylabel = 'y [µm]' if not swap_xy else 'x [µm]'
        ax0.set(xlabel=xlabel, ylabel=ylabel, title=f'Projected pattern (z={z:.1f}µm)')

        x_slice = np.s_[np.searchsorted(y, coord_x_slice), :]
        y_slice = np.s_[:, np.searchsorted(x, coord_y_slice)]

        ax0.axvline(coord_y_slice, color='black', linestyle='--', linewidth=1)
        ax0.axhline(coord_x_slice, color='black', linestyle='--', linewidth=1)

        (result_x,) = ax1.plot(x, start_drive[x_slice], 'b', lw=2, label='Projection')
        (target_x,) = ax1.plot(x, start_target[x_slice], 'k--', lw=2, label='Target')
        ax1.vlines([xt.min(), xt.max()], 0, 1.1*start_drive.max(), linestyles='--', colors='r', label='Target edge')
        color_list = ['b', 'g', 'c', 'm', 'y', 'r', 'orange', 'purple', 'brown']
        spots_x = [[ax1.plot([], [], linestyle="--", c=color_list[r_idx % len(color_list)], alpha=0.5)[0] for _ in centers_x] for r_idx in range(self.rank)]
        ax1.set(xlabel=xlabel, ylabel='Intensity', title=f'Slice through {xlabel[0]} ({ylabel[0]}={coord_x_slice:.1f}µm)')

        (result_y,) = ax2.plot(y, start_drive[y_slice], 'b', lw=2, label='Projection')
        (target_y,) = ax2.plot(y, start_target[y_slice], 'k--', lw=2, label='Target')
        ax2.vlines([yt.min(), yt.max()], 0, 1.1*start_drive.max(),  linestyles='--', colors='r', label='Target edge')
        spots_y = [[ax2.plot([], [], linestyle="--", c=color_list[r_idx % len(color_list)], alpha=0.5)[0] for _ in centers_y] for r_idx in range(self.rank)]
        ax2.legend(loc='upper right')
        ax2.set(xlabel=ylabel, ylabel='Intensity', title=f'Slice through {ylabel[0]} ({xlabel[0]}={coord_y_slice:.1f}µm)')

        # region param table
        ax3.axis('tight')
        ax3.axis('off')
        table = ax3.table(
            cellText=self.table_print(),
            colLabels=['Parameter', 'Value', 'Parameter', 'Value'],
            cellLoc='left', loc='center',
            colWidths=[0.25]*4
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        # endregion
        fig.tight_layout()

        def update(f):
            result = drive(X, Y, f)
            tgt = target(X, Y, self.time_res*f*self.rank)
            im0.set_data(tgt - result if plot_residual_bool else result)
            target_x.set_ydata(tgt[x_slice])
            result_x.set_ydata(result[x_slice])
            target_y.set_ydata(tgt[y_slice])
            result_y.set_ydata(result[y_slice])
            if not update_spots:
                return [target_x, result_x, target_y, result_y]

            for r_idx in range(self.rank):
                # ---------- Spots along x-slice ----------
                yg = np.linspace(Y[x_slice][0] - 2, Y[x_slice][0] + 2, 20)
                ints_x = self.I_x[:, f*self.rank + r_idx]
                ints_y = self.I_y[:, f*self.rank + r_idx]
                if swap_xy:
                    ints_x, ints_y = ints_y, ints_x

                for s, c, a in zip(spots_x[r_idx], centers_x, ints_x):
                    xg = np.linspace(c - 2, c + 2, 20)
                    Xg, Yg = np.meshgrid(xg, yg)
                    avg_spot = np.zeros_like(Xg)
                    for i in range(len(centers_y)):
                        avg_spot += ints_y[i] * self.spot_func(Xg - c, Yg - centers_y[i])
                        
                    s.set_data(xg, a * avg_spot.mean(axis=0) / self.rank)

                # ---------- Spots along y-slice ----------
                xg = np.linspace(X[y_slice][0] - 2, X[y_slice][0] + 2, 20)
                for s, c, a in zip(spots_y[r_idx], centers_y, ints_y):
                    yg = np.linspace(c - 2, c + 2, 20)
                    Xg, Yg = np.meshgrid(xg, yg)
                    avg_spot = np.zeros_like(Xg)
                    for i in range(len(centers_x)):
                        avg_spot += ints_x[i] * self.spot_func(Xg-centers_x[i], Yg-c)
                    s.set_data(yg, a*avg_spot.mean(axis=1) / self.rank)

            return [target_x, result_x, *[s for xs in spots_x for s in xs], target_y, result_y, *[s for ys in spots_y for s in ys]]
        
        ani = FuncAnimation(fig, update, frames=range(0, self.I_x.shape[1]//self.rank, frames_step), interval=frames_step*3*self.rank*self.time_res, blit=True)
        plt.close(fig)
        return ani


    def apply_correction_factors(self):
        """Apply the AOD correction factors to the spot intensities"""
        fx_array = self.xfreq_start + self.spot_spacing / conv_um_MHz * np.arange(self.I_x.shape[0])
        fy_array = self.yfreq_start + self.spot_spacing*self.spacing_frac / conv_um_MHz * np.arange(self.I_y.shape[0])
        assert (35 <= fx_array).all() & (70 >= fx_array).all() & (35 <= fy_array).all() & (70 >= fy_array).all(), 'Pattern lies outside AOD bandwidth'

        corr_array_x = np.interp(x=fx_array, xp=f_for_correction_factors, fp=correction_factors_x)**-1
        corr_array_y = np.interp(x=fy_array, xp=f_for_correction_factors, fp=correction_factors_y)**-1
        self.I_x *= corr_array_x.reshape((-1, 1))
        self.I_y *= corr_array_y.reshape((-1, 1))
        self.corr_factors_applied = True
    
    def generate_wf_single(self, t_idx:int):
        """
        Generate the waveforms at a given time index. Analogue of create_single_waveform

        :param self: Experiment class
        :param int t_idx: time = t_idx * time_resolution

        :return wfx, wfy: Generated waveforms
        """
        if self.proj_method == 'holographic':
            n_samples_x = round(samplerate * 1e-6 * conv_um_MHz / self.spot_spacing)
            all_freqs_x = rfftfreq(n_samples_x, 1e6 / samplerate)
            idx_x = np.abs(all_freqs_x - self.xfreq_start).argmin()
            assert np.isclose(all_freqs_x[idx_x], self.xfreq_start), 'Problem with xfreqstart lookup'
            # Add all frequencies not present in the signal
            I_x_expanded = np.pad(self.I_x[:, t_idx], (idx_x, all_freqs_x.size-self.I_x.shape[0]-idx_x), 'constant', constant_values=0)
            phase_mask_x = Ger_Sax_algo_1d(I_x_expanded, self.GS_iter, parallel=self.GS_parallel, fixed_seed=self.fixed_seed)
            ampl_x_expanded = np.sqrt(I_x_expanded) * np.exp(phase_mask_x * 1j)
            wfx = all_freqs_x.size * irfft(ampl_x_expanded, n_samples_x)

            # Same structure for y
            n_samples_y = round(samplerate * 1e-6 * conv_um_MHz / (self.spot_spacing * self.spacing_frac))
            all_freqs_y = rfftfreq(n_samples_y, 1e6 / samplerate)
            idx_y = np.abs(all_freqs_y - self.yfreq_start).argmin()
            assert np.isclose(all_freqs_y[idx_y], self.yfreq_start), 'Problem with yfreqstart lookup'
            # Add all frequencies not present in the signal
            I_y_expanded = np.pad(self.I_y[:, t_idx], (idx_y, all_freqs_y.size-self.I_y.shape[0]-idx_y), 'constant', constant_values=0)
            phase_mask_y = Ger_Sax_algo_1d(I_y_expanded, self.GS_iter, parallel=self.GS_parallel, fixed_seed=self.fixed_seed)
            ampl_y_expanded = np.sqrt(I_y_expanded) * np.exp(phase_mask_y * 1j)
            wfy = all_freqs_y.size * irfft(ampl_y_expanded, n_samples_y)

            # Tile the waveforms such that they have equal length
            if self.tight_direction == 'y':
                wfx, wfy = np.tile(wfx, self.DeltaN), np.tile(wfy, max(self.DeltaN - 1, 1))
            else:
                wfx, wfy = np.tile(wfx, max(self.DeltaN - 1, 1)), np.tile(wfy, self.DeltaN)
        elif self.proj_method == 'linescanning':
            wfx, wfy = StaticPatternScan(ampl_array=np.hstack((np.sqrt(self.I_x[:, t_idx]), np.sqrt(self.I_y[:, t_idx]))), fx_array_0=self.xfreq_start + np.arange(self.I_x.shape[0])*self.spot_spacing / conv_um_MHz, 
                                         tsamp=np.linspace(0, 1e-6*conv_um_MHz / self.spot_spacing, int(conv_um_MHz * samplerate *1e-6 / self.spot_spacing), endpoint=False), fy_array_0=self.yfreq_start + np.arange(self.I_y.shape[0])*self.spot_spacing / conv_um_MHz, scan_direction=self.scan_direction, 
                                         spot_amp=1, option_phase=self.phase_option, n_frames=self.n_frames, tdead=self.tdead, reps=self.reps, fixed_seed=self.fixed_seed)
        else:
            raise KeyError('proj method not supported')
        # Add dead time per frame
        wfx = np.append(wfx, np.zeros(int(self.tdead_frame*1e-6*samplerate)))
        wfy = np.append(wfy, np.zeros(int(self.tdead_frame*1e-6*samplerate)))
        assert wfx.size == wfy.size, f'Waveforms have unequal length wfx={wfx.size} wfy={wfy.size}'
        return wfx, wfy
    
    def generate_wf_period(self):
        """Generate the waveforms for 1 period of the driving signal"""
        assert self.I_x.ndim == self.I_y.ndim, 'I_x and I_y have different dimensions'
        if not self.corr_factors_applied:
            print('Warning: Correction factors not applied')
        n_samp_frame = round(self.time_res*1e-6*samplerate)
        # Match the waveform duration to period up to samplerate accuracy instead of time_res accuracy
        # Even with the correction of tdead_frame_extra, there still may be some empty samples at the end to match the desired period as well as possible
        n_extra_samp = round((self.drive_period_ms*1e3 % self.time_res) * samplerate * 1e-6)

        self.wfx_period, self.wfy_period = np.zeros(n_samp_frame*self.I_x.shape[1] + n_extra_samp), np.zeros(n_samp_frame*self.I_x.shape[1] + n_extra_samp)
        for idx in range(self.I_x.shape[1]):
            wfx_temp, wfy_temp = self.generate_wf_single(t_idx=idx)
            self.wfx_period[idx*n_samp_frame:(idx+1)*n_samp_frame] = wfx_temp
            self.wfy_period[idx*n_samp_frame:(idx+1)*n_samp_frame] = wfy_temp
    
    def balance_waveforms(self):
        """
        Balance the amplitudes of x and y waveforms by scaling based on their relative magnitudes.
        
        Adjusts wfx_period and wfy_period so that the maximum amplitudes are balanced
        according to the i_ref parameter.
        """
        beta = np.max(np.abs(self.wfy_period)) / np.max(np.abs(self.wfx_period))
        self.wfx_period *= self.i_ref**0.25 * self.extra_i**0.25 * np.sqrt(beta)
        self.wfy_period *= self.i_ref**0.25 * self.extra_i**0.25 / np.sqrt(beta)
        self.max_wf = np.max(np.abs(self.wfx_period))
        self.print_params.append('max_wf')

    def ramp_wf(self, ramp_profile=lambda x: x):
        """Generate a continuously ramped waveform with previously generated waveforms. 
        
        :param float t_ramp_ms: the duration of the ramping in milliseconds
        :param func ramp_profile: function of ramping, expect inputs between 0 and 1, f(1)=1; 0<=f(x)<=1. Default linear f(x)=x
        """
        n_samp_ramp = int(samplerate * 1e-3 * self.t_ramp_ms)
        n_samp_period = self.wfx_period.size
        n_periods = int(np.ceil(n_samp_ramp / n_samp_period))#.astype('int64')
        ramp = np.append(ramp_profile(np.linspace(0, 1, n_samp_ramp, endpoint=False)), np.ones(n_periods*n_samp_period - n_samp_ramp))
        self.ramp_wfx = ramp**0.25 * np.tile(self.wfx_period, n_periods)
        self.ramp_wfy = ramp**0.25 * np.tile(self.wfy_period, n_periods)

    def convert_binary_waveforms(self):
        """Save waveforms"""
        assert (np.abs(self.wfx_period)<=1).all(), f'wfx out of bounds max={np.max(np.abs(self.wfx_period)):.2f}'
        assert (np.abs(self.wfy_period)<=1).all(), f'wfy out of bounds max={np.max(np.abs(self.wfy_period)):.2f}'
   
        wfx_drive_b = np.ascontiguousarray(self.wfx_period * (2 ** 15 -1), dtype='i2')
        wfy_drive_b = np.ascontiguousarray(self.wfy_period * (2 ** 15 -1), dtype='i2')
 
        # Channel synchronization
        wfy_drive_b = np.roll(wfy_drive_b, round(1.9e-6*samplerate))
 
        if self.verbose: print(self.table_print())
 
        if self.t_ramp_ms:
            self.ramp_wf(ramp_profile=self.ramp_profile)
            n_cycles = int(np.ceil(self.dur_tot_ms / self.drive_period_ms - self.ramp_wfx.size / self.wfx_period.size))#, dtype='int64')
            wfx_tot_b = np.hstack((np.ascontiguousarray(self.ramp_wfx * (2 ** 15 -1), dtype='i2'), wfx_drive_b))
            wfy_tot_b = np.hstack((np.ascontiguousarray(self.ramp_wfy * (2 ** 15 -1), dtype='i2'), wfy_drive_b))
            cycles_per_waveform = [1, n_cycles]
            samples_per_waveform = [self.ramp_wfx.size, self.wfx_period.size]
            del self.ramp_wfx, self.ramp_wfy # Be careful with memory
        else:
            cycles_per_waveform = [int(np.ceil(self.dur_tot_ms / self.drive_period_ms))] # Always round upward. Let RF-amplifiers cut precisely at dur_tot_ms
            samples_per_waveform = [min(self.wfx_period.size, self.dur_tot_ms*1e-3*samplerate)]
            wfx_tot_b = wfx_drive_b
            wfy_tot_b = wfy_drive_b
       
        del wfx_drive_b, wfy_drive_b
        return wfx_tot_b, wfy_tot_b, cycles_per_waveform, samples_per_waveform

    def save_waveforms(self):
        """Save waveforms"""
        assert (np.abs(self.wfx_period)<=1).all(), f'wfx out of bounds max={np.max(np.abs(self.wfx_period)):.2f}'
        assert (np.abs(self.wfy_period)<=1).all(), f'wfy out of bounds max={np.max(np.abs(self.wfy_period)):.2f}'
    
        wfx_drive_b = np.ascontiguousarray(self.wfx_period * (2 ** 15 -1), dtype='i2')
        wfy_drive_b = np.ascontiguousarray(self.wfy_period * (2 ** 15 -1), dtype='i2')

        # Channel synchronization
        wfy_drive_b = np.roll(wfy_drive_b, round(1.9e-6*samplerate))

        if self.verbose: print(self.table_print())

        if self.t_ramp_ms:
            self.ramp_wf(ramp_profile=self.ramp_profile)
            n_cycles = int(np.ceil(self.dur_tot_ms / self.drive_period_ms - self.ramp_wfx.size / self.wfx_period.size))#, dtype='int64')
            wfx_tot_b = np.hstack((np.ascontiguousarray(self.ramp_wfx * (2 ** 15 -1), dtype='i2'), wfx_drive_b))
            wfy_tot_b = np.hstack((np.ascontiguousarray(self.ramp_wfy * (2 ** 15 -1), dtype='i2'), wfy_drive_b))
            cycles_per_waveform = [1, n_cycles]
            samples_per_waveform = [self.ramp_wfx.size, self.wfx_period.size]
            del self.ramp_wfx, self.ramp_wfy # Be careful with memory
        else:
            cycles_per_waveform = [int(np.ceil(self.dur_tot_ms / self.drive_period_ms))] # Always round upward. Let RF-amplifiers cut precisely at dur_tot_ms
            samples_per_waveform = [min(self.wfx_period.size, self.dur_tot_ms*1e-3*samplerate)]
            wfx_tot_b = wfx_drive_b
            wfy_tot_b = wfy_drive_b

        with h5py.File(self.awg_file_name, "w") as file:
            awgs_PUT = file.create_group("awgs")
            app_ID = awgs_PUT.create_group("80 MHz AWG (18)")

            job_dict_i = {'ch_1_waveforms':wfx_tot_b, 'ch_2_waveforms':wfy_tot_b, 'cycles_per_waveform':cycles_per_waveform, 'samples_per_waveform':samples_per_waveform}
            for key_i in job_dict_i.keys():
                app_ID.create_dataset(key_i, data= job_dict_i[key_i])
            app_ID["save"]=json.dumps(False)

        print('Saving done')
        # Account for MemoryError: std::bad_alloc
        del wfx_drive_b, wfy_drive_b, wfx_tot_b, wfy_tot_b

    def save_essentials(self, path):
        # TODO save_essentials
        # Save all init param to a file. Later in init, make a check. If this trips, the code can skip multiple expensive steps
        # Save the wf_period to this file
        # What file to use? json? not good for wfs, except if they are in separate file. h5? Good for datasets and such, but harder to work with? 
        raise ValueError('Method not implemented yet')
 
##########################
#  Hardware limitations  #
##########################
 
def calc_memory(nr_samples, onboardMemory_bits=4096*1E6):
    """
 
    :param nr_samples: int or float, number of sample-points in the waveform.
    :param onboardMemory_bits: int or float, number of bits in the total onboard memory of the AWG channel.
    :return: Boolean, whether the current waveform will fit in the memory of the AWG channel.
    """
    return nr_samples * 16 < onboardMemory_bits
 
 
def calc_nr_durs(dur, samplerate, onboardMemory_bits=4096*1E6):
    """
 
    :param dur: int or float, duration (s) of one period in the waveform.
    :param samplerate: int or float, rate (/s) at which a new sample-point is generated in the waveform.
    :param onboardMemory_bits: int or float, number of bits in the total onboard memory of the AWG channel.
    :return: int or float, number of periods that can fit in the memory of the AWG channel.
    """
    nsamples_bits = dur * samplerate * 16 # Units: s * S/s * bits/S
    return onboardMemory_bits / nsamples_bits

########################################################################################################################

def sum_sine(fs, As, dur, samples, phase_option=0, phis=None):
    """
    Creates a waveform from a sum of sinusoids at given frequency 'fs' with
    a given amplitude 'As', of a given duration 'dur', and points 'samples'
    Commented expressions for random, equal, or uniform spaced phase of sines
    """
    # TODO: adapted the tsamp definition
     #tsamp = np.linspace(0, dur, int(samples) )
    tsamp = np.linspace(0, dur, int(samples),endpoint=False)
    ssine = np.zeros(int(samples))
    for i in range(0, len(fs)):
        if not phis.all():
            if phase_option == 0:
                phi = 0
            elif phase_option == 1:
                phi = np.random.rand()*2*np.pi
            elif phase_option == 2:
                phi = 2 * np.pi * i / len(fs)  # Uniformally Spaced Phase
            elif phase_option == 3:
                delta_phi = 5*np.pi / 6 #*np.random.rand()
                phi = i*delta_phi
                #print(180*phi/np.pi)
            else:
                #if i==0:
                #    print('Unvalid phase option. Proceeding with default=0')
                phi = i*np.deg2rad(phase_option) # for 2spot interference
        else:
            phi = phis[i]
        ssine = ssine + As[i]*np.sin(2*np.pi*fs[i]*tsamp*10**6 + phi)

    return ssine, tsamp


# The latest  correction factors (30/10/2025). They are: measured_power_at_freq / measured_power_at_center_freq
correction_factors_x = np.array([1.31914, 1.32315, 1.32844, 1.31180, 1.35683, 1.35751, 1.37788, 1.38992, 1.45394, 1.41997, 1.46902, 1.41077, 1.45012, 1.48938, 1.45864, 1.46295, 1.50710, 1.47234, 1.45678, 1.47920, 1.44464, 
                                 1.51375, 1.48634, 1.46931, 1.48116, 1.47000, 1.49241, 1.47029, 1.47000, 1.41958, 1.45717, 1.44209, 1.42976, 1.43279, 1.45149, 1.44395, 1.42898, 1.45531, 1.43632, 1.43808, 1.44258, 1.43730, 
                                 1.42095, 1.43495, 1.43524, 1.38894, 1.42947, 1.38835, 1.39158, 1.42457, 1.41341, 1.41645, 1.40724, 1.37239, 1.41028, 1.41263, 1.44200, 1.44033, 1.43417, 1.42663, 1.44581, 1.42545, 1.43632, 
                                 1.44660, 1.41909, 1.40186, 1.42996, 1.44366, 1.41958, 1.42212, 1.40822, 1.43191, 1.43563, 1.44043, 1.45698, 1.44699, 1.46197, 1.46314, 1.47264, 1.47871, 1.47988, 1.47636, 1.50817, 1.48634, 
                                 1.53040, 1.52521, 1.52022, 1.50602, 1.49858, 1.52237, 1.52609, 1.53010, 1.49339, 1.52775, 1.51405, 1.49222, 1.48184, 1.47068, 1.46686, 1.47166, 1.41312, 1.41243, 1.39187, 1.42741, 1.44728, 
                                 1.46481, 1.43299, 1.47186, 1.49183, 1.51238, 1.52746, 1.51307, 1.52237, 1.50886, 1.53392, 1.53891, 1.53206, 1.52345, 1.51326, 1.52423, 1.50857, 1.51336, 1.47988, 1.47469, 1.46804, 1.47019, 
                                 1.46667, 1.46412, 1.44317, 1.42780, 1.41273, 1.37876, 1.36907, 1.34146, 1.33744, 1.32335, 1.31845, 1.30582, 1.31062, 1.30113, 1.29897, 1.31248, 1.30445, 1.28292, 1.27186, 1.29075, 1.28311, 
                                 1.27675, 1.27254, 1.27274, 1.25825, 1.23270, 1.22183, 1.21400, 1.19667, 1.17631, 1.15820, 1.12550, 1.12354, 1.11131, 1.11366, 1.09682, 1.08380, 1.06941, 1.07714, 1.09251, 1.08585, 1.07283, 
                                 1.06207, 1.05658, 1.05228, 1.04709, 1.02976, 1.03289, 1.02085, 1.00000, 1.00734, 0.98463, 0.97366, 0.97866, 0.98101, 0.96281, 0.94662, 0.94428, 0.94328, 0.94145, 0.92242, 0.91290, 0.90006, 
                                 0.91458, 0.90873, 0.90857, 0.90523, 0.88804, 0.88536, 0.87736, 0.86851, 0.86902, 0.86100, 0.84648, 0.85416, 0.85749, 0.84131, 0.84681, 0.83697, 0.83429, 0.82112, 0.81461, 0.81744, 0.81560, 
                                 0.81044, 0.81143, 0.81310, 0.80776, 0.81110, 0.80776, 0.81511, 0.80710, 0.79074, 0.80225, 0.79658, 0.79658, 0.78890, 0.78122, 0.77888, 0.78707, 0.78089, 0.78974, 0.78890, 0.78489, 0.78406, 
                                 0.77989, 0.77505, 0.78189, 0.77238, 0.76420, 0.76754, 0.76536, 0.75819, 0.77438, 0.75502, 0.75970, 0.75402, 0.75970, 0.75435, 0.75853, 0.76153, 0.75970, 0.76654, 0.76487, 0.76737, 0.77571, 
                                 0.77789, 0.78373, 0.77589, 0.77438, 0.78072, 0.78757, 0.78940, 0.78940, 0.78072, 0.77338, 0.77739, 0.76604, 0.76520, 0.76370, 0.76169, 0.76003, 0.75802, 0.75202, 0.75986, 0.75669, 0.75669, 
                                 0.75819, 0.75169, 0.74852, 0.74251, 0.73900, 0.74284, 0.74050, 0.73299, 0.73366, 0.72932, 0.72364, 0.71847, 0.71113, 0.70896, 0.70479, 0.69828, 0.69527, 0.68843, 0.68492, 0.68293, 0.68042, 
                                 0.67057, 0.67024, 0.67275, 0.66891, 0.65838, 0.65889, 0.65889, 0.65705, 0.65088, 0.64353, 0.64437, 0.64521, 0.64003, 0.63335, 0.62734, 0.62101, 0.62317, 0.61066, 0.61099, 0.60131, 0.59664, 
                                 0.58796, 0.58796, 0.57878, 0.57160, 0.55558, 0.54574, 0.53989, 0.53572, 0.53122, 0.52771, 0.52254, 0.51803, 0.51403, 0.50584, 0.49933, 0.49350, 0.48766, 0.47847, 0.47464, 0.46312, 0.46062, 
                                 0.45093, 0.44393, 0.43391, 0.42590, 0.42107, 0.41489, 0.40738, 0.39987, 0.39469, 0.38986, 0.38768, 0.38184, 0.37367, 0.36900, 0.36432])
correction_factors_y = np.array([0.63371, 0.64728, 0.66103, 0.67834, 0.67415, 0.70876, 0.69245, 0.72288, 0.72962, 0.74227, 0.74691, 0.75354, 0.78041, 0.76914, 0.80414, 0.80447, 0.80181, 0.82637, 0.83797, 0.88375, 0.86998, 
                                  0.87363, 0.89255, 0.89487, 0.92224, 0.92705, 0.91859, 0.93634, 0.92854, 0.95392, 0.96569, 0.97383, 0.98015, 0.97283, 0.99309, 1.00798, 1.00769, 1.01975, 1.01148, 1.04699, 1.02578, 1.05507, 
                                  1.03668, 1.06674, 1.07054, 1.06723, 1.06305, 1.09243, 1.10002, 1.12142, 1.12065, 1.12259, 1.12493, 1.12609, 1.11247, 1.12561, 1.13709, 1.11948, 1.12522, 1.14769, 1.16180, 1.16404, 1.19196, 
                                  1.19459, 1.22261, 1.22923, 1.27252, 1.26114, 1.26513, 1.26425, 1.28303, 1.28255, 1.30162, 1.30872, 1.31504, 1.33012, 1.32730, 1.34277, 1.35338, 1.38879, 1.37507, 1.38938, 1.40144, 1.41419, 
                                  1.42032, 1.41273, 1.42868, 1.41817, 1.43131, 1.43180, 1.43549, 1.46235, 1.44357, 1.45933, 1.45904, 1.44853, 1.47480, 1.47645, 1.47140, 1.47558, 1.48939, 1.47772, 1.48044, 1.47577, 1.48434, 
                                  1.48687, 1.48618, 1.46974, 1.45719, 1.48219, 1.47198, 1.47821, 1.46381, 1.46809, 1.45067, 1.45836, 1.42070, 1.45233, 1.42547, 1.42119, 1.43345, 1.41856, 1.41983, 1.40737, 1.39161, 1.37410, 
                                  1.36048, 1.34520, 1.34618, 1.34734, 1.35649, 1.36184, 1.34287, 1.33032, 1.32098, 1.31504, 1.31319, 1.31738, 1.30609, 1.30191, 1.27038, 1.28799, 1.26922, 1.27175, 1.26377, 1.25384, 1.24801, 
                                  1.23156, 1.21930, 1.21006, 1.18749, 1.18262, 1.17640, 1.18184, 1.17533, 1.16628, 1.14166, 1.14156, 1.14332, 1.12327, 1.10469, 1.10605, 1.08941, 1.09739, 1.08494, 1.07501, 1.05789, 1.06178, 
                                  1.04797, 1.05390, 1.03366, 1.04437, 1.03950, 1.01226, 1.01002, 1.00000, 0.99689, 0.99027, 0.99766, 0.98609, 0.97167, 0.97500, 0.97266, 0.95989, 0.94712, 0.93053, 0.93783, 0.93352, 0.92257, 
                                  0.90465, 0.88923, 0.88210, 0.88641, 0.88475, 0.89189, 0.89702, 0.90217, 0.88840, 0.89785, 0.89935, 0.90465, 0.91676, 0.90565, 0.90980, 0.90731, 0.90930, 0.91610, 0.90996, 0.90366, 0.90200, 
                                  0.89968, 0.90399, 0.90532, 0.90349, 0.90797, 0.89669, 0.89437, 0.89918, 0.89504, 0.89271, 0.89586, 0.88674, 0.89620, 0.88591, 0.89238, 0.89304, 0.87447, 0.88127, 0.89321, 0.88890, 0.89088, 
                                  0.86998, 0.87812, 0.88459, 0.88160, 0.87762, 0.88426, 0.87480, 0.87845, 0.88027, 0.88641, 0.89968, 0.90167, 0.91196, 0.90100, 0.90914, 0.90532, 0.90084, 0.90018, 0.89785, 0.90250, 0.89437, 
                                  0.89205, 0.89122, 0.89636, 0.89271, 0.88276, 0.87281, 0.87645, 0.87712, 0.86750, 0.85671, 0.85191, 0.85257, 0.85655, 0.85556, 0.85191, 0.85075, 0.84345, 0.84527, 0.84710, 0.84677, 0.84643, 
                                  0.84063, 0.83300, 0.83035, 0.82802, 0.81708, 0.81625, 0.81425, 0.80762, 0.79585, 0.78888, 0.78971, 0.77876, 0.77544, 0.76714, 0.76118, 0.75305, 0.74940, 0.73944, 0.73314, 0.73530, 0.73117, 
                                  0.71836, 0.71086, 0.70127, 0.69519, 0.68512, 0.67920, 0.67859, 0.66781, 0.66077, 0.65504, 0.64789, 0.63784, 0.63017, 0.62456, 0.62408, 0.61429, 0.60596, 0.60071, 0.58883, 0.57995, 0.56994, 
                                  0.56029, 0.55127, 0.54172, 0.53269, 0.52559, 0.52226, 0.51886, 0.52018, 0.51485, 0.51541, 0.51685, 0.51657, 0.51225, 0.50827, 0.50178, 0.49737, 0.48977, 0.48668, 0.48151, 0.47727, 0.47388, 
                                  0.47104, 0.46410, 0.46393, 0.45896, 0.45504, 0.44988, 0.44605, 0.44222, 0.43500, 0.43289, 0.42671, 0.42505, 0.42484, 0.41834, 0.41449])
f_for_correction_factors = np.linspace(35, 70, 351, endpoint=True)
assert len(correction_factors_x) == len(correction_factors_y) == 351, 'Error in length correction factor arrays'


def Ger_Sax_algo_1d(img, max_iter, change_rate=1e-18, fixed_seed=0, parallel=3):
    # retrieved from github user 'wxwang0104' edited
    # introduced stopping criterion by 'rdecruyen'
    l = len(img)
    assert (img >= 0).all(), 'img must be non-negative for GS'
    if fixed_seed:
        # Option to fix the seed to make reproducible phase masks
        np.random.seed(fixed_seed)
    # Generate the initial random phases of the signal (_s) (=image) 
    pm_s = 2*np.pi*np.random.rand(l, parallel)
    if max_iter < 0: 
        return np.zeros(l) #np.linspace(0, 2*np.pi, l)#np.ones(l)
    elif max_iter == 0: # return a random phase mask
        return pm_s[:, 0]
    am_s = np.sqrt(img) # signal amplitudes
    am_s = np.tile(am_s, parallel).reshape((parallel, l)).T # Repeat for parralel runs
    # Target amplitudes in frequency (_f) domain
    am_f = np.ones((l, parallel)) #/img.sum() Normalization is not needed since only phases are optimized

    signal_s = am_s*np.exp(pm_s * 1j)
    prev_pm_f = None
    prev_min_ampl = np.inf # init such that if statement triggers on first iteration
    for iter in range(max_iter):
        signal_f = np.fft.fft(signal_s, axis=0)
        pm_f = np.angle(signal_f)
        max_am = np.max(np.abs(signal_f), axis=0)
        signal_f = am_f*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft(signal_f, axis=0)
        pm_s = np.angle(signal_s)
        signal_s = am_s*np.exp(pm_s * 1j)

        if prev_pm_f is not None: # Old stopping criterion, not a good one
            if np.mean(np.abs(pm_f - prev_pm_f)) < change_rate:
                #print(f'GS stopped after {iter} iterations')
                break
        prev_pm_f = pm_f

        if np.min(max_am) < prev_min_ampl:
            # Save the lowest maximal amplitude and the corresponding phase mask
            prev_min_ampl = np.min(max_am)
            pm_out = pm_s[:, np.argmin(max_am)]
    return pm_out

def equalize_amplitudes(wf1, wf2):
    """Equalize the amplitudes of both waveforms while keeping the total intensity constant"""
    beta = np.max(np.abs(wf2)) / np.max(np.abs(wf1))
    wf1 *= np.sqrt(beta)
    wf2 /= np.sqrt(beta)
    return wf1, wf2

def pattern_to_arraysSumSin(I_ny:np.ndarray, I_nx:np.ndarray, dur:float, implement_corr=True, freq_c_x=52.5, freq_c_y=52.5):
    """
    Go from pattern of 2 1D arrays (generated by 'draw pattern') to input of Sum of Sines methods. 

    :param I_ny: Intensity pattern in y direction
    :param I_nx: Intensity pattern in x direction
    :param dur: period of the signal. This fixes the spot spacing to 1/dur [MHz] or 3.864/dur[µs] [µm]
    :param implement_corr: wheter or not to use the AOD correction factors
    :param freq_c_x: center frequency in x direction
    :param freq_c_y: center frequency in y direction

    :return ampl_array: the spot amplitudes 
    :return fx_array: the series of xfreqs present in the driving signal in MHz
    :return fy_array: the series of yfreqs present in the driving signal in MHz
    """
    N_p = len(I_ny)
    assert N_p == len(I_nx), 'I_ny and I_nx must have the same length'
    assert (I_ny >= 0).all(), 'I_ny cannot contain negative'
    assert (I_nx >= 0).all(), 'I_nx cannot contain negative'
    bandwidth = 35 # MHz
    if dur < 1:
        dur *= 10**6 #convert to µs
    corr_for_dur = False
    if corr_for_dur:
        bandwidth_corr = round(bandwidth*dur)/dur
        freq_c_x_corr = round(freq_c_x*dur)/dur
        freq_c_y_corr = round(freq_c_y*dur)/dur
    else:
        bandwidth_corr = bandwidth
        freq_c_x_corr = freq_c_x
        freq_c_y_corr = freq_c_y
    min_freq_x = freq_c_x_corr - bandwidth_corr/2
    max_freq_x = min_freq_x + bandwidth_corr
    min_freq_y = freq_c_y_corr - bandwidth_corr/2
    max_freq_y = min_freq_y + bandwidth_corr
    fx_array = np.linspace(min_freq_x, max_freq_x, num=N_p, endpoint=False)
    fy_array = np.linspace(min_freq_y, max_freq_y, num=N_p, endpoint=False)
    if implement_corr:
        #corr_array_x = np.interp(x=fx_array, xp=np.linspace(45.87125, 59.12875, 51), fp=temp_fy_factors_sqrt)
        #corr_array_y = np.interp(x=fy_array, xp=np.linspace(45.87125, 59.12875, 51), fp=temp_fy_factors_sqrt)
        corr_array_x = np.interp(x=fx_array, xp=f_for_correction_factors, fp=correction_factors_x)**-0.5
        corr_array_y = np.interp(x=fy_array, xp=f_for_correction_factors, fp=correction_factors_y)**-0.5
        #corr_array_y = np.ones_like(fy_array)

    else:
        corr_array_x = np.ones_like(fx_array)
        corr_array_y = np.ones_like(fy_array)
    fx_array = fx_array[I_nx > 0]
    fy_array = fy_array[I_ny > 0]
    ampl_array = np.concatenate(((np.sqrt(I_nx)*corr_array_x)[I_nx > 0], (np.sqrt(I_ny)*corr_array_y)[I_ny > 0]))
    assert (35 <= fx_array).all() & (70 >= fx_array).all() & (35 <= fy_array).all() & (70 >= fy_array).all(), 'Pattern lies outside AOD bandwidth'
    return ampl_array, fx_array, fy_array

def StaticPatternScan(ampl_array:np.ndarray, fx_array_0:np.ndarray, fy_array_0:np.ndarray, tsamp:np.ndarray, scan_direction:str='x', spot_amp:float=1., option_phase:int=0, n_frames:int=1, tdead:float=0., reps:int=1, fixed_seed=0):
    """
    This function is the analogue of StaticPattern which scans in the scan direction

    This should be equal to lineSingleScanning if n_lines=n_spots / n_frames
    Crosstalk is mitigated by dead time
    fx_array is assumed to be sorted

    :param ampl_array: 1D array of spot amplitudes corresponding with spots at locations
    :param fx_array_0: 1D array with frequencies corresponding with spot locations in the x direction
    :param fy_array_0: 1D array with frequencies corresponding with spot locations in the y direction
    :param tsamp: 1D array with the sampling points
    :param scan_direction: ='x' or 'y'
    :param spot_amp: The amplitude of one spot (=i_tot**0.25)
    :param option_phase: Which procedure to use when giving relative phases in the waveform. Use 1 for random phases
    :param n_frames: The number of frames the pattern is split in. Project nr_spots_scan_direction/n_frames lines at once (on average)
    :param tdead: Number of µs between sequential frames where nothing is projected. 
    :param reps: repeat each frame this number of times to mitigate crosstalk

    :return waveforms: wfx, wfy
    """
    assert len(ampl_array) == len(fx_array_0) + len(fy_array_0), 'ampl array has wrong shape'
    
    scan_direction = scan_direction.lower() # Allow for uppercase
    N_x_0 = len(fx_array_0)
    x_amp_0 = ampl_array[:N_x_0]
    y_amp_0 = ampl_array[N_x_0:]
    # write code as if scan direction is x, otherwise change names in beginning and back at the end
    if scan_direction =='x':
        x_amp = x_amp_0
        y_amp = y_amp_0
        fx_array = fx_array_0
        fy_array = fy_array_0
        N_x = N_x_0
    elif scan_direction =='y':
        x_amp = y_amp_0
        y_amp = x_amp_0
        fx_array = fy_array_0
        fy_array = fx_array_0
        N_x = len(ampl_array) - N_x_0
    else:
        raise KeyError(f"'{scan_direction}' is not a valid scan direction")
    samplerate = 1 / tsamp[1]
    dur = len(tsamp) / samplerate
    tdead *= 1e-6 # conversion to s
    if n_frames > N_x:
        message = 'WARNING: used more frames in linescanning than needed'
        print('--'+len(message)*' '+'--\n  '+message+'  \n--'+len(message)*' '+'--')
        n_frames = min(n_frames, N_x)
    wfx_tot = np.zeros(0)

    if fixed_seed:
        # Option to fix the seed to make reproducible phase masks
        #seed = int(np.random.rand() * (2**32 - 1))
        #np.random.seed(seed)
        #print(seed)
        np.random.seed(fixed_seed)

    for i in range(n_frames):
        # add approx the number of lines, split by n_frames
        fx = fx_array[i::n_frames]
        phis_x = np.random.rand(len(fx))*2*np.pi
        wfx_temp, _ = sum_sine(fx, x_amp[i::n_frames]*np.sqrt(n_frames)*np.sqrt(1+tdead/(reps*dur)), dur=dur, samples=len(tsamp), phase_option=option_phase, phis=phis_x)
        wfx_temp = np.tile(wfx_temp, reps)
        wfx_tot = np.hstack((wfx_tot, wfx_temp, np.zeros(int(tdead*samplerate))))
    
    dead_t_in_y = False
    phis_y = np.random.rand(len(fy_array))*2*np.pi
    if dead_t_in_y:
        wfy_tot, _ = sum_sine(fy_array, y_amp, dur=dur, samples=len(tsamp),phase_option=option_phase, phis=phis_y)
        wfy_tot = np.tile(wfy_tot, reps)
        wfy_tot = np.hstack((wfy_tot, np.zeros(int(tdead*1e-6*samplerate))))
    else:
        wfy_tot, _ = sum_sine(fy_array, y_amp, dur=reps*dur+tdead, samples=round(samplerate*(reps*dur+tdead)),phase_option=option_phase, phis=phis_y)
        wfy_tot = np.roll(wfy_tot, int(samplerate*tdead /2)) # Ensure the phase jump is inside the dead time of x
    # match the length of wfy to the length of wfx
    wfy_tot = np.tile(wfy_tot, n_frames)
    assert len(wfx_tot) == len(wfy_tot), f'len wf not equal wfx={len(wfx_tot)} wfy={len(wfy_tot)}'
    wfx_tot *= spot_amp
    wfy_tot *= spot_amp
    wfx_tot, wfy_tot = equalize_amplitudes(wfx_tot, wfy_tot)

    if scan_direction=='x':
        return wfx_tot, wfy_tot
    else:
        return wfy_tot, wfx_tot

def non_negative_svd(X, init='svd+', tol=1e-10, max_iter=1000, check_convergence=False):
    """
    Perform Non-Negative Singular Value Decomposition (NN-SVD) using the exact iterative scheme from the paper.

    :param (np.ndarray) X: Input matrix of shape 
    :param (str) init: Initialisation method for nnsvd, choose from 'svd+', 'svd0', 'random'
    :param (float) tol: Tolerance for convergence.
    :param (int) max_iter: Maximum number of iterations.
    :param (bool) Check_convergence: wheter to return the errors

    :return (np.ndarray) U: Non-negative left singular matrix.
    :return (np.ndarray) S: Diagonal matrix of singular values.
    :return (np.ndarray) Vt: Non-negative right singular matrix (transposed).
    :return (np.ndarray) e:  1D array with relative error in function of iteration number, only returned when check_convergence
    """
    # Initialize with random non-negative matrices
    eps = 1e-15
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    S = np.diag(s)
    if 'svd+' in init:
        U = np.abs(U)
        Vt = np.abs(Vt)
    elif init == 'svd0':
        U = np.where(U>0, U, 1e-4)
        Vt = np.where(Vt>0, Vt, 1e-4)
    elif 'random' in init:
        U = np.abs(np.random.rand(*U.shape))
        S = np.diag(np.ones(s.shape)) 
        Vt = np.abs(np.random.rand(*Vt.shape))
    else:
        raise KeyError('Not a valid initialisation method')
    if 'rank' in init:
        rank = int(init.split('rank')[-1])
        s[rank:] = 0 #eps
        S = np.diag(s)

    errors = np.zeros(max_iter)
    for iteration in range(max_iter):
        # Update U, S and Vt
        U *= (X @ Vt.T @ S) / (U @ S @ Vt @ Vt.T @ S + eps)
        S *= (U.T @ X @ Vt.T) / (U.T @ U @ S @ Vt @ Vt.T + eps)
        Vt *= ((X.T @ U @ S) / (Vt.T @ S @ U.T @ U @ S + eps)).T

        # Check convergence (Frobenius norm)
        reconstruction = U @ S @ Vt
        if 'rank' in init:
            pass #print(np.sqrt(np.sum(reconstruction**2)))
            #print(np.linalg.norm(X, 'fro'))
            #plt.imshow(reconstruction)
            #plt.colorbar()
            #plt.show()
        rel_error = np.linalg.norm(X - reconstruction, 'fro') / np.linalg.norm(X, 'fro')
        errors[iteration] = rel_error
        if rel_error < tol:
            print(f'Converged after {iteration+1} iterations')
            break
    # normalise the U and V columns
    for i in range(S.shape[0]):
        sum_u = np.sum(U[:, i])
        sum_vt = np.sum(Vt[i, :])
        S[i, i] *= sum_u*sum_vt
        if S[i, i] > 1e-300: # prevent division by 0
            U[:, i] /= sum_u
            Vt[i, :] /= sum_vt
    # Reorder S, U, and Vt based on descending order of diagonal values in S
    order = np.argsort(-np.diag(S))
    S = np.diag(np.sort(np.diag(S))[::-1])
    U = U[:, order]
    Vt = Vt[order, :]
    if check_convergence:
        return U, S, Vt, errors
    else:
        return U, S, Vt

def AS_synth_dim(exp_dur_ms=30, i_ref=1.25e-5, detuning=14.54545454, xfreq=52.5, yfreq=52.5, awg_file_name=None, t_ramp_ms=None):
    with np.load('nn_hopping_test.npz', allow_pickle=True) as loaded:
        print(loaded.files)
        drive_raw_positions = loaded['drive_raw_positions']
        drive_raw_amplitudes = loaded['drive_raw_amplitudes']
        hyperparam = loaded['hyperparam'].item()
        l_ho = hyperparam['l_ho']
        size_x = l_ho * (np.ptp(drive_raw_positions) + 2)
        # Other params to edit in aod code
        size_y = 80 # [µm]
        detuning = 14 
        freq = 440 + detuning
        period_us = 1e6 / freq
        phase = 0
        offset = 0 # set higher to possibly increase fit fidelity of spots, has cost of higher power needed
        x = np.linspace(-size_x/2, size_x/2, 51) 
        y = np.linspace(-size_y/2, size_y /2, 11)
        
        def idrive_for_aod_code(x, y, t):
            # Energy units of hbar omega_x --> Need conversion factor from i_ref to energy
            return np.interp(x / l_ho, drive_raw_positions, drive_raw_amplitudes)*np.cos(2*np.pi*t/period_us + phase) + np.maximum(np.abs(np.min(drive_raw_amplitudes)), np.abs(np.max(drive_raw_amplitudes))) + offset
            
        t_list = [time.time()] # to profile where time is spent
        synth_exp = Experiment(title='AS Synth dim', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=1e3/freq, W_MHz=1.75, tight_direction='x', xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, tdead=1, awg_file_name="output.h5")
        t_list.append(time.time())
        drive = synth_exp.construct_ampl_from_func(x, y, idrive_for_aod_code, one_dim='x', one_dim_len=70)
        t_list.append(time.time())
        if False: # Generation and saving of animation takes at least 10s
            ani = synth_exp.animate_proj(x, y, synth_dim, frames_step=3, coord_x_slice=4, coord_y_slice=0, plot_residual_bool=False)
            ani.save('Example synthetic dim.gif')
        synth_exp.apply_correction_factors()
        synth_exp.generate_wf_period()
        t_list.append(time.time())
        synth_exp.balance_waveforms()
        t_list.append(time.time())
        if synth_exp.verbose: print(f'Execution times {np.array2string(np.diff(t_list), precision=2, floatmode='fixed')} s')

        synth_exp.save_waveforms()



def example_synth_dim(exp_dur_ms=30, i_ref=1.25e-5, detuning=14.54545454, xfreq=52.5, yfreq=52.5, awg_file_name=None, t_ramp_ms=None):
    size_y = 5 # µm Keep this small as pattern is constant in this direction
    size_x = 15 # µm
    freq = 440 + detuning # Hz
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 51) 
    y = np.linspace(-size_y/2, size_y /2, 11)
    def synth_dim(x, y, t):
        period_us = 1e6 /freq
        # t += period_us / 4 # More conveniend start
        t %= period_us
        #return np.heaviside(np.sign(period_us/2 -t)*x, 0.5) # Step function in time
        return (1+np.sin(2*np.pi*t / period_us))*np.heaviside(x, 0.5) /2 + (1-np.sin(2*np.pi*t/period_us))*np.heaviside(-x, 0.5) /2 + 1

    t_list = [time.time()] # to profile where time is spent
    synth_exp = Experiment(title='Synth dim', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=1e3/freq, W_MHz=1.75, tight_direction='x', xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, tdead=1)
    t_list.append(time.time())
    drive = synth_exp.construct_ampl_from_func(x, y, synth_dim, one_dim='x', one_dim_len=70)
    t_list.append(time.time())
    if False: # Generation and saving of animation takes at least 10s
        ani = synth_exp.animate_proj(x, y, synth_dim, frames_step=3, coord_x_slice=4, coord_y_slice=0, plot_residual_bool=False)
        ani.save('Example synthetic dim.gif')
    synth_exp.apply_correction_factors()
    synth_exp.generate_wf_period()
    t_list.append(time.time())
    synth_exp.balance_waveforms()
    t_list.append(time.time())
    if synth_exp.verbose: print(f'Execution times {np.array2string(np.diff(t_list), precision=2, floatmode='fixed')} s')

    synth_exp.save_waveforms()
    return {key: getattr(synth_exp, key) for key in synth_exp.print_params}, drive


def example_synth_dim_2drive(nfreq2=3, k2=1, exp_dur_ms=100, i_ref=1.25e-5, detuning=14.54545454, xfreq=52.5, yfreq=52.5, awg_file_name=None, t_ramp_ms=None):
    size_y = 5 # µm Keep this small as pattern is constant in this direction
    size_x = 15 # µm
    freq = 440 + detuning # Hz
    freq2 = nfreq2 * freq
    
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 51) 
    y = np.linspace(-size_y/2, size_y /2, 11)
    def synth_dim(x, y, t):
        period_us = 1e6 /freq
        # t += period_us / 4 # More conveniend start
        t %= period_us
        
        #return np.heaviside(np.sign(period_us/2 -t)*x, 0.5) # Step function in time
        return (1+np.sin(2*np.pi*t / period_us))*np.heaviside(x, 0.5) /2 + (1-np.sin(2*np.pi*t/period_us))*np.heaviside(-x, 0.5) /2 + k2*(1+np.sin(2*np.pi*t / period_us))*np.heaviside(x, 0.5) /2 + (1-np.sin(2*np.pi*t/period_us))*np.heaviside(-x, 0.5) /2 


    t_list = [time.time()] # to profile where time is spent
    synth_exp = Experiment(title='Synth dim', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=1e3/freq, W_MHz=1.75, tight_direction='x', xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, tdead=1)
    t_list.append(time.time())
    drive = synth_exp.construct_ampl_from_func(x, y, synth_dim, one_dim='x', one_dim_len=70)
    t_list.append(time.time())
    if False: # Generation and saving of animation takes at least 10s
        ani = synth_exp.animate_proj(x, y, synth_dim, frames_step=3, coord_x_slice=4, coord_y_slice=0, plot_residual_bool=False)
        ani.save('Example synthetic dim.gif')
    synth_exp.apply_correction_factors()
    synth_exp.generate_wf_period()
    t_list.append(time.time())
    synth_exp.balance_waveforms()
    t_list.append(time.time())
    if synth_exp.verbose: print(f'Execution times {np.array2string(np.diff(t_list), precision=2, floatmode='fixed')} s')

    synth_exp.save_waveforms()

    return {key: getattr(synth_exp, key) for key in synth_exp.print_params}, drive

def create_flat_rectangle(exp_dur_ms=30, exp_gain_ms=0, i_ref=1e-5,  W_MHz=1.75, reps=2, xfreq=52.5, yfreq=52.5, t_ramp_ms=0, fixed_seed=0):
    size_y = 80 # µm
    size_x = 20 # µm
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 21)
    y = np.linspace(-size_y/2, size_y /2, 81)
    def flat_rectangle(x, y, t):  
        return np.ones_like(x)*np.ones_like(y)
 
    t_list = [time.time()] # to profile where time is spent
    spot_spacing = 0.9
    synth_exp = Experiment(title='Flat rect', verbose=True, dur_tot_ms=exp_dur_ms+exp_gain_ms, i_ref=i_ref, spot_spacing=spot_spacing, drive_period_ms=0, W_MHz=W_MHz, tight_direction='x', 
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1,fixed_seed=int(fixed_seed),
                           proj_method='linescanning', tdead=4, reps=int(reps),
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    t_list.append(time.time())
    drive = synth_exp.construct_ampl_from_func(x, y, flat_rectangle, one_dim='')
    t_list.append(time.time())
    if False: # Generation and saving of animation takes at least 10s
        ani = synth_exp.animate_proj(x, y, flat_rectangle, frames_step=3, coord_x_slice=4, coord_y_slice=0, plot_residual_bool=False)
        ani.save('Example synthetic dim.gif')
    synth_exp.apply_correction_factors()
    synth_exp.generate_wf_period()
    t_list.append(time.time())
    #synth_exp.generate_wf_period()
    t_list.append(time.time())
    synth_exp.balance_waveforms()
    t_list.append(time.time())
    synth_exp.save_waveforms()
    return {key: getattr(synth_exp, key) for key in synth_exp.print_params}

def create_2step_flat_rectangle(exp_dur_ms=20,exp_gain_ms=10,  i_ref=1e-5, W_MHz=1.75, xfreq=52.5, yfreq=52.5,fixed_seed=42,  t_ramp_ms=0):
    size_y = 80 # µm
    size_x = 20 # µm
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 21)
    y = np.linspace(-size_y/2, size_y /2, 81)
    def flat_rectangle(x, y, t):  
        return np.ones_like(x)*np.ones_like(y)
 
    t_list = [time.time()] # to profile where time is spent
    exp_ramp = Experiment(title='Flat rectangle', verbose=True, dur_tot_ms=exp_gain_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=0, W_MHz=1.75, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, fixed_seed=int(fixed_seed),
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    exp_drive = Experiment(title='Flat rectangle', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=0, W_MHz=W_MHz, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, fixed_seed=int(fixed_seed),
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    t_list.append(time.time())
    drive_ramp = exp_ramp.construct_ampl_from_func(x, y, flat_rectangle, one_dim='')
    drive_drive = exp_drive.construct_ampl_from_func(x, y, flat_rectangle, one_dim='')
    t_list.append(time.time())
    exp_ramp.apply_correction_factors()
    exp_drive.apply_correction_factors()
    t_list.append(time.time())
    exp_ramp.generate_wf_period()
    exp_drive.generate_wf_period()
    t_list.append(time.time())
    exp_ramp.balance_waveforms()
    exp_drive.balance_waveforms()
    t_list.append(time.time())
    if exp_ramp.verbose: print(f'Execution times {np.array2string(np.diff(t_list), precision=2, floatmode="fixed")} s')
 
    #exp_ramp.save_waveforms()
    wfx_tot_b_ramp, wfy_tot_b_ramp, cycles_per_waveform_ramp, samples_per_waveform_ramp = exp_ramp.convert_binary_waveforms()
    wfx_tot_b_drive, wfy_tot_b_drive, cycles_per_waveform_drive, samples_per_waveform_drive = exp_drive.convert_binary_waveforms()
 
    with h5py.File(exp_ramp.awg_file_name, "w") as file:
        awgs_PUT = file.create_group("awgs")
        app_ID = awgs_PUT.create_group("80 MHz AWG (18)")
 
        job_dict_i = {'ch_1_waveforms':np.hstack((wfx_tot_b_ramp, wfx_tot_b_drive)), 'ch_2_waveforms':np.hstack((wfy_tot_b_ramp, wfy_tot_b_drive)), 'cycles_per_waveform':np.hstack((cycles_per_waveform_ramp, cycles_per_waveform_drive)), 'samples_per_waveform':np.hstack((samples_per_waveform_ramp, samples_per_waveform_drive))}
        for key_i in job_dict_i.keys():
            app_ID.create_dataset(key_i, data= job_dict_i[key_i])
        app_ID["save"]=json.dumps(False)
 
    print('Saving done')
    # Account for MemoryError: std::bad_alloc
    del wfx_tot_b_ramp, wfy_tot_b_ramp, wfx_tot_b_drive, wfy_tot_b_drive

def create_inverse_parabola(exp_dur_ms=30, i_ref=1e-5, W_MHz=1.75, xfreq=52.5, yfreq=52.5, t_ramp_ms=0, y_0=0., r_y=25., fixed_seed=0):
    size_y = 80 # µm
    size_x = 20 # µm
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 21)
    y = np.linspace(-size_y/2, size_y /2, 121)
    def inverse_parabola(x, y, t):  
        return np.maximum(1 - ((y-y_0)/r_y)**2,0) + np.clip(np.where(np.abs(y-y_0) > r_y, 2 - ((y-y_0)/r_y)**2, 0), 0, None)
 
    t_list = [time.time()] # to profile where time is spent
    synth_exp = Experiment(title='Inverse parabola', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=0, W_MHz=W_MHz, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1,fixed_seed=int(fixed_seed),
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    t_list.append(time.time())
    drive = synth_exp.construct_ampl_from_func(x, y, inverse_parabola, one_dim='')
    t_list.append(time.time())
    if False: # Generation and saving of animation takes at least 10s
        ani = synth_exp.animate_proj(x, y, inverse_parabola, frames_step=3, coord_x_slice=4, coord_y_slice=0, plot_residual_bool=False)
        ani.save('Example proj.gif')
    synth_exp.apply_correction_factors()
    t_list.append(time.time())
    synth_exp.generate_wf_period()
    t_list.append(time.time())
    synth_exp.balance_waveforms()
    t_list.append(time.time())
    if synth_exp.verbose: print(f'Execution times {np.array2string(np.diff(t_list), precision=2, floatmode="fixed")} s')
 
    synth_exp.save_waveforms()
    return {key: getattr(synth_exp, key) for key in synth_exp.print_params}

def create_2step_inverse_parabola(exp_dur_ms=20,exp_gain_ms=10,  i_ref=1e-5, W_MHz=1.75, xfreq=52.5, yfreq=52.5,fixed_seed=42,  t_ramp_ms=0, y_0=0., r_y=25.):
    size_y = 80 # µm
    size_x = 20 # µm
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 21)
    y = np.linspace(-size_y/2, size_y /2, 161)
    def inverse_parabola(x, y, t):  
        return np.maximum(1 - ((y-y_0)/r_y)**2,0) + np.where(np.abs(y)>r_y, np.maximum(1 - ((y - np.sign(y)*r_y)/r_y)**2,0), 0)
 
    t_list = [time.time()] # to profile where time is spent
    exp_ramp = Experiment(title='Inverse parabola w barriers', verbose=True, dur_tot_ms=exp_gain_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=0, W_MHz=1.75, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, fixed_seed=int(fixed_seed),
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    exp_drive = Experiment(title='Inverse parabola w barriers', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.9, drive_period_ms=0, W_MHz=W_MHz, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1, fixed_seed=int(fixed_seed),
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    t_list.append(time.time())
    drive_ramp = exp_ramp.construct_ampl_from_func(x, y, inverse_parabola, one_dim='')
    drive_drive = exp_drive.construct_ampl_from_func(x, y, inverse_parabola, one_dim='')
    t_list.append(time.time())
    exp_ramp.apply_correction_factors()
    exp_drive.apply_correction_factors()
    t_list.append(time.time())
    exp_ramp.generate_wf_period()
    exp_drive.generate_wf_period()
    t_list.append(time.time())
    exp_ramp.balance_waveforms()
    exp_drive.balance_waveforms()
    t_list.append(time.time())
    if exp_ramp.verbose: print(f'Execution times {np.array2string(np.diff(t_list), precision=2, floatmode="fixed")} s')
 
    #exp_ramp.save_waveforms()
    wfx_tot_b_ramp, wfy_tot_b_ramp, cycles_per_waveform_ramp, samples_per_waveform_ramp = exp_ramp.convert_binary_waveforms()
    wfx_tot_b_drive, wfy_tot_b_drive, cycles_per_waveform_drive, samples_per_waveform_drive = exp_drive.convert_binary_waveforms()
 
    with h5py.File(exp_ramp.awg_file_name, "w") as file:
        awgs_PUT = file.create_group("awgs")
        app_ID = awgs_PUT.create_group("80 MHz AWG (18)")
 
        job_dict_i = {'ch_1_waveforms':np.hstack((wfx_tot_b_ramp, wfx_tot_b_drive)), 'ch_2_waveforms':np.hstack((wfy_tot_b_ramp, wfy_tot_b_drive)), 'cycles_per_waveform':np.hstack((cycles_per_waveform_ramp, cycles_per_waveform_drive)), 'samples_per_waveform':np.hstack((samples_per_waveform_ramp, samples_per_waveform_drive))}
        for key_i in job_dict_i.keys():
            app_ID.create_dataset(key_i, data= job_dict_i[key_i])
        app_ID["save"]=json.dumps(False)
 
    print('Saving done')
    # Account for MemoryError: std::bad_alloc
    del wfx_tot_b_ramp, wfy_tot_b_ramp, wfx_tot_b_drive, wfy_tot_b_drive

def center_spot(exp_dur_ms=30, i_ref=1e-5, xfreq=52.5, yfreq=52.5, t_ramp_ms=0):
    # TODO Is making a new function for each experiment practical? How to interact with Run_ContactICS?
    c_spot = Experiment(title='center spot', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=0.1*conv_um_MHz, drive_period_ms=0, W_MHz=0, tight_direction='x', xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1)
    c_spot.insert_custom_pattern(xfreq, yfreq, np.ones((1, 1)), np.ones((1, 1)), None)
    c_spot.apply_correction_factors()
    c_spot.generate_wf_period()
    c_spot.balance_waveforms()
    c_spot.save_waveforms()
    return c_spot

def barcode(exp_dur_ms=30, i_ref=1e-6, xfreq=52.5, yfreq=52.5, W_MHz=2.5, case=0, t_ramp_ms=0):
    #W_MHz = np.linspace(0, 2.5, 6)[case // 4]
    case = int(case)
    proj_method = ['holographic', 'linescanning'][case // 2]
    direction = ['x', 'y'][case % 2]
    c_spot = Experiment(title='barcode', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, proj_method=proj_method, spot_spacing=0.25*conv_um_MHz, drive_period_ms=0, W_MHz=W_MHz, tight_direction=direction, scan_direction=direction, xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1)
    from beclab.ics_command.create_waveform_deprecated_functions import draw_pattern
    I_x = draw_pattern('barcode', dur=4.)
    nz = np.nonzero(I_x)[0]
    I_x = I_x[nz[0]:nz[-1]+1]
    c_spot.insert_custom_pattern(xfreq - 0.25*I_x.size/2, yfreq-0.25*20/2, I_x.reshape((-1, 1)), np.ones((20, 1)), None)
    c_spot.apply_correction_factors()
    c_spot.generate_wf_period()
    c_spot.balance_waveforms()
    c_spot.save_waveforms()
    return c_spot

def create_gradient(exp_dur_ms=30, i_ref=1e-5, xfreq=52.5, yfreq=52.5, W_MHz=2.0, t_ramp_ms=0, tdead=1, reps=2):
    size_y = 40 # µm
    size_x = 10 # µm
    # Coordinates where the function will be evaluated
    x = np.linspace(-size_x/2, size_x/2, 21) - 10/3
    y = np.linspace(-size_y/2, size_y /2, 81)
    def gradient(x, y, t):  
        return 2*(y - y.mean()) / size_y + 1
 
    t_list = [time.time()] # to profile where time is spent
    tdead = tdead* round(conv_um_MHz *samplerate*1e-6/ 0.9) / (samplerate*1e-6)
    synth_exp = Experiment(title='gradient', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, 
                           #proj_method='linescanning', tdead=tdead, reps=int(reps), 
                           spot_spacing=0.9, drive_period_ms=0, W_MHz=W_MHz, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1,
                           t_ramp_ms=t_ramp_ms, ramp_profile=lambda x: x)
    drive = synth_exp.construct_ampl_from_func(x, y, gradient, one_dim='')
    synth_exp.apply_correction_factors()
    synth_exp.generate_wf_period()
    synth_exp.balance_waveforms()
 
    synth_exp.save_waveforms()
    return {key: getattr(synth_exp, key) for key in synth_exp.print_params}

def test_mondriaan(exp_dur_ms=30, i_ref=5e-6, xfreq=52.5, yfreq=52.5, W_MHz=1.75, tdead_frame=1, rank=6, t_ramp_ms=0):
    img_path = 'examples/mondriaan.png'
    rank=int(rank) 
    t0 = time.time()
    exp = Experiment(title='Mondriaan', verbose=False, dur_tot_ms=exp_dur_ms, i_ref=i_ref, 
                           #proj_method='linescanning', tdead=tdead, reps=int(reps), 
                           spot_spacing=1.0, drive_period_ms=0, W_MHz=W_MHz, tight_direction='x',
                           xfreq=xfreq, yfreq=yfreq, GS_iter=30, GS_parallel=1,
                           t_ramp_ms=t_ramp_ms, tdead_frame=tdead_frame)
    exp.construct_ampl_from_img(img_path, rank, pxl_pitch=exp.spot_spacing, k=-1)
    exp.apply_correction_factors()
    exp.generate_wf_period()
    exp.balance_waveforms() 
    print(exp.table_print())
    exp.save_waveforms()
    print(round(time.time()-t0, 2))
    return {key: getattr(exp, key) for key in exp.print_params}

def synth_dim_2d(exp_dur_ms=30, i_ref=1.25e-5, detuning=14.54545454, xfreq=52.5, yfreq=52.5, t_ramp_ms=None):
    period_us = 1e6 / (440 + detuning)
    # TODO change lengths and spatial periods before starting experiment
    xt = np.linspace(-5, 5, 21)
    yt = np.linspace(-15, 15, 71)
    def synth_dim_2d_func(x, y, t):
        return (x / xt.max()) * np.cos(2*np.pi*(y/10 + t/period_us)) + 1
    exp = Experiment(title='synth dim 2d', verbose=True, spot_spacing=0.9, W_MHz=1.5, tight_direction='x', drive_period_ms=period_us*1e-3, tdead_frame=1, rank=2, dur_tot_ms=exp_dur_ms, i_ref=i_ref, buffer=2, xfreq=xfreq, yfreq=yfreq, t_ramp_ms=t_ramp_ms)
    exp.construct_ampl_from_nonsep_func(xt, yt, synth_dim_2d_func)
    #exp.animate_proj(xt, yt, synth_dim_2d_func, 1, coord_x_slice=3, coord_y_slice=2).save('synth_dim_2d_func.gif')
    exp.apply_correction_factors()
    exp.generate_wf_period()
    exp.balance_waveforms()
    exp.save_waveforms()
    return {key: getattr(exp, key) for key in exp.print_params}

def spot_clip(exp_dur_ms=30, i_ref=1e-5, xfreq=52, yfreq=52, tau=10, tdead_frame=0, t_ramp_ms=0):
    Delta_fx = 6
    Delta_fy = 10
    c_spot = Experiment(title='center spot', verbose=True, dur_tot_ms=exp_dur_ms, i_ref=i_ref, spot_spacing=conv_um_MHz / tau, drive_period_ms=0, W_MHz=0, tight_direction='x', xfreq=xfreq, yfreq=yfreq, GS_iter=0, tdead_frame=tdead_frame, GS_parallel=1)
    I_x = np.zeros((int(Delta_fx*tau)+1, 2))
    I_x[[0, -1], [0, 1]] = 1
    I_y = np.zeros((int(Delta_fy*tau)+1, 2))
    I_y[[0, -1], [0, 1]] = 1

    c_spot.insert_custom_pattern(xfreq, yfreq, I_x, I_y, None)
    c_spot.apply_correction_factors()
    c_spot.generate_wf_period()
    c_spot.balance_waveforms()
    c_spot.save_waveforms()
    return c_spot

HDF5_basefolder = "C:\\Users\\Ghent-BEC\\OneDrive - UGent\\BEC Lab"
HDF5_folder = "awgs_manual\\20241114\\1020"
HDF5_filename = 'Test Connection ICS4 awgs'

if __name__ == '__main__':
    #HDF5PATH_awg = HDF5_basefolder + '\\' + HDF5_folder + '\\' + HDF5_filename + '.h5'
    HDF5PATH_awg = "C:\\Users\\Ghent-BEC\\Desktop\\Do not change - AOD folder\\Instructions\\Test Connection ICS4 awgs.h5"
    
    #save_waveforms_ICS(exp_dur_ms=24.2, i_y=5e-5, thold_ms=0, ramp_time_ms=None, xfreq=51.31, yfreq=50.8, samplerate=250E6, awg_file_name=HDF5PATH_awg , show_plot_bool=False, case=0)
    #create_flat_rectangle(exp_dur_ms=50, exp_gain_ms=25, i_ref=0.4e-4,  W_MHz=1.5,reps=25, xfreq=53.416, yfreq=52.175, t_ramp_ms=0, fixed_seed=2666715121)#520833852)
    #create_2step_flat_rectangle(exp_dur_ms=3,exp_gain_ms=150,  i_ref=1e-4,  W_MHz=51.75, xfreq=52.837, yfreq=52, t_ramp_ms=0, fixed_seed=42)
    #create_inverse_parabola(exp_dur_ms=305.1, i_ref=3.e-4, W_MHz=1.75,  xfreq=53.021, yfreq=51.939,fixed_seed=42, t_ramp_ms=0, y_0=0., r_y=25.)
    #create_2step_inverse_parabola(exp_dur_ms=50.,exp_gain_ms=300,  i_ref=3e-4, W_MHz=2, xfreq=53.102, yfreq=52.049,fixed_seed=42,  t_ramp_ms=0, y_0=0., r_y=25.)
    #example_synth_dim(exp_dur_ms=37.4, i_ref=2e-4, detuning=14.5454, xfreq=52.9, yfreq=51.4)
    #AS_synth_dim(exp_dur_ms=37.4, i_ref=2e-4, detuning=14.5454, xfreq=52.9, yfreq=51.4)
    #create_gradient(exp_dur_ms=30,i_ref=2e-6, xfreq=52.5, yfreq=52.5, t_ramp_ms=0)
    #test_mondriaan()
    #center_spot(exp_dur_ms=30, i_ref=5e-5, xfreq=52.5, yfreq=52.5)
    #barcode(exp_dur_ms=30, i_ref=3e-6, xfreq=52.5, yfreq=52.5, W_MHz=2.5, case=1.0)
    
