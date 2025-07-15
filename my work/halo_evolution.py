import os
import glob
import pickle
import scipy
import numpy as np
from mpmath import polylog
from numba import njit
from scipy import integrate
from scipy.optimize import root
# from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d, CubicSpline, splrep, splev
from scipy.special import expi
# from scipy.sparse import csc_matrix
from astropy import units as ut
from astropy import constants as ct
# import matplotlib.pyplot as plt
import random
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.linalg import solve_banded
import time
from time import perf_counter as timer
from matplotlib import pyplot as plt
import math

# from timeit import default_timer as timer

np.seterr(invalid='warn', divide='warn', over='warn', under='warn')

###############################################################################

@njit
def validate_TDMA_solver_condiion(ac, bc, cc):
    '''
    This function validates the TDMA solver condition.
    |b_i| > |a_i| + |c_i|
    '''
    maximum_difference = 0
    index = 0
    for i in range(len(bc)):
        b_i = bc[i]
        a_i = 0 if i == 0 else ac[i - 1]
        c_i = 0 if i == len(bc) - 1 else cc[i]
        if abs(b_i) <= abs(a_i) + abs(c_i):
            #print('difference is', a_i, b_i, c_i, i, abs((abs(b_i) - (abs(a_i) + abs(c_i)))/b_i))
            maximum_difference = max(maximum_difference, abs((abs(b_i) - (abs(a_i) + abs(c_i)))/b_i))
            if maximum_difference == abs((abs(b_i) - (abs(a_i) + abs(c_i)))/b_i):
                index = i
    print('Maximum difference is', maximum_difference, ac[index-1], bc[index], cc[index], index)

    return True


@njit
def TDMA_solver(ac, bc, cc, d_arr, nf):
    '''
    This function solves tridiagonal matrices using TDMA algorithm.
    (https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9)
    '''
    # - is required in front of d since d is calculated
    # in the left-hand side in the original hydrostatic eq.
    #aaa = validate_TDMA_solver_condiion(ac, bc, cc)

    dc = -d_arr
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]
    xc = bc
    xc[-1] = dc[-1] / bc[-1]
    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
    return xc


@njit
def TDMA_solver_4(bc, cc, d_arr, nf):
    '''
    This function solves tridiagonal matrices using TDMA algorithm.
    (https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9)
    '''
    # - is required in front of d since d is calculated
    # in the left-hand side in the original hydrostatic eq.

    dc = -d_arr
    xc = bc
    xc[-1] = dc[-1] / bc[-1]
    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]
    return xc


@njit
def set_hydrostatic_coefficients_1(r_ext, gamma, p, m_ext, rho, flag=1):
    """
    Set the elements of the tridiagonal matrix for solving the linearized
    hydrostatic equation.
    """

    r2 = r_ext * r_ext
    r3 = r2 * r_ext

    # a is ok
    a_arr = (
            (12. * gamma * p[1:-1] * r2[1:-2] * r2[2:-1]
             + m_ext[2:-1] * (-3. * r2[1:-2] * r_ext[3:] * rho[1:-1]
                              + r3[1:-2] * (2. * rho[1:-1] - rho[2:])
                              + r3[2:-1] * (rho[1:-1] + rho[2:])))
            / (4. * (r3[1:-2] - r3[2:-1])))

    b_arr = (
            r_ext[1:-1]
            * (-4. * p[:-1] * (2. * r3[:-2] + (-2. + 3. * gamma) * r3[1:-1]) * (r3[1:-1] - r3[2:])
               - 4. * p[1:] * (r3[:-2] - r3[1:-1]) *
               ((-2. + 3. * gamma) * r3[1:-1] + 2. * r3[2:])
               + 3. * m_ext[1:-1] * r_ext[1:-1] *
               (r_ext[:-2] - r_ext[2:])
               * (r3[2:] * rho[:-1] + r3[:-2] * rho[1:] - r3[1:-1] * (rho[:-1] + rho[1:])))
            / (4. * (r3[:-2] - r3[1:-1]) * (r3[1:-1] - r3[2:])))

    c_arr = (
            (12. * gamma * p[1:-1] * r2[1:-2] * r2[2:-1] + m_ext[1:-2] * (
                    r3[1:-2] * (rho[:-2] + rho[1:-1])
                    - r2[2:-1] * (r_ext[2:-1] * rho[:-2] + 3. * r_ext[:-3] * rho[1:-1]
                                  - 2. * r_ext[2:-1] * rho[1:-1])))
            / (4. * (r3[1:-2] - r3[2:-1])))

    d_arr = (
            -p[:-1] * r2[1:-1] + p[1:] * r2[1:-1]
            + 1. / 4. * m_ext[1:-1] * (-r_ext[:-2] + r_ext[2:])
            * (rho[:-1] + rho[1:]))
    return a_arr, b_arr, c_arr, d_arr 


@njit
def set_hydrostatic_coefficients_s(r, rho, s, M,  n_shells, M_tot):
    """
    Set the elements of the tridiagonal matrix for solving the linearized
    hydrostatic equation.
    """
    a_arr = np.empty(n_shells - 2+1, dtype=np.float64)
    b_arr = np.empty(n_shells - 1+1, dtype=np.float64)
    c_arr = np.empty(n_shells - 2+1, dtype=np.float64)
    d_arr = np.empty(n_shells - 1+1, dtype=np.float64)


    for i in range(0, n_shells-1+1):  # i=0...N-2
        # Maybe divide all by M_i or something????
        # multiplied by 6

        if i == n_shells-1:
            r_ip2 = r[i+1]*2-r[i]
            b_arr[i] = +(M[i+1]-M[i])*(
                        M_tot[i+1]*pow(r_ip2, 2)*(r[i]-r_ip2)*3
                        + pow(rho[i], 2.0/3.0)*pow(s[i], 2.0/3.0)*r[i+1]*pow(r_ip2, 2)*(30*r[i+1] - 20*r_ip2)) \
                   + (pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0))*pow(r[i+1], 3)*pow(r_ip2, 2)*(
                            -24*pow(r[i+1], 2)
                            + 20*r[i+1]*(r_ip2+r[i])
                            - 16*r_ip2*r[i])
            
            d_arr[i] = 10*(M[i+1]-M[i])*pow(rho[i], 2.0/3.0)*pow(r[i+1], 2)*pow(r_ip2, 2)*pow(s[i], 2.0/3.0)*(r[i+1]-r_ip2) \
                   + 3*(M[i+1]-M[i])*M_tot[i+1]*pow(r_ip2, 2)*(
                        + pow(r_ip2, 2)
                        - r[i+1]*r_ip2
                        - r_ip2*r[i]
                        + r[i+1]*r[i]) \
                   + 4*(pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0))*pow(r[i+1], 4)*pow(r_ip2, 2)*(
                        - pow(r[i+1], 2)
                        + r[i+1]*r_ip2
                        + r[i+1]*r[i]
                        - r_ip2*r[i])
    
        else:
            b_arr[i] = 2*(M[i+2]-M[i+1])*pow(rho[i+1], 2.0/3.0)*pow(s[i+1], 2.0/3.0)*(
                        pow(r[i+1], 3)*(25*r[i+1]-20*r[i])) \
                   +(M[i+1]-M[i])*(
                        M_tot[i+1]*pow(r[i+2], 2)*(r[i]-r[i+2])*3
                        + pow(rho[i], 2.0/3.0)*pow(s[i], 2.0/3.0)*r[i+1]*pow(r[i+2], 2)*(30*r[i+1] - 20*r[i+2])) \
                   + (pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0) - pow(rho[i+1], 5.0/3.0)*pow(s[i+1], 2.0/3.0))*pow(r[i+1], 3)*pow(r[i+2], 2)*(
                            -24*pow(r[i+1], 2)
                            + 20*r[i+1]*(r[i+2]+r[i])
                            - 16*r[i+2]*r[i])

            d_arr[i] = 10*(M[i+2]-M[i+1])*pow(rho[i+1], 2.0/3.0)*pow(r[i+1], 4)*pow(s[i+1], 2.0/3.0)*(r[i+1]-r[i]) \
                   + 10*(M[i+1]-M[i])*pow(rho[i], 2.0/3.0)*pow(r[i+1], 2)*pow(r[i+2], 2)*pow(s[i], 2.0/3.0)*(r[i+1]-r[i+2]) \
                   + 3*(M[i+1]-M[i])*M_tot[i+1]*pow(r[i+2], 2)*(
                        + pow(r[i+2], 2)
                        - r[i+1]*r[i+2]
                        - r[i+2]*r[i]
                        + r[i+1]*r[i]) \
                   + 4*(pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0) - pow(rho[i+1], 5.0/3.0)*pow(s[i+1], 2.0/3.0))*pow(r[i+1], 4)*pow(r[i+2], 2)*(
                        - pow(r[i+1], 2)
                        + r[i+1]*r[i+2]
                        + r[i+1]*r[i]
                        - r[i+2]*r[i])

    for i in range(0, n_shells-2+1):  # i = 0 ... N-3
        c_arr[i] = r[i+2]*(
                               (M[i]-M[i+1])*(
                                    + M_tot[i+1]*(
                                        + 9*r[i+1]*r[i+2]
                                        - 6*r[i+1]*r[i]
                                        - 12*pow(r[i+2], 2)
                                        + 9*r[i+2]*r[i])
                                    + pow(rho[i], 2.0/3.0)*pow(s[i], 2.0/3.0)*pow(r[i+1], 2)*(
                                        - 20*r[i+1]
                                        + 30*r[i+2]))
                               + (pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0) - pow(rho[i+1], 5.0/3.0)*pow(s[i+1], 2.0/3.0))*pow(r[i+1], 4)*(r[i+1] - r[i])*(
                                    - 8 *r[i+1]
                                    + 12*r[i+2]))

    for i in range(1, n_shells-1+1):
        if i == n_shells - 1:
            r_ip2 = r[i+1]*2-r[i]

            a_arr[i-1] = + (r_ip2 - r[i+1])*pow(r_ip2, 2)*(
                         - 3*(M[i+1]-M[i])*M_tot[i+1]
                         + 4*(- pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0))*pow(r[i+1], 4))

        else:
            a_arr[i-1] = + 10*(M[i+1]-M[i+2])*pow(rho[i+1], 2.0/3.0)*pow(r[i+1], 4)*pow(s[i+1], 2.0/3.0) \
                     + (r[i+2] - r[i+1])*pow(r[i+2], 2)*(
                         - 3*(M[i+1]-M[i])*M_tot[i+1]
                         + 4*(pow(rho[i+1], 5.0/3.0)*pow(s[i+1], 2.0/3.0) - pow(rho[i], 5.0/3.0)*pow(s[i], 2.0/3.0))*pow(r[i+1], 4))

    return a_arr, b_arr, c_arr, d_arr


@njit
def update_delta(rho, gamma, p, r, delta_r, r_minus, temp_eps):
    delta_ph = np.empty(len(rho), dtype=np.float64)
    delta_rho = np.empty(len(rho), dtype=np.float64)
    r3 = r * r * r
    r_minus3 = r_minus ** 3
    epsilon = min(np.min(np.abs(r[:-1] / delta_r[:-1]) * 0.01), 1)
    #print('eps', epsilon)
    delta_r = epsilon * delta_r * temp_eps

    r2_times_dr = r * r * delta_r
    delta_r_minus = 0
    r_minus2_times_dr_minus = r_minus ** 2 * delta_r_minus

    # change
    delta_rho[0] = -rho[0] * \
                   min((r2_times_dr[0] - r_minus2_times_dr_minus) / \
                       ((r3[0] - r_minus3) / 3.), 0.99)
    delta_ph[0] = -p[0] * \
                  min(gamma * (r2_times_dr[0] - r_minus2_times_dr_minus) / \
                      ((r3[0] - r_minus3) / 3.), 0.99)
    delta_rho[1:] = -rho[1:] * \
                    np.minimum((r2_times_dr[1:] - r2_times_dr[:-1]) / ((r3[1:] - r3[:-1]) / 3.), 0.99)

    # update delta_ph
    delta_ph[1:] = - p[1:] * np.minimum(gamma * \
                                        (r2_times_dr[1:] - r2_times_dr[:-1]) / ((r3[1:] - r3[:-1]) / 3.), 0.99)
    return delta_rho, delta_ph, delta_r


@njit
def update_delta_s(rho, p, r, M, s, delta_r, M_tot, max_index):
    delta_grid = r[1:]-r[:-1]
    max_index = min(max_index, len(rho))
    epsilon_1 = 1# min(np.min(np.abs(delta_grid[:-1] / delta_r[:-1]) * 0.4), 1)
    epsilon = min(np.min(np.abs(r[1:-1] / delta_r[:-1]) * 0.01), 1)
    epsilon = min(epsilon, epsilon_1)
    r[1:max_index+1] += delta_r[:max_index]*epsilon # maybe there shouldnt be an epsilon here???
    #r_mid = np.empty(len(r)-1, dtype=np.float64)
    #r_mid[1:] = 10 ** ((np.log10(r[1:-1]) + np.log10(r[2:])) / 2.)
    #r_mid[0] = 10 ** (2 * np.log10(r_mid[1]) - np.log10(r_mid[2]))
    for i in range(max_index):
        if i>0:
            drho = (M[i]-M[i-1])/(r[i+1]-r[i])/((r[i+1]))**2 - rho[i]
            rho[i] = (M[i]-M[i-1])/(r[i+1]-r[i])/(r[i+1])**2
        else:
            drho = (M[i])/(r[i+1]-r[i])/((r[i+1]))**2 - rho[i]
            rho[i] = (M[i])/(r[i+1]-r[i])/r[i+1]**2
        #p[i] = rho[i]**(5./3.) * s[i]**(2./3.)
        p[i] += rho[i]**(2./3.) * s[i]**(2./3.) * drho * 5/3
    #for i in range(len(p)-2, -1, -1):
     #   p[i] = p[i+1] + (rho[i]*M_tot[i+1])/(r[i+1]**2)*(r[i+2]-r[i+1])
    return rho[:max_index], p[:max_index], r[1:max_index+1]

@njit
def update_derived_parameters_1(run_name, p, rho, r, sigma_m, velocity_dependence, a, b, C, m_bh, m,   flag_central_bh=False, flag_lmfp=False, flag_baryon=False, flag_entropy=False):
    """
    Given r, mass, rho, and p, set derived halo quantities.
    """
    # specific energy (energy per unit mass)
    u = (3. / 2.) * p / rho
    v = np.sqrt(2 / 3 * np.abs(u))
    if not flag_entropy:
        s = v ** 3 / rho

    # thermal conductivity
    if flag_central_bh:
        sigma = sigma_m * (v) ** (-velocity_dependence)  # *1.e-8
        Kinv_smfp = sigma / (b * v)

        n_shells = len(r)
        r_mid = np.empty(n_shells, dtype=np.float64)
        r_mid[1:] = 10 ** ((np.log10(r[:-1]) + np.log10(r[1:])) / 2.)
        r_mid[0] = 10 ** (2 * np.log10(r_mid[1]) - np.log10(r_mid[2]))
        r_j2 = (v * v) / rho
        r2 = r_mid ** 2
        if run_name == 'test1_modm':
            h2 =r2 * np.exp(-m/m_bh) + r_j2*np.exp(-m_bh/m)
        else:
            h2 = np.minimum(r_j2, r**2)

        Kinv_lmfp = 1. / \
                         (a * C * v * sigma * rho * rho * h2)

        if flag_lmfp:
            Keff = 1. / (Kinv_lmfp) # only lmfp
        else:
            Keff = 1. / (Kinv_smfp + Kinv_lmfp)
    elif flag_baryon:

        Kinv_smfp = 1/(2.1 *v/sigma_m)
        Kinv_lmfp = 1/(0.27*4*np.pi*0.75*rho*v**3*sigma_m)
        Keff = 1. / (Kinv_smfp + Kinv_lmfp)*2/3 # haibo has v^2 instead of u
    else:
        sigma = sigma_m * (v) ** (-velocity_dependence)  # *1.e-8
        Kinv_smfp = sigma / (b * v)
        Kinv_lmfp = 1. / \
                         (a * C * v * p * sigma)
        if flag_lmfp:
            #print('in lmfp')
            Keff = 1. / (Kinv_lmfp)  # only lmfp
        else:
            #print('in both')
            Keff = 1. / (Kinv_smfp + Kinv_lmfp)
    L = np.zeros(len(r), dtype=np.float64)
    L[1:-1] = -r[1:-1]**2 * (Keff[1:-1] + Keff[2:]) / 2. * (
            u[2:] - u[1:-1]) / ((r[2:] - r[:-2]) / 2.)
    L[0] = (-r[0] * r[0] * \
                          (Keff[0] + Keff[1]) / 2. * (u[1] - u[0]) / ((r[1]) / 2.))

    # Knudsen number
    Kn = 1. / (np.sqrt(p) * sigma)
    return L, u, v, s, Kn

class Halo:
    """
    Main class to define a dark matter halo with self-interacting dark matter
    and calculate its gravothermal evolution.

    Evolve a halo
    _____________
    Simply call the class as follows:
    my_halo = Halo(my_data_directory, **kwargs)

    In the following, set either t_end or rho_factor_end (or both), otherwise it will run in a continuous loop.
    Now, begin evolving your halo:
    my_halo.evolve_halo(t_end=np.inf,rho_factor_end=np.inf,Kn_end=0,save_method='timing',
                        save_value=0.1,t_epsilon=1.e-4,r_epsilon=1.e-14)

    Continue evolution from last saved timestep
    ___________________________________________
    Load data and last timestep with:
    my_halo = Halo(my_data_directory)
    This by default loads the last timestep saved, and will begin from that timestep and run until your set end.

    Loading saved halo with rederived parameters
    ____________________________________________
    Load your data (by default, the last timestep is loaded):
    my_halo = Halo(my_data_directory)

    To load a specific timestep:
    my_halo.load_halo('timeXXXdXXXXX.pickle')

    Now you can get luminosity, sigma/m, rho, etc with:
    my_halo.L
    my_halo.sigma_m
    my_halo.rho
    etc.

    """

    def __init__(self, dir_data, **kwargs):

        """
        Initialize dark matter halo to run gravothermal evolution.

        Parameters
        ----------
        dir_data: string
            Directory to store information for halo evolution.

        **kwargs: dictionary, optional
            Dictionary of inputs to override default halo values.
            Possible inputs are listed below.

        Inputs for halo profile
        -----------------------
        profile: 'NFW', 'Hernquist', or 'Isothermal', default: 'NFW'
            Specify halo density profile.

        r_s: float, default: 2.586
            scale radius of halo in units of [kpc]

        rho_s: float, default: 0.0194
            scale density in units of [M_sun/pc^3]

        t_trunc: float, default: -1
             Dimensionless time to truncate halo.
             If set to 0, halo is initially truncated.
             If set to any value less than 0, no truncation will occur.

        p_trunc: float (> 3), default: 5
             Power-law for truncation of outer region of halo.

        r_trunc: float, default: 3
             Dimensionless radius at which to truncate halo.

        Inputs for fluid properties
        ---------------------------
        a: float, default: 4/sqrt(pi)
            coefficient for hard-sphere scattering

        b: float, default: 25*sqrt(pi)/32
            effective impact parameter

        C: float, default: 0.753
            calibration parameter for LMFP thermal conductivity kappa. Must be calibrated to simulation. Default was calibrated to Pippin halos.

        gamma: float, default: 5/3
            gas constant for monatomic gas

        Inputs for particle properties
        ------------------------------
        model_elastic_scattering: 'constant', 'momentum_transfer', 'scalar', default: 'constant'
            sets the particle physics model for the scattering cross section

        sigma_m_with_units: float, default: 5 [cm^2/g]
            cross section prefactor for elastic scattering

        model_inelastic_scattering: 'none', 'constant', default: 'none'
            sets the particle physics model for the inelastic scattering cross section

        sigma_m_inelastic_with_units: 0 [cm^2/g]
           cross section prefactor for inelastic scattering

        v_loss_units: float, default: 0 [km/s]
            Velocity loss parameter for model_inelastic_scattering = 'constant'.

        w_units: float, default: 560 [km/s]
            best-fit input velocity from Robertson et al

        Inputs for numerical solving
        ----------------------------
        n_shells: integer, default: 400
            number of shells to divide the halo into

        r_min: float, default: 0.01
            minimum dimensionless radius of radial bins

        r_max: float, default: 100
            maximum dimensionless radius of radial bins

        p_rmax_factor: float, default: 10
            Factor to multiply r_max to obtain upper limit of integral when
            numerically integrating to find initial halo pressure.

        n_adjustment_fixed: integer, default: -1
            Number of forced hydrostatic adjustment steps per heat conduction
            steps. A negative value indicates a fixed number should not be used
            and this parameter is ignored; in this case, the number of steps is
            determined dynamically to satisfy a convergence criterion set at
            run time. See implementation of hydrostatic_adjustment().

        flag_timestep_use_relaxation: bool, default: True
            If True, incorporate the minimum local relaxation time of the halo
            to help set the heat conduction time step.

        flag_timestep_use_energy: bool, default: True
            If True, incorporate the relative specific energy change
            to help set the heat conduction time step.

        flag_hydrostatic_initial: bool, default: False
            If True, perform hydrostatic adjustment to initialized halo
            at t=0 before beginning halo evolution process.

        flag_Hiro_code: bool, default: False
            If True, use code that mimics Hiro's (for testing purposes).
        """
        # quantities to initialize halo
        # these values are inherent to a given halo calculation and
        # should not be altered at any point during halo evolution
        self.halo_ini = {
            'profile': 'NFW',
            'r_s': 2.586,  # [kpc]
            'rho_s': 0.0194,  # [M_sun/pc^3]
            't_trunc': -1,
            'p_trunc': 5,
            'r_trunc': 3,
            'm_bh': 0.,  # [M_sun]
            'a': 4. / np.sqrt(np.pi),
            'b': 25. * np.sqrt(np.pi) / 32.,
            'C': 0.753,
            'gamma': 5. / 3.,
            'model_elastic_scattering': 'constant',
            'sigma_m_with_units': 5.,  # [cm^2/g]
            'model_inelastic_scattering': 'none',
            'sigma_m_inelastic_with_units': 0.,  # [cm^2/g]
            'v_loss_units': 0.,  # [km/s] velocity loss
            'w_units': 560.,  # [km/s]
            'n_shells': 400,
            'r_min': 0.01,
            'r_max': 100.,
            'p_rmax_factor': 10.,
            'n_adjustment_fixed': -1,
            'flag_timestep_use_relaxation': True,
            'flag_timestep_use_energy': True,
            'flag_hydrostatic_initial': False,
            'flag_central_bh': False,
            'flag_Hiro_code': False,
            'flag_r_minus': -1,
            'flag_entropy': False,
            'flag_shapiro': False,
            'velocity_dependence': 4,
            'flag_baryon': False,
            'max_index': 0,
            'initial_bh': False,
            'flag_lmfp': False
        }
        self.dir_data = dir_data  # need to define before calling load_halo()
        self.path_ini = os.path.join(dir_data, 'halo_ini.pickle')

        # overwrite halo initialization defaults
        if os.path.isfile(self.path_ini):
            # load halo initialization file
            self.load_halo()

            # warn that user input is ignored
            if kwargs != {}:
                print(
                    'Warning: all user input is ignored. Obtaining halo information from saved initialization file.')
        else:
            # verify user input
            bad_keys = np.setdiff1d(
                list(kwargs.keys()), list(self.halo_ini.keys()))
            if len(bad_keys) > 0:
                print('Warning: ignoring the following unknown inputs: ' +
                      ', '.join(bad_keys))

            # update parameters from user input
            self.halo_ini.update(kwargs)

            # ensure n_adjustment_fixed is an integer
            self.halo_ini['n_adjustment_fixed'] = round(
                self.halo_ini['n_adjustment_fixed'])

            # attribute the defined values to this class
            for key, value in self.halo_ini.items():
                setattr(self, key, value)

            # ensure at least one criterion for defining the time step
            if (self.flag_timestep_use_relaxation is False) and (self.flag_timestep_use_energy is False):
                raise IOError('At least one time step criterion must be used.')

        # handle units (dimensionful inputs: r_s, rho_s, sigma_m_with_units)
        # dimensionful halo scales
        self.scale_r = self.r_s * ut.kpc  # scale radius
        self.scale_rho = self.rho_s * ut.M_sun / ut.pc ** 3  # scale density
        self.scale_m = 4. * np.pi * self.scale_rho * self.scale_r ** 3  # mass scale
        self.scale_u = ct.G * self.scale_m / self.scale_r  # specific energy scale
        self.scale_p = self.scale_u * self.scale_rho  # pressure scale
        self.scale_v = np.sqrt(self.scale_u)  # velocity dispersion scale
        # dynamical time scale
        self.scale_t = 1. / np.sqrt(4. * np.pi * self.scale_rho * ct.G)
        self.scale_L = ct.G * self.scale_m ** 2 / \
                       (self.scale_r * self.scale_t)  # luminosity scale
        # cross section per mass scale
        self.scale_sigma_m = 1. / (self.scale_r * self.scale_rho)

        # convert dimensionful quantities to be dimensionless
        self.rs = float(2 * ct.G * ut.M_sun * self.m_bh / ct.c ** 2 / self.scale_r) # ss radius
        self.r_acc = 2 * self.rs  # accretion radius 4M_bh

        print('4M radius is: ', 2 * self.rs)
        #time.sleep(1)

        print('m_bh, m_scale', self.m_bh, self.scale_m)
        self.m_bh = float(self.m_bh / (self.scale_m / ut.M_sun))
        print('dimless m_bh= ', self.m_bh)
        if self.flag_shapiro:
            self.m_bh = 0.01
        self.sigma_m = (self.sigma_m_with_units * ut.cm ** 2 /
                        ut.g).to_value(self.scale_sigma_m)
        print('sigma dimensionless', self.sigma_m)
        self.sigma_m_inelastic = (
                self.sigma_m_inelastic_with_units * ut.cm ** 2 / ut.g).to_value(self.scale_sigma_m)
        self.v_loss = (self.v_loss_units * ut.km / ut.s).to_value(self.scale_v)
        self.w = (self.w_units * ut.km / ut.s).to_value(self.scale_v)

        # initialize quantities for evolution, saved in snapshot files
        # this list should coincide with that in save_halo()
        self.rho_center = 0  # initial density of innermost shell
        self.n_adjustment = 0  # counter for hydrostatic adjustment steps
        self.n_conduction = 0  # counter for heat conduction steps
        self.n_save = 0  # counter for total saved files
        # number of truncations; either 0 (no truncation) or 1 (one truncation)
        self.n_trunc = 0
        self.t_epsilon = None  # tolerance level for heat conduction time step
        self.r_epsilon = None  # tolerance level for satisfying hydrostatic adjustment

        self.t = 0  # current dimensionless time
        self.t_before = 0  # dimensionless time from previous time step

        self.r = None  # array of outer radii for each shell
        self.m = None  # total mass enclosed within radii, given by r
        self.rho = None  # average density for each shell
        self.p = None  # average pressure for each shell
        self.ln_old = 0

        # initialize derived quantities (not saved to file, set in update_derived_parameters)
        # generic quantities needed for evolution
        self.u = np.empty(self.n_shells, dtype=np.float64)  # average specific energy for each shell
        self.v = np.empty(self.n_shells, dtype=np.float64)  # average 1d velocity dispersion for each shell
        self.L = np.empty(self.n_shells, dtype=np.float64)  # luminosity at radii, given by r
        self.s = np.empty(self.n_shells, dtype=np.float64)

        # quantities for inelastic scattering
        # initialization corresponds to no inelastic processes
        self.D = np.zeros(self.n_shells)  # cooling term C/rho

        # other useful quantities
        self.Kn = None  # Knudsen number
        self.F = None  # velocity dependence for momentum_transfer model
        self.Finv = None  # velocity dependence for scalar model (inverse)
        self.Kinv_lmfp = None  # lmfp conductivity (inverse)
        self.Kinv_smfp = None  # smfp conductivity (inverse)

        # either load existing halo save state or create a new one
        if os.path.isfile(self.path_ini):
            # obtain latest time step file
            file_list = glob.glob(os.path.join(dir_data, 'time*.pickle'))
            time_list = [float(
                fl.split('time')[-1].rstrip('.pickle').replace('d', '.')) for fl in file_list]
            file_list_1 = glob.glob(os.path.join(dir_data, 'acc_*.pickle'))
            time_list += [float(
                fl.split('acc_')[-1].rstrip('.pickle').replace('d', '.')) for fl in file_list_1]
            file_list += file_list_1
            s = sorted(zip(time_list, file_list))
            time_list_sorted, file_list_sorted = map(list, zip(*s))

            # load data file
            data_halo = os.path.basename(file_list_sorted[-1])
            self.load_halo(data_halo)
            print('~~~~~ Recovered halo state from file {}'.format(
                os.path.join(dir_data, data_halo)))
            print('recovered t', self.t)
        else:
            # set initial halo radius, mass, density, and pressure profiles
            if self.profile == 'Pippin':
                file_Pippin = './else/elbert_halos/pippin/pippin_ini.pickle'
                # keys: 'n_shells','r','rho','m','p_from_disp','p_integrated'
                with open(file_Pippin, 'rb') as fopen:
                    data = pickle.load(fopen)

                # verify n_shells matches
                if data['n_shells'] != self.n_shells:
                    raise IOError('Set input n_shells={} to match Pippin for proper initialization of halo.'.format(
                        data['n_shells']))

                for key, value in data.items():
                    setattr(self, key, value)
                self.p = self.p_from_disp
            else:
                # initial radius (self.r[i] is location of outermost edge of shell i)
                print('r_min = ', self.r_min)
                self.r = np.logspace(np.log10(self.r_min), np.log10(self.r_max),
                                     num=self.n_shells, endpoint=True, base=10)

                print('len r = ', len(self.r))
                # location of midpoints of shells
                self.r_mid = self.get_shell_midpoints()

                # set mass
                self.m = self.get_initial_mass(self.r)

                # set rho
                self.rho = np.empty(self.n_shells, dtype=np.float64)
                self.rho = self.get_initial_rho(self.r_mid)

                # set pressure
                if self.flag_Hiro_code:
                    self.p = self.get_initial_pressure(self.r, numeric=True) # baryon pressure
                    if self.flag_baryon:
                        self.p[1:] += self.get_initial_pressure(self.r_mid, numeric=False)[1:] # and normal pressure
                        self.p[0] += self.get_initial_pressure(self.r[0], numeric=False)
                else:
                    self.p = self.get_initial_pressure(self.r_mid, numeric=False)
                    #self.p[0] = self.get_initial_pressure(
                        #self.r[0], numeric=False)
            
            # store initial density of innermost shell
            self.rho_center = self.rho[0]

            # with necessary quantities initialized, set derived quantities
            #self.update_derived_parameters()
            self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r, self.sigma_m, self.velocity_dependence, self.a, self.b, self.C, self.m_bh, self.m, self.flag_central_bh, self.flag_lmfp, self.flag_baryon)

            print('~~~~~ Created new halo state in {}'.format(dir_data))

        # initialize arrays needed for evolution (not saved to file)
        # radius array, extended by 1
        self.r_ext = np.empty(self.n_shells + 1, dtype=np.float64)
        # mass array, extended by 1
        self.m_ext = np.empty(self.n_shells + 1, dtype=np.float64)
        self.rho_ext = np.empty(self.n_shells + 0, dtype=np.float64)
        self.p_ext = np.empty(self.n_shells + 0, dtype=np.float64)
        # tridiagonal matrix element array
        self.a_arr = np.empty(self.n_shells - 2, dtype=np.float64)
        # tridiagonal matrix element array
        self.b_arr = np.empty(self.n_shells - 1, dtype=np.float64)
        # tridiagonal matrix element array
        self.c_arr = np.empty(self.n_shells - 2, dtype=np.float64)
        # tridiagonal matrix element array
        self.d_arr = np.empty(self.n_shells - 1, dtype=np.float64)
        self.d_arr_1 = np.empty(self.n_shells - 1, dtype=np.float64)
        self.d_arr_2 = np.empty(self.n_shells - 1, dtype=np.float64)
        # shell location change for adjustment
        self.delta_r = np.empty(self.n_shells, dtype=np.float64)
        # specific energy change for adjustment
        self.delta_uc = np.empty(self.n_shells, dtype=np.float64)
        self.delta_s = np.empty(self.n_shells, dtype=np.float64)
        self.delta_uc_old = np.zeros(self.n_shells, dtype=np.float64)
        self.delta_s_old = np.zeros(self.n_shells, dtype=np.float64)
        self.old_timestep = 0
        # density change for adjustment
        self.delta_rho = np.empty(self.n_shells, dtype=np.float64)
        # pressure change for adjustment
        self.delta_ph = np.empty(self.n_shells, dtype=np.float64)

        # times for benchmark
        self.t_solver = 0  # time solver takes
        self.t_compute_drho = 0  # time to compute d(...)
        self.t_update_drho = 0  # time to update variables
        self.t_coeff = 0  # time to set up coefficcients
        self.it_counter = 0  # number of hydrostatic adustments
        self.acc_counter = 0  # number of accretions adustments
        self.n_update_derived_parameters = 0  # number of updates of derived quantities
        self.t_abcd = 0  # compute abcd
        self.t_ext = 0  # extended array
        self.t_r2r3 = 0  # mutliply r*r, r*r*r
        self.t_heat = 0  # time for heat conduction
        self.t_timestep = 0  # time to get timestep
        self.t_derived = 0  # time to update derived quantities
        self.t_hydro = 0
        self.t_all = 0
        self.t_step = 0
        self.n_heat = 0  # counter for heat conduction that always starts at 0
        self.q = 0
        self.min_r = np.inf
        self.flag_rho_big = False
        print('-----------------------------')
        return

    def load_halo(self, file_halo='halo_ini.pickle'):
        """
        Load halo information from saved file.

        Parameters
        ----------
        file_halo: string, default: 'halo_ini.pickle'
            Name of data file to load. If default name is given, this function
            loads the halo initialization file. Otherwise, this function
            expects the name of a file produced by `save_halo`.

        Returns
        -------
        dictionary of halo information
        """
        path_halo = os.path.join(self.dir_data, file_halo)
        with open(path_halo, 'rb') as fopen:
            data = pickle.load(fopen)
            # print(data)
        for key, value in data.items():
            setattr(self, key, value)

        if file_halo == 'halo_ini.pickle':
            # save dictionary of halo information
            self.halo_ini = data
        else:
            # recover derived parameters if halo save state is loaded
            #self.update_derived_parameters()
            self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r, self.sigma_m, self.velocity_dependence, self.a, self.b, self.C, self.m_bh, self.m, self.flag_central_bh, self.flag_lmfp, self.flag_baryon)
            self.s = self.v**3/self.rho
        return

    def save_halo(self, prefix='time', initialization=False):
        """
        Save halo information to file.

        Parameters
        ----------
        prefix: string, default: 'time'
           File name prefix for save state of the halo. The dimensionless time
           follows the prefix. The default name must be used to restart the
           evolution calculation from a save state. Other prefixes can be used
           to define save states at particular times of interests during
           evolution.

        initialization: bool, default: False
            If True, prefix input is ignored and the halo initialization file
            is saved.
        """
        if initialization:
            # make data directory
            os.makedirs(self.dir_data, exist_ok=True)
            # save halo initialization file
            with open(self.path_ini, 'wb') as fopen:
                pickle.dump(self.halo_ini, fopen)
            return

        # update the counter if prefix is default
        #if prefix == 'time':
        self.n_save += 1

        # set the name of the output file
        # fname = os.path.join(
        #   self.dir_data, prefix+'{0:0>9}.pickle'.format(format(self.n_adjustment/10000000000, '.10f').replace('.', 'q')))
        # print(prefix+'{0:0>9}.pickle'.format(format(self.n_adjustment/10000000000, '.10f').replace('.', 'q')))
        fname = os.path.join(self.dir_data, prefix + '{0:0>9}.pickle'.format(format(self.t, '.18f').replace('.', 'd')))
        print(fname)
        slope = -np.gradient(np.log10(self.rho), np.log10(self.r))
        print('saved', self.n_adjustment, self.t, self.get_timestep(), self.r[0], self.rho[:3], np.average(slope[5:40]))

        # here i want to also check certain stuff, that monitoring throughout the evolution would be too expensive

        # save relevant variables (derived parameters can be recovered)
        data = {
            'rho_center': self.rho_center,
            'n_adjustment': self.n_adjustment,
            'n_conduction': self.n_conduction,
            'n_save': self.n_save,
            'n_trunc': self.n_trunc,
            't_epsilon': self.t_epsilon,
            'r_epsilon': self.r_epsilon,
            't': self.t, 't_before': self.t_before,
            'm': self.m, 'm_bh': self.m_bh, 'r': self.r, 'r_mid': self.r_mid,
            'rho': self.rho, 'p': self.p, 'it': self.it_counter, 'r_acc': self.r_acc}

        with open(fname, 'wb') as fopen:
            pickle.dump(data, fopen)
        return

    def check_errors(self):
        print('CFL', f"{self.get_timestep() * np.amax(self.v[:-1] / (self.r[1:] - self.r[:-1])):.2e}")

        dmdr = np.gradient(self.m, self.r)
        rhs = self.r ** 2 * self.rho
        eq1_err = abs(dmdr - rhs) / rhs
        print(f"EQ1 rel. error: {eq1_err[4]:.2e},{ np.amax(eq1_err):.2e}")

        dpdr = np.gradient(self.p, self.r)
        rhs = -self.rho * (self.m + 0.1*self.r**0.6) / (self.r) ** 2
        eq2_err = abs(dpdr - rhs) / abs(rhs)
        print(f"EQ2 rel. error: {eq2_err[4]:.2e}, {np.amax(eq2_err):.2e}")

        dLdr = np.gradient(self.L, self.r)
        rhs = -self.r**2*self.rho*self.v**2*np.log(self.v**3/self.rho)/self.get_timestep()
        eq3_err = abs(dLdr-rhs)/rhs
        print(f"EQ3 rel. error: {eq3_err[4]:.2e},  {np.amax(eq3_err):.2e}")

    def get_r_minus(self):
        try:
            if self.flag_r_minus < 0:
                return 0
            elif self.flag_r_minus == 0:
                temp = abs(self.r[0] + (self.r[0] - self.r[1]))
                if temp < self.r[0]:
                    return temp
                else:
                    return self.r[0] / 2
            elif self.flag_r_minus == 1:
                return self.r[0] / 2
            elif self.flag_r_minus == 2:
                return 10 ** (np.log10(self.r[0]) - (-np.log10(self.r[0]) + np.log10(self.r[1])))
        except AttributeError:
            return 0

    def get_shell_midpoints(self, base=10):
        """
        Obtain radii of central location of shells. Midpoints of shells are
        taken to be the linear average of the outer and inner shell radii.
        """
        if False and self.flag_entropy:
            return self.r
        else:
            n_shells = len(self.r)
            r_mid = np.empty(n_shells, dtype=np.float64)
            r_mid[1:] = 10 ** ((np.log10(self.r[:-1]) + np.log10(self.r[1:])) / 2.)
            r_mid[0] = 10 ** (2 * np.log10(r_mid[1]) - np.log10(r_mid[2]))
            return r_mid

    def get_initial_mass(self, x):
        """
        Obtain halo mass profile.
        """
        if self.profile == 'NFW':
            mass = (1. / (1. + x) - 1.) + np.log(1. + x)
        elif self.profile == 'Hernquist':
            mass = x ** 2 / (2. * (1. + x) ** 2)
        elif self.profile == 'Isothermal':
            mass = x - np.arctan(x)
        elif self.profile == 'Generic Alpha':
            mass = x ** (3. - self.alpha) / (3. - self.alpha)
        elif self.profile == 'Plummer':
            mass = x**3/(1+x**2)**(3/2)
        elif self.flag_shapiro:
            if isinstance(x, float):
                x = np.array([x])
            res = np.zeros(len(x))
            r_in = 9.5*1.e-4
            rh = 0.01
            rho_h = 0.32
            for i, r in enumerate(x):
                if r < r_in:
                    res[i] = 0
                elif r_in <= r < rh:
                    res[i] = rho_h*rh**(7/3)*3/2*(r**(2/3)-r_in**(2/3))
                elif r >= rh:
                    res[i] = rho_h*(rh**(7/3)*3/2*(rh**(2/3)-r_in**(2/3)) + rh/2*(r**2-rh**2))
                else:
                    print('something"s wrong, I can feel it. 2')
            mass = res
        else:
            raise IOError(
                'Profile {} is not recognized for mass calculation'.format(self.profile))
        return mass

    def get_initial_rho(self, x):
        """
        Obtain halo density profile.
        """
        if self.profile == 'NFW':
            print('indeed NFW')
            rho = 1. / (x * (1. + x) ** 2)
        elif self.profile == 'Hernquist':
            rho = 1. / (x * (1. + x) ** 3)
        elif self.profile == 'Isothermal':
            rho = 1. / (1. + x ** 2)
        elif self.profile == 'Generic Alpha':
            rho = x ** (-self.alpha)
        elif self.profile == 'Plummer':
            rho = 3/(1+x**2)**(5/2)
        elif self.flag_shapiro:
            if isinstance(x, float):
                x = np.array([x])
            res = np.zeros(len(x))
            r_in = 9.5*1.e-4
            rh = 0.01
            rho_h = 0.32
            for i, r in enumerate(x):
                if r < r_in:
                    print(r, r_in, x[:3])
                    res[i] = 0
                elif r_in <= r < rh:
                    res[i] = rho_h*(rh/r)**(7./3.)
                elif r >= rh:
                    res[i] = rho_h*(rh/r)
                else:
                    print('something"s wrong, I can feel it')
            rho = res
        else:
            raise IOError(
                'Profile {} is not recognized for density calculation'.format(self.profile))
        return rho

    def get_initial_pressure(self, x, numeric=False, xmax=None):
        """
        Obtain halo pressure profile.
        If numeric is True, calculate by integrating rather than using the
        analytic expression. Numeric integration extends to xmax.
        """
        if numeric:
            if xmax is None:
                xmax = self.p_rmax_factor * self.r_max

            if self.flag_baryon:
                def p_integrand(u): return 0.1/(np.exp(u*1.4)*(1+np.exp(u))**2) # integral over d ln(r)
            else:
                def p_integrand(r): return self.get_initial_mass(r)*self.get_initial_rho(r)/(r*r)
            it = np.nditer([x, None], flags=['buffered'], op_dtypes=np.float64)
            print('numerical integration', min(x), max(x))
            # print(self.get_initial_mass(x)[:100])
            # print(self.get_initial_rho(x)[:100])
            #print(x[:100])
            # time.sleep(10)
            for (xi, y) in it:
                b = xmax
                res = integrate.quad(p_integrand, np.log(xi), np.log(b))
                while abs(res[1] / res[0]) > 1.e-12:
                    b = b / 1.1
                    res = integrate.quad(p_integrand, np.log(xi), np.log(self.r_max * b))
                y[...] = res[0]
            return it.operands[1]

        if self.profile == 'NFW':
            it = np.nditer([x, None], op_dtypes=np.float64)
            for (xi, y) in it:
                y[...] = polylog(2., -xi)
            plog = it.operands[1]
            p = (np.pi ** 2. - (1. + x * (9. + 7. * x)) / (x * (1. + x) ** 2.) - np.log(x) + np.log(1. + x)
                 * (1. + x ** (-2.) - 4. / x - 2. / (1. + x) + 3. * np.log(1. + x)) + 6. * plog) / 2.
            if self.initial_bh:
                p += self.m_bh*(((6 * x**3 + 6 * x**2) * np.log((x + 1)/x)  - 6 * x**2 - 3 * x + 1) / (2 * x**2 * (x + 1)))
        elif self.profile == 'Hernquist':
            p = (np.log(1. + 1. / x) - (25. + 2. * x * (26. + 3. * x * (7. + 2. * x))) /
                 (12. * (1. + x) ** 4)) / 2.
        elif self.profile == 'Isothermal':
            p = np.pi ** 2 / 8. - np.arctan(x) * (2. + x * np.arctan(x)) / (2. * x)
        elif self.profile == 'Generic Alpha':
            p = x ** (2. - 2. * self.alpha) / ((3. - self.alpha) * (2. * self.alpha - 2.))
        elif self.profile == 'Plummer':
            p = self.rho/(6*np.sqrt(1+x**2)) + self.m_bh*((8*x**4 + 12*x**2 + 3)/((1+x**2)**(3/2)*x) - 8)
        elif self.flag_shapiro:
            if isinstance(x, float):
                x = np.array([x])
            res = np.zeros(len(x))
            r_in = 9.5*1.e-4
            rh = 0.01
            rho_h = 0.32
            x_max = max(self.r)
            m_bh = self.m_bh
            print('m_bh', m_bh)
            #print(x)
            for i, r in enumerate(x):
                if r < r_in:
                    res[i] = 0
                elif r_in <= r < rh:
                    res[i] = (1/80)*rho_h*(45*rho_h*r**3*rh*x_max**2*(rh/r)**(14/3) - 40*rho_h*r*rh**5 - 40*rho_h*r*rh**3*x_max**2*math.log(rh) + 40*rho_h*r*rh**3*x_max**2*math.log(x_max) - 5*rho_h*r*rh**3*x_max**2 + 60*rho_h*r*rh**2*r_in**3*(rh/r_in)**(7/3) - 24*rho_h*r*r_in**3*x_max**2*(rh/r_in)**(7/3) - 36*rho_h*rh*r_in**3*x_max**2*(rh/r)**(7/3)*(rh/r_in)**(7/3) - 40*m_bh*r*rh**2 + 16*m_bh*r*x_max**2 + 24*m_bh*rh*x_max**2*(rh/r)**(7/3))/(r*rh*x_max**2)
                elif r >= rh:
                    res[i] = -1/4*rho_h*rh*(2*rho_h*r**2*rh*x_max**2*(math.log(r) - math.log(x_max)) + r**2*(2*rho_h*rh**3 - 3*rho_h*r_in**3*(rh/r_in)**(7/3) + 2*m_bh) - x_max**2*(2*rho_h*rh**3 - 3*rho_h*r_in**3*(rh/r_in)**(7/3) + 2*m_bh))/(r**2*x_max**2)
                else:
                    print('something"s wrong, I can feel it')
            p = res
        else:
            raise IOError(
                'Profile {} is not recognized for pressure calculation'.format(self.profile))
        return p

    def update_derived_parameters(self):
        """
        Given r, mass, rho, and p, set derived halo quantities.
        """
        # specific energy (energy per unit mass)
        self.u[:] = (3. / 2.) * self.p[:] / self.rho[:]
        self.v[:] = np.sqrt(2 / 3 * abs(self.u))[:]
        if not self.flag_entropy:
            #print(self.s[0])
            self.s = self.v ** 3 / self.rho
            #print('in here', self.s[0])

        # elastic scattering (define after u, v and before L)
        if self.model_elastic_scattering == 'constant':
            pass  # avoid repeatedly assigning a trivial velocity dependence
        elif self.model_elastic_scattering == 'momentum_transfer':
            xi = (self.v * self.v) / (self.w * self.w)
            self.F = 2. / (xi * xi) * \
                     (2. * np.log(1. + xi / 2.) - np.log(1. + xi))
            self.a = self.halo_ini['a'] * self.F
            self.b = self.halo_ini['b'] / self.F
        elif self.model_elastic_scattering == 'scalar':
            xi = (self.w * self.w) / (4. * self.v * self.v)
            if np.any(xi >= 696.):
                val = 1
                self.a = np.empty(self.n_shells, dtype=np.float64)
                self.b = np.empty(self.n_shells, dtype=np.float64)
            else:
                val = 0
            if val == 0:
                self.Finv = -xi * xi * \
                            (np.exp(xi) * expi(-xi) * (2. + 4. * xi +
                                                       xi * xi * np.exp(xi) * expi(-xi)) + 3.)
                self.a = 3. / (np.sqrt(np.pi) * self.Finv)
                self.b = 25. * np.sqrt(np.pi) / 32. * self.Finv
            elif val == 1:
                for i in range(len(xi)):
                    if xi[i] >= 696.:
                        self.Finv = 1
                    else:
                        self.Finv = -xi[i] * xi[i] * (np.exp(xi[i]) * expi(-xi[i]) * (
                                xi[i] * xi[i] * np.exp(xi[i]) * expi(-xi[i]) + 4. * xi[i] + 2.) + 3.)
                    self.a[i] = 3. / (np.sqrt(np.pi) * self.Finv)
                    self.b[i] = 25. * np.sqrt(np.pi) / 32. * self.Finv
        # elif self.model_elastic_scattering == 'scalar':
        #     # self.a = np.empty(self.n_shells,dtype=np.float64)
        #     # self.b = np.empty(self.n_shells, dtype=np.float64)
        #     xi = (self.w * self.w) / (4. * self.v * self.v)
        #     Finv = -xi * xi * (np.exp(xi) * expi(-xi) * (2. + 4. * xi + xi * xi * np.exp(xi) * expi(-xi)) + 3.)
        #     self.a = 3. / (np.sqrt(np.pi) * Finv)
        #     self.b = 25. * np.sqrt(np.pi) / 32. * Finv
        else:
            raise IOError('Elastic scattering model {} is not recognized'.format(
                self.model_elastic_scattering))

        # inelastic scattering (define after u and before L)
        if self.model_inelastic_scattering == 'none':
            pass
        elif self.model_inelastic_scattering == 'constant':
            # Cooling term D = C/rho with cooling rate C (see Essig et al., 1809.01144, eqn.(2)).
            x = self.v_loss / self.v
            y = self.sigma_m_inelastic / self.sigma_m
            self.D = (4. / np.sqrt(np.pi)) * (1. / self.a) * self.rho * y * \
                     self.v * self.v_loss ** 2. * (1. + x ** 2.) * np.exp(-x ** 2.)
        else:
            raise IOError('Inelastic scattering model {} is not recognized'.format(
                self.model_inelastic_scattering))

        # thermal conductivity
        if self.flag_central_bh:
            # print(self.v[0], self.v[100], self.v[200], self.v[399], np.amin(self.v), np.amax(self.v))
            sigma = self.sigma_m * (self.v) ** (-self.velocity_dependence)  # *1.e-8
            self.Kinv_smfp = sigma / (self.b * self.v)

            r_j2 = (self.v * self.v) / self.rho
            #r2 = (self.get_shell_midpoints())**2
            h2 = np.minimum(r_j2, (self.get_shell_midpoints() ** 2)[:])
            # print(r_j2[:3], h2[:3], self.r[:3]**2, self.r[:3])
            #   print('min k', np.amin( (self.a * self.C * self.v * sigma * self.rho * self.rho * h2)))
            self.Kinv_lmfp = 1. / \
                             (self.a * self.C * self.v * sigma * self.rho * self.rho * h2)
            '''plt.semilogx(self.r, (np.gradient(np.log10(self.v), np.log10(self.r))), label='v')
            plt.semilogx(self.r, (np.gradient(np.log10(sigma), np.log10(self.r))), label='sigma')
            plt.semilogx(self.r, (np.gradient(np.log10(self.rho), np.log10(self.r))), label='rho')
            plt.semilogx(self.r, (np.gradient(np.log10(h2), np.log10(self.r))), label='h2')
            plt.semilogx(self.r, (np.gradient(np.log10(1/self.Kinv_lmfp), np.log10(self.r))), label='k')
            plt.legend()
            plt.grid()
            plt.show()
'''

            if self.flag_lmfp:
                Keff = 1. / (self.Kinv_lmfp) # only lmfp
            else:
                Keff = 1. / (self.Kinv_smfp + self.Kinv_lmfp)
        # print('keff   ', Keff[0], self.Kinv_smfp[0], self.Kinv_lmfp[0])
        elif self.flag_baryon:

            self.Kinv_smfp = 1/(2.1 *self.v/self.sigma_m)
            self.Kinv_lmfp = 1/(0.27*4*np.pi*0.75*self.rho*self.v**3*self.sigma_m)
            Keff = 1. / (self.Kinv_smfp + self.Kinv_lmfp)*2/3 # haibo has v^2 instead of u
            #plt.loglog(self.r, self.Kinv_smfp, label='haibo smfp')
            #plt.loglog(self.r, self.Kinv_lmfp, label='haibo lmfp')

        else:
            sigma = self.sigma_m * (self.v[:]) ** (-self.velocity_dependence)  # *1.e-8
            self.Kinv_smfp = sigma / (self.b * self.v)
            self.Kinv_lmfp = 1. / \
                             (self.a * self.C * self.v * self.p * sigma)
            if self.flag_lmfp:
                Keff = 1. / (self.Kinv_lmfp)  # only lmfp
            else:
                Keff = 1. / (self.Kinv_smfp + self.Kinv_lmfp)
            #plt.loglog(self.r, self.Kinv_smfp, label='rest smfp')
            #plt.loglog(self.r, self.Kinv_lmfp, label='rest lmfp')
            #plt.legend()
            #plt.grid()
            #plt.show()
            # print('keff  2 ', Keff[:2], self.Kinv_smfp[:2], self.Kinv_lmfp[:2])
            # luminosity
        r_minus = self.r_acc*0

        self.L[1:-1] = -self.r[1:-1]**2 * (Keff[1:-1] + Keff[2:]) / 2. * (
                self.u[2:] - self.u[1:-1]) / ((self.r[2:] - self.r[:-2]) / 2.)
        self.L[0] = (-self.r[0] * self.r[0] * \
                              (Keff[0] + Keff[1]) / 2. * (self.u[1] - self.u[0]) / ((self.r[1] - r_minus) / 2.))
        self.L[-1] = 0
        # print('L0, L1', self.L[0], self.L[1])

        # Knudsen number
        self.Kn = 1. / (np.sqrt(self.p[:]) * sigma)

        return

    def get_timestep(self):
        """
        Determine heat conduction time step.
        """

        L_minus = 0  # self.L[0] + (self.L[1] - self.L[0]) / (self.r[1] - self.r[0]) * (r_minus - self.r[0]) if self.flag_r_minus >= 0 else 0
        m_minus = 0  # self.get_initial_mass(10**(np.log10(self.r_min) - (-np.log10(self.r_min) + np.log10(self.r_max))/self.n_shells)) if self.flag_r_minus >= 0 else 0

        # print(len(self.u), len(self.L), len(self.m), len(self.D))
        delta_t1 = min(np.absolute(self.u[0] / ((self.L[0] - L_minus) / (self.m[0] - m_minus) + self.D[0])),
                       np.amin(np.absolute(
                           self.u[1:] / ((self.L[1:] - self.L[:-1]) / (self.m[1:] - self.m[:-1]) + self.D[1:]))))



        delta_t2 = np.amin(1. / (self.rho * self.v)[1:])

        #r_j2[:] = (self.v * self.v)[:] / self.rho[:]
        #h2 = np.minimum(r_j2[:], (self.get_shell_midpoints() ** 2)[:])
        #D = self.r ** 4 * self.rho ** 2 / 3 / self.v * (h2 / r_j2)

        #delta_t3 = np.amin((self.m[1:] - self.m[:-1]) ** 2 / D[1:]) * 0.5
        # print((self.L[0]-L_minus)/(self.r[0]-r_minus) / (self.r[0]**2*self.p[0])/self.s[0])
        delta_t4 = min(1 / abs((self.L[0] - L_minus) / (self.m[0] - m_minus) / (self.v[0] ** 2)), np.amin(
            1 / abs((self.L[1:] - self.L[:-1]) / (self.m[1:] - self.m[:-1]) / (self.v[1:]**2))))
        # print(delta_t4)
        # determine minimum time step
        if self.flag_timestep_use_relaxation and self.flag_timestep_use_energy:
            delta_t = min(delta_t1, delta_t2)
        elif self.flag_timestep_use_relaxation:
            delta_t = delta_t2
        elif self.flag_timestep_use_energy:
            delta_t = delta_t1
        else:
            raise IOError('Something went wrong. At least one time step criterion must be used.')
        if self.flag_entropy:
            delta_t = delta_t4#min(delta_t4, delta_t1)
        return (delta_t * self.t_epsilon)  # delta_t*self.t_epsilon

    def conduct_heat(self, flag=None):
        """
        Perform heat conduction step.
        """
        # update conduction counter
        self.n_conduction += 1
        self.n_heat += 1
        if np.any(np.isnan(self.s)):
            print(self.t, self.v[0], self.rho[0], np.amax(self.s), np.amin(self.s))
            print('nan before conduction')
            raise SystemExit

        # determine minimum time step
        delta_t = self.get_timestep()
        # calculate delta_uc
        self.delta_uc[1:] = -delta_t * \
                                       ((self.L[1:] - self.L[:-1]) / (self.m[1:] - self.m[:-1]) + self.D[1:])

        r_minus = self.get_r_minus()
        L_minus = 0  # self.L[0] + (self.L[1] - self.L[0]) / (self.r[1] - self.r[0]) * (r_minus - self.r[0]) if self.flag_r_minus >= 0 else 0 #\
        # + (((self.L[2] - self.L[1]) / (self.r[2] - self.r[1])) - (
        # (self.L[1] - self.L[0]) / (self.r[1] - self.r[0]))) \
        # / (self.r[1] - self.r[0]) * (r_minus - self.r[0]) ** 2 / 2.
        m_minus = 0  # self.get_initial_mass(10**(np.log10(self.r_min) - (-np.log10(self.r_min) + np.log10(self.r_max))/self.n_shells)) if self.flag_r_minus >= 0 else 0
        self.delta_uc[0] = -delta_t * ((self.L[0] - L_minus) / (self.m[0] - m_minus) + self.D[0])

        # calculate delta_pc
        #delta_pc = self.p * self.delta_uc / self.u

        if flag == 'skip':
            pass
        else:
            # update variables
            if self.profile == 'Plummer':
                self.u[:-1] += self.delta_uc[:-1]
                self.p[:-1] = (2. / 3.) * self.u[:-1] * self.rho[:-1]
            else:
                self.u[:] += self.delta_uc[:]
                self.p[:] = (2. / 3.) * self.u[:] * self.rho[:]

            # update the current dimensionless time
            self.delta_uc_old[:] = self.delta_uc[:]
            self.old_timestep = delta_t
            self.t_before = self.t
            self.t += delta_t

            self.v[:] = np.sqrt(self.p[:] / self.rho[:])
            self.s = self.v**3/self.rho
        if np.any(np.isnan(self.s)) or np.any(np.isnan(self.p)):
            print(self.t, self.v[0], self.rho[0], np.amax(self.s), np.amin(self.s))
            print('nan in conduction')
            raise SystemExit
        return

    def conduct_heat_s(self):
        """
        Perform heat conduction step.
        """
        # update conduction counter
        self.n_conduction += 1
        self.n_heat += 1

        #self.s = self.v ** 3 / self.rho

        # determine minimum time step
        delta_t = self.get_timestep()
        # calculate delta_s
        self.delta_s[1:] = -delta_t * (self.L[1:] - self.L[:-1]) / (self.m[1:] - self.m[:-1]) / (
                self.v[1:]**2) #* self.s[1:]

        r_minus = self.get_r_minus()
        m_minus = 0
        L_minus = 0
        self.delta_s[0] = -delta_t * (self.L[0] - L_minus) / (self.m[0] - m_minus) / (self.v[0] ** 2) #* self.s[0]

        if self.old_timestep>0:
            dlns = self.delta_s + 0.5*(self.delta_s/delta_t - self.delta_s_old/self.old_timestep)/self.old_timestep*delta_t**2
        else:
            dlns = self.delta_s
        # plt.loglog(self.r, self.s, label='s before conduction', marker='x')
        # plt.loglog(self.r, abs(self.delta_s), label='delta_s due to conduction', marker='.')
        if self.profile == 'Plummer':
            self.s[:-1] = np.exp(np.log(self.s[:-1]) + dlns[:-1])
            self.p[:-1] = self.rho[:-1]**(5/3)*self.s[:-1]**(2/3)
        else:
            self.s[:] = np.exp(np.log(self.s[:]) + dlns[:])
            self.p[:] = self.rho[:]**(5/3)*self.s[:]**(2/3)

        #self.v = (self.s * self.rho) ** (1. / 3.)
        #self.u = 3. / 2. * self.v ** 2
        #self.p = self.v ** 2 * self.rho

        # update the current dimensionless time
        self.delta_s_old = self.delta_s
        self.old_timestep = delta_t
        self.t_before = self.t
        self.t += delta_t

        return

    def bh_accretion_1(self, r_acc=0, max_index=0, show=False):

        # plt.loglog(self.r, self.s, label='before accretion' + str(self.t), marker='<')
        # print('1', min(self.rho))

        if self.t == 0:
            self.rho[0] = 10**(2*np.log10(self.rho[1])-np.log10(self.rho[2]))
            self.p[0] = 10**(2*np.log10(self.p[1])-np.log10(self.p[2]))
            index = -1
        else:
            index = max(np.where(self.r < r_acc)[0])
            self.m_bh += self.m[index]
            self.m -= self.m[index]
        if show:
            plt.loglog(self.r, self.rho, marker='x', label='before')
            plt.axvline(r_acc, color='red')
            plt.axvline(self.r[max_index-1], color='black')

        max_index = max(index+10, max_index)
        max_index = min(max_index, self.n_shells-1)

        p_interp = interp1d(np.log10(self.r[index+1:max_index+1]), np.log10(self.p[index+1:max_index+1]), kind='linear')
        rho_interp = interp1d(np.log10(self.r[index+1:max_index+1]), np.log10(self.rho[index+1:max_index+1]), kind='linear')
        m_interp = interp1d(np.log10(self.r[index+1:max_index+1]), np.log10(self.m[index+1:max_index+1]), kind='linear')
        r_grid = np.linspace(np.log10(self.r[index+1]), np.log10(self.r[max_index-1]), max_index)
        self.rho[:max_index] = 10**rho_interp(r_grid)
        self.p[:max_index] = 10**p_interp(r_grid)
        self.m[:max_index] = 10**m_interp(r_grid)
        self.r[:max_index] = 10**r_grid

        self.m_ext[1:] = self.m + self.m_bh
        self.m_ext[0] = self.m_bh
        if show:
            plt.loglog(self.r, self.rho, marker='.', label='after')
            plt.grid()
            plt.legend()
            plt.show()
        #self.update_derived_parameters()
        self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r, self.sigma_m, self.velocity_dependence, self.a, self.b, self.C, self.m_bh, self.m, self.flag_central_bh, self.flag_lmfp, self.flag_baryon)

        self.s = self.v ** 3 / self.rho
        if self.r[0] < r_acc:
            print('---------------------------------------', self.r[0])
        return

    def hydrostatic_adjustment_step(self, use_entropy_adjustment=False, max_index=0):
        """
        Perform one hydrostatic adjustment process. Adiabaticity is satisified
        in the hydrostatic process after each conduction step. The tridiagonal
        matrix is derived by keeping the central and outermost shells fixed.
        """
        # update the counter
        # rhs = -self.m_ext[1:]*self.rho/self.r**2
        # lhs = np.gradient(self.p, self.r)
        # print('-', np.amax(abs((lhs-rhs)/rhs)[1:]), (abs((lhs-rhs)/rhs)[:3]), np.average(abs((lhs-rhs)/rhs)[:100]))

        self.n_adjustment += 1
        if self.r[0]<1.e-20:
            print(self.r[0])


        self.r_ext[1:] = self.r
        self.r_ext[0] = 0
        if self.flag_baryon:
            temp_m = self.m_ext + 0.1 * self.r_ext ** 0.6
        else:
            temp_m = self.m_ext

        if not self.flag_entropy and not use_entropy_adjustment: # no entropy
            self.a_arr, self.b_arr, self.c_arr, self.d_arr = set_hydrostatic_coefficients_1(
                self.r_ext, self.gamma, self.p, temp_m, self.rho, self.flag_r_minus)
            if np.any(np.isnan(self.a_arr)):
                print('nan in a_arr')
            if np.any(np.isnan(self.b_arr)):
                print('nan in b_arr')
            if np.any(np.isnan(self.c_arr)):
                print('nan in c_arr')
            if np.any(np.isnan(self.d_arr)):
                print('nan in d_arr')
            delta_r = TDMA_solver(self.a_arr, self.b_arr,
                                  self.c_arr, self.d_arr, self.n_shells - 1)
            self.delta_r[:-1] = delta_r
            self.delta_r[-1] = 0

        if np.any(np.isnan(self.delta_r)):
            self.a_arr, self.b_arr, self.c_arr, self.d_arr = set_hydrostatic_coefficients_1(
                self.r_ext, self.gamma, self.p, temp_m, self.rho, self.flag_r_minus)
            ab = np.zeros((3, len(self.b_arr)))
            ab[0, 1:] = self.c_arr  # Superdiagonal (shifted right)
            ab[1, :] = self.b_arr  # Main diagonal
            ab[2, :-1] = self.a_arr  # Subdiagonal (shifted left)
            if np.any(np.isnan(ab)):
                print("NaN values found in 'ab'")
            if np.any(np.isinf(ab)):
                print("Infinite values found in 'ab'")
                print(np.amin(abs(self.r[1:] - self.r[:-1])))
                self.save_halo()
            if np.any(np.isnan(self.d_arr)):
                print("NaN values found in 'self.d_arr'")
            if np.any(np.isinf(self.d_arr)):
                print("Infinite values found in 'self.d_arr'")
            if np.any(np.isnan(ab)) or np.any(np.isnan(self.d_arr)):
                plt.loglog(self.r, self.rho)
                print('lololaoaooaosdolasdoadoadda')
                plt.savefig('./data/baryon_sidm/image.png')
            delta_r = solve_banded((1, 1), ab, -self.d_arr)
            self.delta_r[:-1] = delta_r
            self.delta_r[-1] = 0
            exit()

        if self.flag_entropy or use_entropy_adjustment:  # entropy
            m = np.zeros(len(self.m_ext))
            m[1:] = self.m
            self.a_arr, self.b_arr, self.c_arr, self.d_arr = set_hydrostatic_coefficients_s(
                self.r_ext, self.rho, self.s, m, self.n_shells, temp_m)
            self.a_arr[-1] = 0
            self.b_arr[-1] = 1
            self.c_arr[-1] = 0
            self.d_arr[-1] = 0
            delta_r = TDMA_solver(self.a_arr, self.b_arr,
                                  self.c_arr, self.d_arr, self.n_shells - 1 +1)
            self.delta_r = delta_r

        temp_eps = 1

        if not self.flag_entropy and not use_entropy_adjustment: # entropy
            self.delta_rho, self.delta_ph, delta_r_var = update_delta(self.rho, self.gamma, self.p, self.r, self.delta_r,
                                                                      self.r_ext[0], temp_eps)
            self.r += delta_r_var

            if self.profile=='Plummer':
                self.rho[:-1] += self.delta_rho[:-1]

                self.p[:-1] += self.delta_ph[:-1]
            else:
                self.rho += self.delta_rho
                self.p += self.delta_ph
            if np.any(self.rho<=0) or np.any(np.isnan(self.rho)):
                print(np.where(self.rho<=0))
                print(self.rho[:10])
                print(self.m[:10])
                print('guess, I"m here???')
                raise SystemExit

            #self.p = self.rho**(5/3)*self.s**(2/3)
            #self.rho = (self.p**(3./5.)/self.s**(2./5.))

        else:
            if self.profile == 'Plummer':
                max_index = min(max_index, self.n_shells-1)
            if np.any(self.rho <= 0):
                print(np.where(self.rho <= 0))
                print(self.rho[:10])
                print(self.m[:10])
                print('rho is negative or zero, before')
            self.rho[:max_index], self.p[:max_index], self.r[:max_index] = update_delta_s(self.rho,
                                                         self.p,
                                                         self.r_ext,
                                                         self.m,
                                                         self.s,
                                                         self.delta_r, temp_m, max_index)
            if np.any(self.rho <= 0):
                print(np.where(self.rho <= 0))
                print(self.rho[-10:])
                print(self.m[-10:])
                print(self.m_ext[-10:], self.m_bh)
                print('rho is negative or zero, after ')

        if self.min_r > self.r[0]:
            self.min_r = self.r[0]

        assert not np.any(np.isnan(self.rho)), f"{self.s[:10]}, {self.v[:10]}, {self.rho[:10]} after everything: Result contained NaN: "

        return

    def hydrostatic_adjustment(self, epsilon_b=1, mass=None):
        """
                Perform series of hydrostatic adjustment steps to achieve hydrostatic
                condition for halo to specified tolerance. If the parameter
                n_adjustment_fixed>=0, use a fixed number of adjustment steps.
                """
        if self.t==0:
            self.s = self.v**3/self.rho
        if self.n_adjustment_fixed < 0:  # adjust until convergence is met
            if self.flag_baryon:
                self.hydrostatic_adjustment_step(use_entropy_adjustment=False)
            else:
                self.hydrostatic_adjustment_step(use_entropy_adjustment=False)
            adjustment_counter = 1
            self.it_counter += 1
            if np.any(self.rho<=0):
                print(np.where(self.rho<=0))
                print(self.rho[:10])
                print(self.m[:10])
                print('rho is negative or zero, after first step')
            while np.amax(np.abs(self.delta_r/self.r)) > self.r_epsilon and adjustment_counter < 1.e3:
                adjustment_counter += 1
                self.it_counter += 1
                self.hydrostatic_adjustment_step()
                if adjustment_counter%200==0:
                    self.hydrostatic_adjustment_step(use_entropy_adjustment=True, max_index=20)
                    print('mass conservation was needed in hydrsotatic adjustment', adjustment_counter)
                    print(np.any(self.rho<0))
            for idx in range(self.n_adjustment_fixed):
                self.hydrostatic_adjustment_step(mass)
        # print(f'after adjustment: {self.t:.2e}, {self.get_timestep():.2e}, {self.rho[0]:.2e}, {adjustment_counter:.2e}, {self.r[0]:.2e}, {self.min_r:.2e}, {self.acc_counter}, {self.r_acc:.2e}, {self.t_epsilon:.2e}', end="\r")

        if np.any(self.rho<=0):
            print('rho is negative or zero')

        r_acc = self.r_acc
        if self.r[0] <= r_acc and self.flag_central_bh:
            delta_t = self.get_timestep()*100
            n_scatter = (self.rho*self.sigma_m*self.v**(-self.velocity_dependence+1)*(delta_t))
            if np.amax(n_scatter)>=1:
                max_integer = np.where(n_scatter>=1)[0][-1]
                #print('~~~~~~~~', max_integer)
                max_integer = max(max_integer, 25) #umin#(max_integer+10, umin)
            else:
                if self.t>1.e-16:
                    try:
                        umin = np.where(self.u[:-1]-self.u[1:] < 0)[0][0]
                    except IndexError:
                        umin = np.where(self.u == np.amin(self.u[:int(self.n_shells/2)]))[0][0]
                else:
                    umin = int(self.n_shells/4)
                max_integer = umin
                #print('here', max_integer, self.r[max_integer], self.m[max_integer])

            self.bh_accretion_1(r_acc, max_index=max_integer)
            self.acc_counter += 1
            if self.acc_counter%1009 == 0:
                self.hydrostatic_adjustment_step(use_entropy_adjustment=True, max_index=max_integer)
            for i in range(11):
                self.hydrostatic_adjustment_step()
            if self.acc_counter%40007 == 0:
                self.save_halo(prefix='acc_')
                self.n_save -= 1
            if self.r_acc > self.rs*2:#  not self.flag_shapiro:
                self.r_acc *= 0.995

        #self.update_derived_parameters()
        self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r, self.sigma_m, self.velocity_dependence, self.a, self.b, self.C, self.m_bh, self.m, self.flag_central_bh, self.flag_lmfp, self.flag_baryon)
        self.n_update_derived_parameters += 1
        return

    def evolve_halo(self, t_end=np.inf, rho_factor_end=np.inf, Kn_end=0,
                    save_frequency_rate=None,
                    save_frequency_timing=None,
                    save_frequency_density=None,
                    t_epsilon=1.e-4, r_epsilon=1.e-14):
        """
        Run the halo evolution.

        Parameters
        ----------
        t_end: float, default: np.inf
            Maximum dimensionless time to run the halo evolution.
            (Set to np.inf to render this criterion ineffectual.)

        rho_factor_end: float, default: np.inf
            Factor to determine halo evolution termination, based on central
            density. If the central density reaches this factor times the
            initial central density, evolution stops.
            (Set to np.inf to render this criterion ineffectual.)

        Kn_end: float, default: 0
            Minimum Knudsen number to determine halo evolution termination.
            If the Knudsen number anywhere in the halo drops to this value
            or lower, evolution stops.
            (Set to 0 to render this criterion ineffectual.)

        save_frequency_rate: float, default: None
            Frequency in which to save halo files, in terms of (dimensionless)
            time. An input of N means that N save files will be created for
            each elapsed unit of time. If None, this criteria will not be used
            in saving files.

        save_frequency_timing: float, default: None
            Frequency in which to save halo files, in terms of the fractional
            change in elapsed time. An input of N means that a new save file
            will be created if the time changes by a fractional amount N from
            the previous save. If None, this criteria will not be used in
            saving files.

        save_frequency_density: float, default: None
            Frequency in which to save halo files, in terms of the fractional
            change in the central energy density. An input of N means that a
            new save file will be created if the energy density of the
            innermost shell changes by a fractional amount N from the previous
            save. If None, this criteria will not be used in saving files.

        t_epsilon: float, default: 1e-4
            Tolerance level for heat conduction time step.

        r_epsilon: float, default: 1e-14
            Tolerance level for hydrostatic adjustments.
        """
        print('~~~~~ Running evolution')
        if (save_frequency_rate is None) and (save_frequency_timing is None) and (save_frequency_density is None):
            print('      WARNING: No intermediate halo files will be saved')
        if save_frequency_rate is not None:
            print('      Saving {} files per unit time'.format(save_frequency_rate))
        if save_frequency_timing is not None:
            print('      Saving files as time changes by fraction {}'.format(
                save_frequency_timing))
        if save_frequency_density is not None:
            print('      Saving files as central density changes by fraction {}'.format(
                save_frequency_density))

        # set tolerance for each evolutionary step
        self.t_epsilon = t_epsilon
        self.r_epsilon = r_epsilon

        print('eps_r,eps_t, rminus', self.r_epsilon,self.t_epsilon, self.flag_r_minus)

        # initialization for new evolution calculation
        if self.t == 0:
            print('time', self.t)
            # check if halo is initially truncated
            if self.t_trunc == 0:
                self.truncate_halo()
            # enforce hydrostatic condition initially (but not for Pippin)
            if self.flag_hydrostatic_initial and self.profile != 'Pippin' and self.m_bh > 0 and not self.initial_bh:
                for mass in np.linspace(0., self.m_bh, 100_000):
                    print('mass', float(mass / self.m_bh), end="\n")
                    self.m_ext[1:] = self.m + mass  # moved here
                    self.m_ext[0] = 0 + mass  # to increase speed
                    self.hydrostatic_adjustment(mass)
            else:
                self.m_ext[1:] = self.m + self.m_bh  # moved here
                self.m_ext[0] = 0 + self.m_bh  # to increase speed
                # self.hydrostatic_adjustment()
            # save halo initialization file
            self.r_acc = 0.995*self.r[0]
            if self.flag_shapiro:
                self.r_acc = 0.0009
            # save initial state
            self.save_halo(initialization=True)
            self.save_halo()
            #raise SystemExit
        else:
            self.s = self.v**3/self.rho
            self.m_ext[1:] = self.m + self.m_bh  # moved here
            print('hello there', self.m[0] + self.m_bh, float(self.m_bh), self.m_ext[1:3])
            self.m_ext[0] = 0 + self.m_bh  # to increase speed


        # initialize reference values for determining save frequency
        last_save_time = self.t
        last_save_rho = self.rho[0]

        #self.it_counter = 1

        self.conduct_heat()

        print('starting main')
        # evolve the halo
        ii=0
        u = interp1d(np.log10(self.r), np.log10(self.u), kind='linear', fill_value='extrapolate')
        self.hydrostatic_adjustment()
        deltau0 = (self.u - 10 ** u(np.log10(self.r)))

        while True:
            if self.m_bh>0:
                self.t_epsilon = 1.e-4
                if self.flag_entropy:
                    self.conduct_heat_s()
                else:
                    self.conduct_heat('skip')

                deltau1 = self.delta_uc

                temp_res = np.average(abs(deltau0/deltau1)[2:50]) *self.t_epsilon
                self.t_epsilon = max(1.e-4, min(temp_res, 0.01))
                rep = 1
                if self.t_epsilon == 0.01:
                    if self.velocity_dependence in [0, 1]:
                        self.t_epsilon = 0.005
                        rep = min(int(temp_res / self.t_epsilon), 50)
                    else:
                        rep = min((temp_res / self.t_epsilon), 10)
                        rep = int(rep)

                for someindex in range(rep):
                    if self.flag_entropy:
                        self.conduct_heat_s()
                    else:
                        self.conduct_heat()
                    self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r,
                                                                                          self.sigma_m,
                                                                                          self.velocity_dependence,
                                                                                          self.a, self.b, self.C, self.m_bh, self.m,
                                                                                          self.flag_central_bh,
                                                                                          self.flag_lmfp,
                                                                                          self.flag_baryon)

                u = interp1d(np.log10(self.r), np.log10(self.u), kind='linear', fill_value='extrapolate')
                self.hydrostatic_adjustment()

                deltau0 = (self.u - 10 ** u(np.log10(self.r)))

            else:
                if self.flag_entropy:
                    self.conduct_heat_s()
                    # self.s = self.v**3/self.rho
                else:
                    self.conduct_heat()
                # self.update_derived_parameters()
                self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r,
                                                                                      self.sigma_m,
                                                                                          self.velocity_dependence,
                                                                                          self.a, self.b, self.C, self.m_bh, self.m,
                                                                                      self.flag_central_bh,
                                                                                      self.flag_lmfp, self.flag_baryon)
                self.hydrostatic_adjustment()

            # truncation during the evolution
            if (self.t_trunc > 0) and (self.n_trunc == 0):
                if self.t_trunc < self.t:
                    self.truncate_halo()
                    self.hydrostatic_adjustment()

            # determine if halo has met conditions for terminating evolution
            if self.t >= t_end:
                self.save_halo()
                print(
                    'Success: Evolution has reached dimensionless time {}'.format(self.t))
                break
            elif self.rho[0] > rho_factor_end * self.rho_center:
                self.save_halo()
                print(
                    'Success: Central density reached its maximum requested value of {} times its initial value'.format(
                        rho_factor_end))
                break
            elif np.any(self.Kn < Kn_end):
                self.save_halo()
                print('Success: Smallest Knudsen number in halo is {}'.format(
                    np.min(self.Kn)))
                break


            # save halo state at intermediate stages of evolution
            save_bool = False
            if save_frequency_rate is not None:
                if self.t >= self.n_save / save_frequency_rate:
                    save_bool = True
                    print('save cause rate')
            if save_frequency_timing is not None:
                delta_t = (self.t - last_save_time) / last_save_time
                if delta_t >= save_frequency_timing:
                    save_bool = True
                    print('save cause timing')
            if save_frequency_density is not None:
                delta_rho = np.abs((self.rho[0] - last_save_rho) / min(last_save_rho, self.rho[0]))
                if delta_rho >= save_frequency_density:
                    print('save caue density')
                    save_bool = True

            if save_bool:
                self.save_halo()
                last_save_time = self.t
                last_save_rho = self.rho[0]
        return

    def get_rho_trunc(self, r, rho_at_r_trunc):
        """
        Obtain density profile at truncation time beyond truncation radius.

        Parameters
        ----------
        r: array-like
            Dimensionless radii.

        rho_at_r_trunc: float
            Value of rho at truncation radius at truncation time.

        Returns
        -------
        array-like object with same dimension as r
        """
        return rho_at_r_trunc * (self.r_trunc / r) ** self.p_trunc

    def get_m_trunc(self, r, rho_at_r_trunc, m_at_r_trunc):
        """
        Obtain mass profile at truncation time beyond truncation radius.

        Parameters
        ----------
        r: array-like
            Dimensionless radii.

        rho_at_r_trunc: float
            Value of rho at truncation radius at truncation time.

        m_at_r_trunc: float
            Value of mass at truncation radius at truncation time.

        Returns
        -------
        array-like object with same dimension as r
        """
        a = self.p_trunc - 3
        return m_at_r_trunc + rho_at_r_trunc * self.r_trunc ** 3 * (1. - (self.r_trunc / r) ** a) / a

    def get_p_trunc(self, r, rho_at_r_trunc, m_at_r_trunc, numeric=False):
        """
        Obtain pressure profile at truncation time beyond truncation radius.

        Parameters
        ----------
        r: array-like
            Dimensionless radii.

        rho_at_r_trunc: float
            Value of rho at truncation radius at truncation time.

        m_at_r_trunc: float
            Value of mass at truncation radius at truncation time.

        Returns
        -------
        array-like object with same dimension as r
        """
        if numeric:
            def p_integrand(x):
                return self.get_m_trunc(
                    x, rho_at_r_trunc, m_at_r_trunc) * self.get_rho_trunc(x, rho_at_r_trunc) / x ** 2

            it = np.nditer([r, None], flags=['buffered'], op_dtypes=np.float64)
            for (ri, y) in it:
                y[...] = integrate.quad(
                    p_integrand, ri, self.p_rmax_factor * self.r_max)[0]
            ans = it.operands[1]
        else:
            a = self.p_trunc - 3
            a1 = self.p_trunc + 1
            a2 = 2. * (self.p_trunc - 1)

            factor1 = pow(self.r_trunc / r, a1) / a1 * (m_at_r_trunc *
                                                        a / (rho_at_r_trunc * self.r_trunc ** 3) + 1.)
            factor2 = -pow(self.r_trunc / r, a2) / a2
            ans = (rho_at_r_trunc * self.r_trunc) ** 2 / a * (factor1 + factor2)
        return ans

    def truncate_halo(self):
        """
        Truncate outer region of halo.
        """
        # update the counter
        self.n_trunc += 1

        # find shell near truncation
        i_trunc = (np.abs(self.r - self.r_trunc)).argmin()
        if self.r[i_trunc] <= self.r_trunc:
            i_trunc += 1  # ensure i_trunc is location of first truncated shell
        if i_trunc > self.n_shells - 1:
            print(
                'Truncation requested is at or beyond outermost shell. Skipping truncation.')
            return

        # location of midpoints of shells
        r_mid = self.get_shell_midpoints()

        if self.flag_Hiro_code:
            r_mid = self.r

        # find quantities at r_trunc
        if self.t == 0:
            rho_at_r_trunc = self.get_initial_rho(self.r_trunc)
            m_at_r_trunc = self.get_initial_mass(self.r_trunc)
            p_orig_at_r_trunc = self.get_initial_pressure(
                self.r_trunc, numeric=False)
        else:
            rho_interp1d = interp1d(r_mid[i_trunc - 2:i_trunc + 2],
                                    self.rho[i_trunc - 2:i_trunc + 2],
                                    kind='linear')
            m_interp1d = interp1d(r_mid[i_trunc - 2:i_trunc + 2],
                                  self.m[i_trunc - 2:i_trunc + 2],
                                  kind='linear')
            p_interp1d = interp1d(r_mid[i_trunc - 2:i_trunc + 2],
                                  self.p[i_trunc - 2:i_trunc + 2],
                                  kind='linear')
            rho_at_r_trunc = rho_interp1d(self.r_trunc)
            m_at_r_trunc = m_interp1d(self.r_trunc)
            p_orig_at_r_trunc = p_interp1d(self.r_trunc)

        if self.flag_Hiro_code:
            p_at_r_trunc = self.get_p_trunc(
                self.r_trunc, rho_at_r_trunc, m_at_r_trunc, numeric=True)
        else:
            p_at_r_trunc = self.get_p_trunc(
                self.r_trunc, rho_at_r_trunc, m_at_r_trunc, numeric=False)

        # modify mass
        mask = (self.r > self.r_trunc)
        self.m[mask] = self.get_m_trunc(
            self.r[mask], rho_at_r_trunc, m_at_r_trunc)

        # modify density at radii beyond r_trunc for fully truncated shells
        if i_trunc < self.n_shells - 1:
            self.rho[i_trunc + 1:] = self.get_rho_trunc(
                (self.r[i_trunc + 1:] + self.r[i_trunc:-1]) / 2., rho_at_r_trunc)

        # modify density for shell containing r_trunc
        if self.t == 0:
            self.rho[i_trunc] = (self.get_initial_rho(
                self.r[i_trunc - 1]) + self.get_rho_trunc(self.r[i_trunc], rho_at_r_trunc)) / 2.
        else:
            self.rho[i_trunc] = (rho_interp1d(
                self.r[i_trunc - 1]) + self.get_rho_trunc(self.r[i_trunc], rho_at_r_trunc)) / 2.

        if self.flag_Hiro_code:
            if self.t == 0:
                self.rho[i_trunc] = self.get_rho_trunc(
                    (self.r[i_trunc] + self.r[i_trunc - 1]) / 2., rho_at_r_trunc)
            else:
                self.rho[i_trunc] = self.get_rho_trunc(
                    (self.r[i_trunc] + self.r_trunc) / 2., rho_at_r_trunc)

        # modify pressure at radii beyond r_trunc for fully truncated shells
        if i_trunc < self.n_shells - 1:
            self.p[i_trunc + 1:] = self.get_p_trunc(
                r_mid[i_trunc + 1:], rho_at_r_trunc, m_at_r_trunc, numeric=False)
            if self.flag_Hiro_code:
                self.p[i_trunc + 1:] = self.get_p_trunc(
                    r_mid[i_trunc + 1:], rho_at_r_trunc, m_at_r_trunc, numeric=True)

        # modify pressure for shells below and containing r_trunc
        if self.flag_Hiro_code:
            self.p[mask] = self.get_p_trunc(
                self.r[mask], rho_at_r_trunc, m_at_r_trunc, numeric=True)
            if self.t == 0:
                self.p[~mask] = self.get_initial_pressure(
                    self.r[~mask], numeric=True, xmax=self.r_trunc) + p_at_r_trunc
            else:
                self.p[i_trunc - 1] = (-integrate.trapz(
                    [(self.m[i_trunc - 1] * self.rho[i_trunc - 1] / self.r[i_trunc - 1] ** 2),
                     (m_at_r_trunc * rho_at_r_trunc / self.r_trunc ** 2)],
                    [self.r[i_trunc - 1], self.r_trunc]) + p_at_r_trunc)
                self.p[:i_trunc - 1] = (-integrate.cumtrapz(
                    (self.m[i_trunc - 1::-1] * self.rho[i_trunc - 1::-1]
                     / self.r[i_trunc - 1::-1] ** 2), self.r[i_trunc - 1::-1])[::-1]
                                        + self.p[i_trunc - 1])
        else:
            self.p[:i_trunc] = self.p[:i_trunc] + \
                               p_at_r_trunc - p_orig_at_r_trunc
            if self.t == 0:
                self.p[i_trunc] = (self.get_initial_pressure(self.r[i_trunc - 1]) + p_at_r_trunc - p_orig_at_r_trunc +
                                   self.get_p_trunc(self.r[i_trunc], rho_at_r_trunc, m_at_r_trunc, numeric=False)) / 2.
            else:
                self.p[i_trunc] = (p_interp1d(self.r[i_trunc - 1]) + p_at_r_trunc - p_orig_at_r_trunc +
                                   self.get_p_trunc(self.r[i_trunc], rho_at_r_trunc, m_at_r_trunc, numeric=False)) / 2.

        # update derived parameters
        #self.update_derived_parameters()
        self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r,
                                                                                          self.sigma_m,
                                                                                          self.velocity_dependence,
                                                                                          self.a, self.b, self.C, self.m_bh, self.m,
                                                                                          self.flag_central_bh,
                                                                                          self.flag_lmfp,
                                                                                          self.flag_baryon)

        # save the data in special file right after truncation
        self.save_halo(prefix='truncated_time')

        return

    def resample_shells(self, save=False, spl_bool=False, midpts=False):
        """
        Create newly-spaced shells and update halo parameters accordingly.
        This function is useful as shell spacing becomes large (particularly
        at small radii at late times). Use with caution, as interpolations will
        not perform well if spacing is too large.

        Parameters
        ----------
        save: bool
            if True, save adjusted halo state with prefix 'resample'
        """
        if midpts:
            ### obtain radii at midpoints of shells
            r_mid = self.get_shell_midpoints()

        ### create interpolation functions in log10 of quantities
        # omit innermost shell to ensure functions are smooth
        if spl_bool:
            interp_func = splrep
        else:
            interp_func = CubicSpline

        M10_interp_func = interp_func(np.log10(self.r[1:]), np.log10(self.m[1:]))
        if midpts:
            p10_interp_func = interp_func(np.log10(r_mid[1:]), np.log10(self.p[1:]))
            rho10_interp_func = interp_func(np.log10(r_mid[1:]), np.log10(self.rho[1:]))
        else:
            p10_interp_func = interp_func(np.log10(self.r[1:]), np.log10(self.p[1:]))
            rho10_interp_func = interp_func(np.log10(self.r[1:]), np.log10(self.rho[1:]))

        ### resample radii based on even sampling of log10(pressure)
        if midpts:
            r10_interp_func = interp_func(np.log10(self.p[:0:-1]), np.log10(r_mid[:0:-1]))
        else:
            r10_interp_func = interp_func(np.log10(self.p[:0:-1]), np.log10(self.r[:0:-1]))

        if spl_bool:
            def M10_interp(x):
                return splev(x, M10_interp_func)

            def p10_interp(x):
                return splev(x, p10_interp_func)

            def rho10_interp(x):
                return splev(x, rho10_interp_func)

            def r10_interp(x):
                return splev(x, r10_interp_func)
        else:
            M10_interp = M10_interp_func
            p10_interp = p10_interp_func
            rho10_interp = rho10_interp_func
            r10_interp = r10_interp_func

        ### create new locations of shells and reassign pressure
        # maintain number of shells and original min/max locations
        p10 = np.linspace(np.log10(self.p[1]), np.log10(self.p[-1]),
                          num=self.n_shells - 1, endpoint=True)
        self.p[1:-1] = pow(10., p10[:-1])
        if midpts:
            r_mid[1:] = pow(10., r10_interp(p10))
            ### shift shell locations to outer edges
            self.r[1:-1] = (r_mid[1:-1] + r_mid[2:]) / 2.
            self.r[-1] = r_mid[-1] + (r_mid[-1] - r_mid[-2]) / 2.
        else:
            self.r[1:] = pow(10., r10_interp(p10))

        ### readjust r[1]
        # only readjust if new r[1] is larger, to avoid extrapolations
        if not midpts:
            r10_new_1 = (np.log10(self.r[0]) + np.log10(self.r[2])) / 2.
            if r10_new_1 > np.log10(self.r[1]):
                self.r[1] = pow(10., r10_new_1)
                self.p[1] = pow(10., p10_interp(r10_new_1))

        ### save resampled quantities and update halo
        self.m[1:-1] = pow(10., M10_interp(np.log10(self.r[1:-1])))
        self.rho[1:-1] = pow(10., rho10_interp(np.log10(self.r[1:-1])))

        #self.update_derived_parameters()
        self.L, self.u, self.v, self.s, self.Kn = update_derived_parameters_1(self.dir_data.split('/')[-2],  self.p, self.rho, self.r, self.sigma_m, self.velocity_dependence, self.a, self.b, self.C, self.m_bh, self.m, self.flag_central_bh, self.flag_lmfp, self.flag_baryon)

        if save:
            self.save_halo(prefix='resample')
        return

    def get_dimensionful_quantity(self, quantity, units, value=None):
        """
        Obtain dimensionful radius.

        Parameters
        ----------
        quantity: string
            'r', 'rho', 'm', 'u', 'p', 'v', 't', 'L', 'sigma_m'
        units: Astropy unit
            Units in which quantity will be output.
        value: array-like object, default: None
            Dimensionless quantity to be converted.
            If None, use the relevant quantity of the current halo state.

        Returns
        -------
        array-like object
        """
        scale = getattr(self, 'scale_' + quantity)
        if value is None:
            value = getattr(self, quantity)
        return (value * scale).to_value(units)
