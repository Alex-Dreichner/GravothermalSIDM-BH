import os,sys
import time
import numpy as np
import halo_evolution as he
from astropy import units as ut
import cProfile
import pstats

''' Set default halo parameters (do not change) '''
halo_defaults = {
    'profile': 'NFW',
    'r_s': 2.586, # [kpc]
    'rho_s': 0.0194, # [M_sun/pc^3]
    't_trunc': -1,
    'p_trunc': 5,
    'r_trunc': 3,
    'm_bh': 0.,  # [M_sun]
    'a': 4./np.sqrt(np.pi),
    'b': 25.*np.sqrt(np.pi)/32.,
    'C': 0.753,
    'gamma': 5./3.,
    'model_elastic_scattering_lmfp': 'constant',
    'model_elastic_scattering_smfp': 'constant',
    'sigma_m_with_units': 5., # [cm^2/g]
    'w_units': 1., # [km/s]
    'model_inelastic_scattering': 'none',
    'sigma_m_inelastic_with_units': 0., # [cm^2/g]
    'v_loss_units': 0., # [km/s] velocity loss
    'n_shells': 400,
    'r_min': 0.01,
    'r_max': 100.,
    'p_rmax_factor': 10.,
    'n_adjustment_fixed': -1,
    'n_adjustment_max': -1,
    'flag_timestep_use_relaxation': True,
    'flag_timestep_use_energy': False,
    'flag_hydrostatic_initial': False,
    'flag_central_bh': False,
    'flag_Hiro_code': False,
    'flag_r_minus': -1,
    'flag_entropy': False,
    'flag_shapiro': False,
    'velocity_dependence': 4,
    'max_index': 0,
    'initial_bh': False,
    'flag_lmfp': False
}

''' Set default run parameters (do not change) '''
run_defaults = {
    # parameters to determine how long to run evolution
    't_end': np.inf,
    'rho_factor_end': np.inf,
    'Kn_end': 0,
    # parameters to determine how often to save files
    'save_frequency_rate': None,
    'save_frequency_timing': None,
    'save_frequency_density': None,
    # precision parameters
    't_epsilon': 1.e-4,
    'r_epsilon': 1.e-12
}

def get_halo_parameters(run_name):
    """
    Obtain halo and run parameters for new evolution runs, given an input run
    name. Defaults are explicitly defined above in order to preserve exact
    parameters, in case defaults within the evolution code are changed.
    """
    hdict = halo_defaults.copy()
    rdict = run_defaults.copy()
    if run_name=='default':
        rdict['t_end'] = 20.
        pass
    elif run_name == 'shapiro':
        rdict['t_end'] = 10*10
        hdict['n_shells'] = 200
        hdict['r_min'] = 1.e-3
        hdict['r_max'] = 25*10
        hdict['profile'] = False
        hdict['m_bh'] = 0.01*(4*np.pi*hdict['r_s']*hdict['rho_s']*1.e9)
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        hdict['flag_shapiro'] = True
        hdict['flag_entropy'] = False
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'shapiro_static':
        a = np.sqrt(2)
        rdict['t_end'] = 1
        hdict['n_shells'] = 150
        hdict['r_s'] = a
        hdict['rho_s'] =1/(4*np.pi*a**3)
        hdict['r_min'] = 3.81*1.e-5
        hdict['r_max'] = 1.12*1.e-2*10#*100000
        hdict['profile'] = 'Plummer'
        hdict['m_bh'] = 0.942*1.e-4*1.e9
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        hdict['flag_entropy'] = False
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'shapiro_static_300':
        a = np.sqrt(2)
        rdict['t_end'] = 1
        hdict['n_shells'] = 300
        hdict['r_s'] = a
        hdict['rho_s'] =1/(4*np.pi*a**3)
        hdict['r_min'] = 3.81*1.e-5
        hdict['r_max'] = 1.12*1.e-2*10#*100000
        hdict['profile'] = 'Plummer'
        hdict['m_bh'] = 0.942*1.e-4*1.e9
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        hdict['flag_entropy'] = False
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'shapiro_static_e':
        a = np.sqrt(2)
        rdict['t_end'] = 1
        hdict['n_shells'] = 150
        hdict['r_s'] = a
        hdict['rho_s'] =1/(4*np.pi*a**3)
        hdict['r_min'] = 3.81*1.e-5
        hdict['r_max'] = 1.12*1.e-2*10#*100000
        hdict['profile'] = 'Plummer'
        hdict['m_bh'] = 0.942*1.e-4*1.e9
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        hdict['flag_entropy'] = True
        rdict['save_frequency_rate'] = 30
        # rdict['save_frequency_timing'] = 0.1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'alonso_bh':
        rdict['t_end'] = 10
        hdict['r_s'] = 2 * 1.e3
        hdict['rho_s'] = 3 * 1.e-4
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.00001
        hdict['r_max'] = 10
        hdict['m_bh'] = 1.e9 * 6
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
    elif run_name == 'alonso_bh_lmfp':
        rdict['t_end'] = 10
        hdict['r_s'] = 2 * 1.e3
        hdict['rho_s'] = 3 * 1.e-4
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.00001
        hdict['r_max'] = 10
        hdict['m_bh'] = 1.e9 * 6
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'alonso_bh_lmfp_sidmish':
        rdict['t_end'] = 10
        hdict['r_s'] = 2 * 1.e3
        hdict['rho_s'] = 3 * 1.e-4
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.00001
        hdict['r_max'] = 10
        hdict['m_bh'] = 1.e9 * 6*1.e-200
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'alonso_bh_sidmish':
        rdict['t_end'] = 10
        hdict['r_s'] = 2 * 1.e3
        hdict['rho_s'] = 3 * 1.e-4
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.00001
        hdict['r_max'] = 10
        hdict['m_bh'] = 1.e9 * 6*1.e-200
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'alonso_bh_lmfp_sidm':
        rdict['t_end'] = 10
        hdict['r_s'] = 2 * 1.e3
        hdict['rho_s'] = 3 * 1.e-4
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.00001
        hdict['r_max'] = 10
        hdict['m_bh'] = 0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'alonso_bh_sidm':
        rdict['t_end'] = 10
        hdict['r_s'] = 2 * 1.e3
        hdict['rho_s'] = 3 * 1.e-4
        hdict['n_shells'] = 423*10
        hdict['r_min'] = 0.00001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 1
        #rdict['save_frequency_density'] = 0.15
    elif run_name == 'baryon':
        rdict['t_end'] = 10
        hdict['r_s'] = 0.073
        hdict['rho_s'] = 2.6
        hdict['n_shells'] = 201
        hdict['r_min'] = 1.e-4
        hdict['r_max'] = 100
        hdict['sigma_m_with_units'] = 5.046
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 40
        rdict['t_epsilon'] = 0.001
        rdict['save_frequency_density'] = 3
        hdict['velocity_dependence'] = 0
        hdict['flag_Hiro_code'] = True
        hdict['flag_baryon'] = True
    elif run_name == 'baryon_e':
        rdict['t_end'] = 10
        hdict['r_s'] = 0.073
        hdict['rho_s'] = 2.6
        hdict['n_shells'] = 182*2
        hdict['r_min'] = 1.e-3
        hdict['r_max'] = 100
        hdict['sigma_m_with_units'] = 5.046
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = True
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        hdict['velocity_dependence'] = 0
        hdict['flag_Hiro_code'] = True
        hdict['flag_baryon'] = True
        rdict['save_frequency_rate'] = 40
        rdict['save_frequency_density'] = 3
    elif run_name == 'baryon_e_sidm':
        rdict['t_end'] = 10
        hdict['r_s'] = 0.073
        hdict['rho_s'] = 2.6
        hdict['n_shells'] = 201
        hdict['r_min'] = 1.e-3
        hdict['r_max'] = 100
        hdict['sigma_m_with_units'] = 5.046
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = True
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        hdict['velocity_dependence'] = 0
        hdict['flag_Hiro_code'] = False
        hdict['flag_baryon'] = False
        rdict['save_frequency_rate'] = 4
        rdict['save_frequency_density'] = 3
    elif run_name == 'baryon_sidm':
        rdict['t_end'] = 10
        hdict['r_s'] = 0.073
        hdict['t_epsilon'] = 0.001
        hdict['rho_s'] = 2.6
        hdict['n_shells'] = 201
        hdict['r_min'] = 1.e-3
        hdict['r_max'] = 100
        hdict['sigma_m_with_units'] = 5.046
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        hdict['velocity_dependence'] = 0
        hdict['flag_Hiro_code'] = False
        hdict['flag_baryon'] = False
        rdict['save_frequency_rate'] = 1
        #rdict['save_frequency_density'] = 3
    elif run_name == 'test1':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 1
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_old_both':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 100
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 4
        rdict['save_frequency_timing'] = 0.3
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_modm':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 100
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 4
        rdict['save_frequency_timing'] = 0.3
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_bh10':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2*10
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 100
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 4
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_old_bh01':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2*0.1
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 100
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 4
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_old_sigma':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.1
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 4
        rdict['save_frequency_timing'] = 0.3
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_sigma01':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.001
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 4
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_old_minr':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-7
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_min':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_initial_bh':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [kpc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['initial_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_old_300':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['max_index'] = 300
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_n_600':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400*2
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['max_index'] = 300*2
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_n_200':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400*2
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['max_index'] = 100*2
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_n_400':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400*2
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['max_index'] = 200*2
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_100':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['max_index'] = 100
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_200':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['max_index'] = 200
        # rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_sidm':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 201
        hdict['m_bh'] = 1.e2*0
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_sidm_both':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 201
        hdict['m_bh'] = 1.e2*0
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 0.1
    elif run_name == 'rmove':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 401
        hdict['m_bh'] = 1.e2*0
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test1_old_sidmish':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 201
        hdict['m_bh'] = 1.e-200
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 1
    elif run_name == 'test1_old_sidmish_both':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 201
        hdict['m_bh'] = 1.e-200
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = False
        rdict['save_frequency_rate'] = 1
        # rdict['save_frequency_density'] = 3
        rdict['save_frequency_timing'] = 1
    elif run_name == 'test1_0_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 3
        hdict['velocity_dependence'] = 0
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_3_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 3
        hdict['velocity_dependence'] = 3
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_5_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 20
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 5
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_6_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 6
        rdict['save_frequency_timing'] = 1
    elif run_name == 'test1_6_old_sidm':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2*0
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 1
        hdict['velocity_dependence'] = 6
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_2_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 3
        hdict['velocity_dependence'] = 2
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_1_old':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 10
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = False
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 3
        hdict['velocity_dependence'] = 1
        rdict['save_frequency_timing'] = 0.3
    elif run_name == 'test1_s_001':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 1
        hdict['sigma_m_with_units'] = 5*0.01*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 30
        rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name == 'test2':
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e4
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] = 2.586  # [pc]
        rdict['t_end'] = 0.1
        hdict['sigma_m_with_units'] = 5*0.01
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_entropy'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = -1
        hdict['flag_lmfp'] = True
        rdict['save_frequency_rate'] = 3000
        rdict['save_frequency_density'] = 3
        # rdict['save_frequency_timing'] = 0.1
    elif run_name=='essig_no_cooling':
        # parameters to reproduce no-cooling run in 1809.01144
        hdict['r_s'] = 6.5
        hdict['rho_s'] = 0.0128
        hdict['C'] = 0.6
        hdict['sigma_m_with_units'] = 3.
        hdict['n_shells'] = 150
        hdict['r_min'] = 0.01
        hdict['r_max'] = 1000.
        hdict['n_adjustment_fixed'] = 10
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_timestep_use_energy'] = True
        rdict['Kn_end'] = 0.1
        rdict['t_epsilon'] = 1.e-3
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name == 'hiro_NFW_3':
        # parameters to reproduce NFW run in 1901.00499
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 400
        hdict['m_bh'] = 1.e2
        hdict['r_max'] = 1.e2
        hdict['r_min'] = 1.e-6
        hdict['r_s'] =  2.586  # [pc]
        # hdict['M_BH_div_M_DM'] = 1e-2
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_timestep_use_relaxation'] = False
        #hdict['n_adjustment_fixed'] = 100
        hdict['flag_central_bh'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_Hiro_code'] = False
        rdict['t_end'] = 10.561
        #rdict['save_frequency_rate'] = 10000.#/1.51844308
        hdict['flag_r_minus'] = -1
        rdict['save_frequency_density'] = 0.1
        
    elif run_name == 'comp_eda':
        # test profile Lukas
        # rdict['t_end'] = 5.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.01
        hdict['r_max'] = 100
        hdict['m_bh'] = 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = 1
        #rdict['save_frequency_rate'] = 3.
        #rdict['save_frequency_timing'] = 0.1
        #rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_4':
        # test profile Lukas
        # rdict['t_end'] = 5.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        #hdict['sigma_m_with_units'] = 0.1
        hdict['n_shells'] = 600
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_central_bh'] = True
        hdict['flag_r_minus'] = 1
        #rdict['save_frequency_rate'] = 3.
        #rdict['save_frequency_timing'] = 0.1
        #rdict['save_frequency_density'] = 0.005
    
    elif run_name == 'comp_eda_s1':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 1.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 3.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s132':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 1.32
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 3.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s5':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 5.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 3.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s10':
        # test profile Lukas
        rdict['t_end'] = 10./5633*100
        hdict['sigma_m_with_units'] = 10.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 100
        #rdict['save_frequency_timing'] = 0.1
        #rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s10_e':
        # test profile Lukas
        rdict['t_end'] = 10./5633*100
        hdict['sigma_m_with_units'] = 10.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_entropy'] = True
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 3
        #rdict['save_frequency_timing'] = 0.1
        #rdict['save_frequency_density'] = 0.005

    elif run_name == 'comp_eda_s5_rmin':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 5.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.01
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s10_rmin':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 10.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.01
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s2_rmin':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 2.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.01
        hdict['r_max'] = 100
        hdict['m_bh'] = 0*1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005

    elif run_name == 'comp_eda_s2_rmin1':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 2.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 200
        hdict['r_min'] = 0.01
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'comp_eda_s2_rminmin':
        # test profile Lukas
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 2.
        hdict['r_s'] = 0.0231
        hdict['rho_s'] = 5.615
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.1
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.005

    elif run_name == 'alvarez_s1':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 1.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 10
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s2':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 2.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 10
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s10':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 10.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 10
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s10_max':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 10.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s5':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 5.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s_1':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.1
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s001':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.001
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s01':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.01
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s20':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 20.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'alvarez_s50':
        # SIDM from Alvarez paper
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 50.
        hdict['r_s'] = 1.94
        hdict['rho_s'] = 0.0168
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e3
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005

    elif run_name == 'draco_s5':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 5.
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s_01':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.01
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s_1':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.1
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s1':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 1.
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s10':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 11.
        hdict['sigma_m_with_units'] = 10.
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s_1':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.1
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s_001':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 0.001
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s15':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 15.
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    elif run_name == 'draco_s20':
        # SIDM for Draco from arxiv:1904.04939v2
        rdict['t_end'] = 10.
        hdict['sigma_m_with_units'] = 20.
        hdict['r_s'] = 1.46
        hdict['rho_s'] = 0.0277
        hdict['n_shells'] = 400
        hdict['r_min'] = 0.0001
        hdict['r_max'] = 100
        hdict['m_bh'] = 0 * 1.e5
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = False
        hdict['flag_central_bh'] = False
        hdict['flag_r_minus'] = 2
        rdict['save_frequency_rate'] = 1.
        rdict['save_frequency_timing'] = 0.001
        rdict['save_frequency_density'] = 0.005
    
    





    elif run_name=='laura_NFW':
        # parameters to reproduce Laura's code
        hdict['r_s'] = 2.59
        hdict['rho_s'] = 0.019
        hdict['C'] = 0.75
        hdict['n_shells'] = 100
        rdict['t_end'] = 431
        rdict['t_epsilon'] = 1.e-2
        rdict['r_epsilon'] = 1.e-5
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='hiro_NFW':
        # parameters to reproduce NFW run in 1901.00499
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 200
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_Hiro_code'] = True
        rdict['t_end'] = 374.561
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='hiro_TNFW':
        # parameters to reproduce TNFW run in 1901.00499
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['n_shells'] = 300
        hdict['t_trunc']  = 0
        hdict['r_trunc'] = 1
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_Hiro_code'] = True
        rdict['t_end'] = 53
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='hiro_TNFW3Gyr':
        # parameters to reproduce TNFW3Gyr run in 1901.00499
        hdict['a'] = 2.257
        hdict['b'] = 1.385
        hdict['C'] = 0.75
        hdict['t_trunc']  = 11.74659423
        hdict['r_trunc'] = 1
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_Hiro_code'] = True
        rdict['t_end'] = 60.11
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='essig_no_cooling':
        # parameters to reproduce no-cooling run in 1809.01144
        hdict['r_s'] = 6.5
        hdict['rho_s'] = 0.0128
        hdict['C'] = 0.6
        hdict['sigma_m_with_units'] = 3.
        hdict['n_shells'] = 150
        hdict['r_min'] = 0.01
        hdict['r_max'] = 1000.
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_timestep_use_energy'] = True
        rdict['Kn_end'] = 0.01
        rdict['t_epsilon'] = 1.e-3
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='essig_no_cooling_n_adjustment_fixed_10':
        # parameters to reproduce no-cooling run in 1809.01144
        hdict['r_s'] = 6.5
        hdict['rho_s'] = 0.0128
        hdict['C'] = 0.6
        hdict['sigma_m_with_units'] = 3.
        hdict['n_shells'] = 150
        hdict['r_min'] = 0.01
        hdict['r_max'] = 1000.
        hdict['n_adjustment_fixed'] = 10
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_timestep_use_energy'] = True
        rdict['Kn_end'] = 0.01
        rdict['t_epsilon'] = 1.e-3
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='essig_with_cooling':
        # parameters to reproduce inleastic/cooling run in 1809.01144
        hdict['r_s'] = 6.5
        hdict['rho_s'] = 0.0128
        hdict['C'] = 0.6
        hdict['sigma_m_with_units'] = 3.
        hdict['model_inelastic_scattering'] = 'constant'
        hdict['sigma_m_inelastic_with_units'] = 3. # [cm^2/g]
        hdict['v_loss_units'] = 13. # [km/s] velocity loss
        hdict['n_shells'] = 150
        hdict['r_min'] = 0.01
        hdict['r_max'] = 1000.
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_timestep_use_energy'] = True
        #rdict['Kn_end'] = 1.e-3 
        rdict['t_epsilon'] = 1.e-3
        rdict['save_frequency_rate'] = 10./1.51844308
    elif run_name=='essig_with_cooling_n_adjustment_fixed_10':
        # parameters to reproduce inelastic/cooling run in 1809.01144
        hdict['r_s'] = 6.5
        hdict['rho_s'] = 0.0128
        hdict['C'] = 0.6
        hdict['sigma_m_with_units'] = 3.
        hdict['model_inelastic_scattering'] = 'constant'
        hdict['sigma_m_inelastic_with_units'] = 3. # [cm^2/g]
        hdict['v_loss_units'] = 13. # [km/s] velocity loss
        hdict['n_shells'] = 150
        hdict['r_min'] = 0.01
        hdict['r_max'] = 1000.
        hdict['n_adjustment_fixed'] = 10
        hdict['flag_timestep_use_relaxation'] = False
        hdict['flag_timestep_use_energy'] = True
        #rdict['Kn_end'] = 1.e-3
        rdict['t_epsilon'] = 1.e-3
        rdict['save_frequency_rate'] = 10./1.51844308

    elif run_name=='NFW_sigmam_5_const':
        # 5 cm^2/g to reproduce previous run
        hdict['n_shells'] = 400
        # hdict['r_s'] = 3.
        # hdict['rho_s'] = 0.02
        # hdict['C'] = 0.6 # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['sigma_m_with_units'] = 5.
        # hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        # rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    ###################################
    ###### LMFP paper runs below ######
    ###################################

    elif run_name=='NFW_wlarge_sigmac1_LMFPpaper':
        # params for large w giving behavior like constant cross-section sigma/m = 1 cm^2/g
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6 # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = (2/3) * 1.5 # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_wlarge_sigmac1_LMFPpaper_n_shells800':
        # params for large w giving behavior like constant cross-section sigma/m = 1 cm^2/g
        hdict['n_shells'] = 800
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6 # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = (2/3) * 1.5 # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_wlarge_sigmac1_LMFPpaper_n_shells1200':
        # params for large w giving behavior like constant cross-section sigma/m = 1 cm^2/g
        hdict['n_shells'] = 1200
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6 # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = (2/3) * 1.5 # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_wlarge_sigmac5_LMFPpaper':
        # params for large w giving behavior like constant cross-section sigma/m = 5 cm^2/g
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = (2 / 3) * 7.5  # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_wlarge_sigmac5_LMFPpaper_nshells800':
        # params for large w giving behavior like constant cross-section sigma/m = 5 cm^2/g
        hdict['n_shells'] = 800
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = (2 / 3) * 7.5  # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_wlarge_sigmac5_LMFPpaper_nshells1200':
        # params for large w giving behavior like constant cross-section sigma/m = 5 cm^2/g
        hdict['n_shells'] = 1200
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = (2 / 3) * 7.5  # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w1_sigmac1_LMFPpaper':
        # params for velocity dependent cross-section with w = 1 km/s and sigma(v)/m = 1 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 1e6 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w1_sigmac5_LMFPpaper':
        # params for velocity dependent cross-section with w = 1 km/s and sigma(v)/m = 5 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 6e6 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w1_sigmac90_LMFPpaper':
        # params for v-dep cross-section, w = 1 km/s, sigma(v)/m = 168 cm^2/g at 25 km/s (sigma(100 km/s) = 1 cm^2/g)
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 1e8 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w1_sigmac90_LMFPpaper_nshells800':
        # params for v-dep cross-section, w = 1 km/s, sigma(v)/m = 168 cm^2/g at 25 km/s (sigma(100 km/s) = 1 cm^2/g)
        hdict['n_shells'] = 800
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 1e8 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w95_sigmac5_LMFPpaper':
        # params for v-dependent cross-section with w = 11.18 km/s and sigma(v)/m = 5 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 15 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 95.12
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w40_sigmac5_LMFPpaper':
        # params for v-dependent cross-section with w = 5 km/s and sigma(v)/m = 5 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 60 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 40.21
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_w10_sigmac5_LMFPpaper':
        # params for v-dependent cross-section with w = 2.24 km/s and sigma(v)/m = 5 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 1.5e3 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10.81
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_rhos4_rs4_wlarge_LMFPpaper':
        # params for halo with rhos = 4 M_sun/kpc^3 and rs = 4 kpc, and ~constant cross-section sigma/m = 5 cm^2/g
        hdict['n_shells'] = 400
        hdict['r_s'] = 4.
        hdict['rho_s'] = 0.04
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 7.5 * (2 / 3)  # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_rhos4_rs4_w1_LMFPpaper':
        # params for halo with rhos = 4 M_sun/kpc^3 and rs = 4 kpc, and
        # v-dependent cross-section with w = 1 km/s and sigma(v)/m = 5 cm^2/g at 25 km/s
        hdict['n_shells'] = 400
        hdict['r_s'] = 4.
        hdict['rho_s'] = 0.04
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 6e6 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_rhos40_rs08_wlarge_LMFPpaper':
        # params for halo with rhos = 40 M_sun/kpc^3 and rs = 0.8 kpc, and ~constant cross-section sigma/m = 5 cm^2/g
        hdict['n_shells'] = 400
        hdict['r_s'] = 0.8
        hdict['rho_s'] = 0.4
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 7.5 * (2 / 3)  # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_rhos40_rs08_w1_LMFPpaper':
        # params for halo with rhos = 40 M_sun/kpc^3 and rs = 0.8 kpc, and
        # v-dependent cross-section with w = 1 km/s and sigma(v)/m = 5 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 0.8
        hdict['rho_s'] = 0.4
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 6e6 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_rhos01_rs10_wlarge_LMFPpaper':
        # params for halo with rhos = 0.1 M_sun/kpc^3 and rs = 10 kpc, and ~constant cross-section sigma/m = 5 cm^2/g
        hdict['n_shells'] = 400
        hdict['r_s'] = 10.
        hdict['rho_s'] = 0.001
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 7.5 * (2 / 3)  # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['w_units'] = 10000
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10./1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01
    elif run_name=='NFW_rhos01_rs10_w1_LMFPpaper':
        # params for halo with rhos = 0.1 M_sun/kpc^3 and rs = 10 kpc, and
        # v-dependent cross-section with w = 1 km/s and sigma(v)/m = 5 cm^2/g at vc
        hdict['n_shells'] = 400
        hdict['r_s'] = 10.
        hdict['rho_s'] = 0.001
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApproxTchannel_K3_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApproxTchannel_K5_order1'
        hdict['sigma_m_with_units'] = 6e6 * (2/3) # multiplied by 2/3 since sigma_m_with_units = (2/3) * sigma0
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['rho_factor_end'] = 1000
        rdict['save_frequency_rate'] = 10. / 1.51844308
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    #############################
    ###### SMFP paper runs ######
    #############################

    elif run_name=='n1_sigmac5_SMFP':
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 103.9
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 11
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 10
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n2_sigmac5_SMFP':
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 41.7
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 42
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 80
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n3_sigmac5_SMFP':
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 11.8
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 1050
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 300
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n3_7_sigmac5_SMFP':
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 1
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 5e6
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 800
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n1_sigmac90_SMFP':
        # hdict['n_shells'] = 600
        # hdict['r_min'] = 0.0001
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 103.9
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 195
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 50
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n2_sigmac90_SMFP':
        # hdict['n_shells'] = 600
        # hdict['r_min'] = 0.0001
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 41.7
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 770
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 50
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n3_sigmac90_SMFP':
        # hdict['n_shells'] = 600
        # hdict['r_min'] = 0.0001
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 11.8
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 1.9e4
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 100
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    elif run_name=='n3_7_sigmac90_SMFP':
        # hdict['n_shells'] = 600
        # hdict['r_min'] = 0.0001
        hdict['n_shells'] = 500
        hdict['r_min'] = 0.001
        hdict['r_max'] = 100
        hdict['r_s'] = 3.
        hdict['rho_s'] = 0.02
        hdict['C'] = 0.6  # to match Essig et al findings of calibration parameter matching N-body sims at late times
        hdict['w_units'] = 1
        hdict['model_elastic_scattering_lmfp'] = 'YukawaBornViscosityApprox_K5_order1'
        hdict['model_elastic_scattering_smfp'] = 'YukawaBornViscosityApprox_K5_order2'
        hdict['sigma_m_with_units'] = 9e7
        hdict['flag_timestep_use_energy'] = True
        hdict['flag_hydrostatic_initial'] = True
        hdict['flag_timestep_use_relaxation'] = False
        rdict['save_frequency_rate'] = 50
        rdict['save_frequency_timing'] = 0.1
        rdict['save_frequency_density'] = 0.01

    else:
        raise IOError('Run name {} is not recognized'.format(run_name))
    return hdict,rdict

def perform_run(run_name,dir_output=None):
    """
    Based on run_name, obtain halo and run parameters and run halo
    evolution. Data is saved under a directory matching the run name.
    A top-level directory can be specified with dir_output; otherwise,
    the top-level directory is assumed to be the current directory.
    """
    # get halo and run parameters
    hdict,rdict = get_halo_parameters(run_name)

    # set data directory, based on run name
    if dir_output is None:
        dir_data = run_name
    else:
        dir_data = os.path.join(dir_output,run_name)

    # create halo
    sidmhalo = he.Halo(dir_data,**hdict)
    rdict['t_end'] = rdict['t_end']/sidmhalo.get_dimensionful_quantity('t',ut.Gyr,value=1.)
    rdict['save_frequency_rate'] = rdict['save_frequency_rate']#*sidmhalo.get_dimensionful_quantity('t',ut.Gyr,value=1.)
    print('t_end in dimensionless quantity is: {}'.format(rdict['t_end']))
    # evolve halo
    start=time.time()
    sidmhalo.evolve_halo(**rdict)
    end=time.time()

    print('time elapsed for run {} = {}'.format(run_name,end-start))
    return sidmhalo

###############################################################################
if __name__=="__main__":

    ''' Set location of top level output directory. '''
    dir_output = 'data'

    ''' Set name of run. '''
    if len(sys.argv) > 1:
        for run_name in sys.argv[1:]:
            try:
                with cProfile.Profile() as pr:
                    halo = perform_run(run_name, dir_output=dir_output)
            except KeyboardInterrupt:
                print('smmth')
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.dump_stats(filename=run_name + '_stats.prof')
    else:
        print('non valid input. Try something like "python run.py test1_old"')
        raise SystemExit

