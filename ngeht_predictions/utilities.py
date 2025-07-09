import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import pandas as pd
import scipy.optimize as opt

c = 299792458 
D = 6378137  # Radius of the Earth in meters

station_code_key = {
    'AA' : 0,
    'AP' : 1,
    'AZ' : 2,
    'JC' : 3,
    'LM' : 4,
    'PV' : 5,
    'SM' : 6,
    'SP' : 7,
}
#[SM, PV, AZ]

def core_shift_freq_k(nu, R_0, a):
    return a + R_0 * nu ** (-1)

def core_shift_freq_k_R(nu, R_0):
    return R_0 * nu ** (-1)

def core_shift_freq_k_1(nu, R_0, a, nu_offset, k):
    """
    Returns the core shift in mas for a given frequency in GHz.
    R_0 is the core shift at 1 GHz in mas.
    k is the power law index.
    """
    return a + R_0 * ((nu - nu_offset) ** (-k))

def time_geometric_delay(mas, mas_error):
    """
    Returns
    -------
        Geometric time delay tau_g in seconds with its error.
    """
    arcsec = mas / 1000 # Convert mas to arcseconds
    theta_deg = arcsec / 3600  # Convert arcseconds to degrees 
    theta_rad = np.pi / 180 * theta_deg  # Convert degrees to radians

    arcsec_error = mas_error / 1000
    theta_error = arcsec_error / 3600  # Convert arcseconds to degrees
    theta_error_rad = np.pi / 180 * theta_error  # Convert degrees to radians

    time_delay = (D/c) * np.sin(theta_rad)

    #error propagation for time delay
    time_delay_error = (D/c) * np.sin(theta_rad) * (1/np.tan(theta_rad)) * theta_error_rad

    return time_delay, time_delay_error

def time_delay_freq_error(nu):
    """
    In hertz, the time delay is given. 1% error of the wavelength,
    converted to frequency.
    """

    # Convert frequency from GHz to Hz
    nu = nu * 1e9  # Convert GHz to Hz
    return 0.01 / nu

def time_delay_to_observing_angle(tau_g):
    """
    Converts the geometric time delay in seconds to the observing angle in radians.
    """

    return np.arcsin(c * tau_g / D)

def core_position_offset(delta_r_mas, z, D_L, nu_1, nu_2, k_r):
    
    delta_r_mas = (delta_r_mas * u.mas)
    D_L = (D_L * u.pc)
    nu_1 = (nu_1 * u.GHz)
    nu_2 = (nu_2 * u.GHz)

    a = 4.85 * 1e-9 * ((delta_r_mas * D_L)/(1 + z)**2) 
    b = (((nu_1**(1/k_r))*(nu_2**(1/k_r)))/((nu_2**(1/k_r)) - (nu_1**(1/k_r))))

    return a * b

def magnetic_field_1pc(core_shift, z, beta_app):
    
    return 0.042 * core_shift ** (3/4) * (1 + z)**(1/2) * (1 + beta_app**2)**(1/8)

def magnetic_field_1pc_prime(core_shift, z, beta_app):
 
    return 0.042 * (3/4) * core_shift ** (-1/4) * (1 + z)**(1/2) * (1 + beta_app**2)**(1/8)

def radians_to_mas(radians):
    """
    Converts radians to milliarcseconds (mas).
    """
    return radians * (3600 * 180) / np.pi * 1000

def atm_phase_error(delta_phase, freq_GHz, unit='mas'):

    # Delta phase is given in radians, freq_GHz is the frequency in GHz)
    freq_215 = freq_GHz * 1e9

    geometric_delay_phase_error = (delta_phase / (2 * np.pi)) * (1 / freq_215)
    geometric_delay_phase_error_mas = time_delay_to_observing_angle(geometric_delay_phase_error) * (3600 * 180) / np.pi * 1000

    if unit == 'sec':
        return geometric_delay_phase_error

    elif unit == 'mas':
        return geometric_delay_phase_error_mas

def randomize_coreshifts(ngEHT_coreshifts_real, multiply=True, sigma=0.1):
    ngEHT_coreshifts_simulated_real = [] 
    ngEHT_coreshifts_simulated_real_err = []

    for i in range(len(ngEHT_coreshifts_real)):
        if multiply:
            err = ngEHT_coreshifts_real[i] * sigma
        else:
            err = sigma
        cp_sim = np.random.normal(ngEHT_coreshifts_real[i], err)
        ngEHT_coreshifts_simulated_real_err.append(err)
        ngEHT_coreshifts_simulated_real.append(cp_sim)

    ngEHT_coreshifts_simulated_real = np.array(ngEHT_coreshifts_simulated_real)
    ngEHT_coreshifts_simulated_real_err = np.array(ngEHT_coreshifts_simulated_real_err)
    ngEHT_coreshifts_simulated_real_err[-1] = 0.00001

    return ngEHT_coreshifts_simulated_real, ngEHT_coreshifts_simulated_real_err

def fit_coreshifts(ngEHT_freq_real, ngEHT_coreshifts_simulated_real, ngEHT_coreshifts_simulated_real_err, sigma=0.1):
    ngEHT_params_real, covariance_real = opt.curve_fit(core_shift_freq_k, ngEHT_freq_real, ngEHT_coreshifts_simulated_real, sigma=ngEHT_coreshifts_simulated_real_err, absolute_sigma=True)
    ngEHT_perr_real = np.sqrt(np.diag(covariance_real))

    return ngEHT_params_real, ngEHT_perr_real

def inverse_var_weight(theta_obs, theta_obs_err):
    
    num = np.sum(theta_obs/theta_obs_err**2)
    denom = np.sum(1/theta_obs_err**2)

    return num / denom