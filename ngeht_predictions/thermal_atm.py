import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
import pandas as pd
from .utilities import *

class thermal_atm:
    def __init__(self):
        pass

    ############
    # THERMAL DELAY
    ############

    def thermal_noise_dualband(self, freq1, freq2, snr_hi=False, snr_low=False, baseline=False):

        if freq1 > freq2:
            freq_hi = freq1
            freq_low = freq2 
        elif freq2 > freq1:
            freq_hi = freq2
            freq_low = freq1

        # if snr_hi & snr_low:
        #     snr_hi = snr_hi
        #     snr_low = snr_low
        # else:
        #     snr = self.table['SNR']

        # if snr_hi & snr_low:
        #     n = snr_hi / snr_low
        # else:
        #     snr_hi, snr_low = snr, snr
        #     n = snr_hi / snr_low
        n = 1


        wavelength_hi = c / (freq_hi * 1e9)  # Convert GHz to Hz for wavelength calculation

        snr = (1/snr_hi) + n * (1/snr_low)

        delta_theta = 0.5 * (wavelength_hi / baseline) * snr

        delta_theta_mas =  delta_theta * (3600 * 180) / np.pi * 1000

        return delta_theta_mas
    
    def calculate_thermal_error(self, freq1, freq2, params):

        self.values_coreshifts = []
        self.errors_coreshifts = []
        self.thermal_noise_dualband_per_time = []
        self.all_thermal_noise_dualband = []


        for i in self.table_grouped:
            single_table = i[1]

            baseline = single_table['Baseline Length (m)']
            snr = single_table['SNR']

            eht_thermal_noise_dualband = self.thermal_noise_dualband(freq1, freq2, snr, snr, baseline)
            eht_thermal_noise_dualband_master = np.sqrt(1/(np.sum((eht_thermal_noise_dualband**(-2)))))

            self.thermal_noise_dualband_per_time.append(eht_thermal_noise_dualband_master)
            self.all_thermal_noise_dualband.append(eht_thermal_noise_dualband)

            ngEHT_corepositions_real = np.array([core_shift_freq_k(230,*params), core_shift_freq_k(345,*params),])

            ngEHT_freq_real = np.array([230., 345.])  # Frequencies in GHz

            r_0 = ngEHT_corepositions_real[1]

            ngEHT_coreshifts_real = []

            for i in range(len(ngEHT_corepositions_real)):
                core_position = ngEHT_corepositions_real[i] - r_0
                ngEHT_coreshifts_real.append(core_position)

            ngEHT_coreshifts_real = np.array(ngEHT_coreshifts_real)

            ngEHT_coreshifts_simulated_real, ngEHT_coreshifts_simulated_real_err = randomize_coreshifts(ngEHT_coreshifts_real, sigma=eht_thermal_noise_dualband_master, multiply=False)
            #ngEHT_coreshifts_simulated_real, ngEHT_coreshifts_simulated_real_err = ngEHT_coreshifts_real, eht_thermal_noise_dualband_master
            #ngEHT_coreshifts_simulated_real[1] = 2.10262717e-03

            ngEHT_params_real, ngEHT_perr_real = fit_coreshifts(ngEHT_freq_real, ngEHT_coreshifts_simulated_real, ngEHT_coreshifts_simulated_real_err, sigma=eht_thermal_noise_dualband_master)

            value = core_shift_freq_k(freq2, ngEHT_params_real[0], ngEHT_params_real[1])

            error = core_shift_freq_k(freq2, ngEHT_perr_real[0],ngEHT_perr_real[1])
                
            self.values_coreshifts.append(value)
            self.errors_coreshifts.append(error)

            self.thermal_values_coreshifts = np.array(self.values_coreshifts)
            self.thermal_errors_coreshifts = np.array(self.errors_coreshifts)

    
    ##########
    # ATMOSPHERIC DELAY
    ##########

    
    def generate_rand_atm_delay(self, num_ant, delta_phase, freq_GHz, unit='sec'):
        tau_atm = []
        
        geometric_delay_phase_error_mas = atm_phase_error(delta_phase, freq_GHz, unit=unit)

        for i in range(num_ant):
            t_mas = np.random.normal(0, geometric_delay_phase_error_mas/np.sqrt(num_ant))
            tau_atm.append(t_mas)

        return np.array(tau_atm)
    
    def calculate_obs_angles(self, single_table, tau_geo, tau_atm):

        theta_obs_list = []
        theta_error_list = []

        tau_geo = 0.0 # Geometric delay in seconds, assuming no geometric delay for simplicity

        for i,j,k,l in zip(single_table['tele1'], single_table['tele2'], single_table['Baseline Length (m)'], single_table['psi']):
            #print(i, j, station_code_key[i], station_code_key[j])

            tau_obs = tau_geo + tau_atm[station_code_key[i]] - tau_atm[station_code_key[j]]

            theta_obs = tau_obs * (c / (k * np.cos(l)))
            theta_error = (1/(k * np.cos(l)))

            # print(i, j, k, l)
            # print(tau_obs, theta_obs, theta_error)

            theta_obs_list.append(theta_obs)
            theta_error_list.append(theta_error)

        theta_obs_array = np.array(theta_obs_list)
        theta_error_array = np.array(theta_error_list)

        theta_obs_mas = radians_to_mas(theta_obs_array)
        theta_obs_error_mas = radians_to_mas(theta_error_array)
        
        
        inverse_var_weight(theta_obs_mas, theta_obs_error_mas)

        return theta_obs_mas, theta_obs_error_mas

    def calculate_atm_error(self, n_runs=500):
        results_per_scan = []
        for i in self.table_grouped:
            single_table = i[1]
            results = []
            for i in range(n_runs):
                    tau_atm = self.generate_rand_atm_delay(8, 1, 215, unit='sec')
                    theta, theta_err = self.calculate_obs_angles(single_table, 0, tau_atm)
                    results.append(inverse_var_weight(theta, theta_err))

            results_per_scan.append(results)

        atm_obs_angles = np.array(results_per_scan)
        self.atm_obs_angles = atm_obs_angles

thermal_atm_instance = thermal_atm()