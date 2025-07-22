import numpy as np
import ehtim as eh
import pandas as pd
from sklearn.linear_model import LinearRegression

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

        theta_source_list = []
        theta_source_error_list = []

        tau_geo = 0.0 # Geometric delay in seconds, assuming no geometric delay for simplicity

        for i,j,k,l,p in zip(single_table['tele1'], single_table['tele2'], single_table['Baseline Length (m)'], single_table['psi'], single_table['phase_seconds']):
            #print(i, j, station_code_key[i], station_code_key[j])
            
            tau_obs = tau_geo + tau_atm[station_code_key[i]] - tau_atm[station_code_key[j]]

            tau_source = p

            theta_obs = tau_obs * (c / (k * np.cos(l)))
            theta_error = (1/(k * np.cos(l)))

            theta_source = tau_source * (c / (k * np.cos(l)))
            theta_source_error = (1/(k * np.cos(l)))

            # print(i, j, k, l)
            # print(tau_obs, theta_obs, theta_error)

            theta_obs_list.append(theta_obs)
            theta_error_list.append(theta_error)

            theta_source_list.append(theta_source)
            theta_source_error_list.append(theta_source_error)

        theta_obs_array = np.array(theta_obs_list)
        theta_error_array = np.array(theta_error_list)

        theta_source_array = np.array(theta_source_list)
        theta_source_error_array = np.array(theta_source_error_list)

        self.theta_obs_array = theta_obs_array
        self.theta_error_array = theta_error_array

        self.theta_source_array = theta_source_array
        self.theta_source_error_array = theta_source_error_array

        theta_obs_mas = radians_to_mas(theta_obs_array)
        theta_obs_error_mas = radians_to_mas(theta_error_array)

        theta_source_mas = radians_to_mas(theta_source_array)
        theta_source_error_mas = radians_to_mas(theta_source_error_array)

        self.theta_obs_mas = theta_obs_mas
        self.theta_obs_error_mas = theta_obs_error_mas  

        self.theta_source_mas = theta_source_mas
        self.theta_source_error_mas = theta_source_error_mas


        # inverse_var_weight(theta_obs_mas, theta_obs_error_mas)

        return theta_obs_mas, theta_obs_error_mas

    def calculate_atm_error(self, n_runs=500, num_ant=8, phase_error=1, freq_GHz=215, num_of_scans=1):
        results_per_scan = []
        results_source_per_scan = []

        num_of_scans = num_of_scans - 1

        self.list_of_tables_multiple_scans = []


        for i in range(len(self.list_of_tables_perscan) - num_of_scans):
            # single_table = i[1]
            list_of_tables_for_combined_scans = []

            results = []
            results_source = []

            # thetas = []
            # theta_errors = []

            for iter in range(n_runs):

                thetas = []
                theta_errors = []

                thetas_source = []
                theta_errors_source = []

                for n in range(num_of_scans + 1):
                    list_of_tables_for_combined_scans.append(self.list_of_tables_perscan[i + n])
                    
                    tau_atm = self.generate_rand_atm_delay(num_ant, phase_error, freq_GHz, unit='sec')
                    theta, theta_err = self.calculate_obs_angles(self.list_of_tables_perscan[i + n], 0, tau_atm) # for loop
                    
                    thetas.extend(theta)
                    theta_errors.extend(theta_err)

                    thetas_source.extend(self.theta_source_mas)
                    theta_errors_source.extend(self.theta_source_error_mas)

                results.append(inverse_var_weight(np.array(thetas), np.array(theta_errors)))
                results_source.append(inverse_var_weight(np.array(thetas_source), np.array(theta_errors_source)))

            # print(thetas, len(thetas))

            # try:
            #     print(thetas.shape, theta_errors.shape)
            # except:
            #     print("Error in shape of thetas or theta_errors")
            
            self.list_of_tables_multiple_scans.append(pd.concat(list_of_tables_for_combined_scans, ignore_index=True))
            results_per_scan.append(results)
            results_source_per_scan.append(results_source)

        atm_obs_angles = np.array(results_per_scan)

        self.atm_obs_angles = atm_obs_angles
        self.source_obs_angles = np.array(results_source_per_scan)

    def calculate_antenna_based_delay_obs_angles_2D(self, single_table, tau_atm):

        freq = 215

        antenna_based_delay_phase = []
        for i, j in zip(single_table['tele1'], single_table['tele2'], ):
            # Calculate the antenna-based delay phase for each baseline
            delay_phase = tau_atm[station_code_key[i]] - tau_atm[station_code_key[j]]
            antenna_based_delay_phase.append(delay_phase) # in seconds

        u = single_table['u_coords']
        v = single_table['v_coords']
        
        antenna_based_delay_phase = np.array(antenna_based_delay_phase) * (2 * np.pi * freq * 1e9) # Convert seconds to radians

        # print(u.shape, v.shape, antenna_based_delay_phase.shape)
        # print(antenna_based_delay_phase)

        reg = LinearRegression().fit(2*np.pi*np.column_stack((u, v)), antenna_based_delay_phase)

        return radians_to_mas(reg.coef_)
    

    def calculate_source_structure_delay_obs_angles_2D(self, single_table):

        u = single_table['u_coords']
        v = single_table['v_coords']
        
        antenna_based_delay_phase = single_table['phase'] # Convert seconds to radians

        print(u.shape, v.shape, antenna_based_delay_phase.shape)
        print(antenna_based_delay_phase)

        reg = LinearRegression().fit(2*np.pi*np.column_stack((u, v)), antenna_based_delay_phase)

        return radians_to_mas(reg.coef_)
    
    def calculate_antenna_based_delay_2D(self, n_runs=500, num_ant=8, phase_error=1, freq_GHz=215, num_of_scans=1):

        num_of_scans = num_of_scans - 1

        results_per_run = []
        for iter in range(n_runs):
            results_per_scan = []
            for i in range(len(self.list_of_tables_perscan) - num_of_scans):
                tables = []
                for n in range(num_of_scans + 1):

                    tables.append(self.list_of_tables_perscan[i + n])

                concat_table = pd.concat(tables, ignore_index=True)

                tau_atm = self.generate_rand_atm_delay(num_ant, phase_error, freq_GHz, unit='sec')
                theta = self.calculate_antenna_based_delay_obs_angles_2D(concat_table, tau_atm) # for loop

                results_per_scan.append(theta)

            results_per_run.append(results_per_scan)
            

        self.antenna_based_delay_results = np.array(results_per_run)

    def calculate_source_structure_delay_2D(self, num_of_scans=1):

        results_per_scan = []

        num_of_scans = num_of_scans - 1

        for i in range(len(self.list_of_tables_perscan) - num_of_scans):
            tables = []
            for n in range(num_of_scans + 1):
                tables.append(self.list_of_tables_perscan[i + n])
                
            concat_table = pd.concat(tables, ignore_index=True)

            theta = self.calculate_source_structure_delay_obs_angles_2D(concat_table) # for loop
            results_per_scan.append(theta)

        self.source_structure_delay_results = np.array(results_per_scan)

thermal_atm_instance = thermal_atm()