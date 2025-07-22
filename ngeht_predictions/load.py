import numpy as np
import ehtim as eh
import pandas as pd

from .utilities import *
from .thermal_atm import thermal_atm

class load:
    def __init__(self, uvfits=False, scans=True):

        self.thermal_atm = thermal_atm()

        if uvfits:
            self.uvfits_str = str(uvfits)

        else: 
            self.uvfits = None

    def load_uvfits(self, scans=True, ):

        self.obs = eh.obsdata.load_uvfits(self.uvfits_str)

        if scans:
            self.obs.add_scans()
            self.obs = self.obs.avg_coherent(0, scan_avg=True)


        EHT_freq = 230
        EHT_wavelength = (c / (EHT_freq * 1e9))

        u_coords_meters = self.obs.unpack(['u', 'v']).u * EHT_wavelength   # Convert to meters
        v_coords_meters = self.obs.unpack(['u', 'v']).v * EHT_wavelength  # Convert to meters

        self.u_coords = self.obs.unpack(['u', 'v']).u
        self.v_coords = self.obs.unpack(['u', 'v']).v

        baseline_lengths = np.sqrt(u_coords_meters**2 + v_coords_meters**2)
        baseline_angles = np.arctan2(u_coords_meters , v_coords_meters)

        snr_eht_array = self.obs.unpack(['snr']).astype(np.float64) 

        tele1 = self.obs.unpack(['t1']).astype('str')
        tele2 = self.obs.unpack(['t2']).astype('str')

        phase = self.obs.unpack(['phase']).astype(np.float64)

        # Convert phase in degrees to radians
        phase = phase * (2*np.pi / 360)

        source_structure_delay = (phase/(2*np.pi)) * (1/(230 * 1e9))

        times = self.obs.unpack(['time'])['time']

        self.jet_angle = np.pi / 4

        psi = baseline_angles - self.jet_angle

        self.table = pd.DataFrame({
            'Baseline Length (m)': baseline_lengths,
            'Baseline Angle (rad)': baseline_angles,
            'SNR': snr_eht_array,
            'tele1' : tele1,
            'tele2' : tele2,
            'Time': times,
            'psi' : psi,
            'u_coords' : self.u_coords,
            'v_coords' : self.v_coords,
            'phase': phase,
            'phase_seconds': source_structure_delay,
            
        })

        # Provide the table to thermal_atm instance so its methods can use it
        self.thermal_atm.table = self.table


    def add_columns_random_thermal_atm_delay(self,number_of_scans=5, phase_error=0.55, freq=215):
        for i in range(number_of_scans):
            tau_str_tenna1 = 'tau1_scan_' + str(i+1)
            tau_str_tenna2 = 'tau2_scan_' + str(i+1)
            tau = self.thermal_atm.generate_rand_atm_delay(8, phase_error, freq)
            tenna1_tau = []
            tenna2_tau = []
            for ant1, ant2 in zip(self.table['tele1'], self.table['tele2']):
                tenna1_tau.append(tau[station_code_key[ant1]])
                tenna2_tau.append(tau[station_code_key[ant2]])

            self.table[tau_str_tenna1] = tenna1_tau
            self.table[tau_str_tenna2] = tenna2_tau

            self.table['delta_tau_scan_' + str(i+1)] = self.table[tau_str_tenna1] - self.table[tau_str_tenna2]
    
        self.thermal_atm.table = self.table


    def select_antennas(self, tenna_list):

        self.table = self.table[self.table['tele1'].isin(tenna_list) & self.table['tele2'].isin(tenna_list)]
        
        if self.table.empty:
            print("No data available for the selected antennas.")
    
    def remove_antennas(self, tenna_list):
        self.table = self.table[~(self.table['tele1'].isin(tenna_list) | self.table['tele2'].isin(tenna_list))]
        
        if self.table.empty:
            print("No data available after removing the selected antennas.")

    def grouped_table_time(self, group_by='Time'):
        table_grouped = self.table.groupby(group_by)

        self.table_grouped = table_grouped
        self.thermal_atm.table_grouped = table_grouped

        self.list_of_tables_perscan = []
        self.thermal_atm.list_of_tables_perscan = []

        for i in table_grouped:
            single_table = i[1]
            self.list_of_tables_perscan.append(single_table)
            
        self.thermal_atm.list_of_tables_perscan = self.list_of_tables_perscan
        
    def remove_scans_tenna_set(self, tenna_list):
        """
        Remove scans that do not contain the specified set of antennas.
        """
        scans_with_tenna_set = []

        for i, j in enumerate(self.list_of_tables_perscan):
            #ants_is_scan = set(self.list_of_tables_perscan[i]['tele1']).union(set(self.list_of_tables_perscan[i]['tele2']))
            set_tenna_current = set(np.unique(np.hstack((np.unique(self.list_of_tables_perscan[i]['tele1']), np.unique(self.list_of_tables_perscan[i]['tele2'])))))

            if set_tenna_current == set(tenna_list):
                scans_with_tenna_set.append(j)

        self.list_of_tables_perscan = scans_with_tenna_set
        self.thermal_atm.list_of_tables_perscan = self.list_of_tables_perscan
        self.table_grouped = pd.concat(scans_with_tenna_set).groupby('Time')
        self.thermal_atm.table_grouped = self.table_grouped


    # def add_random_antenna_based_delay_columns(self, tau_atm_1, tau_atm_2):
    #     """
    #     Add random antenna-based delay columns to the table.
    #     """

    #     self.thermal_atm.table = self.table