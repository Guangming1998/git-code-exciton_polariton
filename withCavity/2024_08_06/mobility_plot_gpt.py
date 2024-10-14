import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import os

# Define constants
AU_TO_S = 2.41888e-17
TIME_DIFF = 0.025e-14
KBT_IN_EV = 0.02585
LATTICE_CONSTANT = 4e-8
PS_TO_AU = 4.13414e4
AU_TO_WAVENUMBER = 2.19475e5

# Function to load data from files
def load_data(f, alpha, more=False):
    folder = f"dynamic{alpha}"
    more_str = "/more" if more else ""
    # Fixed: filenames in /more/ subdirectory start from 1
    filename = f'CJJmol_eph_{f*3 if more else f}.csv'
    filepath = f'{os.getcwd()}/{folder}{more_str}/{filename}'
    return pd.read_csv(filepath, index_col=False, header=None)

#Function to load data from files CJJcav_*.csv
def load_data_cavity(f, alpha, more=False):
    folder = f"dynamic{alpha}"
    more_str = "/more" if more else ""
    # Fixed: filenames in /more/ subdirectory start from 1
    filename = f'CJJcav_eph_{f*3 if more else f}.csv'
    filepath = f'{os.getcwd()}/{folder}{more_str}/{filename}'
    return pd.read_csv(filepath, index_col=False, header=None)

# Function to compute mobility for a given dataset
def compute_mobility(Cjjmol_array_combined, lattice_constant, time_diff, kBT_in_eV,more=False):
    time = np.arange(np.size(Cjjmol_array_combined[:, 0])) * 0.25e-3
    time_in_au = time * PS_TO_AU
    freq_wavenumber = np.fft.fftfreq(time_in_au.shape[-1], d=time_in_au[1] - time_in_au[0]) * AU_TO_WAVENUMBER * 2 * np.pi**2
    
    Cjjmol_freq = np.fft.fft(Cjjmol_array_combined, axis=0)
    if more:
        wavenumber_axis = 30 * (np.arange(Cjjmol_array_combined.shape[1])+1)
    else:
        wavenumber_axis = 100 * np.arange(Cjjmol_array_combined.shape[1])
    
    mobility = np.real(Cjjmol_freq[0, :]) * lattice_constant**2 * AU_TO_S**-2 * time_diff / kBT_in_eV
    return wavenumber_axis, mobility

# Function to concatenate data and compute mobility for a given alpha
def process_alpha(alpha):
    Cjjmol_list = [load_data(f, alpha) for f in range(7)]
    Cjjmol_more_list = [load_data(f, alpha, more=True) for f in range(1, 7)]  # Fixed: f starts from 1 in /more/

    Cjjmol_combined = pd.concat(Cjjmol_list, axis=1).to_numpy(complex)
    Cjjmol_combined_more = pd.concat(Cjjmol_more_list, axis=1).to_numpy(complex)
    
    wavenumber_axis, mobility = compute_mobility(Cjjmol_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
    wavenumber_axis_more, mobility_more = compute_mobility(Cjjmol_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV,more = True)
    
    return np.append(wavenumber_axis, wavenumber_axis_more), np.append(mobility, mobility_more)

# Main function to plot mobility
def plot_mobility_CJJ(alphas):
    plt.figure()
    
    for alpha in alphas:
        wavenumber_axis, mobility = process_alpha(alpha)
        sorted_indices = np.argsort(wavenumber_axis)  # Ensure argsort applies to both

        # Apply sorted indices to both arrays
        wavenumber_axis_sorted = wavenumber_axis[sorted_indices]
        mobility_sorted = mobility[sorted_indices]
        
        plt.plot(wavenumber_axis_sorted, mobility_sorted, label=f'$\\alpha = {alpha}$')
    
    plt.xlabel('$g/cm^{-1}$', fontsize=12, fontweight='bold')
    plt.ylabel('$\\mu_{mol} cm^2*V/s$', fontsize=12, fontweight='bold')
    plt.xlim(0, None)
    plt.yscale('log')
    plt.legend()
    plt.show()

# Example usage: Plot for 500, 700, 995, 1400, and new 1800
plot_mobility_CJJ([500, 700, 995, 1400])
