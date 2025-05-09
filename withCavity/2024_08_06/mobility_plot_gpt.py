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
au_to_ev = 27.2114

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

def compute_mobility_imag(Cjjmol_array_combined, lattice_constant, time_diff, kBT_in_eV,more=False):
    time = np.arange(np.size(Cjjmol_array_combined[:, 0])) * 0.25e-3
    time_in_au = time * PS_TO_AU
    freq_wavenumber = np.fft.fftfreq(time_in_au.shape[-1], d=time_in_au[1] - time_in_au[0]) * AU_TO_WAVENUMBER * 2 * np.pi**2
    
    Cjjmol_freq = np.fft.fft(Cjjmol_array_combined, axis=0)
    if more:
        wavenumber_axis = 30 * (np.arange(Cjjmol_array_combined.shape[1])+1)
    else:
        wavenumber_axis = 100 * np.arange(Cjjmol_array_combined.shape[1])
    
    mobility = 2*np.sum(np.imag(time_in_au[:,np.newaxis]*Cjjmol_array_combined),axis = 1) * lattice_constant**2 * AU_TO_S**-2 * time_diff / au_to_ev
    return wavenumber_axis, mobility

def compute_fft(data_array):
    # Perform Fourier Transform on time-domain data
    
    fft_result = np.fft.fft(data_array)
    return fft_result

def mobility_analytical(wavenumber_array,cavity=False):
    ps_to_au = 4.13414e4 # ps to atomic time
    wavenumber_to_au = 4.55634e-6 # energy from wavenumber to atomic unit
    kBT_to_au = 3.16681e-6
    au_to_eV = 27.2114
    total_timestep = 4e4
    number_of_lattice = 100
    decay_rate = 100*wavenumber_to_au
    
    staticCoup = 300 *wavenumber_to_au
    # couplingstrength = wavenumber *wavenumber_to_au
    # time_array = np.arange(total_timestep*10)* 0.25e-4 * ps_to_au
    time_array = np.arange(total_timestep*1e3)* 0.25e-5 * ps_to_au
    number_array = np.arange(1,100,dtype = float)
    angle_n = 2*np.pi*number_array/100
    mobility_constant_part = LATTICE_CONSTANT**2*AU_TO_S**-2*TIME_DIFF/(300*kBT_to_au*au_to_eV)
    
    mobility_analytical_mol = []
    mobility_analytical_cav = []
    
    for couplingstrength in wavenumber_array *wavenumber_to_au:
        #Calculate partition function
        partition_function = np.sum(np.exp(2*staticCoup*np.cos(angle_n)/(300*kBT_to_au)))+ \
            np.exp(( 2*staticCoup-couplingstrength*np.sqrt(number_of_lattice))/(300*kBT_to_au)) + \
            np.exp(( 2*staticCoup+couplingstrength*np.sqrt(number_of_lattice))/(300*kBT_to_au))
        
        #Separate contribution of molecular chain and polariton
        molecular_partition = np.exp(2*staticCoup*np.cos(angle_n)/(300*kBT_to_au))
        upper_polariton_partitioin = np.exp(( 2*staticCoup-couplingstrength*np.sqrt(number_of_lattice))/(300*kBT_to_au))
        lower_polariton_partition = np.exp(( 2*staticCoup+couplingstrength*np.sqrt(number_of_lattice))/(300*kBT_to_au))
        if cavity:
            CJJ_cav = number_of_lattice*couplingstrength**2 *(np.exp((-1j*2*np.sqrt(number_of_lattice)*couplingstrength-decay_rate/2)*time_array)*\
                upper_polariton_partitioin+np.exp((1j*2*np.sqrt(number_of_lattice)*couplingstrength-decay_rate/2)*time_array)*\
                lower_polariton_partition) / partition_function
            # CJJ_cav = number_of_lattice*couplingstrength**2 *(np.exp((-1j*2*np.sqrt(number_of_lattice)*couplingstrength-decay_rate/2)*time_array)*\
            #     1+np.exp((1j*2*np.sqrt(number_of_lattice)*couplingstrength-decay_rate/2)*time_array)*\
            #     1)
            # CJJ_cav = number_of_lattice*couplingstrength**2 *np.exp((-1j*2*np.sqrt(number_of_lattice)*couplingstrength-decay_rate)*time_array)
            CJJ_cav_freq = compute_fft(CJJ_cav)
            mobility_cav = np.sum(CJJ_cav.real)*mobility_constant_part
            mobility_analytical_cav.append(mobility_cav)
        else:
            CJJ_mol = np.ones(int(total_timestep),float)*4*staticCoup**2*np.sum(np.dot(np.sin(angle_n)**2, molecular_partition))/partition_function
            CJJ_mol_freq = compute_fft(CJJ_mol)
            mobility_mol = np.real(CJJ_mol_freq[0]) * mobility_constant_part
            mobility_analytical_mol.append(mobility_mol)
    
    return mobility_analytical_cav if cavity else mobility_analytical_mol

# Function to concatenate data and compute mobility for a given alpha
# def process_alpha(alpha):
#     Cjjmol_list = [load_data(f, alpha) for f in range(7)]
#     Cjjmol_more_list = [load_data(f, alpha, more=True) for f in range(1, 7)]  # Fixed: f starts from 1 in /more/

#     Cjjmol_combined = pd.concat(Cjjmol_list, axis=1).to_numpy(complex)
#     Cjjmol_combined_more = pd.concat(Cjjmol_more_list, axis=1).to_numpy(complex)
    
#     wavenumber_axis, mobility = compute_mobility(Cjjmol_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
#     wavenumber_axis_more, mobility_more = compute_mobility(Cjjmol_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV,more = True)
    
#     return np.append(wavenumber_axis, wavenumber_axis_more), np.append(mobility, mobility_more)

def process_alpha(alpha,imag = False):
    if alpha == 100:
        Cjjmol_list = [load_data(f, alpha) for f in range(4)]  # Only load _0 and _1 for alpha = 100
        Cjjmol_more_list = []
    elif alpha ==0:
        Cjjmol_list = [load_data(f, alpha) for f in range(3)]
        Cjjmol_more_list = [load_data(f, alpha, more=True) for f in range(1, 7)]
    else:
        Cjjmol_list = [load_data(f, alpha) for f in range(7)]
        Cjjmol_more_list = [load_data(f, alpha, more=True) for f in range(1, 7)]
    
    Cjjmol_combined = pd.concat(Cjjmol_list, axis=1).to_numpy(complex)
    
    if imag:
        if Cjjmol_more_list:
            Cjjmol_combined_more = pd.concat(Cjjmol_more_list, axis=1).to_numpy(complex)
            wavenumber_axis, mobility = compute_mobility_imag(Cjjmol_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
            wavenumber_axis_more, mobility_more = compute_mobility_imag(Cjjmol_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV, more=True)
            return np.append(wavenumber_axis, wavenumber_axis_more), np.append(mobility, mobility_more)
        else:
            wavenumber_axis, mobility = compute_mobility_imag(Cjjmol_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
            return wavenumber_axis, mobility
    else:
        if Cjjmol_more_list:
            Cjjmol_combined_more = pd.concat(Cjjmol_more_list, axis=1).to_numpy(complex)
            wavenumber_axis, mobility = compute_mobility(Cjjmol_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
            wavenumber_axis_more, mobility_more = compute_mobility(Cjjmol_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV, more=True)
            return np.append(wavenumber_axis, wavenumber_axis_more), np.append(mobility, mobility_more)
        else:
            wavenumber_axis, mobility = compute_mobility(Cjjmol_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
            return wavenumber_axis, mobility

# def process_alpha_cavity(alpha):
#     Cjj_list = [load_data_cavity(f, alpha) for f in range(7)]
#     Cjj_more_list = [load_data_cavity(f, alpha, more=True) for f in range(1, 7)]  # Fixed: f starts from 1 in /more/

#     Cjj_combined = pd.concat(Cjj_list, axis=1).to_numpy(complex)
#     Cjj_combined_more = pd.concat(Cjj_more_list, axis=1).to_numpy(complex)
    
#     wavenumber_axis, mobility = compute_mobility(Cjj_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
#     wavenumber_axis_more, mobility_more = compute_mobility(Cjj_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV,more = True)
    
#     return np.append(wavenumber_axis, wavenumber_axis_more), np.append(mobility, mobility_more)
def process_alpha_cavity(alpha,imag=False):
    # Set the number of expected files based on alpha value
    if alpha ==100: # Use only 2 files for alpha=100
        num_files = 4
    else:
        num_files = 7
    
    # Load the main set of files
    Cjj_list = []
    for f in range(num_files):
        try:
            Cjj_list.append(load_data_cavity(f, alpha))
        except FileNotFoundError:
            print(f"File not found: CJJcav_eph_{f}.csv for alpha={alpha}")
            continue
    
    # Load the files in the /more subdirectory, if they exist
    Cjj_more_list = []
    for f in range(1, num_files):  # Start from 1 for the /more files
        try:
            Cjj_more_list.append(load_data_cavity(f, alpha, more=True))
        except FileNotFoundError:
            print(f"File not found in /more: CJJcav_eph_{f}.csv for alpha={alpha}")
            continue
    
    # Combine loaded data into arrays, if any files were loaded
    Cjj_combined = pd.concat(Cjj_list, axis=1).to_numpy(complex) if Cjj_list else np.array([])
    Cjj_combined_more = pd.concat(Cjj_more_list, axis=1).to_numpy(complex) if Cjj_more_list else np.array([])
    
    # Calculate mobility based on the data loaded
    if imag:
        if Cjj_combined.size > 0:
            wavenumber_axis, mobility = compute_mobility_imag(Cjj_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
        else:
            wavenumber_axis, mobility = np.array([]), np.array([])

        if Cjj_combined_more.size > 0:
            wavenumber_axis_more, mobility_more = compute_mobility_imag(Cjj_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV, more=True)
        else:
            wavenumber_axis_more, mobility_more = np.array([]), np.array([])
    else:
        if Cjj_combined.size > 0:
            wavenumber_axis, mobility = compute_mobility(Cjj_combined, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV)
        else:
            wavenumber_axis, mobility = np.array([]), np.array([])

        if Cjj_combined_more.size > 0:
            wavenumber_axis_more, mobility_more = compute_mobility(Cjj_combined_more, LATTICE_CONSTANT, TIME_DIFF, KBT_IN_EV, more=True)
        else:
            wavenumber_axis_more, mobility_more = np.array([]), np.array([])

    # Return concatenated results if both exist; otherwise return the one with data
    return (np.append(wavenumber_axis, wavenumber_axis_more) if wavenumber_axis.size > 0 and wavenumber_axis_more.size > 0
            else wavenumber_axis if wavenumber_axis.size > 0
            else wavenumber_axis_more), \
           (np.append(mobility, mobility_more) if mobility.size > 0 and mobility_more.size > 0
            else mobility if mobility.size > 0
            else mobility_more)



# Main function to plot mobility
def plot_mobility_CJJ(alphas):
    # plt.figure()
    # fig,ax = plt.subplots(1,2,figsize = (10,4), sharey='row')
    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(1, 2, wspace=0)
    ax = gs.subplots(sharey='row')
    
    # new_wavenumber_array = np.arange(10,600,10)
    # mobility_cav_new = mobility_analytical(new_wavenumber_array,cavity=True)
    # mobility_mol_new = mobility_analytical(new_wavenumber_array,cavity=False)
    # ax[0].plot(new_wavenumber_array, mobility_mol_new, linestyle='dashed',label=f'$\\alpha = 0$')
    # ax[1].plot(new_wavenumber_array, mobility_cav_new, linestyle='dashed')
    
    for alpha in alphas:
        if alpha == 100:
            wavenumber_axis, mobility = process_alpha(alpha)
            wavenumver_axis_cavity, mobility_cavity = process_alpha_cavity(alpha)
            ax[0].plot(wavenumber_axis, mobility, label=f'$\\alpha = {alpha}$')
            ax[1].plot(wavenumber_axis,mobility_cavity, label = f'$\\alpha = {alpha}$')
        else:
            # wavenumber_axis, mobility = process_alpha(alpha)
            # wavenumver_axis_cavity, mobility_cavity = process_alpha_cavity(alpha)
            wavenumber_axis, mobility = process_alpha(alpha,imag=True)
            wavenumver_axis_cavity, mobility_cavity = process_alpha_cavity(alpha,imag = True)
            sorted_indices = np.argsort(wavenumber_axis)  # Ensure argsort applies to both

            # Apply sorted indices to both arrays
            wavenumber_axis_sorted = wavenumber_axis[sorted_indices]
            mobility_sorted = mobility[sorted_indices]
            mobility_cavity_sorted = mobility_cavity[sorted_indices]
            
            # plt.plot(wavenumber_axis_sorted, mobility_sorted, label=f'$\\alpha = {alpha}$')
            if alpha !=0 :
                ax[0].plot(wavenumber_axis_sorted, mobility_sorted, label=f'$\\alpha = {alpha}$')
            else:
                ax[0].plot(wavenumber_axis_sorted, mobility_sorted, linestyle='dashed',label=f'$\\alpha = {alpha}$')
            if alpha != 0:
                ax[1].plot(wavenumber_axis_sorted,mobility_cavity_sorted, label = f'$\\alpha = {alpha}$') 
        
    
    ax[0].set_xlabel('$g/cm^{-1}$', fontsize=8, fontweight='bold')
    ax[1].set_xlabel('$g/cm^{-1}$', fontsize = 8, fontweight = 'bold')
    ax[0].set_ylabel('$ cm^2*V/s$', fontsize=8, fontweight='bold')
    # ax[1].set_ylabel('$\\mu_{cav} cm^2*V/s$', fontsize=8, fontweight='bold')
    ax[0].set_title('$\\mu_{mol}$', fontsize=12, fontweight='bold')
    ax[1].set_title('$\\mu_{cav}$', fontsize=12, fontweight='bold')
    ax[0].set_xlim(10,1000)
    ax[1].set_xlim(10,1000)
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].set_yscale('log') 
    ax[1].set_yscale('log')
    # ax[0].set_ylim(1e-3,None)
    # ax[1].set_ylim(1e-3,None)
    ax[0].legend()
    # ax[1].legend()
    # plt.savefig('mobility_log_log_final.pdf')
    plt.show()
    
    for alpha in alphas:
        wavenumber_axis, mobility = process_alpha(alpha)
        wavenumver_axis_cavity, mobility_cavity = process_alpha_cavity(alpha)
        sorted_indices = np.argsort(wavenumber_axis)  # Ensure argsort applies to both

        # Apply sorted indices to both arrays
        wavenumber_axis_sorted = wavenumber_axis[sorted_indices]
        mobility_sorted = mobility[sorted_indices]
        mobility_cavity_sorted = mobility_cavity[sorted_indices]
        
        # plt.plot(wavenumber_axis_sorted, mobility_sorted, label=f'$\\alpha = {alpha}$')
        plt.plot(wavenumber_axis_sorted, mobility_sorted+mobility_cavity_sorted, label=f'$\\alpha = {alpha}$')
        # ax[1].plot(wavenumber_axis_sorted,mobility_cavity_sorted, label = f'$\\alpha = {alpha}$')
    plt.xlabel('$g/cm^{-1}$', fontsize = 12, fontweight = 'bold')
    plt.ylabel('$\mu\ cm^2*V/s$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,1000)
    plt.legend()
    # plt.savefig('mobility_log_log_4.pdf')
    plt.show()
    

# Example usage: Plot for 500, 700, 995, 1400, and new 1800
# plot_mobility_CJJ([500, 700, 995, 1400])

# Example usage: Plot for 100,500, 700, 995, 1400, and new 1800
plot_mobility_CJJ([500, 700, 995, 1400])