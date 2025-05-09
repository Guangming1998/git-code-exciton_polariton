import sys
import numpy as np
import pandas as pd
import os # try to use 'os' module to create new file directory for output files.
from Cavity_ssh import Trajectory_SSHmodel
from matplotlib import pyplot as plt
path = os.getcwd()
# print(path)

plotResult = False
printOutput = False
SanityCheck = True

if '--print' in sys.argv:
    printOutput = True
if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt
    #plt.style.use('classic')
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='Times New Roman', size='10')

'unit conversion'
ps_to_au = 4.13414e4 # ps to atomic time
wavenumber_to_au = 4.55634e-6 # energy from wavenumber to atomic unit
Adot_to_au = 1.88973 #angstrom to atomic length
wavenumber_to_amuAps = 2180.66
hbar_to_amuAps = 1.1577e4
amu_to_auMass = 1822.88 #amu is Dalton mass!
kBT_to_au = 3.16681e-6
au_to_eV = 27.2144

if 'param.in' in path:
    exec(open('param.in').read())
    # atomic_unit = True
    
    # if atomic_unit:
    #     staticCoup = 300 *wavenumber_to_au
    #     dynamicCoup = 995/Adot_to_au * wavenumber_to_au
    #     kBT = 104.3*wavenumber_to_au
    #     mass = 100
    #     Kconst = 14500/(ps_to_au**2)
    #     hbar = 1
    #     dt = 0.025e-3*ps_to_au
    # else: #"use amu*A^2*ps^-2 unit"
    #     staticCoup = 300 *wavenumber_to_amuAps
    #     dynamicCoup = 995 * wavenumber_to_amuAps
    #     kBT = 104.3*wavenumber_to_amuAps
    #     mass = 100
    #     Kconst = 14500
    #     hbar = 1*hbar_to_amuAps
    #     dt = 0.025e-3
else:
    Ehrenfest = False # define a bool variable for Ehrenfest force switch
    atomic_unit = True #define a bool variable for atomic unit switch
    Runge_Kutta = True # define a bool variable for switch from velocity verlet to Runge Kutta.
    # ps_to_au = 4.13414e4 # ps to atomic time
    # wavenumber_to_au = 4.55634e-6 # energy from wavenumber to atomic unit
    # Adot_to_au = 1.88973 #angstrom to atomic length
    # dt = 1#0.025e-3*ps_to_au
    Ntimes = 40000
    Nskip = 10
    
    # hbar_to_amuAps = 1.1577e4
    
    #conversion 
    Nmol = 64
    
    if atomic_unit:
        staticCoup = 300 *wavenumber_to_au
        site_energy = -1j * staticCoup/4
        # dynamicCoup = 995/Adot_to_au * wavenumber_to_au
        kBT = 300 * kBT_to_au
        mass = 250*amu_to_auMass
        Kconst = 14500*amu_to_auMass/(ps_to_au**2)
        hbar = 1
        dt = 0.025e-3*ps_to_au
        couplingstrength = 150 *wavenumber_to_au
        cavitydecayrate = 600*wavenumber_to_au
        cavityFrequency = -2*staticCoup - 1j*cavitydecayrate/2
    else: #"use amu*A^2*ps^-2 unit"
        staticCoup = 300 *wavenumber_to_amuAps
        dynamicCoup = 995 * wavenumber_to_amuAps
        kBT = 104.3*wavenumber_to_amuAps
        mass = 100
        Kconst = 14500
        hbar = 1*hbar_to_amuAps
        dt = 0.025e-3
    
    useDiagonalDisorder = False
    useCavityHamiltonian = True
    useDynamicalDisorder = True
    dynamicCoup = 995/Adot_to_au * wavenumber_to_au if useDynamicalDisorder else 0

model1 = Trajectory_SSHmodel(Nmol,hbar)
'check if param.in valids'
# print(model1.Nmol)
model1.initialGaussian(kBT,mass,Kconst)
model1.initialHamiltonian(staticCoup,dynamicCoup,site_energy)
# model1.initialCj_disorder_trajectory(kBT)
# Prob = model1.Prob

# model1.initialHamiltonianCavity(couplingstrength,cavityFrequency)
# model1.initialState(hbar,kBT,most_prob=False)
# model1.initialstate_equal()
# print(model1.Htot)
# eigen_energy = model1.eigenvalue()
if useDiagonalDisorder:
    model1.updateDiagonalStaticDisorder(1e-4*staticCoup)
    
if useCavityHamiltonian:
    model1.initialHamiltonianCavity(couplingstrength,cavityFrequency)

if useDiagonalDisorder or SanityCheck or useCavityHamiltonian:
    
    # fig, ax= plt.subplots()

    # Initialize C_dia
    C_dia = np.zeros((Nmol+1,Nmol),complex)
    for i in range(Nmol):   C_dia[i,i] = 1.0
    # Initialize storage for GF(k,t)
    ka_list = np.arange(0.0,1.02,0.02)*np.pi
    
    'Theta version:'
    # Exp1jnka = np.zeros((len(ka_list),Nmol),complex)
    # for ka in range(len(ka_list)):
    #     for n in range(Nmol): 
    #         Exp1jnka[ka,n] = np.exp(1j*ka_list[ka]*n)
            
    'GPT version:'
    n_values = np.arange(Nmol)
    Exp1jnka_new = np.exp(1j * np.outer(ka_list, n_values))
    '''
    np.outer(ka_list, n_values) computes the outer product between ka_list and n_values, 
    creating a 2D grid of ka * n values.
    np.exp(1j * ...) efficiently applies the complex exponential function to 
    the entire grid at once.
    '''
    
    # test_array = np.array([1,2,3])
    # print(np.outer(test_array,test_array))
    '''result: [[1,2,3]
                [2,4,6]
                [3,6,9]
    '''
    # print(Exp1jnka_new-Exp1jnka)

    'Theta version:'
    # times = []
    # Akat_list = []        
    # for ka in range(len(ka_list)):
    #     Akat_list.append([])
    
    'update version:'
    times = np.arange(Ntimes)*dt
    Akat_array = np.zeros((len(ka_list), Ntimes), dtype=complex)
    

    for it in range(Ntimes):

        if useDynamicalDisorder:
            model1.updateHmol()
            model1.updateHamiltonianCavity()
            model1.old_Aj(Ehrenfest=False)

        ### RK4 propagation 
        K1 = -1j*np.dot(model1.Htot,C_dia)
        K2 = -1j*np.dot(model1.Htot,C_dia+dt*K1/2)
        K3 = -1j*np.dot(model1.Htot,C_dia+dt*K2/2)
        K4 = -1j*np.dot(model1.Htot,C_dia+dt*K3)
        C_dia += (K1+2*K2+2*K3+K4)*dt/6


        Ak = np.dot(Exp1jnka_new,np.dot(C_dia[:-1,:],np.conj(Exp1jnka_new).T))

        'Theta version:'
        # times.append( it*dt )
        # for ka in range(len(ka_list)):
        #     Akat_list[ka].append(Ak[ka,ka])
        # Ak_list.append(np.exp(-1j*Wgrd*it*dt)*Ak[0,0])
        'update version:'
        Akat_array[:, it] = np.diag(Ak)
        if useDynamicalDisorder:
            model1.velocityVerlet(dt,Ehrenfest=False)
            

    
    # ax.plot(times,np.real(Ak_list),'-bo',label = 'cavity')
    'Theta Version:'
    freq = np.fft.fftshift(np.fft.fftfreq(len(times)))*2*np.pi /(dt)/ staticCoup 
    # Akw_list = []
    # for ka in range(len(ka_list)):
    #     Akw = np.fft.fftshift(np.fft.fft(Akat_list[ka]))
    #     Akw_list.append(Akw)
        # Akw_list.append(Akw[int(len(Akw)/2-10):int(len(Akw)/2+10)])
    'update version:'
    Akw_array = np.fft.fftshift(np.fft.fft(Akat_array, axis=1), axes=1)/Nmol
    
    mag = np.abs(Akw_array)/au_to_eV*1e-3
    
    # build a DataFrame with k’s as the index and ω’s as the columns
    df = pd.DataFrame(
    data= mag,
    index= ka_list/np.pi,
    columns= -freq
    )
    
    #name my index
    df.index.name = 'k/π'
    df.columns.name = 'ω/τ'
    
    # save the DataFrame to a CSV file
    df.to_csv(f'{path}/data_AKW_decay600/dynamic{int(dynamicCoup*Adot_to_au / wavenumber_to_au)}_g{int(couplingstrength/wavenumber_to_au)}'+'.csv')

# #Plotting
    'Theta version:'
    # x,y = np.meshgrid(ka_list,freq)
    # ax.pcolormesh(x,y,np.abs(Akw_list).T)
    # ax.set_ylim([-Vndd,Vndd])
    
    'update version:'
    fig, ax = plt.subplots(figsize=(8, 5))   # e.g. 8″ wide by 5″ tall
    c = ax.pcolormesh(ka_list/np.pi, -freq, np.abs(Akw_array).T/au_to_eV*1e-3,
                      vmin = 0.0, vmax = 0.12, cmap = 'hot')
    ax.set_ylim([-8, 8])
    

    ax.set_xlabel('ka/$\pi$')
    ax.set_ylabel('$\omega/\\tau$')
    colorbar = fig.colorbar(
    c, ax=ax,
    orientation='vertical',    # make it horizontal
    pad=0.2,                     # how much space to leave (fraction of axis)
    fraction=0.05,               # width/thickness of the bar
    #location='top'            # (matplotlib ≥3.4) explicitly place at top
    )
    colorbar.set_label("$|A(k, ω)|/meV^{-1}$")
    # plt.savefig(f'{path}/dynamic{int(dynamicCoup*Adot_to_au / wavenumber_to_au)}_g{int(couplingstrength/wavenumber_to_au)}'+'_hot.jpeg',dpi = 300)
    plt.savefig(f'{path}/dynamic{int(dynamicCoup*Adot_to_au / wavenumber_to_au)}_g{int(couplingstrength/wavenumber_to_au)}_decay{int(cavitydecayrate/ wavenumber_to_au)}'+'_hot.png',dpi = 300)
    plt.show()

exit()





