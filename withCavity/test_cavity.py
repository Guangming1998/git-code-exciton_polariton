import sys
import numpy as np
import pandas as pd
import os # try to use 'os' module to create new file directory for output files.
from Cavity_ssh import Trajectory_SSHmodel
path = sys.argv[-1].replace('test_cavity.py','')
'check my path'
# print(path)

directory_names = ['csv_output','dat_output']

try:
    for directory_name in directory_names:
        try:
            os.makedirs(directory_name)
        except FileExistsError:
            print(f"Error: '{directory_name}' already exists")

except :
    print("Error occure while creating the directories.")
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
    Ehrenfest = True # define a bool variable for Ehrenfest force switch
    atomic_unit = True #define a bool variable for atomic unit switch
    Runge_Kutta = True # define a bool variable for switch from velocity verlet to Runge Kutta.
    # ps_to_au = 4.13414e4 # ps to atomic time
    # wavenumber_to_au = 4.55634e-6 # energy from wavenumber to atomic unit
    # Adot_to_au = 1.88973 #angstrom to atomic length
    # dt = 1#0.025e-3*ps_to_au
    Ntimes = 2000
    Nskip = 1
    
    # hbar_to_amuAps = 1.1577e4
    
    #conversion 
    Nmol = 20
    
    if atomic_unit:
        staticCoup = 0.5#300 *wavenumber_to_au
        dynamicCoup = 0*995/Adot_to_au * wavenumber_to_au
        kBT = 0.1#104.3*wavenumber_to_au
        mass = 250*amu_to_auMass
        Kconst = 14500*amu_to_auMass/(ps_to_au**2)
        hbar = 1
        dt = 0.01#0.025e-3*ps_to_au
        couplingstrength = 2#50 *wavenumber_to_au
        cavityFrequency = 1#1500*wavenumber_to_au
    else: #"use amu*A^2*ps^-2 unit"
        staticCoup = 300 *wavenumber_to_amuAps
        dynamicCoup = 995 * wavenumber_to_amuAps
        kBT = 104.3*wavenumber_to_amuAps
        mass = 100
        Kconst = 14500
        hbar = 1*hbar_to_amuAps
        dt = 0.025e-3
    
    useDiagonalDisorder=True

    # Nmol = 101
    # Wmol =  0.0
    # Wgrd = -1.0
    # Vndd = -0.3

    # Wcav = 0.0 + 2.0*Vndd
    # Vcav = 0.0
    # Kcav = 0  #Kcav*pi/Nmol

    # useStaticNeighborDisorder = False
    # useDynamicNeighborDisorder = False
    # DeltaNN = 0.0
    # TauNN = 0.0

    # useStaticDiagonalDisorder = False
    # useDynamicDiagonalDisorder = False
    # DeltaDD = 0.0
    # TauDD = 0.0

    
# 'check temperature'
# print(kBT)
# data = pd.read_csv(path+'XV_100_600.csv',index_col=False)
# Xj = pd.DataFrame(data,columns=['Xj']).to_numpy(dtype=float).flatten()
# Vj = pd.DataFrame(data,columns=['Vj']).to_numpy(dtype=float).flatten()
model1 = Trajectory_SSHmodel(Nmol,hbar)
'check if param.in valids'
# print(model1.Nmol)
model1.initialGaussian(kBT,mass,Kconst)
model1.initialHamiltonian(staticCoup,dynamicCoup)
# model1.initialCj_disorder_trajectory(kBT)
# Prob = model1.Prob

# model1.initialHamiltonianCavity(couplingstrength,cavityFrequency)
# model1.initialState(hbar,kBT,most_prob=False)
# model1.initialstate_equal()
# print(model1.Htot)
# eigen_energy = model1.eigenvalue()
if useDiagonalDisorder:
    model1.updateDiagonalStaticDisorder(1e-9*staticCoup)
    
if useDiagonalDisorder or SanityCheck:
    model1.initialCj_disorder()
    model1.initialCj_disorder_trajectory(kBT)

    # model1.initial_densityMatrix(kBT)
    
    CJJavg1 = model1.getCurrentCorrelation_static(dt,Ntimes,kBT,0.0)
    times = []
    CJJavg2_list = np.zeros((Ntimes,Nmol),complex)
    CJJavg1_list = []
    Prob = model1.Prob
    J0Cj = np.zeros((Nmol,Nmol),complex)
    for i in range(Ntimes):
        times.append(dt*i)
        CJJavg1_list.append(CJJavg1[i])
    
    for j in range(Nmol):
        J0Cj[:,j] = np.dot(model1.Jt0,model1.Cj[:,j])
    CJJavg1_list = np.array(CJJavg1_list,complex)
    
    # CJJavg1_list = np.array 
    for i in range(Nmol):
        for it in range(Ntimes):
            # J0Cj = np.dot(model1.Jt0,model1.Cj[i])
            if it ==0:
                CJJavg2_list[it,i] = model1.getCurrentCorrelation_disorder_trajectory(model1.Cj[:,i],J0Cj[:,i])
            model1.Cj[:,i] = model1.RK4_Cj_trajectory(dt,model1.Cj[:,i])
            J0Cj[:,i] = model1.propagateJ0Cj_RK4_trajectory(dt,J0Cj[:,i])
            if it%Nskip==0 and it!=0 :
                CJJavg2_list[it,i] = model1.getCurrentCorrelation_disorder_trajectory(model1.Cj[:,i],J0Cj[:,i])
                
            
    CJJavg2_list = np.dot(CJJavg2_list,Prob)
    if SanityCheck:
        for it in range(len(times)):
            print(it,CJJavg1_list[it], CJJavg2_list[it], np.abs(CJJavg1_list[it]-CJJavg2_list[it]))
            
    
        fig,ax = plt.subplots()
        ax.plot(times,np.real(CJJavg1_list),'-r+')
        ax.plot(times,np.real(CJJavg2_list),'-bx')
        ax.set_xlabel('time')
        ax.set_ylabel('CJJ')
        plt.show()


'check if the derivation of current operator is correct or not. Result shows that derivation is correct'
# model1.polarization_operator()
# J = model1.Current_operator()
# model1.shift_Rj()

# print(model1.Jt0)
# print(J)
# print((np.dot(model1.Hmol,model1.Jt0)-np.dot(model1.Jt0,model1.Hmol)))
# print((np.dot(model1.Hmol,J)-np.dot(J,model1.Hmol)))

# plt.hist(eigen_energy*27.2)
# plt.show()
'check current operator Jmol_0 and Jcav_0. result is correct'
# print(model1.Jmol_0)
# print(model1.Jcav_0)

'check commutator'
# print(model1.Hint)
# print(np.dot(model1.Jmol_0,model1.Hint)-np.dot(model1.Hint,model1.Jmol_0))

# print(np.dot(model1.Jtot_0,model1.Htot)-np.dot(model1.Htot,model1.Jtot_0))
# print(model1.Htot)
# print(np.dot(model1.Jcav_0,model1.Hint)-np.dot(model1.Hint,model1.Jcav_0))
# print(np.dot(model1.Jcav_0,model1.Hint)-np.dot(model1.Hint,model1.Jcav_0))
# print(np.dot(model1.Jt0,model1.H_disorder)-np.dot(model1.H_disorder,model1.Jt0))


