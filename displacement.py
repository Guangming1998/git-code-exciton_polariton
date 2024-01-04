import sys
import numpy as np
import pandas as pd
import os # try to use 'os' module to create new file directory for output files.
from trajectory import Trajectory_SSHmodel
path = sys.argv[-1].replace('displacement.py','')
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
wavenumber_to_amuAps = 2180.66
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
    Runge_Kutta = False # define a bool variable for switch from velocity verlet to Runge Kutta.
    # ps_to_au = 4.13414e4 # ps to atomic time
    # wavenumber_to_au = 4.55634e-6 # energy from wavenumber to atomic unit
    # Adot_to_au = 1.88973 #angstrom to atomic length
    # dt = 1#0.025e-3*ps_to_au
    Ntimes = 6000
    Nskip = 10
    
    # hbar_to_amuAps = 1.1577e4
    
    #conversion 
    Nmol = 300
    if atomic_unit:
        staticCoup = 300 *wavenumber_to_au
        dynamicCoup = 995/Adot_to_au * wavenumber_to_au
        kBT = 104.3*wavenumber_to_au
        mass = 100
        Kconst = 14500/(ps_to_au**2)
        hbar = 1
        dt = 0.025e-3*ps_to_au
    else: #"use amu*A^2*ps^-2 unit"
        staticCoup = 300 *wavenumber_to_amuAps
        dynamicCoup = 995 * wavenumber_to_amuAps
        kBT = 104.3*wavenumber_to_amuAps
        mass = 100
        Kconst = 14500
        hbar = 1*hbar_to_amuAps
        dt = 0.025e-3
    
    

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

    
# data = pd.read_csv(path+'XV_100_600.csv',index_col=False)
# Xj = pd.DataFrame(data,columns=['Xj']).to_numpy(dtype=float).flatten()
# Vj = pd.DataFrame(data,columns=['Vj']).to_numpy(dtype=float).flatten()
model1 = Trajectory_SSHmodel(Nmol,hbar)
'check if param.in valids'
# print(model1.Nmol)
model1.initialGaussian(kBT,mass,Kconst)
model1.initialHamiltonian(staticCoup,dynamicCoup)


model1.initialState(hbar,kBT,most_prob=False)
# model1.initialstate_equal()

model1.shift_Rj()
# plt.plot(model1.Cj)
# plt.show()

# model1.initialHamiltonian_Cavity_nonHermitian(Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0)

# if useStaticNeighborDisorder:
#     model1.updateNeighborStaticDisorder(DeltaNN)
# if useDynamicNeighborDisorder:
#     model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
# if useStaticDiagonalDisorder:
#     model1.updateDiagonalStaticDisorder(DeltaDD)
# if useDynamicDiagonalDisorder:
#     model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

# model1.initialCj_Bright()
# model1.initialCj_middle()
# model1.initialCj_middle()
# model1.initialCj_Gaussian(2.0)
# model1.initialCj_Cavity()
# model1.initialCj_Boltzman(hbar,kBT,most_prob=True)
# model1.initialCj_Polariton()

times_1 = [0]
Displacement_list_1 = [model1.getDisplacement()]
energy_elec,energy_vib =[model1.getEnergy()[0]],[model1.getEnergy()[1]]
CurrentCorrelation_list = [model1.getCurrentCorrelation()[1]]
Prob_list = [model1.Prob]
Cj_list =[np.sum(np.abs(model1.Cj.T)**2)]

'calculate for energy conservation in Ehrenfest'
# sum_KXjVj,sum_mVjAj = [],[]
sum_dHel_dt = []
sum_dHnuc_dt = []
sum_dH_dt = []

'check the position and velocity of first site'
# Xj_0,Vj_0 = [],[]

def dynamics():
    for it in range(Ntimes):
        # print(it)
        print('This is energy before update Hmol:',model1.getEnergy())
        model1.updateHmol()
        model1.old_Aj(Ehrenfest)
        if Runge_Kutta:
            model1.RK4(dt)
        
        else:
            #for i in range(10):
            old_energy = np.array(model1.getEnergy())
            print('This is energy:',model1.getEnergy())
            # print((model1.Xj[1]-model1.Xj[0])*model1.dynamicCoup)
            # print(((model1.Xj[1]-model1.Xj[0])*model1.dynamicCoup+model1.getEnergy()[0])/model1.getEnergy()[0])
            # model1.propagateCj(dt)
                
            
            model1.velocityVerlet(dt,Ehrenfest)
            model1.updateHmol()
            for i in range(10):
                model1.propagateCj(dt*0.1)
            # model1.Newton(dt)
            print('this is energy difference:',np.array(model1.getEnergy())-old_energy)
            
            
        # model1.propagateJ0Cj_RK4(dt)
        
        
        
        if it%Nskip ==0:
            times_1.append( it*dt )
            
            Displacement_list_1.append(model1.getDisplacement())
            CurrentCorrelation_list.append(model1.getCurrentCorrelation()[1])
            Cj_list.append(np.sum(np.abs(model1.Cj.T)**2))
            energy_elec.append(model1.getEnergy()[0])
            energy_vib.append(model1.getEnergy()[1])
            
            'energy conservation part'
            sum_dHel_dt.append(np.real(np.einsum('i,i',np.conjugate(model1.Cj),np.einsum('ij,j',model1.Hmol_dt,model1.Cj))))
            sum_dHnuc_dt.append (np.einsum('i,i',model1.Vj,model1.accelerate)*mass + np.einsum('i,i',model1.Xj,model1.Vj)*Kconst)
            sum_dH_dt.append(np.einsum('i,i',model1.Vj,model1.accelerate)*mass + np.einsum('i,i',model1.Xj,model1.Vj)*Kconst + 
                                 np.real(np.einsum('i,i',np.conjugate(model1.Cj),np.einsum('ij,j',model1.Hmol_dt,model1.Cj))))
            print(sum_dHel_dt[0]*dt)
            'position and velocity of first site'
            # Xj_0.append(model1.Xj[0]),Vj_0.append(model1.Vj[0])
            
            


dynamics()
# verlet()

if not plotResult:
    
    dici = {"Displacement":Displacement_list_1,'elec_energy':energy_elec,'vib_energy':energy_vib}
    data = pd.DataFrame(dici)
    #os.getcwd()
    dici_1 = {"CurrentCorrelation":CurrentCorrelation_list,'Cj':Cj_list}
    data_1 = pd.DataFrame(dici_1)
    
    dici_2 = {"Probability":Prob_list}
    data_2 = pd.DataFrame(dici_2)
    
    'store the energy conservation file'
    dici_3 = {"dHel_dt":sum_dHel_dt,"dHnuc_dt":sum_dHnuc_dt,"dH_dt":sum_dH_dt}
    data_3 = pd.DataFrame(dici_3)
    
    'position and velocity of first site'
    # dici_XV = {'Xj_0':Xj_0,'Vj_0':Vj_0}
    # data_XV = pd.DataFrame(dici_XV)
    
    'Here I create a new directory to store the output .csv files'
    data.to_csv('csv_output/'+'Displacement_250_600.csv')
    data_1.to_csv('csv_output/'+'CurrentCorrelation_250_600.csv')
    data_2.to_csv('csv_output/'+'Probability.csv')
    data_3.to_csv('csv_output/'+'Econservation.csv')
    # data_XV.to_csv('csv_output/'+'XV.csv')

    fdis = open('Displacement_250_600.dat'+sys.argv[-1], 'w')
    fdis_1 = open('CurrentCorrelation_250_600.dat'+sys.argv[-1], 'w')
    fdis_2 = open('Probability.dat'+sys.argv[-1], 'w')
    fdis_2.write("{Probability}\n".format(Probability=Prob_list[0]))
    
    for it in range(len(times_1)):
        fdis.write("{t}\t{Displacement}\n".format(t=times_1[it],Displacement=Displacement_list_1[it]))
        fdis_1.write("{t}\t{CurrentCorrelation}\n".format(t=times_1[it],CurrentCorrelation=CurrentCorrelation_list[it]))

# times = []
# Pmol1, Pmol2 = [], []
# IPR1, IPR2 = [], []

# Xj_list, Vj_list = [], []
# distr_list = []
# Displacement_list = []
# Correlation_list = []
# Current_list = []

# for it in range(Ntimes):
    

#     # if useDynamicNeighborDisorder:
#     #     model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
#     # if useDynamicDiagonalDisorder:
#     #     model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

#     Javg, CJJ = model1.getCurrentCorrelation()

#     model1.propagateCj_RK4(dt)
#     model1.propagateJ0Cj_RK4(dt)
    
#     if it%Nskip==0:
#         times.append( it*dt )
#         Pmol1.append( model1.getPopulation_system()  )
#         # Pmol2.append( model2.getPopulation_system() )

#         IPR1.append( model1.getIPR() )
#         # IPR2.append( model2.getIPR() )


#         # Xj_list.append(model1.Xj)
#         # Vj_list.append(model1.Vj)
#         distr = np.abs(model1.Cj[model1.Imol:model1.Imol+model1.Nmol])**2
#         distr_list.append(distr[:,0])
#         Displacement_list.append(model1.getDisplacement())
#         Correlation_list.append(CJJ)
#         Current_list.append(Javg)
#         if printOutput:
#             print("{t}\t{d}\t{dP}".format(t=it*dt,d=model1.getDisplacement(),dP=model1.getPopulation_system()))
#                                                 # dE=model1.getEnergy()-E0 ))

# if not plotResult:
#     # write to output 
#     fpop = open('Pmol.dat'+sys.argv[-1], 'w')
#     for it in range(len(times)):
#         fpop.write("{t}\t{Pmol}\n".format(t=times[it],Pmol=Pmol1[it]))

#     fdis = open('Displacement.dat'+sys.argv[-1], 'w')
#     for it in range(len(times)):
#         fdis.write("{t}\t{Displacement}\n".format(t=times[it],Displacement=Displacement_list[it]))

#     fcorr = open('Correlation.dat'+sys.argv[-1], 'w')
#     for it in range(len(times)):
        # fcorr.write("{t}\t{Corr_real}\t{Corr_imag}\n".format(t=times[it],Corr_real=np.real(Correlation_list[it]),Corr_imag=np.imag(Correlation_list[it])))
