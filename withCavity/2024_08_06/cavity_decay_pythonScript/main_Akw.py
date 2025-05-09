import sys
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from models import SingleExcitationWithCollectiveCoupling


plotResult = False
SanityCheck = False
printOutput = False
if '--print' in sys.argv:
    printOutput = True
if '--test' in sys.argv:
    SanityCheck = True
    from matplotlib import pyplot as plt
if '--plot' in sys.argv: 
    plotResult=True
    from matplotlib import pyplot as plt
    #plt.style.use('classic')
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='Times New Roman', size='10')


if 'param.in' in sys.argv:
    exec(open('param.in').read())
else:
    dt = 0.01
    Ntimes = 10000
    Nskip = 10

    Nmol = 100
    Wmol =  0.0 - 1j*0.1
    Wgrd = -1.0
    Vndd = 0.5

    Wcav = 0.0 + 2.0*Vndd
    Vcav = 0.1
    Kcav = 0  #Kcav*pi/Nmol

    useStaticNeighborDisorder = False
    useDynamicNeighborDisorder = False
    DeltaNN = 0.0
    TauNN = 0.0

    useStaticDiagonalDisorder = False
    useDynamicDiagonalDisorder = False
    DeltaDD = 0.0
    TauDD = 0.0

    kBT = 20.0
    useThermalStaticNeighborDisorder = False
    useThermalDyanmicNeighborDisorder = False
    Kconst = 100.0
    DeltaNN = kBT/Kconst
    TauNN = 100.0
    # hbar = 63.508
    # mass = 10000.0  # Joul/mol(ps/A)^2
    # Kconst = 145000.0  # Joul/mol/A^2
    # staticCoup = 0.0 # 1/ps
    # dynamicCoup = 0.0 # 1/ps/A
    # kBT = 1245.0 #Joul/mol
    
model1 = SingleExcitationWithCollectiveCoupling(Nmol,0)
model1.initialHamiltonian_nonHermitian(Wgrd,Wcav,Wmol,Vndd,Vcav,Kcav,Gamma=0.0)

useNodisorder=False
if useStaticNeighborDisorder:
    model1.updateNeighborStaticDisorder(DeltaNN)
elif useDynamicNeighborDisorder:
    model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
elif useStaticDiagonalDisorder:
    model1.updateDiagonalStaticDisorder(DeltaDD)
elif useDynamicDiagonalDisorder:
    model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)
elif useThermalStaticNeighborDisorder:
    model1.updateNeighborStaticDisorder(DeltaNN)
elif useThermalDyanmicNeighborDisorder:
    model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
else:
    useNodisorder=True

# if useDynamicNeighborDisorder or useDynamicDiagonalDisorder or useThermalDyanmicNeighborDisorder or SanityCheck:
if True:
    
    fig, ax= plt.subplots()

    # Initialize C_dia
    C_dia = np.zeros((Nmol+2,Nmol),complex)
    for i in range(Nmol):   C_dia[i+2,i] = 1.0

    # Generate Exp1jnka (left)
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

        if useDynamicNeighborDisorder:
            model1.updateNeighborDynamicDisorder(DeltaNN,TauNN,dt)
        if useDynamicDiagonalDisorder:
            model1.updateDiagonalDynamicDisorder(DeltaDD,TauDD,dt)

        ### RK4 propagation 
        K1 = -1j*np.dot(model1.Ht,C_dia)
        K2 = -1j*np.dot(model1.Ht,C_dia+dt*K1/2)
        K3 = -1j*np.dot(model1.Ht,C_dia+dt*K2/2)
        K4 = -1j*np.dot(model1.Ht,C_dia+dt*K3)
        C_dia += (K1+2*K2+2*K3+K4)*dt/6


        Ak = np.dot(Exp1jnka_new,np.dot(C_dia[2:,:],np.conj(Exp1jnka_new).T))

        'Theta version:'
        # times.append( it*dt )
        # for ka in range(len(ka_list)):
        #     Akat_list[ka].append(Ak[ka,ka])
        # Ak_list.append(np.exp(-1j*Wgrd*it*dt)*Ak[0,0])
        'update version:'
        Akat_array[:, it] = np.diag(Ak)

    
    # ax.plot(times,np.real(Ak_list),'-bo',label = 'cavity')
    'Theta Version:'
    freq = np.fft.fftshift(np.fft.fftfreq(len(times))) /dt/np.pi/ Vndd
    # Akw_list = []
    # for ka in range(len(ka_list)):
    #     Akw = np.fft.fftshift(np.fft.fft(Akat_list[ka]))
    #     Akw_list.append(Akw)
        # Akw_list.append(Akw[int(len(Akw)/2-10):int(len(Akw)/2+10)])
    'update version:'
    Akw_array = np.fft.fftshift(np.fft.fft(Akat_array, axis=1), axes=1)
    

#Plotting
    'Theta version:'
    # x,y = np.meshgrid(ka_list,freq)
    # ax.pcolormesh(x,y,np.abs(Akw_list).T)
    # ax.set_ylim([-Vndd,Vndd])
    
    'update version:'
    c = ax.pcolormesh(ka_list/np.pi, freq, np.abs(Akw_array).T)
    ax.set_ylim([-Vndd, Vndd])
    

    ax.set_xlabel('ka/$\pi$')
    ax.set_ylabel('$\omega\ \\tau$')
    colorbar = fig.colorbar(c, ax=ax)
    colorbar.set_label("|A(k, ω)|")
    plt.show()

exit()
