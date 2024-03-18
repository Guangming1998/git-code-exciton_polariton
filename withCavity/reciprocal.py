from math import gamma
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

class Multimode():
    
    def __init__(self,Nmol,Nrad,seed=None):
        self.Nmol = Nmol
        self.Nrad = Nrad
        np.random.seed(seed)
        
    
    def initialHamiltonian(self,g_c):
        '''constructing matrix of Hamiltonian
        for single layer material in cavity
        h_(N+1) = [\epislon_x(k_x), \Omega_(\pi/L_y), ...
                      \Omega(\pi/_Ly), \omega_c(k_x,\pi/L_y),...
                      ...,               ...,    \omega_c(k_x,2*\pi/L_y),... ]
        1 mode from molecule, N cavity modes.
        '''
        Nm = 100
        Ly =20000
        Lx = 50*Ly
        d_lx = Lx/Nm
        # Kx = 2*np.pi*Nx/Lx
        self.Hmol = np.eye(self.Nmol) * 1.0 #exciton mode
        self.Wmol = np.zeros(self.Nmol)
        for i in range(self.Nmol):
            self.Wmol[i] = 0.62-2*0.0186*np.cos(2*np.pi*i/Nm) #Nmol sites on band 
            self.Hmol[i][i] = self.Wmol[i]
        # self.Wrad = Wrad
        # e_kx= w0 - 2*tao_x*np.sin(2*np.pi*Nx/self.Nmol)
        

        self.Hrad = np.zeros((self.Nrad,self.Nrad),complex) #cavity mode
        
        
        Vradmol = np.zeros((self.Nrad,self.Nmol),complex)

        for Ny in range(0,self.Nrad,1):
            g_k = g_c#np.sqrt(np.sqrt((2*Nx/50)**2 + (Ny+1)**2))  #coupling between cavity modes and exciton
            w_c = 0.62#np.sqrt((2*Nx/50)**2 + (Ny+1)**2) #cavity modes frequency.
            self.Hrad[Ny,Ny] = w_c
            for i in range(self.Nmol):
                Vradmol[Ny,i] = g_k ##np.sin((np.pi*(Ny+1))/2)
        
        self.H0 = np.vstack((np.hstack((self.Hmol,   Vradmol.T)),
                             np.hstack((Vradmol, self.Hrad))))
        # return self.H0

    def initial_energy(self):
        W,V = np.linalg.eigh(self.H0)
        self.E =np.sort(W, axis= None)
        # return self.E
    
    def energy_noCavity(self):
        W,V = np.linalg.eigh(self.Hmol)
        self.E0 = np.sort(W, axis=None)
    

model1 = Multimode(100,1)
model1.initialHamiltonian(25e-3)
model1.initial_energy()
model1.energy_noCavity()

X_aix = np.zeros(model1.Nmol+1)
X_aix_1 = np.zeros(model1.Nmol)
for i in range(model1.Nmol):
    X_aix[i] = np.cos(2*np.pi*i/model1.Nmol)
    X_aix_1[i] = np.cos(2*np.pi*i/model1.Nmol)

# plt.hist(model1.E)
plt.scatter(X_aix,model1.E,label='with cavity')
plt.scatter(X_aix_1,model1.E0,label='without cavity')
plt.legend()
plt.show()