import numpy as np
import matplotlib.pyplot as plt

# Constants
tau = 100  # meV
a = 1  # Lattice constant, assume a = 1 for simplicity
Gamma = 2.5  # Broadening in meV
k_values = np.linspace(0, np.pi/a, 500)  # k-values in the range from -π/a to π/a
omega_values = np.linspace(-400, 400, 1000)  # Omega (frequency) range in meV

# Dispersion relation E(k) = -2 * tau * cos(k * a)
E_k = -2 * tau * np.cos(k_values * a)

# Create the spectral function matrix A(k, ω)
A_kw = np.zeros((len(k_values), len(omega_values)))

# Calculate the spectral function A(k, ω) for each k and ω
for i, k in enumerate(k_values):
    for j, omega in enumerate(omega_values):
        A_kw[i, j] = Gamma / ((omega - E_k[i])**2 + Gamma**2)

# Plotting with k on X-axis and ω on Y-axis
plt.figure(figsize=(8, 6))
plt.imshow(A_kw.T, extent=[k_values[0], k_values[-1], omega_values[0], omega_values[-1]],
           aspect='auto', origin='lower', cmap='viridis')
plt.xlabel(r'Wave vector $k$ (1/a)', fontsize=14)
plt.ylabel(r'Frequency $\omega$ (meV)', fontsize=14)
plt.title('Spectral Function $A(k, \omega)$', fontsize=16)
plt.colorbar(label=r'$A(k, \omega)$', fraction=0.02, pad=0.04)
plt.show()
