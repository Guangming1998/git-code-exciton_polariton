import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

# Constants for unit conversions
ps_to_au = 4.13414e4  # ps to atomic time
wavenumber_to_au = 4.55634e-6  # energy from wavenumber to atomic unit
kBT_to_au = 3.16681e-6
au_to_eV = 27.2114
total_timestep = 4e4
number_of_lattice = 100

# Read A(k, t) from CSV
df = pd.read_csv("GF_k_t.csv")

# Extract k and A(k, t)
k_values = df["Wave Vector k"].values
A_kt = df.iloc[:, 1:].values  # Exclude the "Wave Vector k" column

# Perform FFT along the time axis
dt = 0.25e-3 * ps_to_au  # Time step in atomic units (adjust as needed)
omega_values = np.fft.fftfreq(A_kt.shape[1], d=dt) * au_to_eV  # Frequency in eV
A_kw = np.fft.fft(A_kt, axis=1)  # Perform FFT along the time axis
A_kw = np.array(A_kw, dtype = np.complex128)

# Save A(k, ω) to CSV
df_fft = pd.DataFrame(A_kw, columns=[f"ω_{i}" for i in range(len(omega_values))])
df_fft.insert(0, "Wave Vector k", k_values)  # Insert k as the first column
df_fft.to_csv("GF_k_omega.csv", index=False)

# Read A(k, ω) from CSV
df_fft = pd.read_csv("GF_k_omega.csv")

# Extract k, ω, and A(k, ω)
k_values = df_fft["Wave Vector k"].values
omega_values = np.fft.fftfreq(A_kt.shape[1], d=dt) * au_to_eV  # Correct frequency in eV
A_kw = np.imag(df_fft.iloc[:, 1:].values) / (np.pi * number_of_lattice)  # Normalize
A_kw = np.array(A_kw, dtype = np.float64)

# Plot A(k, ω)
plt.figure(figsize=(8, 6))
plt.imshow(A_kw, extent=[k_values[0], k_values[-1], omega_values[0], omega_values[-1]],
           aspect='auto', origin='lower', cmap='viridis')
plt.xlabel("Wave Vector k (units of 1/a)")
plt.ylabel("Frequency ω (eV)")
plt.colorbar(label="|A(k, ω)|")
plt.title("Spectral Data A(k, ω)")
plt.show()
# print(int(4e5))