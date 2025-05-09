import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# # Read A(k, ω) from CSV
# df_fft = pd.read_csv("GF_k_omega.csv")

# # Extract k, ω, and A(k, ω)
# k_values = df_fft["Wave Vector k"].values
# omega_values = np.fft.fftfreq(A_kt.shape[1], d=dt)
# A_kw = df_fft.iloc[:, 1:].values

# # Plot A(k, ω)
# plt.figure(figsize=(8, 6))
# plt.imshow(np.abs(A_kw), extent=[k_values[0], k_values[-1], omega_values[0], omega_values[-1]],
#            aspect='auto', origin='lower', cmap='viridis')
# plt.xlabel("Wave Vector k")
# plt.ylabel("Frequency ω")
# plt.colorbar(label="|A(k, ω)|")
# plt.title("Spectral Data A(k, ω)")
# plt.show()
