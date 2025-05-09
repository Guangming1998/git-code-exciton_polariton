import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#get my path
path = os.getcwd()

# 1) Define your parameter grids
dynamic_list = [498, 995, 1407]   # your “dynamic***” values
g_list       = [30, 90, 150]   # your “g**” values

# 2) Prepare the 3×3 subplots
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(6, 9),
    sharex='col', sharey='row',
    constrained_layout=True
)

# 3) Loop over rows (dynamic) and cols (g) to load & plot
for i, dynamic in enumerate(dynamic_list):
    for j, g in enumerate(g_list):
        ax = axes[i, j]

        # load your "grid" CSV
        fname = f"{path}/dynamic{dynamic}_g{g}.csv"
        df = pd.read_csv(fname, index_col=0)

        # recover k, ω and the 2D magnitude array
        ka   = df.index.values.astype(float)
        omega= df.columns.values.astype(float)
        Akw  = df.values      # shape (len(ka), len(omega))
        # print(ka[0], omega[0], Akw[0])


        # mesh & pcolormesh
        K, W = np.meshgrid(ka, omega, indexing='ij')
        pcm = ax.pcolormesh(
            K, W, Akw,
            
            vmin=0, vmax=0.12,      # adjust to your data range
            cmap='hot',
            shading='auto',       # matplotlib ≥3.4
        )

        # only label the left‐most column…
        # if j == 0:
            # ax.set_ylabel("$Frequency \omega/\\tau$")
        # only label the bottom row…
        if i == 2:
            # ax.set_xlabel(rf"Wave vector $ka/\pi$")
            ax.set_xticks([0, 0.5, 1])

        # annotate each subplot
        if i == 0:
            ax.set_title(rf"$g = {g}$")
        if j == 0:
            ax.text(
                -0.3, 0.5, 
                rf"$\alpha = {dynamic}$", 
                transform=ax.transAxes,
                rotation=90, va='center'
            )
        ax.set_ylim([-8, 8])

# 4) Add a single horizontal colorbar across the top
cbar = fig.colorbar(
    pcm, 
    ax=axes,                      # span all subplots
    orientation='horizontal',
    location='top',               # matplotlib ≥3.4
    fraction=0.05, pad=0.07
)
cbar.set_label(r"$|A(k,\omega)|\ (\mathrm{meV}^{-1})$", fontsize = 14)
cbar.ax.tick_params(labelsize=12)

# --- one global x- and y-label ---
# If you have Matplotlib ≥3.4:
fig.supxlabel(r"Wave vector $kL / \pi$", fontsize=16)
fig.supylabel(r"Frequency $\omega$ (meV)", fontsize=16)

plt.savefig(f'{path}/spectral_function.png', dpi=300)
plt.show()
