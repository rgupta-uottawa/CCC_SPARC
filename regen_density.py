"""Fig 2: reconstructed density profiles (smooth CCC) for the 7 benchmark
galaxies, 4+3 layout, showing rho_o and rho_bX with R^-2 / R^-4 asymptotes."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sparc_io as io, ccc_core as cc, smooth_ccc as sm
import pandas as pd
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})
t2 = io.load_table2()
fp = pd.read_csv("/mnt/user-data/outputs/fit_parameters.csv").set_index("Galaxy")
GALS = ["D631-7","NGC1090","NGC3198","NGC3521","NGC6503","NGC7814","UGC09133"]

fig = plt.figure(figsize=(15, 7.2))
# 4 panels row 1, 3 panels row 2 (centred): use a 2x8 grid, each panel spans 2 cols
# row1: cols 0-1,2-3,4-5,6-7 ; row2 centred: cols 1-2,3-4,5-6
positions = [(0,0),(0,2),(0,4),(0,6),(1,1),(1,3),(1,5)]
gs = fig.add_gridspec(2, 8)
for (rr, cc_), g in zip(positions, GALS):
    ax = fig.add_subplot(gs[rr, cc_:cc_+2])
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    rho_t = fp.loc[g, "CCC_rhot"] / cc.RHO_CGS
    sp = cc.smooth_vobs(R, V, eV, 0.1)
    rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=1500)
    keep = rg >= R[0]
    rg, rho = rg[keep], rho[keep]
    u = rho / rho_t
    W = sm.W_of_u(u, "standard")
    rho_bX = rho * W
    ax.plot(rg, rho*cc.RHO_CGS, "-", color="tab:blue", lw=1.5, label=r"$\rho_o$")
    ax.plot(rg, rho_bX*cc.RHO_CGS, "-", color="tab:red", lw=1.5, label=r"$\rho_{bX}$")
    ax.axhline(rho_t*cc.RHO_CGS, color="0.6", ls="--", lw=0.8, label=r"$\rho_t$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(R[0]*0.8, R[-1])
    ax.set_xlabel("R (kpc)"); ax.set_ylabel(r"$\rho$ (g cm$^{-3}$)")
    ax.set_title(g, fontsize=9.5)
    m = rg > np.median(rg)
    if m.sum() > 8:
        po = np.polyfit(np.log10(rg[m]), np.log10(rho[m]*cc.RHO_CGS), 1)[0]
        pb = np.polyfit(np.log10(rg[m]), np.log10(np.maximum(rho_bX[m]*cc.RHO_CGS,1e-40)),1)[0]
        ax.text(0.04, 0.05, f"$\\rho_o\\propto R^{{{po:.1f}}}$\n$\\rho_{{bX}}\\propto R^{{{pb:.1f}}}$",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(fc="w", ec="0.7", alpha=0.85, pad=2))
    ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/ccc_density_profiles.png", bbox_inches="tight")
print("regenerated Fig 2: 7 galaxies, 4+3 layout")
