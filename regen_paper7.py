"""Fig 3: benchmark rotation curves, smooth + sharp CCC, 4+3 layout, with
smooth chi2/nu annotated on each panel."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sparc_io as io, ccc_core as cc, smooth_ccc as sm
plt.rcParams.update({"font.size": 8.5, "figure.dpi": 140, "savefig.dpi": 140})
t1, t2 = io.load_table1(), io.load_table2()
fp = pd.read_csv("/mnt/user-data/outputs/fit_parameters.csv").set_index("Galaxy")
ic = pd.read_csv("/mnt/user-data/outputs/ic_smooth_ccc.csv").set_index("Galaxy")
TYPES = {0:"S0",1:"Sa",2:"Sab",3:"Sb",4:"Sbc",5:"Sc",6:"Scd",7:"Sd",8:"Sdm",9:"Sm",10:"Im",11:"BCD"}
G7 = ["D631-7","NGC1090","NGC3198","NGC3521","NGC6503","NGC7814","UGC09133"]

fig = plt.figure(figsize=(15, 7.6))
gs = fig.add_gridspec(2, 8)
positions = [(0,0),(0,2),(0,4),(0,6),(1,1),(1,3),(1,5)]
for (rr, cpos), g in zip(positions, G7):
    ax = fig.add_subplot(gs[rr, cpos:cpos+2])
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    rt_s = ic.loc[g, "CCC_standard_param"]
    rt_sharp = ic.loc[g, "CCC_sharp_param"]
    s0 = cc.inner_slope_of(cc.smooth_vobs(R, V, eV, 0.1), R)
    vs = sm.vbx_smooth(R, V, eV, rt_s, kind="standard", inner_slope=s0)
    vsh = sm.vbx_smooth(R, V, eV, rt_sharp, kind="sharp", inner_slope=s0)
    chi_s = ic.loc[g, "CCC_standard_chi2r"]
    chi_sh = ic.loc[g, "CCC_sharp_chi2r"]
    ax.errorbar(R, V, yerr=eV, fmt="o", ms=3, lw=0.7, color="0.25", label=r"$V_{\rm obs}$", zorder=3)
    ax.plot(R, Vb, "s--", ms=3, lw=1, color="tab:blue", label=r"$V_b$ (SPARC)")
    ax.plot(R, vs, "-", lw=1.8, color="tab:red", label=r"$V_{bX}$ smooth")
    ax.plot(R, vsh, ":", lw=1.6, color="tab:green", label=r"$V_{bX}$ sharp")
    ax.set_title(f"{g} ({TYPES.get(int(t1.loc[g,'T']),'?')})", fontsize=9.5)
    ax.set_xlabel("R (kpc)"); ax.set_ylabel("V (km/s)")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.text(0.96, 0.06,
            r"$\chi^2_\nu$: %.2f (sm), %.2f (sh)" % (chi_s, chi_sh),
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(fc="w", ec="0.7", alpha=0.85, pad=1.5))
    if g == G7[0]:
        ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/ccc_paper7_v2.png", bbox_inches="tight")
print("done; chi2 smooth:", {g: round(float(ic.loc[g,'CCC_standard_chi2r']),2) for g in G7})
