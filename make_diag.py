import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sparc_io as io, ccc_core as cc, ccc_fit2 as cf
from run_paper7 import ORDER, PUB

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})
OUT = "/mnt/user-data/outputs/"
t1, t2 = io.load_table1(), io.load_table2()
SF = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))

# ---- (a) residual autocorrelation vs smoothing
ax = axs[0, 0]
for g in ORDER:
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    a = []
    for sf in SF:
        sp = UnivariateSpline(R, V, w=1 / np.maximum(eV, 1e-3), k=3, s=sf * len(R))
        r = (V - sp(R)) / np.maximum(eV, 1e-3)
        a.append(np.corrcoef(r[:-1], r[1:])[0, 1])
    ax.plot(SF, a, "o-", ms=3, lw=1, label=g)
ax.axhline(0, color="k", lw=0.8)
ax.axhspan(-0.3, 0.3, color="green", alpha=0.10)
ax.axvline(0.1, color="tab:red", ls="--", lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("smoothing $s/N$")
ax.set_ylabel("lag-1 autocorr. of residuals")
ax.set_title("(a) over-smoothing test\ngreen band = residuals consistent with noise", fontsize=9)
ax.legend(fontsize=6, ncol=2)

# ---- (b) density roughness vs smoothing
ax = axs[0, 1]
for g in ORDER:
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    rr = []
    for sf in [0.0] + SF:
        sp = cc.smooth_vobs(R, V, eV, s_factor=sf)
        rg, rho, _, _ = cc.density_profile(sp, R, n_grid=400)
        m = (rg >= R[0]) & (rho > 0)
        rr.append(np.sqrt(np.mean(np.diff(np.log(rho[m]), 2) ** 2)))
    ax.plot([0.02] + SF, rr, "o-", ms=3, lw=1, label=g)
ax.axvline(0.1, color="tab:red", ls="--", lw=1.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("smoothing $s/N$  (leftmost point = interpolating spline)")
ax.set_ylabel(r"roughness of $\log\rho_o$")
ax.set_title("(b) differentiation noise\nred line = adopted $s/N=0.1$", fontsize=9)

# ---- (c) D631-7 anatomy
ax = axs[1, 0]
d = t2[t2.Galaxy == "D631-7"].sort_values("R")
R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
r = cf.fit(R, V, eV, d.Vgas.values, d.Vdisk.values, d.Vbul.values)
ax.errorbar(R, V, yerr=eV, fmt="o", ms=3, color="0.25", label=r"$V_o$")
ax.plot(R, r["Vb0"], "s--", ms=3, color="tab:blue", label=r"$V_b$ (SPARC)")
ax.plot(R, r["VbX"], "-", lw=1.8, color="tab:red", label=r"$V_{bX}$ (CCC)")
ax.fill_between(R, r["VbX"] - r["sig_mod"], r["VbX"] + r["sig_mod"],
                color="tab:red", alpha=0.22, lw=0, label=r"MC $\sigma$ from $e_{V_{obs}}$")
ax.set_xlabel("R (kpc)"); ax.set_ylabel("V (km/s)")
ax.set_title("(c) D631-7: not a bad fit — an undetermined one\n"
             r"model $\sigma$ = %.0f%% of $V_b$" % (100 * r["power"]), fontsize=9)
ax.legend(fontsize=7); ax.set_ylim(0, None)

# ---- (d) rho_t - Upsilon degeneracy
ax = axs[1, 1]
for g in ORDER:
    d = t2[t2.Galaxy == g].sort_values("R")
    a = cf.fit(d.R.values, d.Vobs.values, d.e_Vobs.values, d.Vgas.values,
               d.Vdisk.values, d.Vbul.values, ups_free=False)
    b = cf.fit(d.R.values, d.Vobs.values, d.e_Vobs.values, d.Vgas.values,
               d.Vdisk.values, d.Vbul.values, ups_free=True)
    ax.annotate("", xy=(b["ups_scale"], b["rho_t_cgs"] / 1e-24),
                xytext=(1.0, a["rho_t_cgs"] / 1e-24),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="tab:purple", alpha=0.8))
    ax.plot(1.0, a["rho_t_cgs"] / 1e-24, "o", ms=5, color="tab:blue")
    ax.plot(b["ups_scale"], b["rho_t_cgs"] / 1e-24, "s", ms=5, color="tab:orange")
    ax.text(b["ups_scale"], b["rho_t_cgs"] / 1e-24 * 1.12, g, fontsize=6, ha="center")
ax.axvspan(10**-0.1, 10**0.1, color="green", alpha=0.12)
ax.plot([], [], "o", color="tab:blue", label=r"$\Upsilon$ fixed (0.5/0.7)")
ax.plot([], [], "s", color="tab:orange", label=r"$\Upsilon$ profiled")
ax.set_yscale("log"); ax.set_xlabel(r"$\Upsilon$ scaling relative to 0.5/0.7")
ax.set_ylabel(r"$\rho_t\ (10^{-24}\,\mathrm{g\,cm^{-3}})$")
ax.set_title("(d) $\\rho_t$–$\\Upsilon$ degeneracy\ngreen band = 0.1 dex prior", fontsize=9)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT + "ccc_diagnostics.png", bbox_inches="tight")
print("done")
