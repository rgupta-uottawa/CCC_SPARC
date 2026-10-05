import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sparc_io as io, ccc_core as cc, ccc_fit2 as cf
from run_paper7 import ORDER, PUB

plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.dpi": 140})
t1, t2 = io.load_table1(), io.load_table2()
K = 1e10 / cc.KPC_CM

res = {}
for g in ORDER:
    d = t2[t2.Galaxy == g].sort_values("R")
    res[g] = cf.fit(d.R.values, d.Vobs.values, d.e_Vobs.values,
                    d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    res[g]["Vobs"] = d.Vobs.values
    res[g]["eVobs"] = d.e_Vobs.values

fig, axes = plt.subplots(3, 3, figsize=(11.5, 10))
for ax, g in zip(axes.ravel(), ORDER):
    r = res[g]
    R = r["R"]
    ax.errorbar(R, r["Vobs"], yerr=r["eVobs"], fmt="o", ms=3, lw=0.8,
                color="0.25", label=r"$V_o$ (obs)", zorder=4)
    ax.plot(R, r["Vb0"], "s--", ms=3, lw=1.1, color="tab:blue",
            label=r"$V_b$ (SPARC, $\Upsilon$=0.5/0.7)")
    ax.plot(R, r["VbX"], "-", lw=1.8, color="tab:red", label=r"$V_{bX}$ (CCC)")
    ax.fill_between(R, r["VbX"] - r["sig_mod"], r["VbX"] + r["sig_mod"],
                    color="tab:red", alpha=0.22, lw=0)
    ax.axvline(r["Rt"], color="tab:green", ls=":", lw=1.2)
    ax.set_title(f"{g}  ({io.TYPES[int(t1.loc[g,'T'])]})", fontsize=9.5)
    ax.text(0.97, 0.05,
            r"$\rho_t=%.2f$ (pub %.2f)" % (r["rho_t_cgs"] / 1e-24, PUB[g][0]) + "\n"
            + r"$R_t=%.2f$ (pub %.2f)" % (r["Rt"], PUB[g][1]) + "\n"
            + r"$\chi^2_\nu=%.2f$,  fRMS=%.0f%%" % (r["chi2r"], 100 * r["frms"]),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(fc="w", ec="0.7", alpha=0.9, pad=2))
    ax.set_xlabel("R (kpc)"); ax.set_ylabel("V (km/s)")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    if g == ORDER[0]:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)

ax = axes.ravel()[7]
y = np.arange(len(ORDER))
mine = np.array([res[g]["rho_t_cgs"] / 1e-24 for g in ORDER])
lo = np.array([res[g]["rho_t_lo"] * cc.RHO_CGS / 1e-24 for g in ORDER])
hi = np.array([res[g]["rho_t_hi"] * cc.RHO_CGS / 1e-24 for g in ORDER])
ax.errorbar(mine, y, xerr=[mine - lo, hi - mine], fmt="o", ms=5, color="tab:red",
            label="this work")
ax.plot([PUB[g][0] for g in ORDER], y, "D", ms=5, mfc="none", color="tab:blue",
        label="Gupta (2025)")
ax.set_yticks(y); ax.set_yticklabels(ORDER, fontsize=7.5); ax.invert_yaxis()
ax.set_xlabel(r"$\rho_t\ (10^{-24}\,\mathrm{g\,cm^{-3}})$")
ax.set_title("turn-off density", fontsize=9.5)
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="x")

ax = axes.ravel()[8]
for g in ORDER:
    r = res[g]
    c = r["c2s"] / r["dof"]
    m = np.isfinite(c)
    ax.plot(10 ** r["grid"][m] * cc.RHO_CGS / 1e-24, c[m], lw=1.1, label=g)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\rho_t\ (10^{-24}\,\mathrm{g\,cm^{-3}})$")
ax.set_ylabel(r"$\chi^2/\nu$"); ax.set_title(r"$\chi^2$ profiles", fontsize=9.5)
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/ccc_paper7_v2.png", bbox_inches="tight")

print(f"{'galaxy':10s} {'rho_t':>7s} {'pub':>6s} {'Rt':>6s} {'a_t':>6s} {'chi2n':>7s} {'fRMS':>6s} {'sig/Vb':>7s}  flag")
for g in ORDER:
    r = res[g]
    at = t1.loc[g, "Vflat"] ** 2 / r["Rt"] * K / 1e-8
    print(f"{g:10s} {r['rho_t_cgs']/1e-24:7.2f} {PUB[g][0]:6.2f} {r['Rt']:6.2f} "
          f"{at:6.2f} {r['chi2r']:7.2f} {100*r['frms']:5.0f}% {100*r['power']:6.0f}%  {r['flag']}")
