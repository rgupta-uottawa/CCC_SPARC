import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ccc_core as cc
from run_paper7 import run, ORDER, PUB

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "figure.dpi": 150, "savefig.dpi": 150})
OUT = "/mnt/user-data/outputs/"
res = {g: run(g) for g in ORDER}

# ---------------------------------------------------------- Fig 1: RC panels
fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
for ax, g in zip(axes.ravel(), ORDER):
    r = res[g]
    R, rg = r["R"], r["rg"]
    ax.errorbar(R, r["Vobs"], yerr=r["e_Vobs"], fmt="o", ms=3, lw=0.8,
                color="0.25", label=r"$V_o$ (obs)", zorder=3)
    ax.plot(rg[rg >= R[0]], r["spline"](rg[rg >= R[0]]), "-", color="0.25",
            lw=0.9, alpha=0.6)
    ax.plot(R, r["Vb"], "s--", ms=3, lw=1.1, color="tab:blue",
            label=r"$V_b$ (SPARC)")
    ax.fill_between(R, r["Vb"] - r["sVb"], r["Vb"] + r["sVb"],
                    color="tab:blue", alpha=0.18, lw=0)
    ax.plot(R, r["VbX"], "-", lw=1.8, color="tab:red", label=r"$V_{bX}$ (CCC)")
    ax.axvline(r["Rt"], color="tab:green", ls=":", lw=1.2)
    ax.set_title(f"{g} ({r['T']})", fontsize=9.5)
    ax.text(0.97, 0.06,
            r"$\rho_t=%.2f$" % (r["rho_t_cgs"] / 1e-24) + "\n"
            + r"$R_t=%.2f$ kpc" % r["Rt"] + "\n"
            + r"$\chi^2_\nu=%.2f$" % r["chi2r"],
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(fc="w", ec="0.7", alpha=0.85, pad=2))
    ax.set_xlabel("R (kpc)"); ax.set_ylabel("V (km/s)")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    if g == ORDER[0]:
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

# panel 8: rho_t comparison
ax = axes.ravel()[7]
y = np.arange(len(ORDER))
mine = np.array([res[g]["rho_t_cgs"] / 1e-24 for g in ORDER])
lo = np.array([res[g]["rho_t_lo_cgs"] / 1e-24 for g in ORDER])
hi = np.array([res[g]["rho_t_hi_cgs"] / 1e-24 for g in ORDER])
pub = np.array([PUB[g][0] for g in ORDER])
ax.errorbar(mine, y, xerr=[mine - lo, hi - mine], fmt="o", ms=5,
            color="tab:red", label=r"this work ($\chi^2$)")
ax.plot(pub, y, "D", ms=5, mfc="none", color="tab:blue", label="Gupta (2025)")
ax.set_yticks(y); ax.set_yticklabels(ORDER, fontsize=7.5)
ax.set_xlabel(r"$\rho_t\ (10^{-24}\,\mathrm{g\,cm^{-3}})$")
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="x"); ax.invert_yaxis()
ax.set_title(r"turn-off density: fitted vs published", fontsize=9.5)

# panel 9: chi2 profiles
ax = axes.ravel()[8]
for g in ORDER:
    r = res[g]
    c = r["c2s"] / r["dof"]
    m = np.isfinite(c)
    ax.plot(10 ** r["grid"][m] * cc.RHO_CGS / 1e-24, c[m], lw=1.1, label=g)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\rho_t\ (10^{-24}\,\mathrm{g\,cm^{-3}})$")
ax.set_ylabel(r"$\chi^2/\nu$")
ax.set_title(r"$\chi^2$ profiles (1 free parameter)", fontsize=9.5)
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "ccc_paper7_rotation_curves.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------- Fig 2: density profile NGC3198
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, g in zip(axs, ["NGC3198", "NGC7814"]):
    r = res[g]
    rg, rho = r["rg"], r["rho"] * cc.RHO_CGS
    X = r["X"]
    ax.plot(rg, rho, "-", color="tab:blue", lw=1.5, label=r"$\rho_o$")
    ax.plot(rg, rho * (1 - X) ** 2, "-", color="tab:red", lw=1.5,
            label=r"$\rho_{bX}$")
    ax.axvline(r["Rt"], color="tab:green", ls=":", lw=1.2)
    ax.axhline(r["rho_t_cgs"], color="0.6", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(r["R"][0] * 0.8, r["R"][-1])
    ax.set_xlabel("R (kpc)"); ax.set_ylabel(r"$\rho$ (g cm$^{-3}$)")
    ax.set_title(f"{g}: $R^{{-2}}$ vs $R^{{-4}}$ test", fontsize=9.5)
    m = rg > 2 * r["Rt"]
    if m.sum() > 10:
        po = np.polyfit(np.log10(rg[m]), np.log10(rho[m]), 1)[0]
        pb = np.polyfit(np.log10(rg[m]),
                        np.log10(np.maximum(rho[m] * (1 - X[m]) ** 2, 1e-40)), 1)[0]
        ax.text(0.04, 0.06, f"slopes: $\\rho_o\\propto R^{{{po:.2f}}}$\n"
                            f"$\\rho_{{bX}}\\propto R^{{{pb:.2f}}}$",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(fc="w", ec="0.7", alpha=0.85, pad=2))
    ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT + "ccc_density_profiles.png", bbox_inches="tight")
plt.close(fig)
print("figures written")
for g in ORDER:
    r = res[g]
    print(f"{g:10s} rho_t={r['rho_t_cgs']/1e-24:6.2f}  Rt={r['Rt']:5.2f}  "
          f"chi2nu={r['chi2r']:7.2f}  flag={r['flag']}")
