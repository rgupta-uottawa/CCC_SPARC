import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})

# ---- a_t figure ----
df = pd.read_csv("at_compare.csv")
r = df[df.Rt > df.Rmin].copy()
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axs[0]
for col, lab, c in [("rho_t", r"CCC $\rho_t$ (density)", "tab:red"),
                    ("a_t", r"CCC $a_t$ (accel.)", "tab:orange"),
                    ("a0", r"MOND $a_0$ (accel.)", "tab:blue")]:
    x = np.log10(r[col] / r[col].median())
    ax.hist(x, bins=np.linspace(-1.2, 1.2, 32), density=True, alpha=0.45, color=c,
            label=f"{lab}: {x.std():.2f} dex")
ax.axvline(0, color="0.4", lw=0.8, ls=":")
ax.set_xlabel(r"$\log_{10}$(parameter / median)")
ax.set_ylabel("density")
ax.set_title(f"Turn-off scale: density vs acceleration units\n(resolved sample, N={len(r)})", fontsize=9.5)
ax.legend(fontsize=8)
ax = axs[1]
ax.scatter(np.log10(r.a0), np.log10(r.a_t), s=14, c="tab:purple", alpha=0.6)
lim = [-11, -9]
ax.plot(lim, lim, "k--", lw=1, label="1:1")
ax.axhline(np.log10(1.2e-10), color="tab:blue", ls=":", lw=1, label=r"$a_0=1.2\times10^{-10}$")
ax.set_xlabel(r"$\log_{10} a_0$ (MOND) [m s$^{-2}$]")
ax.set_ylabel(r"$\log_{10} a_t$ (CCC) [m s$^{-2}$]")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.legend(fontsize=8)
ax.set_title(r"CCC $a_t$ vs MOND $a_0$, per galaxy", fontsize=9.5)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/at_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---- correlations figure ----
d = pd.read_csv("corr_table.csv")
d["dwarf"] = d["T"].values >= 8
panels = [("Reff", r"log $R_{\rm eff}$ (kpc)", True), ("Mbar", r"log $M_{\rm bar}$ ($M_\odot$)", True),
          ("fgas", "gas fraction", False), ("Vflat", r"log $V_{\rm flat}$ (km/s)", True)]
fig, axs = plt.subplots(2, 4, figsize=(14, 6.6))
for row, (ycol, ylab, color) in enumerate([("lrt", r"CCC $\log\rho_t$", "tab:red"),
                                           ("la0", r"MOND $\log a_0$", "tab:blue")]):
    for col, (xc, xl, lg) in enumerate(panels):
        ax = axs[row][col]
        x = np.log10(d[xc]) if lg else d[xc]
        y = d[ycol]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m & ~d.dwarf], y[m & ~d.dwarf], s=12, c="0.4", alpha=0.6, label="spiral")
        ax.scatter(x[m & d.dwarf], y[m & d.dwarf], s=12, c=color, alpha=0.6, label="dwarf")
        rr, p = stats.spearmanr(x[m], y[m])
        b = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 20)
        ax.plot(xs, np.polyval(b, xs), "k-", lw=1.3)
        ax.set_title(rf"$\rho_{{\rm S}}$={rr:+.2f}" + (r" $\dagger$" if p < 0.003 else ""), fontsize=9)
        ax.set_xlabel(xl, fontsize=8)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)
fig.suptitle(r"Smooth-CCC $\rho_t$ (top) and MOND $a_0$ (bottom) vs baryonic properties", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/parameter_correlations.png", bbox_inches="tight")
plt.close(fig)
print("at + correlation figures regenerated")
print(f"a_t scatter {np.log10(r.a_t).std():.3f}, a0 {np.log10(r.a0).std():.3f}, rho_t {np.log10(r.rho_t).std():.3f}")
