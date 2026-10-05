import numpy as np, pandas as pd, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sparc_io as io

plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.dpi": 140})
OUT = "/mnt/user-data/outputs/"
df = pd.read_csv(OUT + "ccc_sparc175_results.csv")
store = pickle.load(open("/home/claude/ccc/store.pkl", "rb"))
t1 = io.load_table1()
T2 = io.load_table2()
PAPER7 = ["D631-7", "NGC1090", "NGC3198", "NGC3521", "NGC6503", "NGC7814", "UGC09133"]

df["lr"] = np.log10(df.rho_t / 1e-24)
dw, sp = df.dwarf.values, ~df.dwarf.values

# ================================================== Figure 1: histogram + trends
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

ax = fig.add_subplot(gs[0, :2])
bins = np.linspace(df.lr.min() - 0.1, df.lr.max() + 0.1, 34)
ax.hist(df.lr[sp], bins=bins, color="tab:blue", alpha=0.65, label=f"spirals, T<8 (N={sp.sum()})")
ax.hist(df.lr[dw], bins=bins, color="tab:orange", alpha=0.65, label=f"dwarfs, T$\\geq$8 (N={dw.sum()})")
for g in PAPER7:
    v = df.loc[df.Galaxy == g, "lr"].values
    if v.size:
        ax.axvline(v[0], color="tab:red", lw=1.0, alpha=0.8)
ax.plot([], [], color="tab:red", lw=1.0, label="Gupta (2025) galaxies")
ax.axvline(df.lr.median(), color="k", ls="--", lw=1.4,
           label=f"median = {10**df.lr.median():.1f}")
ax.set_xlabel(r"$\log_{10}\,[\rho_t / 10^{-24}\,\mathrm{g\,cm^{-3}}]$")
ax.set_ylabel("number of galaxies")
ax.set_title(r"Turn-off density for all 175 SPARC galaxies    "
             r"(scatter %.2f dex $\Rightarrow$ $\rho_t$ is not universal)" % df.lr.std(),
             fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.hist(np.log10(df.chi2r.clip(1e-2, 1e3)), bins=28, color="0.5")
ax.axvline(0, color="k", ls="--", lw=1)
ax.axvline(np.log10(3), color="tab:red", ls=":", lw=1.2)
ax.set_xlabel(r"$\log_{10}\,\chi^2/\nu$"); ax.set_ylabel("N")
ax.set_title(f"fit quality\n{100*(df.chi2r<3).mean():.0f}% have $\\chi^2_\\nu<3$", fontsize=9)

for k, (xk, xl, logx) in enumerate([("Vflat", r"$V_{\rm flat}$ (km/s)", True),
                                    ("L36", r"$L_{[3.6]}\ (10^9 L_\odot)$", True),
                                    ("Rt_over_Rdisk", r"$R_t / R_{\rm disk}$", True)]):
    ax = fig.add_subplot(gs[1, k])
    x = df[xk].replace(0, np.nan)
    ax.scatter(x[sp], df.lr[sp], s=14, c="tab:blue", alpha=0.7, label="spirals")
    ax.scatter(x[dw], df.lr[dw], s=14, c="tab:orange", alpha=0.7, label="dwarfs")
    m = x.notna() & df.lr.notna() & (x > 0)
    if m.sum() > 10:
        lx = np.log10(x[m]) if logx else x[m]
        r = np.corrcoef(lx, df.lr[m])[0, 1]
        p = np.polyfit(lx, df.lr[m], 1)
        xs = np.linspace(lx.min(), lx.max(), 20)
        ax.plot(10**xs if logx else xs, np.polyval(p, xs), "k-", lw=1.4)
        ax.set_title(f"r = {r:+.2f},  slope {p[0]:+.2f}", fontsize=9)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xl); ax.set_ylabel(r"$\log_{10}\rho_t$")
    if k == 0:
        ax.legend(fontsize=7)

fig.savefig(OUT + "ccc_sparc175_summary.png", bbox_inches="tight")
plt.close(fig)

# ================================================== Figure 2: a_t and Upsilon
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axs[0]
a = (df.a_t / 1e-8).dropna()
ax.hist(np.log10(a), bins=26, color="tab:green", alpha=0.75)
ax.axvline(np.log10(1.2), color="tab:red", lw=1.6, label=r"MOND $a_0=1.2$")
ax.axvline(np.log10(a.median()), color="k", ls="--", lw=1.4,
           label=f"median = {a.median():.1f}")
ax.set_xlabel(r"$\log_{10}\,[a_t/10^{-8}\,\mathrm{cm\,s^{-2}}]$"); ax.set_ylabel("N")
ax.set_title(f"turn-off acceleration (N={len(a)})", fontsize=9.5); ax.legend(fontsize=7.5)

ax = axs[1]
u = df.ups_scale.dropna()
ax.hist(np.log10(u), bins=26, color="tab:purple", alpha=0.75)
ax.axvspan(-0.1, 0.1, color="green", alpha=0.15, label="0.1 dex prior")
ax.axvline(0, color="k", lw=1.2)
ax.axvline(np.log10(u.median()), color="k", ls="--", lw=1.4,
           label=f"median = {u.median():.2f}")
ax.set_xlabel(r"$\log_{10}$ ($\Upsilon$ scaling)"); ax.set_ylabel("N")
ax.set_title(r"variant B: does CCC want $\Upsilon>0.5$?", fontsize=9.5)
ax.legend(fontsize=7.5)

ax = axs[2]
m = df.rho_t_C.notna()
ax.scatter(df.rho_t[m] / 1e-24, df.rho_t_C[m] / 1e-24, s=14,
           c=np.where(df.dwarf[m], "tab:orange", "tab:blue"), alpha=0.7)
lim = [df.rho_t.min() / 1e-24 * 0.7, df.rho_t.max() / 1e-24 * 1.4]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$\rho_t$  spline"); ax.set_ylabel(r"$\rho_t$  raw finite differences")
rat = np.log10(df.rho_t_C[m] / df.rho_t[m])
ax.set_title(f"reconstruction systematic\nmedian offset {10**rat.median():.2f}x, "
             f"scatter {rat.std():.2f} dex", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT + "ccc_sparc175_at_upsilon.png", bbox_inches="tight")
plt.close(fig)

# ================================================== Figure 3: montage (all 175)
order = sorted(store.keys())
with PdfPages(OUT + "ccc_sparc175_rotation_curves.pdf") as pdf:
    per = 20
    for pg in range(int(np.ceil(len(order) / per))):
        chunk = order[pg * per:(pg + 1) * per]
        fig, axes = plt.subplots(5, 4, figsize=(11.5, 13))
        for ax, g in zip(axes.ravel(), chunk):
            r = store[g]
            R = r["R"]
            row = df[df.Galaxy == g].iloc[0]
            d = r
            ax.plot(R, r["Vb0"], "s--", ms=2.5, lw=0.9, color="tab:blue")
            if r["spline"] is not None:
                rr = np.linspace(R[0], R[-1], 300)
                ax.plot(rr, r["spline"](rr), "-", color="0.35", lw=0.9)
            ax.plot(R, r["VbX"], "-", lw=1.6, color="tab:red")
            ax.axvline(r["Rt"], color="tab:green", ls=":", lw=1.0)
            ax.set_title(f"{g}  ({row.Type})", fontsize=7.5)
            ax.text(0.96, 0.05,
                    r"$\rho_t$=%.1f  $\chi^2_\nu$=%.1f" % (row.rho_t / 1e-24, row.chi2r),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
            ax.tick_params(labelsize=6)
            ax.set_xlim(0, None); ax.set_ylim(0, None)
        # observed points drawn last so they sit on top
        for ax, g in zip(axes.ravel(), chunk):
            o = T2[T2.Galaxy == g].sort_values("R")
            ax.errorbar(o.R.values, o.Vobs.values, yerr=o.e_Vobs.values, fmt="o",
                        ms=2, lw=0.6, color="0.15", zorder=5)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.suptitle("SPARC CCC+TL fits — grey: smoothing spline, black: $V_{obs}$, "
                     "blue: $V_b$ (SPARC), red: $V_{bX}$ (CCC), green: $R_t$",
                     fontsize=9, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        pdf.savefig(fig); plt.close(fig)
print("figures written")
