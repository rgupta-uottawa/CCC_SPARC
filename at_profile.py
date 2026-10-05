import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150, "savefig.dpi": 150})
df = pd.read_csv("at_compare.csv"); r = df[df.Rt > df.Rmin].copy().reset_index(drop=True)
lrt = np.log10(r.rho_t); lat = np.log10(r.a_t); la0 = np.log10(r.a0)

fig, axs = plt.subplots(1, 4, figsize=(19, 4.5))

# Panel 1: rho_t alone
ax = axs[0]
x = lrt - lrt.median()
ax.hist(x, bins=np.linspace(-1.2, 1.2, 28), density=True, alpha=0.55, color="tab:red",
        label=f"CCC $\\rho_t$: {x.std():.2f} dex")
ax.axvline(0, color="0.4", ls=":", lw=0.8)
ax.set_xlabel(r"$\log_{10}(\rho_t/\mathrm{median})$"); ax.set_ylabel("probability density")
ax.set_title(f"(a) Turn-off density $\\rho_t$\nscatter {x.std():.2f} dex (not universal)", fontsize=9.5)
ax.legend(fontsize=9)

# Panel 2: a_t & a_0 centred, Gaussian fits, KS
ax = axs[1]; bins = np.linspace(-1.0, 1.0, 24)
for xx, lab, c in [(lat - lat.median(), r"CCC $a_t$", "tab:orange"),
                   (la0 - la0.median(), r"MOND $a_0$", "tab:blue")]:
    ax.hist(xx, bins=bins, density=True, alpha=0.45, color=c, label=f"{lab}: {xx.std():.2f} dex")
    xs = np.linspace(-1, 1, 200); ax.plot(xs, stats.norm.pdf(xs, 0, xx.std()), color=c, lw=1.6)
ks = stats.ks_2samp(lat - lat.median(), la0 - la0.median())
ax.axvline(0, color="0.4", ls=":", lw=0.8)
ax.set_xlabel(r"$\log_{10}(a/\mathrm{median})$"); ax.set_ylabel("probability density")
ax.set_title(f"(b) $a_t$ vs $a_0$: same shape\nGaussian fits; KS $p={ks.pvalue:.2f}$", fontsize=9.5)
ax.legend(fontsize=9)

# Panel 3: absolute offset
ax = axs[2]
for xx, lab, c in [(lat, r"CCC $a_t$", "tab:orange"), (la0, r"MOND $a_0$", "tab:blue")]:
    ax.hist(xx, bins=np.linspace(-11, -9, 24), density=True, alpha=0.45, color=c,
            label=f"{lab}: {10**xx.median():.2e}")
ax.axvline(np.log10(1.2e-10), color="0.4", ls=":", lw=1, label=r"$a_0=1.2\times10^{-10}$")
ax.set_xlabel(r"$\log_{10} a\ [\mathrm{m\,s^{-2}}]$"); ax.set_ylabel("probability density")
ax.set_title(f"(c) Absolute scales\noffset factor {r.a_t.median()/r.a0.median():.2f}", fontsize=9.5)
ax.legend(fontsize=8.5)

# Panel 4: a_t vs a_0 scatter
ax = axs[3]
ax.scatter(la0, lat, s=20, c="tab:purple", alpha=0.6)
lim = [-11, -9]; ax.plot(lim, lim, "k--", lw=1, label="1:1")
ax.axhline(np.log10(1.2e-10), color="tab:blue", ls=":", lw=1, label=r"$a_0=1.2\times10^{-10}$")
rho, p = stats.spearmanr(la0, lat)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$\log_{10} a_0$ (MOND) $[\mathrm{m\,s^{-2}}]$")
ax.set_ylabel(r"$\log_{10} a_t$ (CCC) $[\mathrm{m\,s^{-2}}]$")
ax.set_title(f"(d) $a_t$ vs $a_0$ per galaxy\n$\\rho_{{\\rm S}}={rho:+.2f}$", fontsize=9.5)
ax.legend(fontsize=8.5)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/at_comparison.png", bbox_inches="tight")
print("saved 4-panel at_comparison.png; KS p=%.3f, offset=%.2f, scatter corr rho=%.2f" %
      (ks.pvalue, r.a_t.median()/r.a0.median(), rho))
