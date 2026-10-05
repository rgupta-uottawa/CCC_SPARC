"""
Regenerate the residual 4-panel. The distributions are strongly leptokurtic
(excess kurtosis +4.6 to +8.3): sharply peaked cores with heavy tails. A
Gaussian therefore cannot fit them -- it sits below the peak and above the
shoulders. This is a real property of the residuals, not a plotting artefact.

We therefore show:
  - the histogram (full range, no percentile trimming of the data),
  - a kernel-density estimate (KDE), which follows the actual shape,
  - a Gaussian of the same median and standard deviation, drawn as a light
    dotted REFERENCE (not a fit) so the peakedness is visible,
and annotate the excess kurtosis.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})

P = pd.read_csv("/mnt/user-data/outputs/residuals_inverse_all.csv")
models = [("CCC", "CCC (smooth)", "tab:red"),
          ("MOND", "MOND (smooth)", "tab:blue"),
          ("NFW1", "NFW 1-param", "tab:green"),
          ("NFW2", "NFW 2-param", "tab:purple")]

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
xs = np.linspace(-0.6, 0.6, 600)
for ax, (k, lab, c) in zip(axs.ravel(), models):
    r = P[k].dropna().values
    # clip only for the plotting window, but compute stats on full data
    rr = r[(r > -0.6) & (r < 0.6)]
    bins = np.linspace(-0.6, 0.6, 61)
    ax.hist(rr, bins=bins, density=True, color=c, alpha=0.40, label="data")
    # KDE (the curve that actually fits)
    kde = stats.gaussian_kde(rr, bw_method=0.25)
    ax.plot(xs, kde(xs), color=c, lw=2.2, label="kernel density")
    # Gaussian REFERENCE (same median & std) -- light dotted, labelled as reference
    mu, sg = np.median(r), r.std()
    ax.plot(xs, stats.norm.pdf(xs, mu, sg), "k:", lw=1.1, alpha=0.7,
            label="Gaussian (reference)")
    loc_c, gam_c = stats.cauchy.fit(rr)
    ax.plot(xs, stats.cauchy.pdf(xs, loc_c, gam_c), color="k", lw=1.6, ls="-",
            label="Lorentzian fit")
    ek = stats.kurtosis(r)
    ax.axvline(0, color="0.5", lw=0.8, ls=":")
    ax.set_title(f"{lab}: $\\sigma$={r.std():.3f} dex, "
                 f"skew {stats.skew(r):+.2f}, exc.\\ kurt.\\ {ek:+.1f}", fontsize=9)
    ax.set_xlabel(r"$\log_{10}(V_{b,\rm pred}/V_{b,\rm SPARC})$")
    ax.set_ylabel("probability density")
    ax.set_xlim(-0.6, 0.6)
    ax.legend(fontsize=7.5)
fig.suptitle("Inverse-mode residual distributions (full sample, 165 galaxies)",
             fontsize=10.5, y=1.0)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/residuals_4panel.png", bbox_inches="tight")
print("regenerated residual 4-panel with KDE + Gaussian reference")
for k, lab, c in models:
    r = P[k].dropna().values
    print(f"  {lab:18s} std {r.std():.3f}  skew {stats.skew(r):+.2f}  exc.kurt {stats.kurtosis(r):+.2f}")
