import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 150})

ic = pd.read_csv("/mnt/user-data/outputs/ic_smooth_ccc.csv")
n2 = pd.read_csv("/mnt/user-data/outputs/ic_nfw2.csv")
m = ic.CCC_standard_chi2r.notna() & ic.MOND_smooth_chi2r.notna()

def cdf(x):
    xs = np.sort(x[np.isfinite(x)])
    return xs, np.arange(1, len(xs) + 1) / len(xs)

fig, ax = plt.subplots(figsize=(7, 5))
series = [("CCC (smooth)", ic.CCC_standard_chi2r[m].values, "tab:red", "-"),
          ("MOND (smooth)", ic.MOND_smooth_chi2r[m].values, "tab:blue", "-"),
          ("NFW (2-param)", n2.NFW2_chi2r.dropna().values, "tab:green", "--"),
          ("NFW (1-param)", ic.NFW_chi2r.dropna().values, "tab:purple", ":")]
for lab, d, c, ls in series:
    xs, ys = cdf(d)
    ax.plot(xs, ys, ls, color=c, lw=2, label=f"{lab}  (mean {np.mean(d):.2f})")
ax.axvline(1, color="0.6", lw=0.8, ls=":")
ax.set_xlim(0, 8); ax.set_ylim(0, 1)
ax.set_xlabel(r"reduced $\chi^2_\nu$")
ax.set_ylabel(r"cumulative fraction of galaxies")
ax.set_title(r"Cumulative distribution of per-galaxy $\chi^2_\nu$")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/chi2_cdf.png", bbox_inches="tight")
print("saved chi2_cdf.png")
for lab, d, c, ls in series:
    xs, ys = cdf(d)
    print(f"  {lab}: median {np.median(d):.2f}, frac chi2<2 = {100*np.mean(d<2):.0f}%")
