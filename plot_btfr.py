import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})

df = pd.read_csv("/mnt/user-data/outputs/btfr.csv")
lv = np.log10(df.Vflat.values)
mv = np.median(df.Vflat.values)
models = [("MB_ccc", "CCC (smooth)", "tab:red"),
          ("MB_mond", "MOND (smooth)", "tab:blue"),
          ("MB_nfw", "NFW (2-param)", "tab:purple")]

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axs[0]
ax.scatter(lv, np.log10(df.MB_data), s=16, c="0.5", alpha=0.5, label="SPARC data", zorder=2)
xs = np.linspace(lv.min(), lv.max(), 20)
for col, lab, c in models:
    lm = np.log10(df[col].values)
    ts = theilslopes(lm, lv)
    ax.plot(xs, ts[0] * xs + ts[1], "-", color=c, lw=1.8, label=f"{lab}: b={ts[0]:.2f}")
med_m = np.median(np.log10(df.MB_data))
ax.plot(xs, 4 * (xs - np.median(lv)) + med_m, "k--", lw=1.2,
        label="b=4 (CCC prediction)", zorder=1)
ax.set_xlabel(r"$\log_{10} V_{\rm flat}$ (km s$^{-1}$)")
ax.set_ylabel(r"$\log_{10} M_B$ ($M_\odot$)")
ax.set_title("Baryonic Tully-Fisher relation", fontsize=10)
ax.legend(fontsize=7.5)

ax = axs[1]
sel = np.abs(np.log10(df.Vflat.values / mv)) > 0.05
bins = np.linspace(0, 8, 21)
bi_all = []
for col, lab, c in models:
    mm = np.median(df[col].values)
    bi = np.log10(df[col].values / mm) / np.log10(df.Vflat.values / mv)
    bi = bi[sel & np.isfinite(bi)]
    bi = bi[(bi > 0) & (bi < 8)]
    bi_all.append((bi, lab, c))
# grouped solid bars
centers = 0.5 * (bins[1:] + bins[:-1])
width = (bins[1] - bins[0]) / (len(models) + 0.5)
for i, (bi, lab, c) in enumerate(bi_all):
    h, _ = np.histogram(bi, bins=bins)
    ax.bar(centers + (i - 1) * width, h, width=width, color=c, alpha=0.9,
           label=f"{lab}: med {np.median(bi):.1f}", edgecolor="white", linewidth=0.3)
ax.axvline(4, color="k", ls="--", lw=1.5, label="b=4")
ax.set_xlabel(r"local exponent $b_i$")
ax.set_ylabel("number of galaxies")
ax.set_title(r"Scatter of local BTFR exponent around $b=4$", fontsize=10)
ax.legend(fontsize=7.5)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/btfr.png", bbox_inches="tight")
print("saved")
