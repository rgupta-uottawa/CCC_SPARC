import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import sparc_io as io
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})

ic = pd.read_csv("/mnt/user-data/outputs/ic_smooth_ccc.csv")
t1 = io.load_table1()
m = (ic.CCC_standard_AIC.notna() & ic.MOND_smooth_AIC.notna()).values
s = ic[m].reset_index(drop=True)
gal = list(s.Galaxy.values)
d = (s.MOND_smooth_AIC - s.CCC_standard_AIC).values   # per-galaxy total dAIC
SB = np.array([float(t1.loc[g, "SBeff"]) for g in gal])
Tv = np.array([float(t1.loc[g, "T"]) for g in gal])
isSp = Tv < 8
dclip = np.clip(d, -40, 40)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) histogram of per-galaxy dAIC
ax = axs[0]
ax.hist(dclip, bins=np.linspace(-40, 40, 41), color="0.6", edgecolor="white", linewidth=0.4)
ax.axvspan(-2, 2, color="gray", alpha=0.2)
ax.axvline(0, color="k", lw=1, ls=":")
ax.axvline(d.mean(), color="tab:red", lw=2, label=f"mean = {d.mean():.1f}")
ax.set_xlabel(r"$\Delta$AIC = AIC$_{\rm MOND}-$AIC$_{\rm CCC}$  (+ favours CCC)")
ax.set_ylabel("number of galaxies")
ax.set_title("(a) Per-galaxy AIC difference\n($|\\Delta$AIC$|>$40 piled at $\\pm$40)", fontsize=9.5)
ax.legend(fontsize=8)

# (b) evidence bands
ax = axs[1]
bands = [("<$-$10\nMOND\ndecisive", (d < -10).sum()),
         ("$-$10..$-$4\nMOND", ((d >= -10) & (d < -4)).sum()),
         ("$-$4..+4\ntie", ((d >= -4) & (d <= 4)).sum()),
         ("+4..+10\nCCC", ((d > 4) & (d <= 10)).sum()),
         (">+10\nCCC\ndecisive", (d > 10).sum())]
counts = [b[1] for b in bands]
colors = ["tab:blue", "#7aa8d0", "0.7", "#e39aa0", "tab:red"]
ax.bar(range(5), counts, color=colors, edgecolor="white")
for i, c in enumerate(counts):
    ax.text(i, c + 1, f"{c}\n({100*c/len(d):.0f}%)", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(5)); ax.set_xticklabels([b[0] for b in bands], fontsize=7.5)
ax.set_ylabel("number of galaxies"); ax.set_ylim(0, max(counts) * 1.25)
ax.set_title(f"(b) Galaxies by evidence band (N={len(d)})", fontsize=9.5)

# (c) scatter: per-galaxy dAIC (clipped +-40) vs surface brightness
ax = axs[2]
ax.scatter(np.log10(SB[isSp]), dclip[isSp], s=20, c="tab:red", alpha=0.6, label="spiral (T$<$8)")
ax.scatter(np.log10(SB[~isSp]), dclip[~isSp], s=20, c="tab:blue", alpha=0.6, label="dwarf (T$\\geq$8)")
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.axhspan(-2, 2, color="gray", alpha=0.15)
ax.set_ylim(-44, 44)
r, p = stats.spearmanr(np.log10(SB), d)
ax.set_xlabel(r"$\log_{10}$ effective surface brightness [$L_\odot\,\mathrm{pc}^{-2}$]")
ax.set_ylabel(r"$\Delta$AIC  (+ favours CCC, clipped $\pm$40)")
ax.set_title(rf"(c) Preference vs surface brightness ($\rho_{{\rm S}}={r:+.2f}$)", fontsize=9.5)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/daic_distribution.png", bbox_inches="tight")
print("saved. mean %.1f (all), %.1f (|d|<=40); rho_S(SB)=%.2f; %d beyond +-40" %
      (d.mean(), d[np.abs(d) <= 40].mean(), r, (np.abs(d) > 40).sum()))
