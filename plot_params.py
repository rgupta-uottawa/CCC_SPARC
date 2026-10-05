import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.dpi": 140})
df = pd.read_csv("/mnt/user-data/outputs/fit_parameters.csv")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axs[0]
for col, lab, c in [("CCC_rhot", r"CCC $\rho_t$", "tab:red"),
                    ("MOND_a0", r"MOND $a_0$", "tab:blue")]:
    x = df[col].dropna()
    r = np.log10(x / x.median())
    ax.hist(r, bins=np.linspace(-1.5, 1.5, 40), density=True, alpha=0.4, color=c,
            label=f"{lab}: {r.std():.2f} dex")
ax.axvline(0, color="0.4", lw=0.8, ls=":")
ax.set_xlabel(r"$\log_{10}$(parameter / median)")
ax.set_ylabel("density")
ax.set_title("Universality test: the two constants\n(narrower = more universal)", fontsize=9.5)
ax.legend(fontsize=8.5)

ax = axs[1]
params = [(r"CCC $\rho_t$", "CCC_rhot", "tab:red", "constant?"),
          (r"MOND $a_0$", "MOND_a0", "tab:blue", "constant?"),
          (r"NFW1 $V_{200}$", "NFW1_V200", "tab:green", "galaxy prop."),
          (r"NFW1 $c$", "NFW1_c", "tab:olive", "from c-M"),
          (r"NFW2 $V_{200}$", "NFW2_V200", "tab:purple", "galaxy prop."),
          (r"NFW2 $c$", "NFW2_c", "tab:pink", "galaxy prop.")]
sc = [np.log10(df[col].dropna()).std() for _, col, _, _ in params]
cols = [c for _, _, c, _ in params]
ax.barh(range(len(params)), sc, color=cols, alpha=0.85)
ax.set_yticks(range(len(params)))
ax.set_yticklabels([p[0] for p in params])
ax.invert_yaxis()
for i, (s, p) in enumerate(zip(sc, params)):
    ax.text(s + 0.01, i, f"{s:.2f} dex  ({p[3]})", va="center", fontsize=7.5)
ax.set_xlabel("log-scatter of fitted parameter (dex)")
ax.set_title("Parameter scatter, all models\n(only rho_t, a0 are meant to be universal)", fontsize=9.5)
ax.set_xlim(0, 1.15)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/parameter_residuals.png", bbox_inches="tight")
print("saved")
print(f"  CCC  rho_t : median {df.CCC_rhot.median()/1e-24:.2f}e-24 g/cm3")
print(f"  MOND a0    : median {df.MOND_a0.median():.2e} m/s2")
print(f"  NFW1 V200  : median {df.NFW1_V200.median():.0f} km/s")
print(f"  NFW2 V200  : median {df.NFW2_V200.median():.0f} km/s, c median {df.NFW2_c.median():.1f}")
