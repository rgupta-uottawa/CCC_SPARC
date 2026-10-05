import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "figure.dpi": 140, "savefig.dpi": 140})

df = pd.read_csv("/home/claude/ccc/params_morph.csv")
RNG = np.random.default_rng(0)

# normalize each parameter to its OWN global median so we compare dimensionless
# deviations; scatter is measured in dex (log10)
df["lrt"] = np.log10(df.CCC_rhot / df.CCC_rhot.median())
df["la0"] = np.log10(df.MOND_a0 / df.MOND_a0.median())

CLASSES = ["0-2 early (S0-Sab)", "3-4 (Sb-Sbc)", "5 (Sc)", "6-7 (Scd-Sd)", "8-11 dwarf/Irr"]


def boot(x, fn, n=3000):
    x = np.asarray(x)
    vals = [fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(vals, [16, 50, 84])


print("=" * 74)
print("MORPHOLOGICAL DEPENDENCE — median parameter (normalised) by class")
print("Question: does the CENTRE shift with type, and more for rho_t than a0?\n")
print(f"{'class':22s} {'N':>3s} {'rho_t median [16-84]':>26s} {'a0 median [16-84]':>26s}")
res = {}
for cl in CLASSES:
    m = df.cls == cl
    N = int(m.sum())
    lo_r, md_r, hi_r = boot(df.lrt[m], np.median)
    lo_a, md_a, hi_a = boot(df.la0[m], np.median)
    res[cl] = dict(N=N, rt=(lo_r, md_r, hi_r), a0=(lo_a, md_a, hi_a))
    print(f"{cl:22s} {N:3d} {md_r:+6.2f} [{lo_r:+.2f},{hi_r:+.2f}]     "
          f"{md_a:+6.2f} [{lo_a:+.2f},{hi_a:+.2f}]")

# trend strength: spread of class medians (early->late). Larger = stronger type dependence
mr = np.array([res[c]["rt"][1] for c in CLASSES])
ma = np.array([res[c]["a0"][1] for c in CLASSES])
print(f"\nrange of class medians:  rho_t {mr.max()-mr.min():.2f} dex   a0 {ma.max()-ma.min():.2f} dex")
print(f"  -> rho_t centre shifts {'MORE' if (mr.max()-mr.min())>(ma.max()-ma.min()) else 'LESS'} across type than a0")

print("\n" + "=" * 74)
print("WITHIN-CLASS scatter (does homogeneous morphology reduce the scatter?)\n")
print(f"{'class':22s} {'N':>3s} {'rho_t scatter':>16s} {'a0 scatter':>16s}")
for cl in CLASSES:
    m = df.cls == cl
    lo_r, md_r, hi_r = boot(df.lrt[m], np.std)
    lo_a, md_a, hi_a = boot(df.la0[m], np.std)
    print(f"{cl:22s} {int(m.sum()):3d} {md_r:6.2f} [{lo_r:.2f},{hi_r:.2f}]  "
          f"{md_a:6.2f} [{lo_a:.2f},{hi_a:.2f}]")
full_r, full_a = df.lrt.std(), df.la0.std()
within_r = np.sqrt(np.mean([df.lrt[df.cls == c].var() for c in CLASSES]))
within_a = np.sqrt(np.mean([df.la0[df.cls == c].var() for c in CLASSES]))
print(f"\nfull-sample scatter:  rho_t {full_r:.2f}  a0 {full_a:.2f} dex")
print(f"mean within-class:    rho_t {within_r:.2f}  a0 {within_a:.2f} dex")
print(f"reduction from binning: rho_t {100*(1-within_r/full_r):.0f}%   a0 {100*(1-within_a/full_a):.0f}%")
print(f"  -> if rho_t reduces MORE, morphological mixing was a bigger part of its scatter")

print("\n" + "=" * 74)
print("BULGE CONTROL (sphericity argument is sharpest here)\n")
for lab, m in [("has bulge", df.bulge), ("bulgeless", ~df.bulge)]:
    print(f"  {lab:10s} N={int(m.sum()):3d}   rho_t scatter {df.lrt[m].std():.2f}   a0 scatter {df.la0[m].std():.2f} dex"
          f"   rho_t median {df.lrt[m].median():+.2f}")

# ---- figure
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(CLASSES))
ax = axs[0]
for col, lab, c, off in [("lrt", r"CCC $\rho_t$", "tab:red", -0.1), ("la0", r"MOND $a_0$", "tab:blue", 0.1)]:
    md = [boot(df[col][df.cls == cl], np.median) for cl in CLASSES]
    md = np.array(md)
    ax.errorbar(x + off, md[:, 1], yerr=[md[:, 1] - md[:, 0], md[:, 2] - md[:, 1]],
                fmt="o-", color=c, capsize=3, label=lab, lw=1.4, ms=5)
ax.axhline(0, color="0.5", lw=0.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(["S0-Sab", "Sb-Sbc", "Sc", "Scd-Sd", "dwarf/Irr"], fontsize=8)
ax.set_ylabel(r"$\log_{10}$(param / global median)")
ax.set_title("Parameter centre vs morphology\n(with bootstrap 16-84% CI)", fontsize=9.5)
ax.legend(fontsize=8.5)

ax = axs[1]
sr = [df.lrt[df.cls == cl].std() for cl in CLASSES]
sa = [df.la0[df.cls == cl].std() for cl in CLASSES]
w = 0.35
ax.bar(x - w/2, sr, w, color="tab:red", alpha=0.8, label=r"CCC $\rho_t$")
ax.bar(x + w/2, sa, w, color="tab:blue", alpha=0.8, label=r"MOND $a_0$")
ax.axhline(full_r, color="tab:red", ls="--", lw=1, alpha=0.6)
ax.axhline(full_a, color="tab:blue", ls="--", lw=1, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(["S0-Sab", "Sb-Sbc", "Sc", "Scd-Sd", "dwarf/Irr"], fontsize=8)
ax.set_ylabel("within-class scatter (dex)")
ax.set_title("Scatter within each class\n(dashed = full-sample scatter)", fontsize=9.5)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/parameter_morphology.png", bbox_inches="tight")
print("\nsaved figure")
