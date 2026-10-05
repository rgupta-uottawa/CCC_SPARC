"""
Sharp-MOND vs smooth-MOND on SPARC, acceleration space, per galaxy.

Models for g_obs(g_bar), each ONE free parameter, fit to the observed g_obs
at every radius (weighted by e_Vobs propagated to g):

  standard   g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))          [a0]   -- SPARC RAR
  simple     g_obs = g_bar/2 + sqrt((g_bar/2)^2 + g_bar a0)      [a0]   -- "simple" nu
  sharp      g_bar>g_t: g_obs=g_bar;  g_bar<g_t: sqrt(g_t g_bar) [g_t]  -- CCC twin

Comparison points:
  * does the SHARP threshold g_t come out near a0=1.2e-10, or is it biased?
  * how much worse is the sharp fit than the smooth ones (the cost of sharpness)?
This is the acceleration-space measure of exactly the simplification CCC makes
in density space.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import sparc_io as io, ccc_core as cc, compare_models as cm

A0 = 1.2e-10
K = cm.KMS2_KPC_TO_MS2


def g_std(gb, a0):
    return gb / (1.0 - np.exp(-np.sqrt(np.clip(gb / a0, 1e-30, None))))


def g_simple(gb, a0):
    return gb / 2.0 + np.sqrt((gb / 2.0) ** 2 + gb * a0)


def g_sharp(gb, gt):
    return np.where(gb > gt, gb, np.sqrt(gt * gb))


t1, t2 = io.load_table1(), io.load_table2()
rows = []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 5:
        continue
    gobs = V ** 2 / R * K
    gbar = Vb ** 2 / R * K
    # weight: d g_obs = 2 V/R * K * dV  -> sigma_g
    sg = np.abs(2.0 * V / R * K) * np.maximum(eV, 1.0)
    w = 1.0 / np.maximum(sg, 1e-14) ** 2
    m = (gobs > 0) & (gbar > 0)
    if m.sum() < 4:
        continue
    gobs, gbar, w = gobs[m], gbar[m], w[m]
    dof = max(m.sum() - 1, 1)

    def fit(fn, lo, hi):
        f = lambda lp: float(np.sum(w * (np.log10(fn(gbar, 10 ** lp)) - np.log10(gobs)) ** 2))
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-4))
        # chi2 in linear g units for comparability
        p = 10 ** r.x
        c2 = float(np.sum(w * (fn(gbar, p) - gobs) ** 2)) / dof
        # dex scatter of this galaxy
        sc = np.std(np.log10(gobs / fn(gbar, p)))
        return p, c2, sc

    rec = dict(Galaxy=g, N=int(m.sum()), T=int(t1.loc[g, "T"]))
    for name, fn in [("std", g_std), ("simple", g_simple), ("sharp", g_sharp)]:
        p, c2, sc = fit(fn, -12.0, -8.0)
        rec[f"{name}_p"] = p
        rec[f"{name}_chi2r"] = c2
        rec[f"{name}_dex"] = sc
    rows.append(rec)

df = pd.DataFrame(rows)
df["dwarf"] = df["T"] >= 8
df.to_csv("/mnt/user-data/outputs/sharp_mond.csv", index=False)

print(f"{df.shape[0]} galaxies fit in acceleration space\n")
print(f"{'model':22s} {'median param':>14s} {'16-84%':>20s} {'med chi2/nu':>12s} {'med dex':>8s}")
for name, lab, unit in [("std", "standard (a0)", "a0"), ("simple", "simple nu (a0)", "a0"),
                        ("sharp", "SHARP (g_t)", "g_t")]:
    p = df[f"{name}_p"]
    print(f"{lab:22s} {p.median():14.2e} [{p.quantile(.16):.1e}, {p.quantile(.84):.1e}]"
          f" {df[f'{name}_chi2r'].median():12.2f} {df[f'{name}_dex'].median():8.3f}")

print(f"\nsharp g_t / standard a0, per galaxy: median ratio {(df.sharp_p/df.std_p).median():.2f}"
      f"  [16-84%: {(df.sharp_p/df.std_p).quantile(.16):.2f}, {(df.sharp_p/df.std_p).quantile(.84):.2f}]")
print(f"cost of sharpness: median chi2/nu  smooth {df.std_chi2r.median():.2f} -> sharp {df.sharp_chi2r.median():.2f}"
      f"  (x{df.sharp_chi2r.median()/df.std_chi2r.median():.2f})")
print(f"cost of sharpness: median dex     smooth {df.std_dex.median():.3f} -> sharp {df.sharp_dex.median():.3f}")
print(f"\nstd a0 median {df.std_p.median():.2e}  (literature 1.2e-10; recovers the RAR scale as a check)")
