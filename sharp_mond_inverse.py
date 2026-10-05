"""
INVERSE-mode sharp vs smooth MOND, matching the CCC methodology exactly.

CCC inverse:  V_obs -> rho_o -> (strip X) -> predicted rho_b/V_b, compare to SPARC V_b.
MOND inverse: g_obs -> (invert law) -> predicted g_bar, compare to SPARC g_bar.

Same direction, same target side (the baryons), same per-galaxy single parameter,
weighted by e_Vobs propagated to acceleration. This is the confirmed
apples-to-apples: baryon side obtained FROM observed velocities in every model.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import sparc_io as io, ccc_core as cc, compare_models as cm

A0 = 1.2e-10
K = cm.KMS2_KPC_TO_MS2
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def inv_std(gobs, a0):
    return a0 * 10 ** np.interp(np.log10(np.clip(gobs / a0, 1e-30, None)), _LY, _LX)


def inv_simple(gobs, a0):
    return gobs ** 2 / (gobs + a0)


def inv_sharp(gobs, gt):
    return np.where(gobs > gt, gobs, gobs ** 2 / gt)


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
    # error on the TARGET (gbar) is not kinematic; weight by propagated e_Vobs on gobs,
    # carried through the inversion (dominant known error), plus 5% floor on gbar.
    sg_obs = np.abs(2.0 * V / R * K) * np.maximum(eV, 1.0)
    m = (gobs > 0) & (gbar > 0)
    if m.sum() < 4:
        continue
    gobs, gbar, sg_obs = gobs[m], gbar[m], sg_obs[m]
    dof = max(m.sum() - 1, 1)

    def fit(inv):
        def chi2(lp):
            gp = inv(gobs, 10 ** lp)
            # propagate: near-unit Jacobian; use log residual weighted by relative err
            rel = sg_obs / gobs
            s = np.sqrt(rel ** 2 + 0.05 ** 2)
            return float(np.sum((np.log(gp / gbar)) ** 2 / s ** 2))
        r = minimize_scalar(chi2, bounds=(-12, -8), method="bounded", options=dict(xatol=1e-4))
        p = 10 ** r.x
        gp = inv(gobs, p)
        c2 = r.fun / dof
        sc = np.std(np.log10(gbar / gp))
        return p, c2, sc

    rec = dict(Galaxy=g, N=int(m.sum()), T=int(t1.loc[g, "T"]))
    for name, inv in [("std", inv_std), ("simple", inv_simple), ("sharp", inv_sharp)]:
        p, c2, sc = fit(inv)
        rec[f"{name}_p"], rec[f"{name}_chi2r"], rec[f"{name}_dex"] = p, c2, sc
    rows.append(rec)

df = pd.DataFrame(rows)
df["dwarf"] = df["T"] >= 8
df.to_csv("/mnt/user-data/outputs/sharp_mond_inverse.csv", index=False)

print("INVERSE MODE (baryon side predicted from observed velocities) — 171 galaxies\n")
print(f"{'model':22s} {'median scale':>14s} {'med chi2/nu':>12s} {'med dex':>9s}")
for name, lab in [("std", "smooth standard"), ("simple", "smooth simple-nu"), ("sharp", "SHARP (CCC-twin)")]:
    print(f"{lab:22s} {df[f'{name}_p'].median():14.2e} {df[f'{name}_chi2r'].median():12.2f} {df[f'{name}_dex'].median():9.3f}")

print(f"\nsharp g_t / smooth a0 (inverse): median {(df.sharp_p/df.std_p).median():.2f}"
      f"  [16-84%: {(df.sharp_p/df.std_p).quantile(.16):.2f}, {(df.sharp_p/df.std_p).quantile(.84):.2f}]")
print(f"cost of sharpness (inverse): chi2/nu x{df.sharp_chi2r.median()/df.std_chi2r.median():.2f},"
      f"  dex {df.std_dex.median():.3f} -> {df.sharp_dex.median():.3f}")
print(f"smooth a0 median {df.std_p.median():.2e} (check vs 1.2e-10)")
