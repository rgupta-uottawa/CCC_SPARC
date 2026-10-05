"""
Full-sample test of the inner (r<R_min) density extrapolation for CCC.

Two prescriptions, fit rho_t for every galaxy with smooth-CCC, inverse mode:
  solid-body : V propto r  ->  constant inner density  (current fiducial)
  power-law  : continue the reconstructed density's log-log slope inward
               rho(r) = rho(R_min) (r/R_min)^s,  s = d ln rho / d ln R at R_min
               (clipped to s > -2.8 for a convergent inner mass)

We report the sample median rho_t, its scatter, and the median chi2/nu for each,
plus the per-galaxy shift, to see whether the choice matters in aggregate.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import sparc_io as io, ccc_core as cc, smooth_ccc as sm

t1, t2 = io.load_table1(), io.load_table2()


def recon(sp, R, mode):
    Rmin, Rmax = R[0], R[-1]
    rg = np.linspace(Rmin, Rmax, 1500)
    Vg = sp(rg); dV = sp.derivative()(rg)
    dMdR = (Vg ** 2 + 2 * rg * Vg * dV) / cc.G
    rho = dMdR / (4 * np.pi * rg ** 2)
    n_in = 200
    rin = np.linspace(1e-4, Rmin, n_in, endpoint=False)
    if mode == "solid":
        Vmin = float(sp(Rmin))
        rho_core = 3 * Vmin ** 2 / (4 * np.pi * cc.G * Rmin ** 2)
        rho_in = np.full_like(rin, rho_core)
        dMdR_in = 3 * Vmin ** 2 * rin ** 2 / (cc.G * Rmin ** 2)
        s = 0.0
    else:  # power-law slope continuation
        s = np.polyfit(np.log(rg[:30]), np.log(np.maximum(rho[:30], 1e-40)), 1)[0]
        s = np.clip(s, -2.8, 0.5)
        rho_in = rho[0] * (rin / Rmin) ** s
        dMdR_in = 4 * np.pi * rin ** 2 * rho_in
    rg = np.concatenate([rin, rg])
    rho = np.concatenate([rho_in, rho])
    dMdR = np.concatenate([dMdR_in, dMdR])
    return rg, rho, dMdR, s


def fit_rhot(R, V, eV, Vb, mode):
    sp = cc.smooth_vobs(R, V, eV, 0.1)
    sig = np.sqrt((0.05 * Vb) ** 2 + 1 + np.maximum(eV, 1) ** 2)
    def predict(rt):
        rg, rho, dMdR, s = recon(sp, R, mode)
        W = sm.W_of_u(rho / rt, "standard")
        integ = W * dMdR
        MbX = np.concatenate([[0], np.cumsum(np.diff(rg) * 0.5 * (integ[1:] + integ[:-1]))])
        return cc.vbx_at(rg, MbX, R), s
    def c2(lp):
        v, _ = predict(10 ** lp)
        return np.inf if (v is None or not np.isfinite(v).all()) else float(np.sum((v - Vb) ** 2 / sig ** 2))
    b = minimize_scalar(c2, bounds=(3, 12), method="bounded", options=dict(xatol=1e-4)).x
    _, s = predict(10 ** b)
    return 10 ** b * cc.RHO_CGS, c2(b) / max(len(R) - 1, 1), s


rows = []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 6:
        continue
    try:
        rt_sb, c_sb, _ = fit_rhot(R, V, eV, Vb, "solid")
        rt_pl, c_pl, s = fit_rhot(R, V, eV, Vb, "power")
    except Exception:
        continue
    rows.append(dict(Galaxy=g, T=int(t1.loc[g, "T"]), RminRd=R[0] / t1.loc[g, "Rdisk"],
                     inner_s=s, rt_sb=rt_sb, rt_pl=rt_pl, chi2_sb=c_sb, chi2_pl=c_pl))

df = pd.DataFrame(rows)
df.to_csv("/mnt/user-data/outputs/inner_extrap_test.csv", index=False)

N = len(df)
lsb, lpl = np.log10(df.rt_sb), np.log10(df.rt_pl)
print(f"Full-sample inner-extrapolation test ({N} galaxies)\n")
print(f"{'quantity':30s} {'solid-body':>14s} {'power-law':>14s}")
print("-" * 60)
print(f"{'median rho_t (1e-24 g/cm3)':30s} {df.rt_sb.median()/1e-24:14.2f} {df.rt_pl.median()/1e-24:14.2f}")
print(f"{'rho_t scatter (dex)':30s} {lsb.std():14.3f} {lpl.std():14.3f}")
print(f"{'median chi2/nu':30s} {df.chi2_sb.median():14.2f} {df.chi2_pl.median():14.2f}")
print()
dlrt = lpl - lsb
print(f"per-galaxy rho_t shift (power-law vs solid-body):")
print(f"  median {dlrt.median():+.3f} dex ({100*(10**dlrt.median()-1):+.1f}%)")
print(f"  16-84% [{dlrt.quantile(.16):+.3f}, {dlrt.quantile(.84):+.3f}] dex")
print(f"  max |shift| {dlrt.abs().max():.3f} dex")
print(f"  galaxies with |shift|>0.05 dex: {(dlrt.abs()>0.05).sum()} ({100*(dlrt.abs()>0.05).mean():.0f}%)")
print()
dchi = df.chi2_pl - df.chi2_sb
print(f"chi2/nu change (power-law minus solid-body):")
print(f"  median {dchi.median():+.3f}")
print(f"  power-law better in {100*(df.chi2_pl<df.chi2_sb).mean():.0f}% of galaxies")
print(f"  |median chi2| solid {df.chi2_sb.median():.2f} vs power {df.chi2_pl.median():.2f}")
print()
# does the shift correlate with Rmin/Rd or inner slope?
from scipy.stats import spearmanr
print("shift correlates with:")
for col in ["RminRd", "inner_s"]:
    r, p = spearmanr(df[col], dlrt)
    print(f"  {col:10s}: Spearman {r:+.2f} (p={p:.1e})")
