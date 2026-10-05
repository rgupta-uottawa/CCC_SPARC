"""
Two-parameter NFW (free V200 AND free concentration c), matched to the same
inverse pipeline and scored on chi2/nu, AIC, BIC, DIC.

Included for context (so the k=2 penalty is judged fairly):
  NFW_1p   V200 only, c from c-M relation     k=1   (previous straw-man)
  NFW_2p   V200 and c independent             k=2   (the honest halo)
  CCC_std  smooth-CCC standard                k=1   (our best one-parameter model)
  MOND     smooth standard                    k=1

AIC = chi2 + 2k ;  BIC = chi2 + k ln N ;  DIC = chi2 + 2 p_D.
p_D for the 2-parameter model is estimated from the 2D 1-sigma slab
(points with delta chi2 <= 2.30, the 68% region for 2 dof).
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
import sparc_io as io, ccc_core as cc, compare_models as cm, smooth_ccc as sm

K = cm.KMS2_KPC_TO_MS2
FLOOR, FRAC, N_MC = 1.0, 0.05, 80
H0 = 0.0671
RNG = np.random.default_rng(5)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def nfw_v(R, V200, c):
    R200 = V200 / (10 * H0)
    x = np.clip(R / R200, 1e-6, None)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    return np.sqrt(np.clip(V200 ** 2 * mu(c * x) / (x * mu(c)), 0, None))


def cM(V200):
    M200 = V200 ** 2 * (V200 / (10 * H0)) / cc.G
    return float(np.clip(10 ** (0.905 - 0.101 * np.log10(M200 / (1e12 / 0.671))), 1, 100))


def vb_nfw1(R, V, eV, V200):
    return np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, cM(V200)) ** 2, 0, None))


def vb_ccc_std(R, V, eV, rt):
    return sm.vbx_smooth(R, V, eV, rt, kind="standard")


def vb_mond(R, V, eV, a0):
    go = V ** 2 / R * K
    gb = a0 * 10 ** np.interp(np.log10(np.clip(go / a0, 1e-30, None)), _LY, _LX)
    return np.sqrt(np.clip(gb / K * R, 0, None))


def smc1(pred, R, V, eV, p):
    out = [pred(R, V + RNG.normal(0, eV), eV, p) for _ in range(N_MC)]
    out = [v for v in out if v is not None and np.isfinite(v).all()]
    if len(out) < 8:
        return np.full(len(R), 5.0)
    A = np.array(out)
    return 0.5 * (np.percentile(A, 84, 0) - np.percentile(A, 16, 0))


def fit1(pred, R, V, eV, Vb, lo, hi):
    grid = np.linspace(lo, hi, 160)
    def c2(pl, sig):
        v = pred(R, V, eV, 10 ** pl)
        return np.inf if (v is None or not np.isfinite(v).all()) else float(np.sum((v - Vb) ** 2 / sig ** 2))
    sig = np.maximum(eV, 1.0)
    best = grid[int(np.nanargmin([c2(x, sig) for x in grid]))]
    for _ in range(2):
        sig = np.sqrt(smc1(pred, R, V, eV, 10 ** best) ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)
        cur = np.array([c2(x, sig) for x in grid]); j = int(np.nanargmin(cur))
        best = minimize_scalar(c2, args=(sig,), bounds=(grid[max(j-1,0)], grid[min(j+1,159)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    cur = np.array([c2(x, sig) for x in grid]); C = c2(best, sig); N = len(R)
    ok = np.isfinite(cur); slab = cur[ok][cur[ok] <= C + 1.0]
    pD = max(np.mean(slab) - C, 0.0) if slab.size > 1 else 0.5
    return C, C / max(N-1, 1), C + 2, C + np.log(N), C + 2 * pD


def fit_nfw2(R, V, eV, Vb):
    # 2D fit: log10 V200 in [1,2.78], c in [1,40]
    def sig_for(V200, c):
        base = lambda vv: np.sqrt(np.clip(vv ** 2 - nfw_v(R, V200, c) ** 2, 0, None))
        out = [base(V + RNG.normal(0, eV)) for _ in range(N_MC)]
        out = [v for v in out if np.isfinite(v).all()]
        if len(out) < 8:
            return np.sqrt((FRAC * Vb) ** 2 + FLOOR ** 2)
        A = np.array(out)
        return np.sqrt((0.5*(np.percentile(A,84,0)-np.percentile(A,16,0)))**2 + (FRAC*Vb)**2 + FLOOR**2)

    def c2(p, sig):
        V200, c = 10 ** p[0], p[1]
        if not (10 <= V200 <= 600 and 1 <= c <= 40):
            return 1e12
        v = np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, c) ** 2, 0, None))
        return float(np.sum((v - Vb) ** 2 / sig ** 2))

    # coarse grid then refine
    gV = np.linspace(1.0, 2.78, 30); gc = np.linspace(1, 40, 25)
    sig = np.maximum(eV, 1.0)
    best, bc = None, np.inf
    for pv in gV:
        for cc_ in gc:
            val = c2([pv, cc_], sig)
            if val < bc:
                bc, best = val, [pv, cc_]
    for _ in range(2):
        sig = sig_for(10 ** best[0], best[1])
        r = minimize(c2, best, args=(sig,), method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-2, maxiter=300))
        best = r.x
    C = c2(best, sig); N = len(R)
    # p_D from 2D slab: sample chi2 around best on a small grid, delta<=2.30
    slab = []
    for dv in np.linspace(-0.15, 0.15, 7):
        for dc in np.linspace(-6, 6, 7):
            val = c2([best[0] + dv, best[1] + dc], sig)
            if val <= C + 2.30:
                slab.append(val)
    pD = max(np.mean(slab) - C, 0.0) if len(slab) > 1 else 1.0
    return C, C / max(N-2, 1), C + 4, C + 2*np.log(N), C + 2 * pD, 10 ** best[0], best[1]


t1, t2 = io.load_table1(), io.load_table2()
rows = []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 6:              # need N>k+3 for a meaningful 2-param fit
        continue
    rec = dict(Galaxy=g, N=len(R), T=int(t1.loc[g, "T"]))
    for nm, pr, lo, hi in [("CCC", vb_ccc_std, 3, 12), ("MOND", vb_mond, -12, -8),
                           ("NFW1", vb_nfw1, np.log10(10), np.log10(600))]:
        try:
            C, cr, a, b, di = fit1(pr, R, V, eV, Vb, lo, hi)
            rec[nm+"_chi2r"], rec[nm+"_AIC"], rec[nm+"_BIC"], rec[nm+"_DIC"] = cr, a, b, di
        except Exception:
            pass
    try:
        C, cr, a, b, di, V200, c = fit_nfw2(R, V, eV, Vb)
        rec["NFW2_chi2r"], rec["NFW2_AIC"], rec["NFW2_BIC"], rec["NFW2_DIC"] = cr, a, b, di
        rec["NFW2_c"] = c
    except Exception:
        pass
    rows.append(rec)

df = pd.DataFrame(rows)
df["dwarf"] = df["T"] >= 8
df.to_csv("/mnt/user-data/outputs/ic_nfw2.csv", index=False)

print(f"{df.shape[0]} galaxies (N>=6), inverse mode, information criteria\n")
print(f"{'model':16s}{'k':>3s}{'chi2/nu':>9s}{'AIC':>8s}{'BIC':>8s}{'DIC':>8s}")
for nm, k in [("CCC", 1), ("MOND", 1), ("NFW1", 1), ("NFW2", 2)]:
    print(f"{nm:16s}{k:>3d}{df[nm+'_chi2r'].median():9.2f}{df[nm+'_AIC'].median():8.1f}"
          f"{df[nm+'_BIC'].median():8.1f}{df[nm+'_DIC'].median():8.1f}")

print("\nhead-to-head vs NFW2 (fraction of galaxies where model is preferred = lower):")
for nm in ["CCC", "MOND", "NFW1"]:
    for ic in ["AIC", "BIC", "DIC"]:
        m = df[nm+"_"+ic].notna() & df["NFW2_"+ic].notna()
        pref = 100 * (df[nm+"_"+ic][m] < df["NFW2_"+ic][m]).mean()
        print(f"  {nm:5s} beats NFW2 on {ic}: {pref:3.0f}%", end="   ")
    print()

print("\nsum over sample (lower wins):")
for ic in ["AIC", "BIC", "DIC"]:
    tot = {nm: df[nm+"_"+ic].sum() for nm in ["CCC", "MOND", "NFW1", "NFW2"]}
    best = min(tot, key=tot.get)
    print(f"  {ic}: " + "  ".join(f"{k}={v:.0f}" for k, v in tot.items()) + f"  -> {best}")

print(f"\nNFW2 fitted concentration: median c = {df['NFW2_c'].median():.1f}"
      f"  [16-84%: {df['NFW2_c'].quantile(.16):.1f}, {df['NFW2_c'].quantile(.84):.1f}]")
