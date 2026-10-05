"""
Per-point INVERSE-mode residuals for all four models, at each galaxy's
best-fit parameter(s). Residual = log10(V_b,pred / V_b,SPARC), i.e. the
predicted baryonic velocity (from V_obs) over the photometric baryonic velocity.

All models inverse:  V_obs -> predicted V_b  ; compared to SPARC V_b.
  CCC   smooth-standard transition, rho_t fit per galaxy
  MOND  smooth-standard RAR, a0 fit per galaxy
  NFW1  V200 fit, c from c-M relation (k=1)
  NFW2  V200 and c both fit (k=2)
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
import sparc_io as io, ccc_core as cc, compare_models as cm, smooth_ccc as sm

K = cm.KMS2_KPC_TO_MS2
FLOOR, FRAC, N_MC, H0 = 1.0, 0.05, 60, 0.0671
RNG = np.random.default_rng(11)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def nfw_v(R, V200, c):
    R200 = V200 / (10 * H0); x = np.clip(R / R200, 1e-6, None)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    return np.sqrt(np.clip(V200 ** 2 * mu(c * x) / (x * mu(c)), 0, None))


def cM(V200):
    M200 = V200 ** 2 * (V200 / (10 * H0)) / cc.G
    return float(np.clip(10 ** (0.905 - 0.101 * np.log10(M200 / (1e12 / 0.671))), 1, 100))


def vb_ccc(R, V, eV, rt):
    s0=cc.inner_slope_of(cc.smooth_vobs(R,V,eV,0.1),R)
    return sm.vbx_smooth(R, V, eV, rt, kind="standard", inner_slope=s0)
def vb_mond(R, V, eV, a0):
    go = V ** 2 / R * K
    gb = a0 * 10 ** np.interp(np.log10(np.clip(go / a0, 1e-30, None)), _LY, _LX)
    return np.sqrt(np.clip(gb / K * R, 0, None))
def vb_nfw1(R, V, eV, V200): return np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, cM(V200)) ** 2, 0, None))


def sig1(pred, R, V, eV, p, Vb):
    out = [pred(R, V + RNG.normal(0, eV), eV, p) for _ in range(N_MC)]
    out = [v for v in out if v is not None and np.isfinite(v).all()]
    A = np.array(out) if len(out) >= 8 else None
    s = 0.5 * (np.percentile(A, 84, 0) - np.percentile(A, 16, 0)) if A is not None else np.full(len(R), 5.0)
    return np.sqrt(s ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)


def fit1(pred, R, V, eV, Vb, lo, hi):
    grid = np.linspace(lo, hi, 140)
    def c2(pl, s):
        v = pred(R, V, eV, 10 ** pl)
        return np.inf if (v is None or not np.isfinite(v).all()) else float(np.sum((v - Vb) ** 2 / s ** 2))
    s = np.maximum(eV, 1.0)
    best = grid[int(np.nanargmin([c2(x, s) for x in grid]))]
    for _ in range(2):
        s = sig1(pred, R, V, eV, 10 ** best, Vb)
        cur = np.array([c2(x, s) for x in grid]); j = int(np.nanargmin(cur))
        best = minimize_scalar(c2, args=(s,), bounds=(grid[max(j-1,0)], grid[min(j+1,139)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    return pred(R, V, eV, 10 ** best)


def fit_nfw2(R, V, eV, Vb):
    def c2(p, s):
        V200, c = 10 ** p[0], p[1]
        if not (10 <= V200 <= 600 and 1 <= c <= 40):
            return 1e12
        v = np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, c) ** 2, 0, None))
        return float(np.sum((v - Vb) ** 2 / s ** 2))
    gV = np.linspace(1.0, 2.78, 26); gc = np.linspace(1, 40, 22)
    s = np.maximum(eV, 1.0); best, bc = [2.0, 8.0], np.inf
    for pv in gV:
        for c_ in gc:
            val = c2([pv, c_], s)
            if val < bc:
                bc, best = val, [pv, c_]
    for _ in range(2):
        V200, c = 10 ** best[0], best[1]
        out = [np.sqrt(np.clip((V + RNG.normal(0, eV)) ** 2 - nfw_v(R, V200, c) ** 2, 0, None)) for _ in range(N_MC)]
        out = [v for v in out if np.isfinite(v).all()]
        A = np.array(out) if len(out) >= 8 else None
        sg = 0.5 * (np.percentile(A, 84, 0) - np.percentile(A, 16, 0)) if A is not None else np.zeros(len(R))
        s = np.sqrt(sg ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)
        best = minimize(c2, best, args=(s,), method="Nelder-Mead",
                        options=dict(xatol=1e-3, fatol=1e-2, maxiter=250)).x
    return np.sqrt(np.clip(V ** 2 - nfw_v(R, 10 ** best[0], best[1]) ** 2, 0, None))


t1, t2 = io.load_table1(), io.load_table2()
pts = []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 6:
        continue
    try:
        vc = fit1(vb_ccc, R, V, eV, Vb, 3, 12)
        vm = fit1(vb_mond, R, V, eV, Vb, -12, -8)
        v1 = fit1(vb_nfw1, R, V, eV, Vb, np.log10(10), np.log10(600))
        v2 = fit_nfw2(R, V, eV, Vb)
    except Exception:
        continue
    for i in range(len(R)):
        if Vb[i] > 0:
            pts.append(dict(Galaxy=g, R=R[i], Vb=Vb[i],
                            CCC=np.log10(vc[i] / Vb[i]) if vc[i] > 0 else np.nan,
                            MOND=np.log10(vm[i] / Vb[i]) if vm[i] > 0 else np.nan,
                            NFW1=np.log10(v1[i] / Vb[i]) if v1[i] > 0 else np.nan,
                            NFW2=np.log10(v2[i] / Vb[i]) if v2[i] > 0 else np.nan))

P = pd.DataFrame(pts)
P.to_csv("/mnt/user-data/outputs/residuals_inverse_all.csv", index=False)
print(f"{len(P)} points, {P.Galaxy.nunique()} galaxies")
for k in ["CCC", "MOND", "NFW1", "NFW2"]:
    r = P[k].dropna()
    print(f"  {k:5s}: scatter {r.std():.3f} dex   median {r.median():+.3f}   N={len(r)}")
