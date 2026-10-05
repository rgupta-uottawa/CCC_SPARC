"""
Consolidated regeneration with the fixed-slope power-law reconstruction.

The inner power-law slope is measured ONCE from each galaxy's fiducial
(unperturbed) reconstruction and held fixed during Monte-Carlo error
propagation, so it no longer amplifies noise. All models share an identical
error model: MC-propagated e_Vobs (robust 16-84 half-width) + 5% Vb + 1 km/s.

Outputs: fit_parameters.csv, ic_smooth_ccc.csv, ic_nfw2.csv  (overwritten).
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
import sparc_io as io, ccc_core as cc, compare_models as cm, smooth_ccc as sm

K = cm.KMS2_KPC_TO_MS2
FLOOR, FRAC, N_MC, H0 = 1.0, 0.05, 100, 0.0671
RNG = np.random.default_rng(3)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def nfw_v(R, V200, c):
    R200 = V200 / (10 * H0); x = np.clip(R / R200, 1e-6, None)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    return np.sqrt(np.clip(V200 ** 2 * mu(c * x) / (x * mu(c)), 0, None))


def cM(V200):
    M200 = V200 ** 2 * (V200 / (10 * H0)) / cc.G
    return float(np.clip(10 ** (0.905 - 0.101 * np.log10(M200 / (1e12 / 0.671))), 1, 100))


# ---- predictors. CCC variants take a fixed inner slope s0.
def make_ccc(kind, s0):
    return lambda R, V, eV, rt: sm.vbx_smooth(R, V, eV, rt, kind=kind, inner_slope=s0)


def vb_mond(R, V, eV, a0):
    go = V ** 2 / R * K
    gb = a0 * 10 ** np.interp(np.log10(np.clip(go / a0, 1e-30, None)), _LY, _LX)
    return np.sqrt(np.clip(gb / K * R, 0, None))


def vb_nfw1(R, V, eV, V200): return np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, cM(V200)) ** 2, 0, None))


def smc(pred, R, V, eV, p):
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
        sig = np.sqrt(smc(pred, R, V, eV, 10 ** best) ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)
        cur = np.array([c2(x, sig) for x in grid]); j = int(np.nanargmin(cur))
        best = minimize_scalar(c2, args=(sig,), bounds=(grid[max(j-1,0)], grid[min(j+1,159)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    cur = np.array([c2(x, sig) for x in grid]); C = c2(best, sig); N = len(R)
    ok = np.isfinite(cur); slab = cur[ok][cur[ok] <= C + 1.0]
    pD = max(np.mean(slab) - C, 0.0) if slab.size > 1 else 0.5
    return dict(param=10 ** best, chi2r=C / max(N-1, 1), AIC=C + 2, BIC=C + np.log(N), DIC=C + 2 * pD)


def fit_nfw2(R, V, eV, Vb):
    def c2(p, s):
        V200, c = 10 ** p[0], p[1]
        if not (10 <= V200 <= 600 and 1 <= c <= 40): return 1e12
        v = np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, c) ** 2, 0, None))
        return float(np.sum((v - Vb) ** 2 / s ** 2))
    gV = np.linspace(1.0, 2.78, 26); gc = np.linspace(1, 40, 22)
    s = np.maximum(eV, 1.0); best, bc = [2., 8.], np.inf
    for pv in gV:
        for c_ in gc:
            val = c2([pv, c_], s)
            if val < bc: bc, best = val, [pv, c_]
    for _ in range(2):
        V200, c = 10 ** best[0], best[1]
        out = [np.sqrt(np.clip((V + RNG.normal(0, eV)) ** 2 - nfw_v(R, V200, c) ** 2, 0, None)) for _ in range(N_MC)]
        out = [v for v in out if np.isfinite(v).all()]
        A = np.array(out) if len(out) >= 8 else None
        sg = 0.5 * (np.percentile(A, 84, 0) - np.percentile(A, 16, 0)) if A is not None else np.zeros(len(R))
        s = np.sqrt(sg ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)
        best = minimize(c2, best, args=(s,), method="Nelder-Mead",
                        options=dict(xatol=1e-3, fatol=1e-2, maxiter=250)).x
    C = c2(best, s); N = len(R)
    slab = []
    for dv in np.linspace(-0.15, 0.15, 7):
        for dc in np.linspace(-6, 6, 7):
            val = c2([best[0] + dv, best[1] + dc], s)
            if val <= C + 2.30: slab.append(val)
    pD = max(np.mean(slab) - C, 0.0) if len(slab) > 1 else 1.0
    return dict(param=10 ** best[0], c=best[1], chi2r=C / max(N-2, 1),
                AIC=C + 4, BIC=C + 2 * np.log(N), DIC=C + 2 * pD)


t1, t2 = io.load_table1(), io.load_table2()
frows, icrows, n2rows = [], [], []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 6:
        continue
    # measure fiducial inner slope ONCE
    sp0 = cc.smooth_vobs(R, V, eV, 0.1)
    s0 = cc.inner_slope_of(sp0, R)

    ccc_std = make_ccc("standard", s0)
    ccc_sim = make_ccc("simple", s0)
    ccc_shp = make_ccc("sharp", s0)
    try:
        r_std = fit1(ccc_std, R, V, eV, Vb, 3, 12)
        r_sim = fit1(ccc_sim, R, V, eV, Vb, 3, 12)
        r_shp = fit1(ccc_shp, R, V, eV, Vb, 3, 12)
        r_mond = fit1(vb_mond, R, V, eV, Vb, -12, -8)
        r_nfw1 = fit1(vb_nfw1, R, V, eV, Vb, np.log10(10), np.log10(600))
        r_nfw2 = fit_nfw2(R, V, eV, Vb)
    except Exception:
        continue

    T = int(t1.loc[g, "T"])
    frows.append(dict(Galaxy=g, T=T, dwarf=T >= 8, inner_slope=s0,
                      CCC_rhot=r_std["param"] * cc.RHO_CGS, MOND_a0=r_mond["param"],
                      NFW1_V200=r_nfw1["param"], NFW1_c=cM(r_nfw1["param"]),
                      NFW2_V200=r_nfw2["param"], NFW2_c=r_nfw2["c"]))
    ic = dict(Galaxy=g, N=len(R), T=T)
    for nm, r in [("CCC_sharp", r_shp), ("CCC_standard", r_std), ("CCC_simple", r_sim),
                  ("MOND_smooth", r_mond), ("NFW", r_nfw1)]:
        for kk in ["chi2r", "AIC", "BIC", "DIC", "param"]:
            ic[f"{nm}_{kk}"] = r.get(kk)
    icrows.append(ic)
    n2 = dict(Galaxy=g, N=len(R), T=T)
    for kk in ["chi2r", "AIC", "BIC", "DIC"]:
        n2[f"NFW2_{kk}"] = r_nfw2[kk]
    n2["NFW2_c"] = r_nfw2["c"]
    # also carry the k=1 CCC/MOND/NFW1 for the same galaxy set
    n2["CCC_chi2r"], n2["CCC_AIC"], n2["CCC_BIC"], n2["CCC_DIC"] = r_std["chi2r"], r_std["AIC"], r_std["BIC"], r_std["DIC"]
    n2["MOND_chi2r"], n2["MOND_AIC"], n2["MOND_BIC"], n2["MOND_DIC"] = r_mond["chi2r"], r_mond["AIC"], r_mond["BIC"], r_mond["DIC"]
    n2["NFW1_chi2r"], n2["NFW1_AIC"], n2["NFW1_BIC"], n2["NFW1_DIC"] = r_nfw1["chi2r"], r_nfw1["AIC"], r_nfw1["BIC"], r_nfw1["DIC"]
    n2rows.append(n2)

pd.DataFrame(frows).to_csv("/mnt/user-data/outputs/fit_parameters.csv", index=False)
pd.DataFrame(icrows).to_csv("/mnt/user-data/outputs/ic_smooth_ccc.csv", index=False)
pd.DataFrame(n2rows).to_csv("/mnt/user-data/outputs/ic_nfw2.csv", index=False)

df = pd.DataFrame(icrows)
print(f"{len(df)} galaxies, fixed-slope power-law, common error model\n")
print(f"{'model':14s}{'chi2/nu':>9s}{'AIC':>8s}{'BIC':>8s}{'DIC':>8s}")
for nm in ["CCC_standard", "CCC_simple", "CCC_sharp", "MOND_smooth", "NFW"]:
    print(f"{nm:14s}{df[nm+'_chi2r'].median():9.2f}{df[nm+'_AIC'].median():8.1f}"
          f"{df[nm+'_BIC'].median():8.1f}{df[nm+'_DIC'].median():8.1f}")
n2 = pd.DataFrame(n2rows)
print(f"{'NFW2':14s}{n2.NFW2_chi2r.median():9.2f}{n2.NFW2_AIC.median():8.1f}"
      f"{n2.NFW2_BIC.median():8.1f}{n2.NFW2_DIC.median():8.1f}")
m = df.CCC_standard_chi2r.notna() & df.MOND_smooth_chi2r.notna()
print(f"\nCCC beats MOND: chi2 {100*(df.CCC_standard_chi2r[m]<df.MOND_smooth_chi2r[m]).mean():.0f}%  "
      f"AIC {100*(df.CCC_standard_AIC[m]<df.MOND_smooth_AIC[m]).mean():.0f}%")
fp = pd.DataFrame(frows)
print(f"CCC rho_t median {fp.CCC_rhot.median()/1e-24:.2f}e-24, scatter {np.log10(fp.CCC_rhot).std():.3f} dex")
