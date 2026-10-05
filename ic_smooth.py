import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import sparc_io as io, ccc_core as cc, compare_models as cm, smooth_ccc as sm

K = cm.KMS2_KPC_TO_MS2
FLOOR, FRAC, N_MC = 1.0, 0.05, 100
RNG = np.random.default_rng(3)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def vb_ccc_sharp(R, V, eV, rt): return sm.vbx_smooth(R, V, eV, rt, kind="sharp")
def vb_ccc_std(R, V, eV, rt): return sm.vbx_smooth(R, V, eV, rt, kind="standard")
def vb_ccc_simple(R, V, eV, rt): return sm.vbx_smooth(R, V, eV, rt, kind="simple")
def vb_from_gbar(gb, R): return np.sqrt(np.clip(gb / K * R, 0, None))


def vb_mond_smooth(R, V, eV, a0):
    go = V ** 2 / R * K
    gb = a0 * 10 ** np.interp(np.log10(np.clip(go / a0, 1e-30, None)), _LY, _LX)
    return vb_from_gbar(gb, R)


def vb_nfw(R, V, eV, v200):
    return np.sqrt(np.clip(V ** 2 - cm.nfw_v(R, v200) ** 2, 0, None))


PRED = {"CCC_sharp": (vb_ccc_sharp, 3, 12), "CCC_standard": (vb_ccc_std, 3, 12),
        "CCC_simple": (vb_ccc_simple, 3, 12), "MOND_smooth": (vb_mond_smooth, -12, -8),
        "NFW": (vb_nfw, np.log10(10), np.log10(600))}


def smc(pred, R, V, eV, p):
    out = []
    for _ in range(N_MC):
        v = pred(R, V + RNG.normal(0, eV), eV, p)
        if v is not None and np.isfinite(v).all():
            out.append(v)
    if len(out) < 8:
        return np.full(len(R), 5.0)
    A = np.array(out)
    return 0.5 * (np.percentile(A, 84, 0) - np.percentile(A, 16, 0))


def fit(name, R, V, eV, Vb):
    pred, lo, hi = PRED[name]
    grid = np.linspace(lo, hi, 160)

    def c2(pl, sig):
        v = pred(R, V, eV, 10 ** pl)
        if v is None or not np.isfinite(v).all():
            return np.inf
        return float(np.sum((v - Vb) ** 2 / sig ** 2))

    sig = np.maximum(eV, 1.0)
    best = grid[int(np.nanargmin([c2(x, sig) for x in grid]))]
    for _ in range(2):
        sig = np.sqrt(smc(pred, R, V, eV, 10 ** best) ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)
        cur = np.array([c2(x, sig) for x in grid])
        j = int(np.nanargmin(cur))
        best = minimize_scalar(c2, args=(sig,), bounds=(grid[max(j-1,0)], grid[min(j+1,159)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    cur = np.array([c2(x, sig) for x in grid])
    C = c2(best, sig); N = len(R); dof = max(N - 1, 1)
    ok = np.isfinite(cur); slab = cur[ok][cur[ok] <= C + 1.0]
    pD = max(np.mean(slab) - C, 0.0) if slab.size > 1 else 0.5
    return dict(chi2r=C / dof, AIC=C + 2, BIC=C + np.log(N), DIC=C + 2 * pD, param=10 ** best)


t1, t2 = io.load_table1(), io.load_table2()
rows = []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 5:
        continue
    rec = dict(Galaxy=g, N=len(R), T=int(t1.loc[g, "T"]))
    for name in PRED:
        try:
            r = fit(name, R, V, eV, Vb)
            for kk in ["chi2r", "AIC", "BIC", "DIC", "param"]:
                rec[f"{name}_{kk}"] = r[kk]
        except Exception:
            pass
    rows.append(rec)

df = pd.DataFrame(rows)
df["dwarf"] = df["T"] >= 8
df.to_csv("/mnt/user-data/outputs/ic_smooth_ccc.csv", index=False)

print(f"{df.shape[0]} galaxies, inverse mode, one parameter each, identical errors\n")
print(f"{'model':14s}{'chi2/nu':>9s}{'AIC':>8s}{'BIC':>8s}{'DIC':>8s}")
for name in PRED:
    print(f"{name:14s}{df[name+'_chi2r'].median():9.2f}{df[name+'_AIC'].median():8.1f}"
          f"{df[name+'_BIC'].median():8.1f}{df[name+'_DIC'].median():8.1f}")

print()
for a in ["CCC_standard", "CCC_simple", "CCC_sharp"]:
    m = df[a+"_chi2r"].notna() & df["MOND_smooth_chi2r"].notna()
    print(f"{a:14s} beats MOND_smooth: chi2/nu {100*(df[a+'_chi2r'][m]<df['MOND_smooth_chi2r'][m]).mean():3.0f}%"
          f"   AIC {100*(df[a+'_AIC'][m]<df['MOND_smooth_AIC'][m]).mean():3.0f}%"
          f"   DIC {100*(df[a+'_DIC'][m]<df['MOND_smooth_DIC'][m]).mean():3.0f}%")

print()
for ic in ["AIC", "BIC", "DIC"]:
    tot = {n: df[n+"_"+ic].sum() for n in PRED}
    best = min(tot, key=tot.get)
    print(f"sum {ic}: " + "  ".join(f"{k}={v:.0f}" for k, v in tot.items()) + f"  -> best: {best}")

print()
for name in ["CCC_standard", "CCC_simple"]:
    r = df[name+"_param"].dropna() * cc.RHO_CGS / 1e-24
    rs = (df["CCC_sharp_param"].dropna() * cc.RHO_CGS / 1e-24)
    print(f"{name} rho_t median {r.median():.2f}   sharp median {rs.median():.2f}   ratio {r.median()/rs.median():.2f}")
