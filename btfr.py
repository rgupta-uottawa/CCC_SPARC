"""
Baryonic Tully-Fisher Relation (BTFR) test.

Gupta (2025) predicts analytically that CCC satisfies the BTFR with slope b=4:
from rho_o = rho(1-X)^2 one has (1-X)^2 propto V^2 for R>R_t, and since the
gravitating baryonic mass carries (1-X)^4, M_B propto V^4.

We test this empirically for three models on equal footing. In the inverse
mode each model predicts a baryonic velocity curve V_b,pred(R) from V_obs. The
model's total baryonic mass is the enclosed mass implied by that curve at the
outermost measured radius,
    M_B^model = V_b,pred(R_max)^2 * R_max / G,
and we fit  log10 M_B = b * log10 V_flat + const  across the sample, per model.

For validation we also fit the SPARC photometric baryonic mass (the data BTFR),
whose slope is known to be ~3.5-4 (Lelli et al. 2016, McGaugh 2012).

We report, per model:
  - the global BTFR slope b (orthogonal-distance / robust fit),
  - the per-galaxy 'local' exponent b_i = log(M_B,i/<M_B>)/log(V_flat,i/<V_flat>)
    and its scatter around b=4, which is what the user asked to plot.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import theilslopes
import sparc_io as io, ccc_core as cc, compare_models as cm, smooth_ccc as sm

K = cm.KMS2_KPC_TO_MS2
G = cc.G
FLOOR, FRAC, N_MC, H0 = 1.0, 0.05, 40, 0.0671
RNG = np.random.default_rng(3)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def nfw_v(R, V200, c):
    R200 = V200 / (10 * H0); x = np.clip(R / R200, 1e-6, None)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    return np.sqrt(np.clip(V200 ** 2 * mu(c * x) / (x * mu(c)), 0, None))


def vb_ccc(R, V, eV, rt):
    s0=cc.inner_slope_of(cc.smooth_vobs(R,V,eV,0.1),R)
    return sm.vbx_smooth(R, V, eV, rt, kind="standard", inner_slope=s0)
def vb_mond(R, V, eV, a0):
    go = V ** 2 / R * K
    gb = a0 * 10 ** np.interp(np.log10(np.clip(go / a0, 1e-30, None)), _LY, _LX)
    return np.sqrt(np.clip(gb / K * R, 0, None))


def fit_ccc(R, V, eV, Vb):
    grid = np.linspace(3, 12, 100)
    s = np.sqrt((FRAC * Vb) ** 2 + FLOOR ** 2 + np.maximum(eV, 1.) ** 2)
    def c2(pl):
        v = vb_ccc(R, V, eV, 10 ** pl)
        return np.inf if (v is None or not np.isfinite(v).all()) else float(np.sum((v - Vb) ** 2 / s ** 2))
    b = grid[int(np.nanargmin([c2(x) for x in grid]))]
    b = minimize_scalar(c2, bounds=(3, 12), method="bounded", options=dict(xatol=1e-4)).x
    return vb_ccc(R, V, eV, 10 ** b)


def fit_mond(R, V, eV, Vb):
    s = np.sqrt((FRAC * Vb) ** 2 + FLOOR ** 2 + np.maximum(eV, 1.) ** 2)
    def c2(pl): return float(np.sum((vb_mond(R, V, eV, 10 ** pl) - Vb) ** 2 / s ** 2))
    b = minimize_scalar(c2, bounds=(-12, -8), method="bounded", options=dict(xatol=1e-4)).x
    return vb_mond(R, V, eV, 10 ** b)


def fit_nfw2(R, V, eV, Vb):
    s = np.sqrt((FRAC * Vb) ** 2 + FLOOR ** 2 + np.maximum(eV, 1.) ** 2)
    def c2(p):
        V200, c = 10 ** p[0], p[1]
        if not (10 <= V200 <= 600 and 1 <= c <= 40): return 1e12
        v = np.sqrt(np.clip(V ** 2 - nfw_v(R, V200, c) ** 2, 0, None))
        return float(np.sum((v - Vb) ** 2 / s ** 2))
    best, bc = [2., 8.], np.inf
    for pv in np.linspace(1, 2.78, 20):
        for c_ in np.linspace(1, 40, 16):
            val = c2([pv, c_])
            if val < bc: bc, best = val, [pv, c_]
    best = minimize(c2, best, method="Nelder-Mead", options=dict(maxiter=200)).x
    return np.sqrt(np.clip(V ** 2 - nfw_v(R, 10 ** best[0], best[1]) ** 2, 0, None))


def enclosed_mass(Vkms, Rkpc):
    """Baryonic mass enclosed at the outermost point, Msun."""
    return Vkms[-1] ** 2 * Rkpc[-1] / G


t1, t2 = io.load_table1(), io.load_table2()
rows = []
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    vf = t1.loc[g, "Vflat"]
    if len(R) < 6 or not np.isfinite(vf) or vf <= 0:
        continue
    try:
        vc = fit_ccc(R, V, eV, Vb)
        vm = fit_mond(R, V, eV, Vb)
        v2 = fit_nfw2(R, V, eV, Vb)
    except Exception:
        continue
    rec = dict(Galaxy=g, Vflat=vf, T=int(t1.loc[g, "T"]),
               MB_data=enclosed_mass(Vb, R),
               MB_ccc=enclosed_mass(vc, R),
               MB_mond=enclosed_mass(vm, R),
               MB_nfw=enclosed_mass(v2, R))
    rows.append(rec)

df = pd.DataFrame(rows)
df = df[(df.MB_ccc > 0) & (df.MB_mond > 0) & (df.MB_nfw > 0) & (df.MB_data > 0)]
df.to_csv("/mnt/user-data/outputs/btfr.csv", index=False)

lv = np.log10(df.Vflat.values)
print(f"BTFR: log10 M_B = b log10 V_flat + const   ({len(df)} galaxies)\n")
print(f"{'source':16s} {'slope b (Theil-Sen)':>22s} {'[95% CI]':>18s} {'scatter (dex)':>14s}")
results = {}
for lab, col in [("SPARC data", "MB_data"), ("CCC (smooth)", "MB_ccc"),
                 ("MOND (smooth)", "MB_mond"), ("NFW (2-param)", "MB_nfw")]:
    lm = np.log10(df[col].values)
    ts = theilslopes(lm, lv)
    b, lo, hi = ts[0], ts[2], ts[3]
    # scatter about the fit
    resid = lm - (b * lv + np.median(lm - b * lv))
    results[lab] = dict(b=b, lo=lo, hi=hi, scatter=np.std(resid))
    print(f"{lab:16s} {b:22.2f} {f'[{lo:.2f},{hi:.2f}]':>18s} {np.std(resid):14.3f}")

print("\nPer-galaxy local exponent b_i = log(M_B/median M_B)/log(V_flat/median V_flat):")
print("(distribution around the predicted b=4)")
mv = np.median(df.Vflat.values)
for lab, col in [("CCC", "MB_ccc"), ("MOND", "MB_mond"), ("NFW", "MB_nfw"), ("data", "MB_data")]:
    mm = np.median(df[col].values)
    bi = np.log10(df[col].values / mm) / np.log10(df.Vflat.values / mv)
    bi = bi[np.isfinite(bi) & (np.abs(np.log10(df.Vflat.values / mv)) > 0.05)]
    print(f"  {lab:6s} median b_i = {np.median(bi):+.2f}  16-84% [{np.percentile(bi,16):.1f}, {np.percentile(bi,84):.1f}]")
