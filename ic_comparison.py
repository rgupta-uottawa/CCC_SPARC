"""
Matched inverse-mode model comparison with information criteria.

Every model predicts the BARYON velocity V_b from the observed V_obs, and is
scored against SPARC's photometric V_b in identical units (km/s), with an
identical error model and identical dof. One free parameter each (k=1).

Models
  CCC        rho_t   V_obs -> rho_o -> X=1-sqrt(rho_o/rho_t) -> M_bX -> V_bX
  MOND_smooth a0     g_obs -> RAR^-1 -> g_bar -> V_b
  MOND_sharp  g_t    g_obs -> sharp^-1 -> g_bar -> V_b   (CCC-twin sharpness)
  NFW        V200    V_b = sqrt(V_obs^2 - V_nfw^2)

Error model (identical for all): sigma_i^2 = sigma_MC_i^2 + (0.05 V_b,i)^2 + 1
  sigma_MC from propagating e_Vobs through each model's own inverse chain
  (Monte Carlo, 16-84 percentile half-width -- the robust estimator).

Criteria per galaxy (lower is better):
  chi2/nu = chi2 / (N-1)
  AIC     = chi2 + 2k
  BIC     = chi2 + k ln N
  DIC     = chi2(theta_hat) + 2 p_D,  p_D = mean(chi2) - chi2(theta_hat)
            over the 1-sigma posterior slab in the single parameter.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import sparc_io as io, ccc_core as cc, compare_models as cm

K = cm.KMS2_KPC_TO_MS2
FLOOR, FRAC = 1.0, 0.05
N_MC = 100
RNG = np.random.default_rng(3)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


# ---- inverse predictors: return predicted V_b (km/s) from V_obs
def vb_ccc(R, Vobs, eV, rho_t):
    sp = cc.smooth_vobs(R, Vobs, eV, s_factor=0.1)
    rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=1200)
    m = cc.ccc_model(rg, rho, dMdR, rho_t)
    return None if m is None else cc.vbx_at(rg, m["MbX"], R)


def vb_from_gbar(gbar, R):
    return np.sqrt(np.clip(gbar / K * R, 0, None))


def vb_mond_smooth(R, Vobs, eV, a0):
    gobs = Vobs ** 2 / R * K
    gbar = a0 * 10 ** np.interp(np.log10(np.clip(gobs / a0, 1e-30, None)), _LY, _LX)
    return vb_from_gbar(gbar, R)


def vb_mond_sharp(R, Vobs, eV, gt):
    gobs = Vobs ** 2 / R * K
    gbar = np.where(gobs > gt, gobs, gobs ** 2 / gt)
    return vb_from_gbar(gbar, R)


def vb_nfw(R, Vobs, eV, V200):
    return np.sqrt(np.clip(Vobs ** 2 - cm.nfw_v(R, V200) ** 2, 0, None))


PRED = {"CCC": (vb_ccc, np.log10(1e3), np.log10(1e12)),
        "MOND_smooth": (vb_mond_smooth, -12, -8),
        "MOND_sharp": (vb_mond_sharp, -12, -8),
        "NFW": (vb_nfw, np.log10(10), np.log10(600))}


def sigma_mc(pred, R, Vobs, eV, p):
    out = []
    for _ in range(N_MC):
        v = pred(R, Vobs + RNG.normal(0, eV), eV, p)
        if v is not None and np.isfinite(v).all():
            out.append(v)
    if len(out) < 8:
        return np.full(len(R), 5.0)
    A = np.array(out)
    return 0.5 * (np.percentile(A, 84, 0) - np.percentile(A, 16, 0))


def fit_model(name, R, Vobs, eV, Vb):
    pred, lo, hi = PRED[name]
    grid = np.linspace(lo, hi, 160)

    def chi2(p_log, sig):
        v = pred(R, Vobs, eV, 10 ** p_log)
        if v is None or not np.isfinite(v).all():
            return np.inf
        return float(np.sum((v - Vb) ** 2 / sig ** 2))

    sig = np.maximum(eV, 1.0)
    best = grid[int(np.nanargmin([chi2(g, sig) for g in grid]))]
    for _ in range(2):
        sig = np.sqrt(sigma_mc(pred, R, Vobs, eV, 10 ** best) ** 2 + (FRAC * Vb) ** 2 + FLOOR ** 2)
        curve = np.array([chi2(g, sig) for g in grid])
        j = int(np.nanargmin(curve))
        best = minimize_scalar(chi2, args=(sig,), bounds=(grid[max(j-1,0)], grid[min(j+1,159)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    curve = np.array([chi2(g, sig) for g in grid])
    c2 = chi2(best, sig)
    N = len(R); dof = max(N - 1, 1)
    # DIC over the 1-sigma slab (delta chi2 <= 1)
    ok = np.isfinite(curve)
    slab = curve[ok][curve[ok] <= c2 + 1.0]
    pD = max(np.mean(slab) - c2, 0.0) if slab.size > 1 else 0.5
    return dict(param=10 ** best, chi2=c2, chi2r=c2 / dof, N=N,
                AIC=c2 + 2, BIC=c2 + np.log(N), DIC=c2 + 2 * pD)


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
            r = fit_model(name, R, V, eV, Vb)
            for kk in ["chi2r", "AIC", "BIC", "DIC", "param"]:
                rec[f"{name}_{kk}"] = r[kk]
        except Exception:
            pass
    rows.append(rec)

df = pd.DataFrame(rows)
df["dwarf"] = df["T"] >= 8
df.to_csv("/mnt/user-data/outputs/ic_comparison.csv", index=False)

print(f"{df.shape[0]} galaxies, inverse mode, one parameter each, identical error model\n")
print(f"{'model':14s} {'med chi2/nu':>12s} {'med AIC':>9s} {'med BIC':>9s} {'med DIC':>9s}")
for name in PRED:
    print(f"{name:14s} {df[f'{name}_chi2r'].median():12.2f} {df[f'{name}_AIC'].median():9.1f} "
          f"{df[f'{name}_BIC'].median():9.1f} {df[f'{name}_DIC'].median():9.1f}")

print("\nhead-to-head chi2/nu (fraction of galaxies where row < column):")
mods = list(PRED)
print("         " + "".join(f"{m:>13s}" for m in mods))
for a in mods:
    line = f"{a:14s}"
    for b in mods:
        if a == b:
            line += f"{'-':>13s}"
        else:
            m = df[f"{a}_chi2r"].notna() & df[f"{b}_chi2r"].notna()
            line += f"{100*(df[f'{a}_chi2r'][m] < df[f'{b}_chi2r'][m]).mean():12.0f}%"
    print(line)

print("\nCCC vs MOND_smooth by information criterion (fraction CCC preferred = lower):")
for ic in ["AIC", "BIC", "DIC"]:
    m = df[f"CCC_{ic}"].notna() & df[f"MOND_smooth_{ic}"].notna()
    print(f"  {ic}: CCC lower in {100*(df[f'CCC_{ic}'][m] < df[f'MOND_smooth_{ic}'][m]).mean():.0f}% of galaxies")

print("\nsum over sample (total AIC/BIC/DIC, lower wins):")
for ic in ["AIC", "BIC", "DIC"]:
    tot = {name: df[f"{name}_{ic}"].sum() for name in PRED}
    best = min(tot, key=tot.get)
    print(f"  {ic}: " + "  ".join(f"{k}={v:.0f}" for k, v in tot.items()) + f"   -> best: {best}")
