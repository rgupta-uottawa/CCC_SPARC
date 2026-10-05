"""
Inverse-mode comparison, mirroring the published CCC methodology.

Every model runs BACKWARDS: take the observed rotation curve, strip out the
model's mass-modification, and predict what the baryonic curve should be.
Compare that against SPARC's photometric V_b.

  CCC   V_obs -> rho_o -> X = 1 - sqrt(rho_o/rho_t) -> M_bX -> V_bX
        free parameter rho_t
  MOND  g_obs = V_obs^2/R, invert the RAR for g_N, V_b,pred = sqrt(g_N R)
        free parameter a0
  NFW   V_b,pred^2 = V_obs^2 - V_NFW(R; V200)^2
        free parameter V200 (concentration from Dutton & Maccio)

Identical error treatment for all three: e_Vobs propagated by Monte Carlo
through each model's own inverse chain (16-84 percentile half-width), plus a
5% photometric term on V_b and a 1 km/s floor.

fRMS is also reported because, unlike chi2, it does not depend on the error
model at all -- important here, because CCC's inverse route differentiates the
data and therefore amplifies noise more than the pointwise MOND/NFW inversions.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, brentq
import sparc_io as io, ccc_core as cc, compare_models as cm

N_MC = 80
FRAC_B, FLOOR = 0.05, 1.0
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- inversions
# The RAR map x = g_N/a0  ->  y = g_obs/a0 = x/(1-exp(-sqrt(x))) is scale-free,
# so it can be inverted once by table lookup instead of root-finding per point.
_LX = np.linspace(-12, 12, 24001)
_LY = np.log10(10.0 ** _LX / (1.0 - np.exp(-np.sqrt(10.0 ** _LX))))


def inv_mond(R, Vobs, a0):
    """Invert the RAR: given g_obs, return the baryonic V_b."""
    gobs = np.clip(Vobs ** 2 / R * cm.KMS2_KPC_TO_MS2, 1e-30, None)
    x = 10.0 ** np.interp(np.log10(gobs / a0), _LY, _LX)
    gN = x * a0
    return np.sqrt(np.clip(gN / cm.KMS2_KPC_TO_MS2 * R, 0, None))


def inv_nfw(R, Vobs, V200):
    return np.sqrt(np.clip(Vobs ** 2 - cm.nfw_v(R, V200) ** 2, 0.0, None))


def inv_ccc(R, Vobs, eVobs, rho_t, s_factor=0.1):
    sp = cc.smooth_vobs(R, Vobs, eVobs, s_factor=s_factor)
    rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=1200)
    m = cc.ccc_model(rg, rho, dMdR, rho_t)
    if m is None:
        return None
    return cc.vbx_at(rg, m["MbX"], R)


# ------------------------------------------------------------------ fitting
def mc_sigma(predict, R, Vobs, eVobs, n_mc=N_MC):
    out = []
    for _ in range(n_mc):
        try:
            v = predict(Vobs + RNG.normal(0, eVobs))
            if v is not None and np.isfinite(v).all():
                out.append(v)
        except Exception:
            pass
    if len(out) < 10:
        return np.full(len(R), 5.0)
    A = np.array(out)
    return 0.5 * (np.percentile(A, 84, axis=0) - np.percentile(A, 16, axis=0))


def fit_inverse(R, Vobs, eVobs, Vb, predict_at, lo, hi):
    """predict_at(param, Vobs_array) -> V_b,pred"""
    dof = max(len(R) - 1, 1)

    def chi2(p, sig):
        v = predict_at(10 ** p, Vobs)
        if v is None or not np.isfinite(v).all():
            return np.inf
        s2 = sig ** 2 + (FRAC_B * Vb) ** 2 + FLOOR ** 2
        return float(np.sum((v - Vb) ** 2 / s2))

    sig = np.maximum(eVobs, 1.0)
    grid = np.linspace(lo, hi, 200)
    best = grid[int(np.nanargmin([chi2(g, sig) for g in grid]))]
    for _ in range(2):
        sig = mc_sigma(lambda vv: predict_at(10 ** best, vv), R, Vobs, eVobs)
        vals = np.array([chi2(g, sig) for g in grid])
        j = int(np.nanargmin(vals))
        best = minimize_scalar(chi2, args=(sig,),
                               bounds=(grid[max(j - 1, 0)], grid[min(j + 1, 199)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    v = predict_at(10 ** best, Vobs)
    c2 = chi2(best, sig)
    frms = float(np.sqrt(np.mean((v - Vb) ** 2)) / max(np.mean(Vb), 1e-6))
    return 10 ** best, c2 / dof, frms, float(np.median(sig))


if __name__ == "__main__":
    t1, t2 = io.load_table1(), io.load_table2()
    rows = []
    for g in sorted(t2.Galaxy.unique()):
        d = t2[t2.Galaxy == g].sort_values("R")
        R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
        Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
        rec = dict(Galaxy=g, N=len(R), T=int(t1.loc[g, "T"]))
        try:
            p, c, f, s = fit_inverse(R, V, eV, Vb,
                                     lambda rt, vv: inv_ccc(R, vv, eV, rt),
                                     np.log10(1e3), np.log10(1e12))
            rec.update(ccc_rho_t=p * cc.RHO_CGS, ccc_chi2r=c, ccc_frms=f, ccc_sig=s)
        except Exception:
            pass
        try:
            p, c, f, s = fit_inverse(R, V, eV, Vb,
                                     lambda a0, vv: inv_mond(R, vv, a0), -12.0, -8.0)
            rec.update(mond_a0=p, mond_chi2r=c, mond_frms=f, mond_sig=s)
        except Exception:
            pass
        try:
            p, c, f, s = fit_inverse(R, V, eV, Vb,
                                     lambda v2, vv: inv_nfw(R, vv, v2),
                                     np.log10(10.0), np.log10(600.0))
            rec.update(nfw_V200=p, nfw_chi2r=c, nfw_frms=f, nfw_sig=s)
        except Exception:
            pass
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["dwarf"] = df["T"] >= 8
    df.to_csv("/mnt/user-data/outputs/ccc_inverse_comparison.csv", index=False)

    print("INVERSE MODE: predict V_b from V_obs, compare against SPARC photometric V_b")
    print("(this is the direction the published CCC analysis uses)\n")
    print(f"{'model':22s} {'med chi2/nu':>12s} {'<3':>6s} {'med fRMS':>9s} {'med sigma':>10s}")
    for lab, k in [("CCC  (rho_t free)", "ccc"), ("MOND (a0 free)", "mond"),
                   ("NFW  (V200 free)", "nfw")]:
        c, f, s = df[k + "_chi2r"].dropna(), df[k + "_frms"].dropna(), df[k + "_sig"].dropna()
        print(f"{lab:22s} {c.median():12.2f} {100*(c<3).mean():5.0f}% "
              f"{100*f.median():8.0f}% {s.median():9.1f} km/s")
    print("\nhead-to-head on fRMS (error-model independent):")
    for a, b in [("ccc", "mond"), ("ccc", "nfw"), ("mond", "nfw")]:
        m = df[a + "_frms"].notna() & df[b + "_frms"].notna()
        print(f"   {a.upper():5s} beats {b.upper():5s} in {100*(df[a+'_frms'][m] < df[b+'_frms'][m]).mean():4.0f}% of galaxies")
