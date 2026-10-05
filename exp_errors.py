"""
Experiment 3: what is the defensible error budget?

V_b and V_bX have *different and independent* error budgets:

  V_bX  is derived from V_obs, so it DOES inherit e_Vobs. The propagation is
        strongly non-linear (spline -> d/dR -> X -> re-integration), so it is
        done by Monte Carlo rather than first-order.

  V_b   is derived from Spitzer photometry and the HI surface density. It has
        no kinematic input at all, so giving it e_Vobs would be unjustifiable.
        Its dominant error is the M/L ratio, and that error is COHERENT across
        radii: one Upsilon shift moves the whole curve together. Treating it as
        independent per point invents constraints that do not exist.

So:  C = diag(sigma_MC^2 + floor^2)  +  sigma_ML sigma_ML^T
     chi2 = r^T C^-1 r
"""
import numpy as np, pandas as pd
import sparc_io as io, ccc_core as cc
from run_paper7 import ORDER

t2 = io.load_table2()
RNG = np.random.default_rng(12345)


def mc_sigma_vbx(R, V, eV, Vgas, Vdisk, Vbul, rho_t, s_factor=0.1, n_mc=200):
    """Scatter in V_bX induced by e_Vobs, holding rho_t fixed."""
    out = []
    for _ in range(n_mc):
        Vp = V + RNG.normal(0.0, eV)
        try:
            sp = cc.smooth_vobs(R, Vp, eV, s_factor=s_factor)
            rg, rho, dMdR, _ = cc.density_profile(sp, R)
            m = cc.ccc_model(rg, rho, dMdR, rho_t)
            if m is None:
                continue
            out.append(cc.vbx_at(rg, m["MbX"], R))
        except Exception:
            continue
    A = np.array(out)
    return A.std(axis=0), A.shape[0]


def chi2_cov(resid, sig_diag, sig_ml):
    C = np.diag(sig_diag**2) + np.outer(sig_ml, sig_ml)
    return float(resid @ np.linalg.solve(C, resid))


rows = []
for g in ORDER:
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vgas, Vdisk, Vbul = d.Vgas.values, d.Vdisk.values, d.Vbul.values

    fit = cc.fit_galaxy(R, V, eV, Vgas, Vdisk, Vbul, s_factor=0.1)
    Vb, VbX = fit["Vb"], fit["VbX"]
    sig_ml = cc.sigma_vbar(Vgas, Vdisk, Vbul)
    sig_mc, nok = mc_sigma_vbx(R, V, eV, Vgas, Vdisk, Vbul, fit["rho_t"])
    r = VbX - Vb
    dof = max(len(R) - 1, 1)

    naive = float(np.sum((r / np.sqrt(sig_ml**2 + (VbX / np.maximum(V, 1e-6) * eV)**2 + 1)) ** 2)) / dof
    full = chi2_cov(r, np.sqrt(sig_mc**2 + 1.0), sig_ml) / dof
    indep = float(np.sum((r / np.sqrt(sig_mc**2 + sig_ml**2 + 1)) ** 2)) / dof

    rows.append(dict(Galaxy=g, N=len(R),
                     med_sigMC=float(np.median(sig_mc)),
                     med_sigML=float(np.median(sig_ml)),
                     med_eVobs=float(np.median(eV)),
                     chi2_naive=naive, chi2_indep=indep, chi2_corr=full,
                     mc_ok=nok))

df = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:8.2f}")
print(df.to_string(index=False))
print("""
med_sigMC  : Monte Carlo scatter in V_bX from e_Vobs        (km/s)
med_sigML  : coherent 0.1 dex M/L error on V_b              (km/s)
chi2_naive : first-order propagation, all errors independent
chi2_indep : MC errors, still treating M/L as independent
chi2_corr  : MC errors + M/L as a fully correlated systematic  <-- defensible
""")
