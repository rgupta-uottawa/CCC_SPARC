"""
Final CCC+TL fitter.

Changes from the first pass:
  * s_factor = 0.1 (was 1.0) -- see exp_smoothing.py. At s=1 the spline
    residuals had lag-1 autocorrelation 0.4-0.8, i.e. systematic runs, the
    signature of over-smoothing. s=0.1 is the lightest smoothing at which
    the reconstructed d(rho)/dR is free of differentiation noise for every
    galaxy tested.
  * V_bX errors from Monte Carlo through the full non-linear pipeline
    instead of a first-order estimate.
  * V_b errors from the M/L ratio only (no kinematic input), treated as a
    fully correlated systematic via a covariance matrix.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import ccc_core as cc

S_FACTOR = 0.1
N_MC = 150
N_GRID = 1500
N_GRID_MC = 700
FLOOR = 1.0          # km/s numerical floor


def build_mc(R, V, eV, n_mc=N_MC, s_factor=S_FACTOR, seed=0):
    """Ensemble of density reconstructions from V_obs perturbed by e_Vobs."""
    rng = np.random.default_rng(seed)
    ens = []
    for _ in range(n_mc):
        try:
            sp = cc.smooth_vobs(R, V + rng.normal(0.0, eV), eV, s_factor=s_factor)
            rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=N_GRID_MC)
            ens.append((rg, rho, dMdR))
        except Exception:
            continue
    return ens


def sigma_mc(ens, R, rho_t):
    out = []
    for rg, rho, dMdR in ens:
        m = cc.ccc_model(rg, rho, dMdR, rho_t)
        if m is not None:
            out.append(cc.vbx_at(rg, m["MbX"], R))
    if len(out) < 10:
        return np.full(len(R), np.nan), 0
    A = np.array(out)
    return A.std(axis=0), A.shape[0]


def chi2_cov(resid, sig_diag, sig_ml):
    C = np.diag(sig_diag ** 2) + np.outer(sig_ml, sig_ml)
    try:
        return float(resid @ np.linalg.solve(C, resid))
    except np.linalg.LinAlgError:
        return np.inf


def fit(R, Vobs, eVobs, Vgas, Vdisk, Vbul, s_factor=S_FACTOR, n_mc=N_MC,
        seed=0, require_resolved=True):
    Vb = cc.sparc_vbar(Vgas, Vdisk, Vbul)
    sig_ml = cc.sigma_vbar(Vgas, Vdisk, Vbul)

    sp = cc.smooth_vobs(R, Vobs, eVobs, s_factor=s_factor)
    rg, rho, dMdR, Mo = cc.density_profile(sp, R, n_grid=N_GRID)

    if require_resolved:
        ins = (rg >= R[0]) & (rg <= R[-1]) & (rho > 0)
        if ins.sum() < 3:
            return dict(flag="no_positive_density", rho_t=np.nan, npts=len(R))
        lo, hi = np.log10(rho[ins].min()), np.log10(rho[ins].max())
    else:
        pos = rho[rho > 0]
        lo, hi = np.log10(pos.min() * 0.5), np.log10(pos.max() * 2.0)
    grid = np.linspace(lo, hi, 400)

    ens = build_mc(R, Vobs, eVobs, n_mc=n_mc, s_factor=s_factor, seed=seed)

    def predict(rt):
        m = cc.ccc_model(rg, rho, dMdR, rt)
        return (None, None) if m is None else (m, cc.vbx_at(rg, m["MbX"], R))

    def objective(logrt, sig_d):
        m, Vm = predict(10.0 ** logrt)
        if Vm is None:
            return np.inf
        return chi2_cov(Vm - Vb, sig_d, sig_ml)

    # ---- pass 1: crude diagonal errors, just to locate the region
    sig_d = np.sqrt(eVobs ** 2 + FLOOR ** 2)
    c2 = np.array([objective(g, sig_d) for g in grid])
    best = grid[int(np.nanargmin(c2))]

    # ---- pass 2/3: Monte Carlo errors evaluated at the running best fit
    for _ in range(2):
        smc, nok = sigma_mc(ens, R, 10.0 ** best)
        if not np.isfinite(smc).all():
            smc = np.full(len(R), np.nanmedian(smc) if np.isfinite(smc).any() else 5.0)
        sig_d = np.sqrt(smc ** 2 + FLOOR ** 2)
        c2 = np.array([objective(g, sig_d) for g in grid])
        j = int(np.nanargmin(c2))
        a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
        r = minimize_scalar(objective, args=(sig_d,), bounds=(a, b),
                            method="bounded", options=dict(xatol=1e-4))
        best = r.x

    rho_t = 10.0 ** best
    m, Vm = predict(rho_t)
    chi2 = chi2_cov(Vm - Vb, sig_d, sig_ml)
    dof = max(len(R) - 1, 1)
    chi2r = chi2 / dof

    band = grid[np.isfinite(c2)][c2[np.isfinite(c2)] <= chi2 + max(chi2r, 1.0)]
    rlo, rhi = (10.0 ** band.min(), 10.0 ** band.max()) if band.size else (np.nan,) * 2

    frms = float(np.sqrt(np.mean((Vm - Vb) ** 2)) / np.mean(Vb))
    # constraining power: how tightly the data pin the prediction
    power = float(np.median(sig_d) / max(np.median(Vb), 1e-6))

    flags = []
    if abs(best - hi) < 3e-3 or m["Rt"] <= R[0] * 1.001:
        flags.append("Rt_unresolved")
    if abs(best - lo) < 3e-3 or m["Rt"] >= R[-1] * 0.999:
        flags.append("Rt_at_outer_edge")
    if m["clip_frac"] > 0.10:
        flags.append("X_clipped")
    if power > 0.30:
        flags.append("uninformative")
    if chi2r > 3.0:
        flags.append("poor_chi2")
    if len(R) < 6:
        flags.append("few_points")

    return dict(rho_t=rho_t, rho_t_lo=rlo, rho_t_hi=rhi, Rt=m["Rt"],
                chi2=chi2, dof=dof, chi2r=chi2r, frms=frms, power=power,
                clip_frac=m["clip_frac"], flag=",".join(flags) or "ok",
                Vb=Vb, VbX=Vm, sig_ml=sig_ml, sig_mc=sig_d, R=R,
                rg=rg, rho=rho, X=m["X"], MbX=m["MbX"], spline=sp,
                grid=grid, c2s=c2, npts=len(R),
                rho_t_cgs=rho_t * cc.RHO_CGS)
