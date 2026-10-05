"""
CCC+TL fitter, v2.

Error budget (answering "should V_b inherit e_Vobs?" -- no):

  V_bX  IS derived from V_obs, so it inherits e_Vobs. Propagated by Monte
        Carlo through spline -> d/dR -> X(R) -> re-integration, because the
        chain is strongly non-linear (first-order propagation underestimated
        sigma by up to 6x on sparse curves).

  V_b   is derived from Spitzer photometry + HI. It has NO kinematic input,
        so e_Vobs must not be attached to it. Its per-point error is taken as
        a 5% photometric/gas-profile term. Its dominant uncertainty, the M/L
        ratio, is COHERENT across radius -- so it is handled as an explicit
        nuisance parameter with a 0.1 dex prior, NOT as a correlated
        covariance term. Burying it in the covariance lets a ~30% overall
        normalisation offset be absorbed for ~free, which silently turns the
        goodness-of-fit into a shape-only test.

Two modes:
  ups_free=False  Upsilon fixed at 0.5/0.7  -- matches the published analysis
  ups_free=True   Upsilon scaling profiled with a 0.1 dex Gaussian prior
"""
import numpy as np
from scipy.optimize import minimize_scalar
import ccc_core as cc

S_FACTOR = 0.1
N_MC = 150
N_GRID = 1500
N_GRID_MC = 700
FLOOR = 1.0            # km/s
FRAC_B = 0.05          # per-point photometric/gas error on V_b
DEX_PRIOR = 0.10       # M/L prior width, dex


def _vb_scaled(Vgas, Vdisk, Vbul, u):
    """V_b with both M/L values scaled coherently by 10**u."""
    s = 10.0 ** u
    v2 = Vgas * np.abs(Vgas) + s * (cc.YD * Vdisk ** 2 + cc.YB * Vbul ** 2)
    return np.sqrt(np.clip(v2, 0.0, None))


def build_mc(R, V, eV, n_mc=N_MC, s_factor=S_FACTOR, seed=0):
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
        return None
    A = np.array(out)
    # robust scatter: the ensemble is heavy-tailed (a few realisations give
    # pathological rho_o), and the plain std is seed-unstable by ~3.6x.
    # The 16-84 percentile half-width is stable to ~15%.
    return 0.5 * (np.percentile(A, 84, axis=0) - np.percentile(A, 16, axis=0))


def fit(R, Vobs, eVobs, Vgas, Vdisk, Vbul, s_factor=S_FACTOR, n_mc=N_MC,
        seed=0, ups_free=False, require_resolved=False):
    Vb0 = cc.sparc_vbar(Vgas, Vdisk, Vbul)

    sp = cc.smooth_vobs(R, Vobs, eVobs, s_factor=s_factor)
    rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=N_GRID)

    ins = (rg >= R[0]) & (rg <= R[-1]) & (rho > 0)
    if ins.sum() < 3:
        return dict(flag="no_positive_density", rho_t=np.nan, npts=len(R))
    if require_resolved:
        lo, hi = np.log10(rho[ins].min()), np.log10(rho[ins].max())
    else:
        pos = rho[rho > 0]
        lo, hi = np.log10(pos.min() * 0.5), np.log10(pos.max() * 2.0)
    grid = np.linspace(lo, hi, 400)

    ens = build_mc(R, Vobs, eVobs, n_mc=n_mc, s_factor=s_factor, seed=seed)

    def vbx(rt):
        m = cc.ccc_model(rg, rho, dMdR, rt)
        return (None, None) if m is None else (m, cc.vbx_at(rg, m["MbX"], R))

    def chi2_at(logrt, sig_mod, want_u=False):
        m, Vm = vbx(10.0 ** logrt)
        if Vm is None:
            return (np.inf, 0.0) if want_u else np.inf

        def c2(u):
            Vb = _vb_scaled(Vgas, Vdisk, Vbul, u)
            s2 = sig_mod ** 2 + (FRAC_B * Vb) ** 2 + FLOOR ** 2
            return float(np.sum((Vm - Vb) ** 2 / s2)) + (u / DEX_PRIOR) ** 2

        if not ups_free:
            return (c2(0.0), 0.0) if want_u else c2(0.0)
        r = minimize_scalar(c2, bounds=(-0.4, 0.4), method="bounded",
                            options=dict(xatol=1e-4))
        return (r.fun, r.x) if want_u else r.fun

    # locate, then refine with Monte Carlo model errors
    sig_mod = np.maximum(eVobs, 1.0)
    best = grid[int(np.nanargmin([chi2_at(g, sig_mod) for g in grid]))]
    for _ in range(2):
        s = sigma_mc(ens, R, 10.0 ** best)
        if s is not None and np.isfinite(s).all():
            sig_mod = s
        c2s = np.array([chi2_at(g, sig_mod) for g in grid])
        j = int(np.nanargmin(c2s))
        a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
        best = minimize_scalar(chi2_at, args=(sig_mod,), bounds=(a, b),
                               method="bounded", options=dict(xatol=1e-4)).x

    rho_t = 10.0 ** best
    m, Vm = vbx(rho_t)
    chi2, u_best = chi2_at(best, sig_mod, want_u=True)
    dof = max(len(R) - (2 if ups_free else 1), 1)
    chi2r = chi2 / dof
    Vb = _vb_scaled(Vgas, Vdisk, Vbul, u_best)

    ok = np.isfinite(c2s)
    band = grid[ok][c2s[ok] <= chi2 + max(chi2r, 1.0)]
    rlo, rhi = (10.0 ** band.min(), 10.0 ** band.max()) if band.size else (np.nan,) * 2

    frms = float(np.sqrt(np.mean((Vm - Vb) ** 2)) / np.mean(Vb))
    power = float(np.median(sig_mod) / max(np.median(Vb0), 1e-6))

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

    return dict(rho_t=rho_t, rho_t_cgs=rho_t * cc.RHO_CGS,
                rho_t_lo=rlo, rho_t_hi=rhi, Rt=m["Rt"],
                chi2=chi2, dof=dof, chi2r=chi2r, frms=frms, power=power,
                ups_scale=10.0 ** u_best, u_best=u_best,
                clip_frac=m["clip_frac"], flag=",".join(flags) or "ok",
                Vb=Vb, Vb0=Vb0, VbX=Vm, sig_mod=sig_mod, R=R,
                rg=rg, rho=rho, X=m["X"], spline=sp, grid=grid, c2s=c2s,
                npts=len(R))
