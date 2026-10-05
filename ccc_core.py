"""
CCC+TL galaxy rotation curve pipeline (Gupta 2025, Galaxies 13, 108).

Method
------
Spherical approximation:            G M_o(R) = V_o(R)^2 R
Total ("observed") density:         4 pi rho_o(R) = dM_o / (R^2 dR)
Inside turn-off radius R_t:         X = 0,  rho = rho_o
Outside R_t:                        rho = rho_t (const) and
                                    rho_o = rho_t (1-X)^2  =>  X = 1 - sqrt(rho_o/rho_t)
Baryon density:                     rho_bX = rho (1-X)^4 = rho_o (1-X)^2
Baryon mass:                        M_bX(R) = Int_0^R (1-X)^2 dM_o
Model prediction:                   V_bX = sqrt(G M_bX / R)

Compared against the SPARC baryonic curve
    V_b = sqrt( Vgas|Vgas| + Yd*Vdisk^2 + Yb*Vbul^2 ),  Yd=0.5, Yb=0.7
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------- constants
G = 4.300917270e-6          # kpc (km/s)^2 / Msun
MSUN_G = 1.98892e33         # g
KPC_CM = 3.0856775814913673e21
RHO_CGS = MSUN_G / KPC_CM**3   # (Msun/kpc^3) -> g/cm^3   = 6.7697e-32

YD, YB = 0.5, 0.7           # nominal [3.6] mass-to-light ratios
DEX_ML = 0.10               # 0.1 dex M/L uncertainty (Schombert+2022)


def sparc_vbar(vgas, vdisk, vbul, yd=YD, yb=YB):
    """SPARC baryonic curve; Vgas sign preserved (negative = inward)."""
    v2 = vgas * np.abs(vgas) + yd * vdisk**2 + yb * vbul**2
    return np.sqrt(np.clip(v2, 0.0, None))


def sigma_vbar(vgas, vdisk, vbul, yd=YD, yb=YB, dex=DEX_ML):
    """Uncertainty on V_b from a coherent `dex` shift of both M/L values."""
    vb = sparc_vbar(vgas, vdisk, vbul, yd, yb)
    d_v2 = dex * np.log(10.0) * (yd * vdisk**2 + yb * vbul**2)
    return np.where(vb > 0, d_v2 / (2.0 * np.maximum(vb, 1e-6)), 0.0)


# ---------------------------------------------------------------- smoothing
def smooth_vobs(R, V, eV, s_factor=1.0, k=3):
    """
    Error-weighted smoothing spline for V_obs(R).
    UnivariateSpline with w = 1/sigma satisfies sum((r_i/sigma_i)^2) <= s,
    so s = s_factor * N targets chi2/N ~ s_factor.
    """
    n = len(R)
    kk = min(k, n - 1)
    w = 1.0 / np.maximum(eV, 1e-3)
    sp = UnivariateSpline(R, V, w=w, k=kk, s=s_factor * n)
    return sp


def density_profile(sp, R, n_grid=2000, inner="powerlaw", inner_slope=None):
    """
    Dense grid of rho_o(R) and dM_o/dR from the smoothed V_o(R).

    For r < R_min the density is extrapolated inward. Two prescriptions:
      inner='powerlaw' (default, fiducial): the reconstructed density's
          log-log slope at R_min is continued inward as a power law,
          rho(r) = rho(R_min) (r/R_min)^s, clipped to s > -2.8 for a convergent
          inner mass. This continues the observed profile smoothly (no knee).
          The slope s is a structural property of the galaxy: pass it via
          `inner_slope` (measured once from the fiducial reconstruction) so that
          it stays FIXED during Monte-Carlo error propagation and does not act as
          a noise amplifier. If inner_slope is None it is measured from the
          current profile (used only for the one fiducial call).
      inner='solidbody': a solid-body inner curve V = V_min r/R_min, giving a
          constant inner density (retained for comparison only).
    """
    Rmin, Rmax = R[0], R[-1]
    rg = np.linspace(Rmin, Rmax, n_grid)
    Vg = sp(rg)
    dV = sp.derivative()(rg)

    dMdR = (Vg**2 + 2.0 * rg * Vg * dV) / G          # Msun/kpc
    rho = dMdR / (4.0 * np.pi * rg**2)               # Msun/kpc^3

    n_in = max(int(n_grid * 0.15), 60)
    rin = np.linspace(1e-4, Rmin, n_in, endpoint=False)
    if inner == "solidbody":
        Vmin = float(sp(Rmin))
        rho_core = 3.0 * Vmin**2 / (4.0 * np.pi * G * Rmin**2)
        rho_in = np.full_like(rin, rho_core)
        dMdR_in = 3.0 * Vmin**2 * rin**2 / (G * Rmin**2)
        rg = np.concatenate([rin, rg])
        rho = np.concatenate([rho_in, rho])
        dMdR = np.concatenate([dMdR_in, dMdR])
    elif inner == "powerlaw":
        if inner_slope is None:
            m = min(30, n_grid)
            s = np.polyfit(np.log(rg[:m]), np.log(np.maximum(rho[:m], 1e-40)), 1)[0]
            s = float(np.clip(s, -2.8, 0.5))
        else:
            s = float(np.clip(inner_slope, -2.8, 0.5))
        rho_in = rho[0] * (rin / Rmin) ** s
        dMdR_in = 4.0 * np.pi * rin**2 * rho_in
        rg = np.concatenate([rin, rg])
        rho = np.concatenate([rho_in, rho])
        dMdR = np.concatenate([dMdR_in, dMdR])
    # inner='none' leaves the profile starting at R_min

    Mo = np.concatenate([[0.0], np.cumsum(np.diff(rg) * 0.5 * (dMdR[1:] + dMdR[:-1]))])
    return rg, rho, dMdR, Mo


def inner_slope_of(sp, R, n_grid=1500):
    """Measure the fiducial inner log-log density slope once, for fixing during MC."""
    Rmin, Rmax = R[0], R[-1]
    rg = np.linspace(Rmin, Rmax, n_grid)
    Vg = sp(rg); dV = sp.derivative()(rg)
    rho = (Vg**2 + 2.0 * rg * Vg * dV) / G / (4.0 * np.pi * rg**2)
    m = min(30, n_grid)
    s = np.polyfit(np.log(rg[:m]), np.log(np.maximum(rho[:m], 1e-40)), 1)[0]
    return float(np.clip(s, -2.8, 0.5))


# ---------------------------------------------------------------- CCC model
def ccc_model(rg, rho, dMdR, rho_t):
    """
    Given a turn-off density rho_t (Msun/kpc^3) return R_t, X(R), M_bX(R).
    """
    below = rho < rho_t
    if not below.any():
        return None                                   # rho_t below whole profile
    i = int(np.argmax(below))                         # first crossing
    if i == 0:
        Rt = rg[0]
    else:
        r0, r1 = rg[i - 1], rg[i]
        d0, d1 = rho[i - 1] - rho_t, rho[i] - rho_t
        Rt = r0 + (r1 - r0) * d0 / (d0 - d1)

    X = np.zeros_like(rg)
    out = rg > Rt
    ratio = np.clip(rho[out] / rho_t, 0.0, None)
    X[out] = np.clip(1.0 - np.sqrt(ratio), 0.0, 1.0)

    # fraction of the outer range where X had to be clipped at 0
    raw = 1.0 - np.sqrt(ratio)
    clip_frac = float(np.mean(raw < 0)) if raw.size else 0.0

    integ = (1.0 - X) ** 2 * dMdR
    MbX = np.concatenate([[0.0], np.cumsum(np.diff(rg) * 0.5 * (integ[1:] + integ[:-1]))])
    return dict(Rt=Rt, X=X, MbX=MbX, clip_frac=clip_frac)


def vbx_at(rg, MbX, Rq):
    M = np.interp(Rq, rg, MbX)
    return np.sqrt(G * M / Rq)


# ---------------------------------------------------------------- fitting
def chi2_for(rho_t, rg, rho, dMdR, Rq, Vb, sVb, Vobs, eVobs):
    m = ccc_model(rg, rho, dMdR, rho_t)
    if m is None:
        return np.inf, None
    Vm = vbx_at(rg, m["MbX"], Rq)
    # propagate V_obs error into V_bX to first order (M_bX ~ V_obs^2)
    sM = np.maximum(Vm / np.maximum(Vobs, 1e-6) * eVobs, 0.0)
    sig = np.sqrt(sVb**2 + sM**2 + 1.0)               # +1 km/s floor
    c2 = float(np.sum(((Vm - Vb) / sig) ** 2))
    return c2, dict(model=m, Vm=Vm, sig=sig)


def fit_galaxy(R, Vobs, eVobs, Vgas, Vdisk, Vbul, s_factor=1.0, n_grid=2000,
               require_resolved=True):
    """
    Full single-galaxy fit.

    If require_resolved, rho_t is restricted so the turn-off radius falls
    inside the observed range [R_min, R_max]: the model needs a Newtonian
    (X = 0) core that the data actually resolve, and it needs the turn-off
    to happen before the last measured point. Galaxies driven to either
    bound are flagged rather than silently accepted.
    """
    Vb = sparc_vbar(Vgas, Vdisk, Vbul)
    sVb = sigma_vbar(Vgas, Vdisk, Vbul)

    sp = smooth_vobs(R, Vobs, eVobs, s_factor=s_factor)
    rg, rho, dMdR, Mo = density_profile(sp, R, n_grid=n_grid)

    if require_resolved:
        inside = (rg >= R[0]) & (rg <= R[-1])
        rho_in = rho[inside]
        rho_in = rho_in[rho_in > 0]
        if rho_in.size < 3:
            return dict(rho_t=np.nan, flag="no_positive_density", npts=len(R))
        lo, hi = np.log10(rho_in.min()), np.log10(rho_in.max())
    else:
        lo = np.log10(max(np.nanmin(rho[rho > 0]) * 0.5, 1e2))
        hi = np.log10(np.nanmax(rho) * 2.0)
    grid = np.linspace(lo, hi, 500)

    c2s = np.array([chi2_for(10.0**g, rg, rho, dMdR, R, Vb, sVb, Vobs, eVobs)[0]
                    for g in grid])
    j = int(np.nanargmin(c2s))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    res = minimize_scalar(
        lambda g: chi2_for(10.0**g, rg, rho, dMdR, R, Vb, sVb, Vobs, eVobs)[0],
        bounds=(a, b), method="bounded", options=dict(xatol=1e-4))
    rho_t = 10.0 ** res.x
    c2, extra = chi2_for(rho_t, rg, rho, dMdR, R, Vb, sVb, Vobs, eVobs)

    Vm = extra["Vm"]
    dof = max(len(R) - 1, 1)
    frms = float(np.sqrt(np.mean((Vm - Vb) ** 2)) / np.mean(Vb)) if np.mean(Vb) > 0 else np.nan

    # chi2 profile -> 1-sigma interval; inflate by sqrt(chi2/dof) if chi2r > 1
    ok = np.isfinite(c2s)
    lohi = (np.nan, np.nan)
    scale = max(c2 / dof, 1.0)
    if ok.sum() > 3:
        band = grid[ok][c2s[ok] <= c2 + scale]
        if band.size:
            lohi = (10.0 ** band.min(), 10.0 ** band.max())

    Rt = extra["model"]["Rt"]
    flags = []
    if abs(res.x - lo) < 2e-3 or Rt >= R[-1] * 0.999:
        flags.append("Rt_at_outer_edge")
    if abs(res.x - hi) < 2e-3 or Rt <= R[0] * 1.001:
        flags.append("Rt_unresolved")           # turn-off at/inside first point
    if extra["model"]["clip_frac"] > 0.10:
        flags.append("X_clipped")
    if c2 / dof > 3.0:
        flags.append("poor_chi2")
    if len(R) < 6:
        flags.append("few_points")

    return dict(rho_t=rho_t, rho_t_lo=lohi[0], rho_t_hi=lohi[1],
                Rt=Rt, chi2=c2, dof=dof, chi2r=c2 / dof,
                frms=frms, clip_frac=extra["model"]["clip_frac"],
                flag=",".join(flags) if flags else "ok",
                Vb=Vb, sVb=sVb, VbX=Vm, spline=sp,
                rg=rg, rho=rho, dMdR=dMdR, Mo=Mo, X=extra["model"]["X"],
                MbX=extra["model"]["MbX"],
                grid=grid, c2s=c2s,
                Rmin=R[0], Rmax=R[-1], npts=len(R))
