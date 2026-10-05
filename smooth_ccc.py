"""
Smooth-CCC: replace the sharp turn-off with a smooth transition, the
density-space analogue of MOND's nu interpolation function.

CCC inverse integrand is  (1-X)^2 = rho_b/rho_o.  Sharp CCC:
    rho_o > rho_t : (1-X)^2 = 1        (Newtonian core, X=0)
    rho_o < rho_t : (1-X)^2 = rho_o/rho_t   [since rho_o=sqrt(rho_t rho_b)]
i.e. (1-X)^2 = min(rho_o/rho_t, 1), a sharp kink at rho_o=rho_t.

Smooth version. Let y = rho_b/rho_t and rho_o = rho_b * nu(y) with nu a standard
MOND interpolation. Then
    (1-X)^2 = rho_b/rho_o = 1/nu(y),     u := rho_o/rho_t = y*nu(y).
Given the observed u we solve y*nu(y)=u for y (monotonic, by table) and set the
integrand W = 1/nu(y). W -> u as u->0 (deep) and W -> 1 as u->inf (Newtonian),
reproducing the sharp limits but with a smooth knee.

M_bX = Int W(rho_o/rho_t) dM_o,  V_bX = sqrt(G M_bX / R).
"""
import numpy as np
import ccc_core as cc


def _build_table(nu):
    y = np.logspace(-10, 10, 60001)
    u = y * nu(y)          # rho_o/rho_t
    W = 1.0 / nu(y)        # (1-X)^2
    return np.log10(u), W


_NU = {
    "standard": lambda y: 1.0 / (1.0 - np.exp(-np.sqrt(np.clip(y, 1e-30, None)))),
    "simple":   lambda y: 0.5 + np.sqrt(0.25 + 1.0 / np.clip(y, 1e-30, None)),
}
_TAB = {k: _build_table(v) for k, v in _NU.items()}


def W_of_u(u, kind):
    """(1-X)^2 as a function of rho_o/rho_t for a smooth transition."""
    lu, W = _TAB[kind]
    return np.interp(np.log10(np.clip(u, 1e-30, None)), lu, W, left=0.0, right=1.0)


def vbx_smooth(R, Vobs, eV, rho_t, kind="standard", s_factor=0.1, n_grid=1200,
               inner_slope=None):
    """Smooth-CCC inverse prediction of V_b from V_obs.

    inner_slope: if given, the inner power-law slope is held FIXED at this value
    rather than re-measured from Vobs. Pass the fiducial slope (from
    cc.inner_slope_of on the unperturbed curve) during Monte-Carlo error
    propagation so the slope does not act as a noise amplifier.
    """
    sp = cc.smooth_vobs(R, Vobs, eV, s_factor=s_factor)
    rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=n_grid, inner_slope=inner_slope)
    if kind == "sharp":
        W = np.clip(rho / rho_t, 0.0, 1.0)
    else:
        W = W_of_u(rho / rho_t, kind)
    integ = W * dMdR
    MbX = np.concatenate([[0.0], np.cumsum(np.diff(rg) * 0.5 * (integ[1:] + integ[:-1]))])
    return cc.vbx_at(rg, MbX, R)
