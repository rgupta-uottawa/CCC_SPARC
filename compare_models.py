"""
Like-for-like model comparison on all 175 SPARC galaxies.

Every model:
  * predicts V_obs from the photometric baryons,
  * has exactly ONE free parameter per galaxy,
  * is fitted by minimising chi2 = sum (V_pred - V_obs)^2 / e_Vobs^2,
  * uses every data point, no cuts, no downweighting,
  * uses identical dof = N - 1.

Models
  CCC   free rho_t.  Forward form: rho_o = rho_b for r<R_t,
        rho_o = sqrt(rho_t * rho_b) for r>R_t, with R_t where rho_b = rho_t.
        (from rho_o = rho(1-X)^2 and rho_b = rho(1-X)^4  =>  rho_o^2 = rho rho_b)
        Upsilon fixed at 0.5 / 0.7.
  MOND-a  free a0 per galaxy, Upsilon fixed at 0.5/0.7.   <- closest analogue to CCC
  MOND-U  free Upsilon per galaxy, a0 fixed at 1.2e-10.   <- the standard literature fit
  NFW     free V200, concentration from the Dutton & Maccio (2014) c-M200
          relation, Upsilon fixed at 0.5/0.7.             <- standard 1-param NFW
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline
import sparc_io as io, ccc_core as cc

G = cc.G
KMS2_KPC_TO_MS2 = 1e6 / 3.0856775814913673e19     # (km/s)^2/kpc -> m/s^2
A0 = 1.2e-10                                       # m/s^2
H0 = 0.0671                                        # km/s/kpc  (h = 0.671)
RHO_CRIT = 3 * H0 ** 2 / (8 * np.pi * G)           # Msun/kpc^3
DELTA = 200.0


# ----------------------------------------------------------------- CCC forward
def rho_from_v(R, V, n_grid=1200, s=0.0):
    """Spherical-approximation density from a velocity curve."""
    n = len(R)
    sp = UnivariateSpline(R, V, k=min(3, n - 1), s=s)
    rg = np.linspace(R[0], R[-1], n_grid)
    Vg, dV = sp(rg), sp.derivative()(rg)
    dMdR = (Vg ** 2 + 2 * rg * Vg * dV) / G
    rho = dMdR / (4 * np.pi * rg ** 2)
    # inner solid-body core, as in the main pipeline
    n_in = 150
    rin = np.linspace(1e-4, R[0], n_in, endpoint=False)
    Vmin = float(sp(R[0]))
    rho_in = np.full(n_in, 3 * Vmin ** 2 / (4 * np.pi * G * R[0] ** 2))
    dM_in = 3 * Vmin ** 2 * rin ** 2 / (G * R[0] ** 2)
    return (np.concatenate([rin, rg]), np.concatenate([rho_in, rho]),
            np.concatenate([dM_in, dMdR]))


def ccc_forward(rg, rho_b, R, rho_t):
    """rho_o = rho_b inside R_t, sqrt(rho_t rho_b) outside. Returns V_pred."""
    rb = np.clip(rho_b, 0.0, None)
    below = rb < rho_t
    Rt = rg[int(np.argmax(below))] if below.any() else rg[-1]
    # Outside R_t: rho_o = sqrt(rho_t rho_b). If rho_b rises back ABOVE rho_t
    # (non-monotonic profiles) this would give X < 0, so clip X at 0, which is
    # exactly rho_o = max(sqrt(rho_t rho_b), rho_b). Inside R_t: X = 0, rho_o = rho_b.
    rho_o = np.where(rg <= Rt, rb, np.maximum(np.sqrt(rho_t * rb), rb))
    integ = 4 * np.pi * rho_o * rg ** 2
    Mo = np.concatenate([[0.0], np.cumsum(np.diff(rg) * 0.5 * (integ[1:] + integ[:-1]))])
    M = np.interp(R, rg, Mo)
    return np.sqrt(np.clip(G * M / R, 0, None)), Rt


# ----------------------------------------------------------------------- MOND
def mond_v(Vb, R, a0):
    """Radial acceleration relation interpolating function."""
    gN = Vb ** 2 / R * KMS2_KPC_TO_MS2
    gN = np.clip(gN, 1e-16, None)
    g = gN / (1.0 - np.exp(-np.sqrt(gN / a0)))
    return np.sqrt(g / KMS2_KPC_TO_MS2 * R)


# ------------------------------------------------------------------------ NFW
def nfw_v(R, V200):
    R200 = V200 / (10 * H0)
    M200 = V200 ** 2 * R200 / G
    c = 10 ** (0.905 - 0.101 * np.log10(M200 / (1e12 / 0.671)))
    c = float(np.clip(c, 1.0, 100.0))
    x = np.clip(R / R200, 1e-6, None)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    return np.sqrt(V200 ** 2 * mu(c * x) / (x * mu(c)))


# ------------------------------------------------------------------------ fit
def fit1d(chi2fn, lo, hi, n=300):
    grid = np.linspace(lo, hi, n)
    vals = np.array([chi2fn(g) for g in grid])
    j = int(np.nanargmin(vals))
    r = minimize_scalar(chi2fn, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, n - 1)]),
                        method="bounded", options=dict(xatol=1e-4))
    return r.x, r.fun


def run():
    t1, t2 = io.load_table1(), io.load_table2()
    rows = []
    for g in sorted(t2.Galaxy.unique()):
        d = t2[t2.Galaxy == g].sort_values("R")
        R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
        Vgas, Vdisk, Vbul = d.Vgas.values, d.Vdisk.values, d.Vbul.values
        Vb = cc.sparc_vbar(Vgas, Vdisk, Vbul)
        w = 1.0 / np.maximum(eV, 1.0) ** 2
        dof = max(len(R) - 1, 1)
        rec = dict(Galaxy=g, N=len(R), T=int(t1.loc[g, "T"]))

        # ---- CCC
        try:
            rg, rho_b, _ = rho_from_v(R, Vb, s=0.5 * len(R))
            pos = rho_b[rho_b > 0]
            f = lambda lg: float(np.sum(w * (ccc_forward(rg, rho_b, R, 10 ** lg)[0] - V) ** 2))
            b, c2 = fit1d(f, np.log10(pos.min() * 0.3), np.log10(pos.max() * 3))
            rec.update(ccc_rho_t=10 ** b * cc.RHO_CGS, ccc_chi2r=c2 / dof,
                       ccc_Rt=ccc_forward(rg, rho_b, R, 10 ** b)[1])
        except Exception:
            rec.update(ccc_rho_t=np.nan, ccc_chi2r=np.nan)

        # ---- MOND, a0 free
        try:
            f = lambda lg: float(np.sum(w * (mond_v(Vb, R, 10 ** lg) - V) ** 2))
            b, c2 = fit1d(f, -12.0, -8.0)
            rec.update(mond_a0=10 ** b, mond_a_chi2r=c2 / dof)
        except Exception:
            rec.update(mond_a0=np.nan, mond_a_chi2r=np.nan)

        # ---- MOND, Upsilon free, a0 fixed
        try:
            def f(lu):
                vb = np.sqrt(np.clip(Vgas * np.abs(Vgas)
                                     + 10 ** lu * (0.5 * Vdisk ** 2 + 0.7 * Vbul ** 2), 0, None))
                return float(np.sum(w * (mond_v(vb, R, A0) - V) ** 2))
            b, c2 = fit1d(f, -1.0, 1.0)
            rec.update(mond_ups=10 ** b, mond_U_chi2r=c2 / dof)
        except Exception:
            rec.update(mond_ups=np.nan, mond_U_chi2r=np.nan)

        # ---- NFW, one parameter
        try:
            f = lambda lv: float(np.sum(w * (np.sqrt(np.clip(Vb ** 2 + nfw_v(R, 10 ** lv) ** 2, 0, None)) - V) ** 2))
            b, c2 = fit1d(f, np.log10(10.0), np.log10(600.0))
            rec.update(nfw_V200=10 ** b, nfw_chi2r=c2 / dof)
        except Exception:
            rec.update(nfw_V200=np.nan, nfw_chi2r=np.nan)

        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df["dwarf"] = df["T"] >= 8
    df.to_csv("/mnt/user-data/outputs/ccc_model_comparison.csv", index=False)
    cols = [("CCC  (rho_t free)", "ccc_chi2r"), ("MOND (a0 free)", "mond_a_chi2r"),
            ("MOND (Ups free)", "mond_U_chi2r"), ("NFW  (V200 free)", "nfw_chi2r")]
    print(f"{'model':20s} {'median':>8s} {'mean':>8s} {'<1':>6s} {'<3':>6s} {'>10':>6s}  "
          f"{'dwarf':>7s} {'spiral':>7s}")
    for lab, k in cols:
        s = df[k].dropna()
        print(f"{lab:20s} {s.median():8.2f} {s.mean():8.1f} "
              f"{100*(s<1).mean():5.0f}% {100*(s<3).mean():5.0f}% {100*(s>10).mean():5.0f}%  "
              f"{df[k][df.dwarf].median():7.2f} {df[k][~df.dwarf].median():7.2f}")
    print()
    print("head-to-head (lower chi2/nu wins, all 175 galaxies):")
    for a, b in [("ccc_chi2r", "mond_a_chi2r"), ("ccc_chi2r", "mond_U_chi2r"),
                 ("ccc_chi2r", "nfw_chi2r"), ("mond_U_chi2r", "nfw_chi2r")]:
        m = df[a].notna() & df[b].notna()
        print(f"   {a:14s} beats {b:14s} in {100*(df[a][m] < df[b][m]).mean():4.0f}% of galaxies")
    print()
    print(f"MOND a0 (free):  median {df.mond_a0.median():.2e} m/s2  "
          f"[16-84%: {df.mond_a0.quantile(.16):.1e}, {df.mond_a0.quantile(.84):.1e}]   "
          f"literature 1.2e-10")
    print(f"MOND Upsilon scaling: median {df.mond_ups.median():.2f} "
          f"[{df.mond_ups.quantile(.16):.2f}, {df.mond_ups.quantile(.84):.2f}]")
