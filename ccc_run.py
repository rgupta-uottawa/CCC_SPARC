"""
Run the CCC+TL analysis over all 175 SPARC galaxies.

Fit variants
  A  fiducial : rho_o from error-weighted smoothing spline (s/N = 0.1),
                Upsilon fixed at 0.5 / 0.7            <- matches the paper
  B  ups_free : as A, but the Upsilon scaling is profiled with a 0.1 dex prior
  C  rawfd    : as A, but rho_o from raw one-sided finite differences on the
                tabulated grid (the convention systematic)

No data points are cut or downweighted in any variant.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
import sparc_io as io, ccc_core as cc

S_FACTOR = 0.1
N_MC = 150
N_GRID = 1500
N_GRID_MC = 700
FLOOR = 1.0
FRAC_B = 0.05
DEX_PRIOR = 0.10
K_ACC = 1e10 / cc.KPC_CM        # (km/s)^2/kpc -> cm/s^2


def reconstruct(R, V, eV, mode="spline", s_factor=S_FACTOR, n_grid=N_GRID):
    """Return (rg, rho_o, dM/dR, spline_or_None)."""
    if mode == "spline":
        sp = cc.smooth_vobs(R, V, eV, s_factor=s_factor)
        rg, rho, dMdR, _ = cc.density_profile(sp, R, n_grid=n_grid)
        return rg, rho, dMdR, sp
    M = V ** 2 * R / cc.G
    rho_n = (np.diff(M) / np.diff(R)) / (4.0 * np.pi * R[:-1] ** 2)
    rg = np.linspace(1e-3, R[-1], n_grid)
    rho = np.interp(rg, R[:-1], rho_n, left=rho_n[0], right=rho_n[-1])
    return rg, rho, rho * 4.0 * np.pi * rg ** 2, None


def vb_scaled(Vgas, Vdisk, Vbul, u):
    v2 = Vgas * np.abs(Vgas) + (10.0 ** u) * (cc.YD * Vdisk ** 2 + cc.YB * Vbul ** 2)
    return np.sqrt(np.clip(v2, 0.0, None))


def build_mc(R, V, eV, mode, n_mc=N_MC, seed=0):
    rng = np.random.default_rng(seed)
    ens = []
    for _ in range(n_mc):
        try:
            rg, rho, dMdR, _ = reconstruct(R, V + rng.normal(0, eV), eV,
                                           mode=mode, n_grid=N_GRID_MC)
            ens.append((rg, rho, dMdR))
        except Exception:
            pass
    return ens


def sigma_mc(ens, R, rho_t):
    out = [cc.vbx_at(rg, m["MbX"], R)
           for rg, rho, dM in ens
           if (m := cc.ccc_model(rg, rho, dM, rho_t)) is not None]
    if len(out) < 10:
        return None
    A = np.array(out)
    return 0.5 * (np.percentile(A, 84, axis=0) - np.percentile(A, 16, axis=0))


def fit(R, Vobs, eVobs, Vgas, Vdisk, Vbul, mode="spline", ups_free=False, seed=0):
    Vb0 = cc.sparc_vbar(Vgas, Vdisk, Vbul)
    rg, rho, dMdR, sp = reconstruct(R, Vobs, eVobs, mode=mode)
    pos = rho[rho > 0]
    if pos.size < 3:
        return None
    grid = np.linspace(np.log10(pos.min() * 0.5), np.log10(pos.max() * 3.0), 400)
    ens = build_mc(R, Vobs, eVobs, mode, seed=seed)

    def chi2_at(g, sig, want_u=False):
        m = cc.ccc_model(rg, rho, dMdR, 10.0 ** g)
        if m is None:
            return (np.inf, 0.0) if want_u else np.inf
        Vm = cc.vbx_at(rg, m["MbX"], R)

        def c2(u):
            Vb = vb_scaled(Vgas, Vdisk, Vbul, u)
            s2 = sig ** 2 + (FRAC_B * Vb) ** 2 + FLOOR ** 2
            return float(np.sum((Vm - Vb) ** 2 / s2)) + (u / DEX_PRIOR) ** 2

        if not ups_free:
            return (c2(0.0), 0.0) if want_u else c2(0.0)
        r = minimize_scalar(c2, bounds=(-0.4, 0.4), method="bounded",
                            options=dict(xatol=1e-4))
        return (r.fun, r.x) if want_u else r.fun

    sig = np.maximum(eVobs, 1.0)
    c2s = np.array([chi2_at(g, sig) for g in grid])
    best = grid[int(np.nanargmin(c2s))]
    for _ in range(2):
        s = sigma_mc(ens, R, 10.0 ** best)
        if s is not None and np.isfinite(s).all():
            sig = s
        c2s = np.array([chi2_at(g, sig) for g in grid])
        j = int(np.nanargmin(c2s))
        best = minimize_scalar(chi2_at, args=(sig,),
                               bounds=(grid[max(j - 1, 0)], grid[min(j + 1, 399)]),
                               method="bounded", options=dict(xatol=1e-4)).x

    rho_t = 10.0 ** best
    m = cc.ccc_model(rg, rho, dMdR, rho_t)
    if m is None:
        return None
    Vm = cc.vbx_at(rg, m["MbX"], R)
    chi2, u = chi2_at(best, sig, want_u=True)
    dof = max(len(R) - (2 if ups_free else 1), 1)
    Vb = vb_scaled(Vgas, Vdisk, Vbul, u)

    ok = np.isfinite(c2s)
    band = grid[ok][c2s[ok] <= chi2 + max(chi2 / dof, 1.0)]
    rlo, rhi = (10.0 ** band.min(), 10.0 ** band.max()) if band.size else (np.nan,) * 2

    flags = []
    if m["Rt"] < R[0]:
        flags.append("Rt_inside_Rmin")
    if m["Rt"] >= R[-1] * 0.999:
        flags.append("Rt_at_outer_edge")
    if m["clip_frac"] > 0.10:
        flags.append("X_clipped")
    if chi2 / dof > 3.0:
        flags.append("poor_chi2")
    if len(R) < 6:
        flags.append("few_points")

    return dict(rho_t=rho_t, rho_t_cgs=rho_t * cc.RHO_CGS,
                rho_t_lo_cgs=rlo * cc.RHO_CGS, rho_t_hi_cgs=rhi * cc.RHO_CGS,
                Rt=m["Rt"], chi2=chi2, dof=dof, chi2r=chi2 / dof,
                frms=float(np.sqrt(np.mean((Vm - Vb) ** 2)) / max(np.mean(Vb), 1e-6)),
                sig_med=float(np.median(sig)), ups_scale=10.0 ** u,
                clip_frac=m["clip_frac"], flag=",".join(flags) or "ok",
                Vb=Vb, Vb0=Vb0, VbX=Vm, sig=sig, R=R, spline=sp,
                rg=rg, rho=rho, X=m["X"], npts=len(R))


if __name__ == "__main__":
    import pickle, time
    t1, t2 = io.load_table1(), io.load_table2()
    gals = sorted(t2.Galaxy.unique())
    rows, store = [], {}
    t0 = time.time()
    for i, g in enumerate(gals, 1):
        d = t2[t2.Galaxy == g].sort_values("R")
        args = (d.R.values, d.Vobs.values, d.e_Vobs.values,
                d.Vgas.values, d.Vdisk.values, d.Vbul.values)
        rec = dict(Galaxy=g, N=len(d), T=int(t1.loc[g, "T"]),
                   Type=io.TYPES[int(t1.loc[g, "T"])], D=t1.loc[g, "D"],
                   Inc=t1.loc[g, "Inc"], L36=t1.loc[g, "L36"],
                   Rdisk=t1.loc[g, "Rdisk"], MHI=t1.loc[g, "MHI"],
                   Vflat=t1.loc[g, "Vflat"], Q=int(t1.loc[g, "Q"]),
                   Rmin=d.R.values[0], Rmax=d.R.values[-1])
        try:
            A = fit(*args, mode="spline", ups_free=False)
        except Exception:
            A = None
        if A is None:
            rec["flag"] = "FIT_FAILED"
            rows.append(rec)
            continue
        store[g] = A
        vf = t1.loc[g, "Vflat"]
        rec.update(rho_t=A["rho_t_cgs"], rho_t_lo=A["rho_t_lo_cgs"],
                   rho_t_hi=A["rho_t_hi_cgs"], Rt=A["Rt"], chi2r=A["chi2r"],
                   frms=A["frms"], sig_med=A["sig_med"], clip=A["clip_frac"],
                   flag=A["flag"],
                   a_t=(vf ** 2 / A["Rt"] * K_ACC) if (vf > 0 and A["Rt"] > 1e-2) else np.nan,
                   Rt_over_Rdisk=A["Rt"] / t1.loc[g, "Rdisk"] if t1.loc[g, "Rdisk"] > 0 else np.nan)
        for key, kw in [("B", dict(mode="spline", ups_free=True)),
                        ("C", dict(mode="rawfd", ups_free=False))]:
            try:
                r = fit(*args, **kw)
            except Exception:
                r = None
            if r is None:
                continue
            rec[f"rho_t_{key}"] = r["rho_t_cgs"]
            rec[f"Rt_{key}"] = r["Rt"]
            rec[f"chi2r_{key}"] = r["chi2r"]
            if key == "B":
                rec["ups_scale"] = r["ups_scale"]
        rows.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(gals)}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/mnt/user-data/outputs/ccc_sparc175_results.csv", index=False)
    with open("/home/claude/ccc/store.pkl", "wb") as f:
        pickle.dump(store, f)
    print(f"\ndone in {time.time()-t0:.0f}s; {df.rho_t.notna().sum()} of {len(df)} fitted")
