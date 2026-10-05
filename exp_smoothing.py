"""Experiment 1: how much smoothing is actually justified?"""
import numpy as np, pandas as pd
from scipy.interpolate import UnivariateSpline
import sparc_io as io, ccc_core as cc
from run_paper7 import ORDER

t2 = io.load_table2()
SF = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def diagnostics(R, V, eV, sf):
    n = len(R)
    w = 1.0 / np.maximum(eV, 1e-3)
    sp = UnivariateSpline(R, V, w=w, k=min(3, n - 1), s=sf * n)
    resid = (V - sp(R)) / np.maximum(eV, 1e-3)
    chi2n = float(np.sum(resid**2) / n)
    # lag-1 autocorrelation of residuals: >0 means systematic runs (over-smoothed)
    r1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 4 else np.nan
    # roughness of the reconstructed density (relative curvature of log rho)
    rg, rho, _, _ = cc.density_profile(sp, R, n_grid=400)
    m = (rg >= R[0]) & (rho > 0)
    lr = np.log(rho[m])
    rough = float(np.sqrt(np.mean(np.diff(lr, 2) ** 2))) if m.sum() > 5 else np.nan
    return chi2n, r1, rough


rows = []
for g in ORDER:
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    for sf in SF:
        c2n, r1, rough = diagnostics(R, V, eV, sf)
        try:
            fit = cc.fit_galaxy(R, V, eV, d.Vgas.values, d.Vdisk.values,
                                d.Vbul.values, s_factor=sf)
            rt = fit["rho_t"] * cc.RHO_CGS / 1e-24
            cx = fit["chi2r"]
        except Exception:
            rt, cx = np.nan, np.nan
        rows.append(dict(Galaxy=g, s=sf, spline_chi2n=c2n, resid_acf1=r1,
                         rough=rough, rho_t=rt, ccc_chi2n=cx))

df = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:7.3f}")

print("=== spline fidelity to the data (chi2/N of the spline itself) ===")
print(df.pivot(index="s", columns="Galaxy", values="spline_chi2n").to_string())
print("\n=== lag-1 autocorrelation of spline residuals (>0.3 => over-smoothed) ===")
print(df.pivot(index="s", columns="Galaxy", values="resid_acf1").to_string())
print("\n=== roughness of reconstructed log rho_o (numerical noise proxy) ===")
print(df.pivot(index="s", columns="Galaxy", values="rough").to_string())
print("\n=== recovered rho_t [1e-24 g/cm3] ===")
print(df.pivot(index="s", columns="Galaxy", values="rho_t").to_string())
print("\n=== CCC fit chi2/nu ===")
print(df.pivot(index="s", columns="Galaxy", values="ccc_chi2n").to_string())
