"""
Density-space comparison of CCC and MOND, on equal footing.

Both models predict a total density rho_o at each radius from the baryons:

  operator (spherical):  rho = (1/(4 pi G)) (2 g / R + dg/dR),   g = V^2 / R

  MOND:  g_o = nu(g_b) via the RAR, then apply the operator to g_o(R).
  CCC :  rho_o = sqrt(rho_t rho_b) for rho_b < rho_t, else rho_b.

The SAME numerical derivative (same spline, same grid) is applied to build
rho_b, rho_o^MOND and rho_o^CCC, so any differentiation noise is common-mode
and cancels in the comparison. We then ask, for each model:

  how tightly does the OBSERVED total density rho_o^data track the PREDICTED
  rho_o, across all points of all galaxies?   -> scatter in dex

rho_o^data and rho_b and the model predictions all use the identical operator,
so this is the density-space analogue of the RAR built symmetrically.
"""
import numpy as np, pandas as pd
from scipy.interpolate import UnivariateSpline
import sparc_io as io, ccc_core as cc, compare_models as cm

G = cc.G
A0 = 1.2e-10
KMS2_KPC_TO_MS2 = cm.KMS2_KPC_TO_MS2
_LX = np.linspace(-12, 12, 24001)
_LY = np.log10(10.0 ** _LX / (1.0 - np.exp(-np.sqrt(10.0 ** _LX))))


def g_of(V, R):
    return V ** 2 / R * KMS2_KPC_TO_MS2          # m/s^2


def rho_from_g_curve(R, gfun_on_grid, rg):
    """rho = (1/4piG)(2g/R + dg/dR) with g in m/s^2 -> rho in Msun/kpc^3.
    Convert: g[m/s2] = (V^2/R)[km2/s2/kpc] * KMS2_KPC_TO_MS2, and
    rho[Msun/kpc3] = (1/(4 pi G)) (2 g/R + dg/dR) with g back in kpc units."""
    g_kpc = gfun_on_grid / KMS2_KPC_TO_MS2       # (km/s)^2/kpc
    dg = np.gradient(g_kpc, rg)
    return (2.0 * g_kpc / rg + dg) / (4.0 * np.pi * G)


def mond_go(gb):
    """RAR: g_o from g_b (both m/s^2)."""
    x = gb / A0
    return gb / (1.0 - np.exp(-np.sqrt(np.clip(x, 1e-30, None)))) 


t1, t2 = io.load_table1(), io.load_table2()

# fitted rho_t per galaxy (inverse fiducial) for the CCC prediction
resdf = pd.read_csv("/mnt/user-data/outputs/ccc_sparc175_results.csv").set_index("Galaxy")
RHO_T_GLOBAL = (resdf.rho_t/cc.RHO_CGS).median()   # 4.59e-24 g/cm3

pts = []   # one row per radial point
for g in sorted(t2.Galaxy.unique()):
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    if len(R) < 5 or g not in resdf.index:
        continue
    rho_t = RHO_T_GLOBAL   # single global constant, matches MOND single global a0

    # common smoothing spline for BOTH velocity curves (same operator)
    n = len(R)
    sb = UnivariateSpline(R, Vb, w=1/np.maximum(0.05*Vb+1, 1), k=min(3, n-1), s=0.1*n)
    so = UnivariateSpline(R, V, w=1/np.maximum(eV, 1), k=min(3, n-1), s=0.1*n)
    rg = np.linspace(R[0], R[-1], 400)

    gb = g_of(sb(rg), rg)                 # baryonic accel on grid
    go = g_of(so(rg), rg)                 # observed accel on grid
    go_mond = mond_go(gb)                 # MOND-predicted observed accel

    rho_b = rho_from_g_curve(rg, gb, rg)
    rho_o_data = rho_from_g_curve(rg, go, rg)
    rho_o_mond = rho_from_g_curve(rg, go_mond, rg)
    rho_o_ccc = np.where(rho_b < rho_t, np.sqrt(np.clip(rho_t*np.clip(rho_b,0,None),0,None)), rho_b)

    # sample back at the observed radii (avoid endpoints where gradient is worst)
    for Ri in R[1:-1]:
        j = np.argmin(np.abs(rg - Ri))
        rb, rod, rom, roc = rho_b[j], rho_o_data[j], rho_o_mond[j], rho_o_ccc[j]
        if rb > 0 and rod > 0 and rom > 0 and roc > 0:
            pts.append((g, Ri, rb, rod, rom, roc,
                        g_of(so(Ri), Ri), g_of(sb(Ri), Ri)))

P = pd.DataFrame(pts, columns=["Galaxy", "R", "rho_b", "rho_o_data",
                               "rho_o_mond", "rho_o_ccc", "g_obs", "g_bar"])
P.to_csv("/mnt/user-data/outputs/ccc_density_relation.csv", index=False)

def scatter(pred, data):
    r = np.log10(data / pred)
    r = r[np.isfinite(r)]
    return np.std(r), np.median(r), len(r)

print(f"points: {len(P)}   galaxies: {P.Galaxy.nunique()}\n")
print("DENSITY-SPACE residual scatter  log10(rho_o_data / rho_o_pred)")
for lab, k in [("CCC   rho_o = sqrt(rho_t rho_b)", "rho_o_ccc"),
               ("MOND  rho_o = D[nu(g_b)]", "rho_o_mond")]:
    s, m, npt = scatter(P[k].values, P.rho_o_data.values)
    print(f"  {lab:34s}  scatter = {s:.3f} dex   bias = {m:+.3f}   (N={npt})")

print("\nFor reference, the SAME points in ACCELERATION space (the RAR):")
s, m, npt = scatter(mond_go(P.g_bar.values), P.g_obs.values)
print(f"  MOND  g_o = nu(g_b)                  scatter = {s:.3f} dex   bias = {m:+.3f}")
# CCC in acceleration space: integrate is hard pointwise; report density only.

print("\nOnly outer points (rho_b < median rho_t, where X>0 and models differ):")
med_rt = (resdf.rho_t/cc.RHO_CGS).median()
mask = P.rho_b < med_rt
for lab, k in [("CCC", "rho_o_ccc"), ("MOND", "rho_o_mond")]:
    s, m, npt = scatter(P[k][mask].values, P.rho_o_data[mask].values)
    print(f"  {lab:6s} scatter = {s:.3f} dex   bias = {m:+.3f}   (N={npt})")
