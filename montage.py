"""
Inverse-mode rotation-curve montage for supplementary material.

One galaxy per row, three columns (Model A / B / C, deliberately unlabelled as
to identity). Each panel shows, in inverse mode:
    - V_obs observed points with error bars (the input)
    - SPARC V_b photometric baryonic curve (grey, the target)
    - the model's predicted V_b from V_obs (coloured line)
with the fitted parameter value annotated.

Model A = smooth CCC (rho_t)     Model B = smooth MOND (a0)     Model C = 2-param NFW
The identity key is written only to a separate text file, not on the plots.
"""
import numpy as np, pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
import sparc_io as io, ccc_core as cc, compare_models as cm, smooth_ccc as sm

plt.rcParams.update({"font.size": 8, "figure.dpi": 110, "savefig.dpi": 110})
K = cm.KMS2_KPC_TO_MS2
FLOOR, FRAC, N_MC, H0 = 1.0, 0.05, 50, 0.0671
RNG = np.random.default_rng(7)
_LX = np.linspace(-13, 12, 50001)
_LY = np.log10(10 ** _LX / (1 - np.exp(-np.sqrt(10 ** _LX))))


def nfw_v(R, V200, c):
    R200 = V200 / (10 * H0); x = np.clip(R / R200, 1e-6, None)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    return np.sqrt(np.clip(V200 ** 2 * mu(c * x) / (x * mu(c)), 0, None))


def vb_ccc(R, V, eV, rt):
    s0=cc.inner_slope_of(cc.smooth_vobs(R,V,eV,0.1),R)
    return sm.vbx_smooth(R, V, eV, rt, kind="standard", inner_slope=s0)
def vb_mond(R, V, eV, a0):
    go = V ** 2 / R * K
    gb = a0 * 10 ** np.interp(np.log10(np.clip(go / a0, 1e-30, None)), _LY, _LX)
    return np.sqrt(np.clip(gb / K * R, 0, None))


def sig_ccc(R, V, eV, rt, Vb):
    out = [vb_ccc(R, V + RNG.normal(0, eV), eV, rt) for _ in range(N_MC)]
    out = [v for v in out if v is not None and np.isfinite(v).all()]
    A = np.array(out) if len(out) >= 8 else None
    s = 0.5*(np.percentile(A,84,0)-np.percentile(A,16,0)) if A is not None else np.full(len(R),5.)
    return np.sqrt(s**2 + (FRAC*Vb)**2 + FLOOR**2)


def fit_ccc(R, V, eV, Vb):
    grid = np.linspace(3, 12, 120)
    def c2(pl, s):
        v = vb_ccc(R, V, eV, 10**pl)
        return np.inf if (v is None or not np.isfinite(v).all()) else float(np.sum((v-Vb)**2/s**2))
    s = np.maximum(eV, 1.)
    best = grid[int(np.nanargmin([c2(x, s) for x in grid]))]
    for _ in range(2):
        s = sig_ccc(R, V, eV, 10**best, Vb)
        cur = np.array([c2(x, s) for x in grid]); j = int(np.nanargmin(cur))
        best = minimize_scalar(c2, args=(s,), bounds=(grid[max(j-1,0)], grid[min(j+1,119)]),
                               method="bounded", options=dict(xatol=1e-4)).x
    return 10**best, c2(best, s)/max(len(R)-1,1)


def fit_mond(R, V, eV, Vb):
    grid = np.linspace(-12, -8, 120)
    def c2(pl, s):
        v = vb_mond(R, V, eV, 10**pl)
        return float(np.sum((v-Vb)**2/s**2))
    s = np.sqrt((FRAC*Vb)**2 + FLOOR**2 + np.maximum(eV,1.)**2)
    best = grid[int(np.nanargmin([c2(x, s) for x in grid]))]
    best = minimize_scalar(c2, args=(s,), bounds=(-12, -8), method="bounded",
                           options=dict(xatol=1e-4)).x
    return 10**best, c2(best, s)/max(len(R)-1,1)


def fit_nfw2(R, V, eV, Vb):
    def c2(p, s):
        V200, c = 10**p[0], p[1]
        if not (10 <= V200 <= 600 and 1 <= c <= 40): return 1e12
        v = np.sqrt(np.clip(V**2 - nfw_v(R, V200, c)**2, 0, None))
        return float(np.sum((v-Vb)**2/s**2))
    s = np.sqrt((FRAC*Vb)**2 + FLOOR**2 + np.maximum(eV,1.)**2)
    gV = np.linspace(1.0, 2.78, 24); gc = np.linspace(1, 40, 20)
    best, bc = [2., 8.], np.inf
    for pv in gV:
        for c_ in gc:
            val = c2([pv, c_], s)
            if val < bc: bc, best = val, [pv, c_]
    best = minimize(c2, best, args=(s,), method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-2, maxiter=200)).x
    return (10**best[0], best[1]), c2(best, s)/max(len(R)-2,1)


t1, t2 = io.load_table1(), io.load_table2()
gals = [g for g in sorted(t2.Galaxy.unique())
        if (t2.Galaxy == g).sum() >= 6]

COLS = [("A", "tab:red"), ("B", "tab:blue"), ("C", "tab:purple")]
PER_PAGE = 6

with PdfPages("/mnt/user-data/outputs/rotation_curves_montage.pdf") as pdf:
    # Pull fitted parameters and chi2 from the consistent fixed-slope, common-error
    # analysis (regen_all.py outputs) so the montage labels match Table 2 exactly,
    # rather than recomputing with a possibly different per-model error treatment.
    _fp = pd.read_csv("/mnt/user-data/outputs/fit_parameters.csv").set_index("Galaxy")
    _ic = pd.read_csv("/mnt/user-data/outputs/ic_smooth_ccc.csv").set_index("Galaxy")
    _n2 = pd.read_csv("/mnt/user-data/outputs/ic_nfw2.csv").set_index("Galaxy")
    for p0 in range(0, len(gals), PER_PAGE):
        chunk = gals[p0:p0 + PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(9.5, 2.0 * len(chunk)),
                                 squeeze=False)
        for row, g in enumerate(chunk):
            d = t2[t2.Galaxy == g].sort_values("R")
            R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
            Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
            s0 = cc.inner_slope_of(cc.smooth_vobs(R, V, eV, 0.1), R)
            # parameters and chi2/nu from the consistent analysis
            rt = _fp.loc[g, "CCC_rhot"] / cc.RHO_CGS
            a0 = _fp.loc[g, "MOND_a0"]
            V200, c = _fp.loc[g, "NFW2_V200"], _fp.loc[g, "NFW2_c"]
            cr_a = _ic.loc[g, "CCC_standard_chi2r"]
            cr_b = _ic.loc[g, "MOND_smooth_chi2r"]
            cr_c = _n2.loc[g, "NFW2_chi2r"]
            preds = [
                (sm.vbx_smooth(R, V, eV, rt, kind="standard", inner_slope=s0), r"$\rho_t=%.1f$" % (rt * cc.RHO_CGS / 1e-24) + "\n" + r"$\chi^2_\nu=%.1f$" % cr_a),
                (vb_mond(R, V, eV, a0), r"$a_0=%.1f$" % (a0 / 1e-10) + "\n" + r"$\chi^2_\nu=%.1f$" % cr_b),
                (np.sqrt(np.clip(V**2 - nfw_v(R, V200, c)**2, 0, None)),
                 r"$V_{200}=%.0f$" % V200 + "\n" + r"$c=%.1f,\ \chi^2_\nu=%.1f$" % (c, cr_c)),
            ]
            for col, ((lab, color), (vp, txt)) in enumerate(zip(COLS, preds)):
                ax = axes[row][col]
                ax.errorbar(R, V, yerr=eV, fmt="o", ms=2, lw=0.5, color="0.15",
                            label=r"$V_{\rm obs}$", zorder=3)
                ax.plot(R, Vb, "-", color="0.55", lw=1.2, label=r"$V_b$ (phot.)")
                ax.plot(R, vp, "-", color=color, lw=1.6, label="predicted", zorder=4)
                ax.set_xlim(0, R.max() * 1.02)
                ax.set_ylim(0, max(V.max(), Vb.max()) * 1.15)
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                        fontsize=6.5, bbox=dict(fc="white", ec="0.7", alpha=1.0, pad=1.5),
                        zorder=6)
                ax.tick_params(labelsize=6)
                if col == 0:
                    ax.set_ylabel(f"{g}\nV (km/s)", fontsize=7)
                if row == 0 and col == 0:
                    leg = ax.legend(fontsize=5.5, loc="lower right", framealpha=1.0)
                    leg.get_frame().set_facecolor("white")
                    leg.set_zorder(6)
                if row == 0:
                    ax.set_title(f"Model {lab}", fontsize=9, color=color)
                if row == len(chunk) - 1:
                    ax.set_xlabel("R (kpc)", fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        print(f"page {p0//PER_PAGE + 1}/{(len(gals)+PER_PAGE-1)//PER_PAGE} done", flush=True)

with open("/mnt/user-data/outputs/montage_KEY.txt", "w") as f:
    f.write("Rotation-curve montage — model identity key (keep separate from figures)\n")
    f.write("  Model A = smooth CCC      (fitted parameter rho_t, 1e-24 g/cm3)\n")
    f.write("  Model B = smooth MOND        (fitted parameter a0, 1e-10 m/s2)\n")
    f.write("  Model C = two-parameter NFW  (fitted V200 km/s and concentration c)\n")
    f.write("\nAll fits are inverse mode: each model predicts V_b from V_obs;\n")
    f.write("grey line is SPARC photometric V_b (the comparison target).\n")
print("montage + key written")
