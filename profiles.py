"""
Build chi2(rho_t) profiles per galaxy (inverse mode, correct-geometry target),
so that GLOBAL laws for rho_t can be fitted afterwards:

   universal   rho_t = rho_0                 1 global parameter
   size-scaled rho_t = rho_0 * Rd^(-alpha)   2 global parameters
   per-galaxy  rho_t free                    175 parameters (the status quo)
"""
import sys, numpy as np, pandas as pd
import sparc_io as io, ccc_core as cc
from compare_inverse import inv_ccc, mc_sigma
from ccc_fit2 import FRAC_B, FLOOR

CHUNK, NCH = int(sys.argv[1]), int(sys.argv[2])
LG = np.linspace(4.0, 11.0, 140)          # log10 rho_t in Msun/kpc^3

t1, t2 = io.load_table1(), io.load_table2()
res = pd.read_csv("/mnt/user-data/outputs/ccc_sparc175_results.csv").set_index("Galaxy")
gals = [g for i, g in enumerate(sorted(t2.Galaxy.unique())) if i % NCH == CHUNK]

out = {}
for g in gals:
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    Vb = cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values)
    try:
        rt_best = res.loc[g, "rho_t"] / cc.RHO_CGS
        sig = mc_sigma(lambda vv: inv_ccc(R, vv, eV, rt_best), R, V, eV)
        s2 = sig ** 2 + (FRAC_B * Vb) ** 2 + FLOOR ** 2
        c2 = np.full(len(LG), np.nan)
        for j, lg in enumerate(LG):
            v = inv_ccc(R, V, eV, 10.0 ** lg)
            if v is not None and np.isfinite(v).all():
                c2[j] = float(np.sum((v - Vb) ** 2 / s2))
        out[g] = dict(c2=c2, N=len(R), Rdisk=t1.loc[g, "Rdisk"],
                      Reff=t1.loc[g, "Reff"], T=int(t1.loc[g, "T"]))
    except Exception:
        pass

np.save(f"/home/claude/ccc/prof{CHUNK}.npy", dict(LG=LG, prof=out), allow_pickle=True)
print(f"chunk {CHUNK}: {len(out)} galaxies profiled")
