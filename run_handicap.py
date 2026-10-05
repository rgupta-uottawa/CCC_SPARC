"""
Spherical handicap control.

Same inverse-mode fits as before, but the comparison target is swapped:

  target = V_b        SPARC's correct thin-disk solution   (status quo)
  target = V_b,sph    spherically-approximated baryons     (handicapped)

Expectation:
  CCC  produces a spherical-equivalent V_bX, so V_b,sph is the geometrically
       CONSISTENT target -> CCC should improve.
  MOND and NFW are built on the correct disk geometry, so V_b,sph is a
       DISTORTED target -> they should degrade.

If the gap closes, the deficit was geometric. If it does not, it is physical.
"""
import numpy as np, pandas as pd
import sparc_io as io, ccc_core as cc, spherical as sph
from compare_inverse import fit_inverse, inv_mond, inv_nfw, inv_ccc

t1, t2 = io.load_table1(), io.load_table2()
MODES = [("Vb_disk", None), ("Vb_sph_keep", "keep"), ("Vb_sph_scale", "scale")]
import sys, os
CHUNK = int(sys.argv[1]); NCH = int(sys.argv[2])

rows = []
GALS = sorted(t2.Galaxy.unique())
GALS = [x for i, x in enumerate(GALS) if i % NCH == CHUNK]
for g in GALS:
    d = t2[t2.Galaxy == g].sort_values("R")
    R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
    rec = dict(Galaxy=g, N=len(R), T=int(t1.loc[g, "T"]))
    for tag, gm in MODES:
        Vb = (cc.sparc_vbar(d.Vgas.values, d.Vdisk.values, d.Vbul.values) if gm is None
              else sph.vb_spherical(R, d.SBdisk.values, d.Vgas.values,
                                    d.Vdisk.values, d.Vbul.values, gas_mode=gm))
        for name, pred, lo, hi in [
                ("ccc", lambda rt, vv: inv_ccc(R, vv, eV, rt), np.log10(1e3), np.log10(1e12)),
                ("mond", lambda a0, vv: inv_mond(R, vv, a0), -12.0, -8.0),
                ("nfw", lambda v2, vv: inv_nfw(R, vv, v2), np.log10(10.0), np.log10(600.0))]:
            try:
                p, c2, fr, s = fit_inverse(R, V, eV, Vb, pred, lo, hi)
                rec[f"{name}_{tag}_chi2r"] = c2
                rec[f"{name}_{tag}_frms"] = fr
            except Exception:
                pass
    rows.append(rec)

df = pd.DataFrame(rows)
df["dwarf"] = df["T"] >= 8
df.to_csv(f"/home/claude/ccc/handicap_part{CHUNK}.csv", index=False)

print(f"chunk {CHUNK} done: {len(df)} galaxies")
