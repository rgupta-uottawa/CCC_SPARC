"""
Spherical handicap control.

SPARC's V_disk comes from solving Poisson's equation in the correct thin-disk
geometry. The CCC analysis instead uses the spherical approximation
M(<R) = V^2 R / G throughout. To find out how much of CCC's apparent deficit is
geometric rather than physical, we build the spherically-approximated stellar
disk curve directly from the tabulated [3.6] surface brightness:

    M_d(<R) = 2 pi * 1e6 * Upsilon_d * Int_0^R SB_disk(R') R' dR'     [Msun]
    V_d,sph(R) = sqrt(G M_d(<R) / R)

i.e. all mass inside cylindrical radius R treated as if it were interior to a
sphere of that radius -- exactly the approximation CCC makes.

The bulge is left alone: SPARC's V_bul already comes from a deprojected
spherical density, so M = V^2 R / G is exact for it and there is no handicap
to apply. The gas has no tabulated surface density in Table 2, so it is
handled two ways to bracket the answer:
    gas_mode='keep' : V_gas untouched  -> conservative, understates the penalty
    gas_mode='scale': V_gas scaled by the stellar disk's geometric factor f(R)
"""
import numpy as np
import ccc_core as cc

YD, YB = cc.YD, cc.YB


def disk_mass_cyl(R, SB, Yd=YD):
    """Cumulative cylindrical mass of the stellar disk, Msun. R in kpc,
    SB in Lsun/pc^2. Inner extrapolation is exponential from the first
    two tabulated points (disks rise inward)."""
    R = np.asarray(R, float)
    SB = np.clip(np.asarray(SB, float), 0.0, None)
    if len(R) > 1 and SB[0] > 0 and SB[1] > 0 and R[1] > R[0]:
        h = (R[1] - R[0]) / max(np.log(SB[0] / SB[1]), 1e-3)
        SB0 = SB[0] * np.exp(R[0] / h)
    else:
        SB0 = SB[0] if len(SB) else 0.0
    rr = np.concatenate([[0.0], R])
    ss = np.concatenate([[SB0], SB])
    integ = ss * rr                                   # SB * R
    cum = np.concatenate([[0.0], np.cumsum(np.diff(rr) * 0.5 * (integ[1:] + integ[:-1]))])
    return 2.0 * np.pi * 1e6 * Yd * cum[1:]           # drop the R=0 entry


def vb_spherical(R, SBdisk, Vgas, Vdisk, Vbul, gas_mode="keep",
                 Yd=YD, Yb=YB, return_parts=False):
    """Spherically-approximated baryonic curve."""
    Md = disk_mass_cyl(R, SBdisk, Yd)
    Vd_sph = np.sqrt(np.clip(cc.G * Md / np.maximum(R, 1e-6), 0.0, None))

    Vd_true = np.sqrt(Yd) * np.asarray(Vdisk, float)       # SPARC disk solution
    f = np.where(Vd_true > 1e-6, Vd_sph / np.maximum(Vd_true, 1e-6), 1.0)

    Vg = np.asarray(Vgas, float)
    Vg_use = Vg * f if gas_mode == "scale" else Vg

    v2 = Vg_use * np.abs(Vg_use) + Vd_sph ** 2 + Yb * np.asarray(Vbul, float) ** 2
    Vb_sph = np.sqrt(np.clip(v2, 0.0, None))
    if return_parts:
        return Vb_sph, Vd_sph, Vd_true, f
    return Vb_sph
