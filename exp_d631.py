"""
Test D631-7 at rho_t = 2.79 (published) and 3.48 (raw left-endpoint FD),
under two different reconstructions of rho_o(R).
"""
import numpy as np
import sparc_io as io, ccc_core as cc, ccc_fit2 as cf

t2 = io.load_table2()
d = t2[t2.Galaxy == "D631-7"].sort_values("R")
R, V, eV = d.R.values, d.Vobs.values, d.e_Vobs.values
Vgas, Vdisk, Vbul = d.Vgas.values, d.Vdisk.values, d.Vbul.values
Vb = cc.sparc_vbar(Vgas, Vdisk, Vbul)


def raw_fd_density(R, V, n_grid=1500):
    """Paper-style: one-sided finite differences on the tabulated grid,
    evaluated at the LEFT endpoint of each interval."""
    M = V ** 2 * R / cc.G
    dM = np.diff(M) / np.diff(R)
    rho_nodes = dM / (4.0 * np.pi * R[:-1] ** 2)
    Rn = R[:-1]
    rg = np.linspace(1e-3, R[-1], n_grid)
    rho = np.interp(rg, Rn, rho_nodes, left=rho_nodes[0], right=rho_nodes[-1])
    dMdR = rho * 4.0 * np.pi * rg ** 2
    return rg, rho, dMdR


# smoothed-spline reconstruction (fiducial)
sp = cc.smooth_vobs(R, V, eV, s_factor=0.1)
rg_s, rho_s, dM_s, _ = cc.density_profile(sp, R, n_grid=1500)

# raw finite-difference reconstruction (paper-style)
rg_f, rho_f, dM_f = raw_fd_density(R, V)

# model uncertainty: Monte Carlo through the fiducial pipeline
ens = cf.build_mc(R, V, eV, n_mc=300, s_factor=0.1, seed=1)

print("max rho_o available:  smoothed spline %.2f   raw FD %.2f  [1e-24 g/cm3]\n"
      % (rho_s.max() * cc.RHO_CGS / 1e-24, rho_f.max() * cc.RHO_CGS / 1e-24))

for label, (rg, rho, dMdR) in [("smoothed spline (f=0.1)", (rg_s, rho_s, dM_s)),
                               ("raw FD at left endpoint", (rg_f, rho_f, dM_f))]:
    print(f"--- rho_o from {label} ---")
    print(f"{'rho_t':>7} {'R_t':>7} {'chi2/nu':>8} {'fRMS':>7} {'<X>out':>7}   note")
    for rt_cgs, note in [(1.50, "fiducial best fit"), (2.10, ""), (2.79, "published"),
                         (3.48, "raw FD at R1"), (5.00, ""), (10.0, "")]:
        rho_t = rt_cgs * 1e-24 / cc.RHO_CGS
        m = cc.ccc_model(rg, rho, dMdR, rho_t)
        if m is None:
            print(f"{rt_cgs:7.2f}      --       --      --       --   rho_t below profile")
            continue
        Vm = cc.vbx_at(rg, m["MbX"], R)
        s = cf.sigma_mc(ens, R, rho_t)
        sig = np.sqrt(s ** 2 + (cf.FRAC_B * Vb) ** 2 + 1.0) if s is not None \
            else np.sqrt((cf.FRAC_B * Vb) ** 2 + 1.0)
        chi2n = float(np.sum((Vm - Vb) ** 2 / sig ** 2)) / (len(R) - 1)
        frms = 100 * np.sqrt(np.mean((Vm - Vb) ** 2)) / np.mean(Vb)
        Rt = m["Rt"]
        Xout = float(np.mean(m["X"][rg > max(Rt, R[0])]))
        rts = f"{Rt:7.2f}" if Rt > 1e-2 else "  <0.01"
        print(f"{rt_cgs:7.2f} {rts} {chi2n:8.2f} {frms:6.1f}% {Xout:7.3f}   {note}")
    print()

# what inner slope would be needed to support rho_t = 2.79 at R_t = 0.35 ?
print("--- can an inward extrapolation support (rho_t, R_t) = (2.79, 0.35)? ---")
for lab, rho_edge in [("smoothed spline", np.interp(R[0], rg_s, rho_s) * cc.RHO_CGS / 1e-24),
                      ("raw FD at R1", rho_f[0] * cc.RHO_CGS / 1e-24)]:
    ratio = 2.79 / rho_edge
    alpha = 1.0 + 0.5 * np.log(ratio) / np.log(0.35 / R[0])
    ok = "physical (V->0 at r=0)" if alpha > 0 else "UNPHYSICAL: needs V rising inward"
    print(f"  rho_o(R_min)={rho_edge:5.2f} via {lab:16s} -> needs V ~ r^{alpha:5.2f}   {ok}")
