import numpy as np, pandas as pd, sparc_io as io, ccc_core as cc

t1, t2 = io.load_table1(), io.load_table2()
PUB = {'D631-7': (2.79, 0.35, 3.10), 'NGC1090': (2.23, 4.7, 1.76),
       'NGC3198': (2.09, 3.37, 2.15), 'NGC3521': (7.96, 2.15, 6.43),
       'NGC6503': (5.17, 1.9, 2.25), 'NGC7814': (5.01, 2.8, 5.33),
       'UGC09133': (3.98, 4.9, 3.33)}
ORDER = ['D631-7', 'NGC1090', 'NGC3198', 'NGC3521', 'NGC6503', 'NGC7814', 'UGC09133']

KMS2_KPC_TO_CGS = 1e10 / cc.KPC_CM      # (km/s)^2/kpc -> cm/s^2


def run(gal, s_factor=1.0):
    d = t2[t2.Galaxy == gal].sort_values('R')
    r = cc.fit_galaxy(d.R.values, d.Vobs.values, d.e_Vobs.values,
                      d.Vgas.values, d.Vdisk.values, d.Vbul.values,
                      s_factor=s_factor)
    r['Galaxy'] = gal
    r['R'] = d.R.values
    r['Vobs'] = d.Vobs.values
    r['e_Vobs'] = d.e_Vobs.values
    r['Vflat'] = t1.loc[gal, 'Vflat']
    r['T'] = io.TYPES[int(t1.loc[gal, 'T'])]
    r['D'] = t1.loc[gal, 'D']
    r['Reff'] = t1.loc[gal, 'Reff']
    r['rho_t_cgs'] = r['rho_t'] * cc.RHO_CGS
    r['rho_t_lo_cgs'] = r['rho_t_lo'] * cc.RHO_CGS
    r['rho_t_hi_cgs'] = r['rho_t_hi'] * cc.RHO_CGS
    r['a_t'] = r['Vflat'] ** 2 / r['Rt'] * KMS2_KPC_TO_CGS if r['Rt'] > 0 else np.nan
    return r


if __name__ == '__main__':
    res = {g: run(g) for g in ORDER}
    rows = []
    for g in ORDER:
        r = res[g]
        pr, pR, pa = PUB[g]
        rows.append(dict(Galaxy=g, Type=r['T'], N=r['npts'],
                         Rt=r['Rt'], Rt_pub=pR,
                         rho_t=r['rho_t_cgs'] / 1e-24, rho_t_pub=pr,
                         lo=r['rho_t_lo_cgs'] / 1e-24, hi=r['rho_t_hi_cgs'] / 1e-24,
                         a_t=r['a_t'] / 1e-8, a_t_pub=pa,
                         chi2r=r['chi2r'], fRMS=100 * r['frms'], flag=r['flag'],
                         clip=100 * r['clip_frac']))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200, 'display.float_format', lambda v: f'{v:8.2f}')
    print(df.to_string(index=False))
    np.save('/home/claude/ccc/res7.npy', res, allow_pickle=True)

    print('\n--- smoothing sensitivity: rho_t [1e-24 g/cm3] ---')
    for sf in [0.25, 0.5, 1.0, 2.0, 4.0]:
        vals = [run(g, sf)['rho_t_cgs'] / 1e-24 for g in ORDER]
        print(f's={sf:4.2f}  ' + '  '.join(f'{v:6.2f}' for v in vals))
