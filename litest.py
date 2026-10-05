import numpy as np, pandas as pd
import sparc_io as io, ccc_core as cc, smooth_ccc as sm
from scipy.optimize import minimize_scalar

t1, t2 = io.load_table1(), io.load_table2()
K = 3.24078e-17 * 1e3            # (km/s)^2/kpc -> m/s^2 : 1 km^2/s^2/kpc
# careful unit constant for g = V^2/R: use cc if available
KMS2_KPC_TO_MS2 = (1e3)**2 / (3.0856775814913673e19)   # (km/s)^2 per kpc -> m/s^2

# Their Table 2 nuisance values
GAL = {
 'NGC2841': dict(Ydisk=0.81, Ybul=0.93, D=15.5, i=82.2, chi2_their=1.515, Dsparc=14.10, isparc=76.0),
 'IC2574' : dict(Ydisk=0.07, Ybul=0.0 , D=3.77, i=64.5, chi2_their=1.44 , Dsparc=3.91 , isparc=75.0),
}

# Their interpolating function (RAR / "simple" nu): g_obs = g_bar / (1 - exp(-sqrt(g_bar/g_dagger)))
def mond_gobs_their(gbar, gdag):
    x = np.sqrt(np.clip(gbar/gdag, 1e-30, None))
    return gbar / (1.0 - np.exp(-x))

# "Our" standard MOND mu-function form: nu(y)=0.5+sqrt(0.25+1/y) style (standard interpolating)
def mond_gobs_ours(gbar, gdag):
    y = np.clip(gbar/gdag, 1e-30, None)
    nu = 0.5 + np.sqrt(0.25 + 1.0/y)
    return gbar*nu

def prep(gal):
    p = GAL[gal]
    d = t2[t2.Galaxy==gal].sort_values('R')
    R = d.R.values.copy(); Vobs = d.Vobs.values.copy(); eV = d.e_Vobs.values.copy()
    Vgas=d.Vgas.values.copy(); Vdisk=d.Vdisk.values.copy(); Vbul=d.Vbul.values.copy()
    # distance rescale: R ~ D, baryonic V ~ sqrt(D). Vobs NOT affected by D.
    fD = p['D']/p['Dsparc']
    R = R*fD
    Vgas*=np.sqrt(fD); Vdisk*=np.sqrt(fD); Vbul*=np.sqrt(fD)
    # inclination: Vobs measured with sin i; adopting their i rescales Vobs by sin(i_sparc)/sin(i_their)
    fI = np.sin(np.radians(p['isparc']))/np.sin(np.radians(p['i']))
    Vobs = Vobs*fI; eV = eV*fI
    # baryonic model velocity with THEIR Y
    Vb2 = Vgas*np.abs(Vgas) + p['Ydisk']*Vdisk**2 + p['Ybul']*Vbul**2
    Vb = np.sqrt(np.clip(Vb2,0,None))
    return R,Vobs,eV,Vb,Vgas,Vdisk,Vbul,p

def chi2_mond_direct(R,Vobs,eV,Vb,gdag,func):
    gbar = Vb**2/R*KMS2_KPC_TO_MS2
    gobs = func(gbar,gdag)
    Vmod = np.sqrt(np.clip(gobs/KMS2_KPC_TO_MS2*R,0,None))
    return np.sum(((Vmod-Vobs)/eV)**2)

def chi2_mond_inverse(R,Vobs,eV,Vb,gdag,func):
    # predict Vb from Vobs: invert gobs->gbar via the same relation, compare to baryonic Vb
    gobs = Vobs**2/R*KMS2_KPC_TO_MS2
    # invert: find gbar such that func(gbar)=gobs
    grid=np.logspace(-13,-7,4000)
    pred=func(grid,gdag)
    gbar_pred=np.interp(gobs,pred,grid)
    Vbmod=np.sqrt(np.clip(gbar_pred/KMS2_KPC_TO_MS2*R,0,None))
    return np.sum(((Vbmod-Vb)/np.maximum(0.05*Vb,1.0))**2)

def fit_scale(fn,lo,hi):
    r=minimize_scalar(fn,bounds=(lo,hi),method='bounded',options=dict(xatol=1e-4))
    return r.x,r.fun

print('%-9s %-13s %-7s %8s %8s %8s'%('galaxy','model/mode','func','scale','chi2/N-1','chi2/N-3'))
for gal in ['NGC2841','IC2574']:
    R,Vobs,eV,Vb,Vgas,Vdisk,Vbul,p=prep(gal)
    N=len(R)
    for func,fname in [(mond_gobs_their,'their'),(mond_gobs_ours,'ours')]:
        # MOND direct: fit log10 gdag
        f=lambda lg: chi2_mond_direct(R,Vobs,eV,Vb,10**lg,func)
        lg,c=fit_scale(f,-11,-9); 
        print('%-9s %-13s %-7s %8.2e %8.2f %8.2f'%(gal,'MOND direct',fname,10**lg,c/(N-1),c/(N-3)))
        # MOND inverse
        f2=lambda lg: chi2_mond_inverse(R,Vobs,eV,Vb,10**lg,func)
        lg2,c2=fit_scale(f2,-11,-9)
        print('%-9s %-13s %-7s %8.2e %8.2f %8.2f'%(gal,'MOND inverse',fname,10**lg2,c2/(N-1),c2/(N-3)))
    print('   (their quoted chi2/nu = %.3f)'%p['chi2_their'])

print()
print('=== ADDING CCC (both modes) with their nuisance parameters ===')
print('%-9s %-13s %8s %8s %8s'%('galaxy','model/mode','rho_t','chi2/N-1','chi2/N-3'))
def chi2_ccc_inverse(R,Vobs,eV,Vb,rt):
    s0=cc.inner_slope_of(cc.smooth_vobs(R,Vobs,eV,0.1),R)
    vpred=sm.vbx_smooth(R,Vobs,eV,rt,kind='standard',inner_slope=s0)
    if vpred is None or not np.isfinite(vpred).all(): return 1e12
    return np.sum(((vpred-Vb)/np.maximum(0.05*Vb,1.0))**2)

def chi2_ccc_direct(R,Vobs,eV,Vb,rt):
    # direct: predict Vobs from Vb. CCC forward: rho_o = sqrt(rho_t * rho_bX)
    # Build baryonic density from Vb, apply forward relation, predict Vobs.
    sp=cc.smooth_vobs(R,Vb,np.maximum(0.05*Vb,1.0),0.1)
    s0=cc.inner_slope_of(sp,R)
    rg,rhob,dMdR,Mb=cc.density_profile(sp,R,n_grid=1200,inner_slope=s0)
    rho_o=np.sqrt(np.clip(rt*np.abs(rhob),0,None))  # forward density relation
    # enclosed mass from rho_o
    integ=4*np.pi*rg**2*rho_o
    Mo=np.concatenate([[0],np.cumsum(np.diff(rg)*0.5*(integ[1:]+integ[:-1]))])
    Vpred=np.interp(R,rg,np.sqrt(np.clip(cc.G*Mo/rg,0,None)))
    if not np.isfinite(Vpred).all(): return 1e12
    return np.sum(((Vpred-Vobs)/eV)**2)

for gal in ['NGC2841','IC2574']:
    R,Vobs,eV,Vb,Vgas,Vdisk,Vbul,p=prep(gal)
    N=len(R)
    fi=lambda lr: chi2_ccc_inverse(R,Vobs,eV,Vb,10**lr)
    lri,ci=fit_scale(fi,-24.5,-22.5)
    print('%-9s %-13s %8.2e %8.2f %8.2f'%(gal,'CCC inverse',10**lri,ci/(N-1),ci/(N-3)))
    fd=lambda lr: chi2_ccc_direct(R,Vobs,eV,Vb,10**lr)
    lrd,cd=fit_scale(fd,-24.5,-22.5)
    print('%-9s %-13s %8.2e %8.2f %8.2f'%(gal,'CCC direct',10**lrd,cd/(N-1),cd/(N-3)))
