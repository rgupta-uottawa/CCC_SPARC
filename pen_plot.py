import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'figure.dpi':150,'savefig.dpi':150})
res=np.array(json.load(open('pen_curve.json')))
mond=2.650
fig,ax=plt.subplots(figsize=(8,5.5))
ax.plot(res[:,0],res[:,1],'o-',color='tab:red',lw=2,ms=5,label=r'CCC mean $\chi^2_\nu$')
ax.axhline(mond,color='tab:blue',lw=1.8,ls='--',label=r'MOND mean $\chi^2_\nu$ = 2.65')
idx=np.where(np.diff(np.sign(res[:,1]-mond)))[0]
if len(idx):
    i=idx[0]; x0,x1=res[i,0],res[i+1,0]; y0,y1=res[i,1],res[i+1,1]
    xc=x0+(mond-y0)*(x1-x0)/(y1-y0)
    ax.axvline(xc,color='0.5',ls=':',lw=1)
    ax.plot(xc,mond,'ks',ms=9)
    ax.annotate('CCC = MOND at\npenalty = %.2f'%xc,xy=(xc,mond),xytext=(xc+0.07,mond+0.22),
                fontsize=10,arrowprops=dict(arrowstyle='->',color='0.4'))
ax.set_xlabel(r'differentiation penalty factor $p$   ($V_{bX}$ err $= p\times e_{V_{obs}}$)')
ax.set_ylabel(r'mean per-galaxy $\chi^2_\nu$')
ax.set_title('CCC fit quality vs differentiation penalty\n(MOND and NFW error models unchanged)')
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_xlim(0.98,1.52)
fig.tight_layout(); fig.savefig('/mnt/user-data/outputs/penalty_curve.png',bbox_inches='tight')
print('crossing at penalty=%.3f'%xc)
