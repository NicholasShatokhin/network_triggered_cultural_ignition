from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; DATA_DIR=ROOT/'data'; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'reproduced')); OUT.mkdir(parents=True,exist_ok=True)
TOPOLOGY_SEEDS={
    'Global':202607280,
    'Ring':202607281,
    'Lattice':202607282,
    'Erdos-Renyi':202607283,
    'Small-world':202607284,
    'Scale-free':202607285,
    'Modular bridges':202607286,
    'Wheel':202607287,
    'Gravity':202607288,
}
raw=pd.read_csv(DATA_DIR/'topology_independent_realisations_raw.csv')

def p50_arrays(x,k,n):
    p=np.maximum.accumulate(k/n)
    if p[-1] < .5: return np.nan
    if p[0] >= .5: return float(x[0])
    j=np.flatnonzero(p>=.5)[0]
    return float(x[j-1]+(.5-p[j-1])*(x[j]-x[j-1])/max(p[j]-p[j-1],1e-12))

def bootstrap_top(g,rng,nboot=2000):
    contacts=np.array(sorted(g.contact_probability.unique()),float)
    cells=[]
    for c in contacts:
        h=g[g.contact_probability==c]
        cells.append((h.ignitions.to_numpy(int),h.agent_replicates.to_numpy(int)))
    vals=np.empty(nboot,float)
    for b in range(nboot):
        ks=np.empty(len(contacts),float); ns=np.empty(len(contacts),float)
        for i,(karr,narr) in enumerate(cells):
            if len(karr)>1:
                idx=rng.integers(0,len(karr),size=len(karr))
                ks[i]=karr[idx].sum(); ns[i]=narr[idx].sum()
            else:
                n=int(narr[0]); p=float(karr[0]/n); ks[i]=rng.binomial(n,p); ns[i]=n
        vals[b]=p50_arrays(contacts,ks,ns)
    vals=vals[np.isfinite(vals)]
    return float(np.median(vals)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),len(vals)

summary=[]; curve_rows=[]
for top,g in raw.groupby('topology'):
    q=g.groupby('contact_probability',as_index=False).agg(k=('ignitions','sum'),n=('agent_replicates','sum')).sort_values('contact_probability')
    x=q.contact_probability.to_numpy(float); k=q.k.to_numpy(float); n=q.n.to_numpy(float)
    for c,kk,nn in zip(x,k,n):curve_rows.append({'topology':top,'contact_probability':c,'k':kk,'n':nn,'probability':kk/nn})
    med,lo,hi,nb=bootstrap_top(g,np.random.default_rng(TOPOLOGY_SEEDS[top]))
    summary.append({'topology':top,'pooled_p50':p50_arrays(x,k,n),'bootstrap_median_p50':med,'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'network_realisations_per_contact':int(g.groupby('contact_probability').network_realization.nunique().median()),'agent_replicates_per_network_contact':int(g.agent_replicates.iloc[0]),'bootstrap_valid':nb})
summary=pd.DataFrame(summary).sort_values('bootstrap_median_p50'); curves=pd.DataFrame(curve_rows)
summary.to_csv(OUT/'topology_independent_realisations_summary.csv',index=False);curves.to_csv(OUT/'topology_independent_realisations_curves.csv',index=False)

stoch=raw[raw.topology.isin(['Erdos-Renyi','Small-world','Scale-free','Modular bridges','Gravity'])].copy()
stoch['ignition_residual']=stoch.ignition_probability-stoch.groupby('contact_probability').ignition_probability.transform('mean')
metric_rows=[]
for col in ['density','clustering','mean_shortest_path','degree_cv','modularity']:
    r,p=spearmanr(stoch[col],stoch.ignition_residual,nan_policy='omit'); metric_rows.append({'metric':col,'spearman_with_contact_adjusted_ignition':r,'p_value':p,'n':len(stoch)})
pd.DataFrame(metric_rows).to_csv(OUT/'topology_metric_associations_independent.csv',index=False)

plt.figure(figsize=(9.2,5.8));s=summary.sort_values('bootstrap_median_p50',ascending=False);y=np.arange(len(s));x=s.bootstrap_median_p50.to_numpy();err=np.vstack([x-s.bootstrap_ci_low.to_numpy(),s.bootstrap_ci_high.to_numpy()-x]);plt.errorbar(x,y,xerr=err,fmt='o',capsize=4);plt.yticks(y,s.topology);plt.xlabel('External-model probability at 50% persistent ignition');plt.title('Topology effects across independent network realisations');plt.tight_layout();plt.savefig(OUT/'topology_independent_realisations_p50.png',dpi=220);plt.close()
plt.figure(figsize=(9.8,5.8));
for top,g in curves.groupby('topology'):plt.plot(g.contact_probability,g.probability,marker='o',label=top)
plt.axhline(.5,ls='--');plt.xlabel('External-model probability');plt.ylabel('Persistent ignition probability');plt.title('Pooled topology phase curves');plt.legend(fontsize=7,ncol=2);plt.tight_layout();plt.savefig(OUT/'topology_independent_realisations_curves.png',dpi=220);plt.close()
print(summary.to_string(index=False));print(pd.DataFrame(metric_rows).to_string(index=False))
