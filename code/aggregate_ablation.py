from pathlib import Path
import os
import math,glob
import numpy as np,pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; DATA_DIR=ROOT/'data'; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'reproduced')); OUT.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(DATA_DIR/'ablation_phase_raw.csv').drop_duplicates(['model','seed'])
raw.to_csv(OUT/'ablation_phase_raw.csv',index=False)
def wilson(k,n,z=1.959963984540054):
 p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return max(0,c-h),min(1,c+h)
def p50(g):
 x=g.contact_probability.to_numpy();k=g.ignitions.to_numpy();n=g.replicates.to_numpy();pr=k/n
 if k.sum()==0:return np.nan,'not_reached'
 if np.all(pr>=.5):return float(x.min()),'below_scan'
 try:
  fit=sm.GLM(np.column_stack([k,n-k]),sm.add_constant(x),family=sm.families.Binomial()).fit();b0,b1=fit.params
  if b1<=0:return np.nan,'nonpositive_slope'
  v=float(-b0/b1);return v,'estimated' if x.min()<=v<=x.max() else ('below_scan' if v<x.min() else 'above_scan')
 except:return np.nan,'fit_failed'
rows=[]
for keys,g in raw.groupby(['model','flag','contact_probability','duration_generations','duration_years']):
 k=int(g.primary_ignition.sum());n=len(g);lo,hi=wilson(k,n)
 first=np.where(g.first_generation>=0,g.first_generation*25,np.nan)
 rows.append(dict(zip(['model','flag','contact_probability','duration_generations','duration_years'],keys))|{'ignitions':k,'replicates':n,'ignition_probability':k/n,'wilson_low':lo,'wilson_high':hi,'median_first_ignition_year_if_any':float(np.nanmedian(first)) if np.isfinite(first).any() else np.nan,'median_final_repertoire':float(g.final_repertoire.median())})
ens=pd.DataFrame(rows);ens.to_csv(OUT/'ablation_phase_ensemble.csv',index=False)
b=[]
for (model,d),g in ens[ens.model!='No contact event'].groupby(['model','duration_years']):
 v,s=p50(g.sort_values('contact_probability'));b.append({'model':model,'duration_years':d,'estimated_p50':v,'status':s,'maximum_observed_probability':float(g.ignition_probability.max())})
bounds=pd.DataFrame(b);bounds.to_csv(OUT/'ablation_phase_boundaries.csv',index=False)
s=[]
for model,g in ens.groupby('model'):
 s.append({'model':model,'total_ignitions':int(g.ignitions.sum()),'total_replicates':int(g.replicates.sum()),'max_ignition_probability':float(g.ignition_probability.max()),'conditions_with_any_ignition':int((g.ignitions>0).sum()),'conditions_with_at_least_50pct':int((g.ignition_probability>=.5).sum())})
summary=pd.DataFrame(s);summary.to_csv(OUT/'ablation_phase_summary.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(15,4.7),sharey=True)
for ax,d in zip(axes,sorted(ens[ens.model!='No contact event'].duration_years.unique())):
 for model,g in ens[(ens.duration_years==d)&(ens.model!='No contact event')].groupby('model'):
  g=g.sort_values('contact_probability');ax.plot(g.contact_probability,g.ignition_probability,marker='o',label=model)
 ax.axhline(.5,ls='--');ax.set_title(f'{int(d):,} years');ax.set_xlabel('External-model probability');ax.set_ylim(-.03,1.03)
axes[0].set_ylabel('Persistent ignition probability');axes[-1].legend(fontsize=7,loc='center left',bbox_to_anchor=(1.02,.5));fig.suptitle('Ablation phase grids: removal may eliminate or shift ignition');fig.tight_layout();fig.savefig(OUT/'ablation_phase_grids.png',dpi=220,bbox_inches='tight');plt.close(fig)
print('rows',len(raw));print(summary.to_string(index=False));print(bounds.to_string(index=False))
