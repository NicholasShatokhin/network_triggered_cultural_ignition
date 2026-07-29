from __future__ import annotations
from pathlib import Path
import os
import json, math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; DATA_DIR=ROOT/'data'; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'reproduced')); OUT.mkdir(parents=True,exist_ok=True); DATA=Path(os.environ.get('CULTURAL_IGNITION_ARCHAEOLOGY_DATA', DATA_DIR/'archaeological_single_chain_data.csv'))
RNG=np.random.default_rng(20260728); df=pd.read_csv(DATA).copy(); GRID=np.arange(25,3000,25,dtype=float)

def ll_counts(k,n):
    k=np.asarray(k,float); n=np.asarray(n,float); p=(k+.5)/(n+1.0)
    return k*np.log(p)+(n-k)*np.log1p(-p)

def best_cp(age,y):
    age=np.asarray(age,float); y=np.asarray(y,int); masks=age[:,None]>GRID[None,:]
    nold=masks.sum(axis=0); ny=len(y)-nold; valid=(nold>=5)&(ny>=5)
    kold=y@masks; ky=y.sum()-kold
    ll=ll_counts(kold,nold)+ll_counts(ky,ny)-ll_counts(y.sum(),len(y)); ll[~valid]=-np.inf
    j=int(np.argmax(ll)); return float(ll[j]),float(GRID[j]),float(kold[j]/nold[j]),float(ky[j]/ny[j])

age=df.age_mid_ka.to_numpy(); y=df.above_noncumulative_baseline.astype(int).to_numpy(); obs_delta,obs_cp,obs_old,obs_young=best_cp(age,y)
# Vectorised permutation test.
N_PERM=10000; masks=age[:,None]>GRID[None,:]; nold=masks.sum(0); ny=len(y)-nold; valid=(nold>=5)&(ny>=5)
Y=np.vstack([RNG.permutation(y) for _ in range(N_PERM)]); kold=Y@masks; ky=Y.sum(1)[:,None]-kold
ll=ll_counts(kold,nold)+ll_counts(ky,ny)-ll_counts(Y.sum(1),len(y))[:,None]; ll[:,~valid]=-np.inf
perm_delta=ll.max(axis=1); perm_p=(1+np.sum(perm_delta>=obs_delta))/(N_PERM+1)

# Date uncertainty in batches.
N_DATE=5000; young=df['KA.young'].to_numpy(float); old=df['KA.old'].to_numpy(float); lo=np.minimum(young,old); hi=np.maximum(young,old)
date_cp=[]; date_delta=[]
for start in range(0,N_DATE,250):
    b=min(250,N_DATE-start); ages=RNG.uniform(lo,hi,size=(b,len(df)))
    mm=ages[:,:,None]>GRID[None,None,:]; no=mm.sum(1); nny=len(y)-no; va=(no>=5)&(nny>=5)
    ko=np.einsum('i,bij->bj',y,mm); kyy=y.sum()-ko
    l=ll_counts(ko,no)+ll_counts(kyy,nny)-ll_counts(y.sum(),len(y)); l[~va]=-np.inf
    jj=np.argmax(l,axis=1); date_cp.extend(GRID[jj]); date_delta.extend(l[np.arange(b),jj])
date_cp=np.asarray(date_cp); date_delta=np.asarray(date_delta)

N_BOOT=2500
def row_bootstrap(n=N_BOOT):
    cps=[];ds=[]
    for _ in range(n):
        idx=RNG.integers(0,len(df),len(df));d,cp,_,_=best_cp(age[idx],y[idx]);cps.append(cp);ds.append(d)
    return np.asarray(cps),np.asarray(ds)
def cluster_bootstrap(col,n=N_BOOT):
    groups={k:g.index.to_numpy() for k,g in df.groupby(col,dropna=False)}; keys=list(groups);cps=[];ds=[]
    for _ in range(n):
        idx=np.concatenate([groups[k] for k in RNG.choice(keys,size=len(keys),replace=True)]);d,cp,_,_=best_cp(age[idx],y[idx]);cps.append(cp);ds.append(d)
    return np.asarray(cps),np.asarray(ds)
row_cp,row_delta=row_bootstrap(); source_cp,source_delta=cluster_bootstrap('Source'); site_cp,site_delta=cluster_bootstrap('Sitename')

loo=[]
for col in ['Source','Sitename']:
    for value in df[col].drop_duplicates():
        mask=df[col]!=value;d,cp,po,py=best_cp(age[mask],y[mask]);loo.append({'cluster_type':col,'omitted_cluster':str(value),'n_remaining':int(mask.sum()),'change_point_ka':cp,'delta_loglik':d,'older_rate':po,'younger_rate':py})
loo_df=pd.DataFrame(loo);loo_df.to_csv(OUT/'archaeology_leave_one_cluster_out.csv',index=False)

sens=[]
for baseline in [5,6,7]:
    yy=(df.procedural_units>baseline).astype(int).to_numpy();d,cp,po,py=best_cp(age,yy)
    for analysis,boundary in [('change_point',cp)]+[('fixed_boundary',v) for v in [500,525,600,700]]:
        a=yy[age<=boundary];b=yy[age>boundary];_,p=fisher_exact([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])
        sens.append({'analysis':analysis,'baseline':baseline,'boundary_ka':boundary,'older_successes':int(b.sum()),'older_n':len(b),'younger_successes':int(a.sum()),'younger_n':len(a),'older_rate':float(b.mean()),'younger_rate':float(a.mean()),'delta_loglik':d if analysis=='change_point' else np.nan,'fisher_p':p})
sens_df=pd.DataFrame(sens);sens_df.to_csv(OUT/'archaeology_threshold_sensitivity.csv',index=False)

def effect_at_boundary(indices,boundary=600,baseline=6):
    a=df.iloc[indices];yy=(a.procedural_units>baseline).astype(int).to_numpy();aa=a.age_mid_ka.to_numpy();om=aa>boundary;ym=~om
    if om.sum()==0 or ym.sum()==0:return (np.nan,)*3
    po=(yy[om].sum()+.5)/(om.sum()+1);py=(yy[ym].sum()+.5)/(ym.sum()+1)
    return py-po,py/po,math.log(py/(1-py))-math.log(po/(1-po))
effect_rows=[]
for label,col in [('row',None),('source','Source'),('site','Sitename')]:
    vals=[]
    if col is None:
        for _ in range(N_BOOT):vals.append(effect_at_boundary(RNG.integers(0,len(df),len(df))))
    else:
        groups={k:g.index.to_numpy() for k,g in df.groupby(col,dropna=False)};keys=list(groups)
        for _ in range(N_BOOT): vals.append(effect_at_boundary(np.concatenate([groups[k] for k in RNG.choice(keys,size=len(keys),replace=True)])))
    vals=np.asarray(vals,float)
    for j,metric in enumerate(['risk_difference','risk_ratio','log_odds_ratio']):
        v=vals[:,j];v=v[np.isfinite(v)];effect_rows.append({'bootstrap':label,'metric':metric,'median':float(np.median(v)),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975)),'n_valid':len(v)})
effect_df=pd.DataFrame(effect_rows);effect_df.to_csv(OUT/'archaeology_cluster_bootstrap_effects.csv',index=False)

def qs(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)];return {'median':float(np.median(v)),'q025':float(np.quantile(v,.025)),'q975':float(np.quantile(v,.975))}
summary={'n_sequences':len(df),'observed_change_point_ka':obs_cp,'observed_delta_loglik':obs_delta,'permutation_replicates':N_PERM,'permutation_p_value':perm_p,'date_uncertainty_change_point':qs(date_cp),'row_bootstrap_change_point':qs(row_cp),'source_cluster_bootstrap_change_point':qs(source_cp),'site_cluster_bootstrap_change_point':qs(site_cp),'leave_one_source_cp_range':[float(loo_df[loo_df.cluster_type=='Source'].change_point_ka.min()),float(loo_df[loo_df.cluster_type=='Source'].change_point_ka.max())],'leave_one_site_cp_range':[float(loo_df[loo_df.cluster_type=='Sitename'].change_point_ka.min()),float(loo_df[loo_df.cluster_type=='Sitename'].change_point_ka.max())]}
(OUT/'archaeology_robustness_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame([{'analysis':name,'median_change_point_ka':qs(v)['median'],'ci_low_ka':qs(v)['q025'],'ci_high_ka':qs(v)['q975']} for name,v in [('Date-interval Monte Carlo',date_cp),('Row bootstrap',row_cp),('Source-cluster bootstrap',source_cp),('Site-cluster bootstrap',site_cp)]]).to_csv(OUT/'archaeology_change_point_robustness.csv',index=False)

plt.figure(figsize=(8.6,5.4));plt.hist(perm_delta,bins=45,density=True,alpha=.75);plt.axvline(obs_delta,linestyle='--',label=f'Observed ΔLL={obs_delta:.2f}');plt.xlabel('Maximum two-regime log-likelihood improvement under permutation');plt.ylabel('Density');plt.title('Permutation test for a temporal regime change');plt.legend();plt.tight_layout();plt.savefig(OUT/'archaeology_permutation_test.png',dpi=220);plt.close()
plt.figure(figsize=(9,5.5))
for v,label in [(date_cp,'Date uncertainty'),(source_cp,'Source-cluster bootstrap'),(site_cp,'Site-cluster bootstrap')]:
    h,e=np.histogram(v[np.isfinite(v)],bins=np.arange(0,1600,50),density=True);plt.plot((e[:-1]+e[1:])/2,h,label=label)
plt.axvline(obs_cp,linestyle='--',label='Midpoint estimate');plt.xlabel('Change point (ka)');plt.ylabel('Density');plt.title('Archaeological change-point robustness');plt.legend();plt.tight_layout();plt.savefig(OUT/'archaeology_change_point_robustness.png',dpi=220);plt.close()
plt.figure(figsize=(8.8,5.4));sub=sens_df[sens_df.analysis=='fixed_boundary']
for baseline,g in sub.groupby('baseline'):plt.plot(g.boundary_ka,g.younger_rate-g.older_rate,marker='o',label=f'Baseline >{baseline} units')
plt.axhline(0,linestyle='--');plt.xlabel('Temporal boundary (ka)');plt.ylabel('Younger-minus-older exceedance rate');plt.title('Sensitivity to procedural-unit and temporal thresholds');plt.legend();plt.tight_layout();plt.savefig(OUT/'archaeology_threshold_sensitivity.png',dpi=220);plt.close()
print(json.dumps(summary,indent=2))
