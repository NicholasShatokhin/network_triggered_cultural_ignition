from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'reproduced'))
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA/'scaling_ensemble.csv')

def p50(g):
    g=g.sort_values('contact_probability')
    x=g.contact_probability.to_numpy(float)
    k=g.ignitions.to_numpy(float)
    n=g.replicates.to_numpy(float)
    p=k/n
    if k.sum()==0:
        return np.nan, f'>{x.max():.2f}'
    if np.all(p>=.5):
        return float(x.min()), f'<{x.min():.2f}'
    try:
        fit=sm.GLM(np.column_stack([k,n-k]),sm.add_constant(x),family=sm.families.Binomial()).fit()
        b0,b1=fit.params
        if b1<=0:
            return np.nan,'nonpositive_slope'
        v=float(-b0/b1)
        status='estimated' if x.min()<=v<=x.max() else (f'<{x.min():.2f}' if v<x.min() else f'>{x.max():.2f}')
        return v,status
    except Exception:
        return np.nan,'fit_failed'

rows=[]
for keys,g in df.groupby(['design','n_groups','group_size','total_population']):
    v,status=p50(g)
    rows.append(dict(zip(['design','n_groups','group_size','total_population'],keys))|{
        'estimated_p50':v,
        'p50_status':status,
        'min_observed_probability':float(g.ignition_probability.min()),
        'max_observed_probability':float(g.ignition_probability.max()),
    })
out=pd.DataFrame(rows)
out.to_csv(OUT/'scaling_summary_refit.csv',index=False)
for design,filename,xcol,xlabel in [
    ('fixed_group_20','scaling_group_count.png','n_groups','Number of groups'),
    ('fixed_groups_16','scaling_group_size.png','group_size','Agents per group'),
    ('fixed_total_320','scaling_fixed_total.png','n_groups','Number of groups (total population = 320)'),
]:
    q=out[out.design==design].sort_values(xcol)
    plt.figure(figsize=(7.8,5.0)); plt.plot(q[xcol],q.estimated_p50,marker='o')
    plt.xlabel(xlabel); plt.ylabel('Estimated external-model probability at 50% ignition')
    plt.title(design.replace('_',' ')); plt.tight_layout(); plt.savefig(OUT/filename,dpi=200); plt.close()
print(out.to_string(index=False))
