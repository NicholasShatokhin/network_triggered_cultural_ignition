import numpy as np,pandas as pd,statsmodels.api as sm,time,os
from pathlib import Path
from stage10_general_model import *
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'new_runs'))
OUT.mkdir(parents=True,exist_ok=True)
def cdf(G):
 w=np.ones((G,G)); np.fill_diagonal(w,0); w/=w.sum(1,keepdims=True); return np.cumsum(w,axis=1)
def p50(f):
 try:
  X=sm.add_constant(f.contact_probability.to_numpy()); k=f.ignitions.to_numpy(); n=f.replicates.to_numpy()
  fit=sm.GLM(np.column_stack([k,n-k]),X,family=sm.families.Binomial()).fit(); b0,b1=fit.params
  return -b0/b1 if b1>0 else np.nan
 except: return np.nan
run_general(0,16,20,cdf(16),.06,60,DEFAULT_PARAMS)
contacts=[.03,.05,.08,.12]; reps=8
sc=[]
for G,s in [(4,80),(8,40),(16,20),(32,10)]: sc.append(('fixed_total_320',G,s))
for G in [4,8,16,32,64]: sc.append(('fixed_group_20',G,20))
for s in [10,20,40,80]: sc.append(('fixed_groups_16',16,s))
rows=[]; st=time.time()
for si,(design,G,size) in enumerate(sc):
 C=cdf(G)
 for contact in contacts:
  hits=0; repscores=[]
  for r in range(reps):
   o=run_general(500000+si*1000+int(contact*1000)*10+r,G,size,C,contact,60,DEFAULT_PARAMS)
   hits+=int(o[0]); repscores.append(o[2])
  rows.append({'design':design,'n_groups':G,'group_size':size,'total_population':G*size,'contact_probability':contact,'ignitions':hits,'replicates':reps,'ignition_probability':hits/reps,'median_final_repertoire':float(np.median(repscores))})
 print(si,design,G,size,'done',round(time.time()-st,1),flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'scaling_ensemble.csv',index=False)
summary=[]
for key,f in df.groupby(['design','n_groups','group_size','total_population']):
 d=dict(zip(['design','n_groups','group_size','total_population'],key));d['estimated_p50']=float(p50(f));summary.append(d)
s=pd.DataFrame(summary);s.to_csv(OUT/'scaling_summary.csv',index=False);print(s.to_string(index=False))
