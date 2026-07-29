import math, time
from pathlib import Path
import os
import numpy as np, pandas as pd
import statsmodels.api as sm
from stage9_continuous_model import DEFAULT_PARAMS, YEARS_PER_GENERATION, run_model
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'new_runs')); OUT.mkdir(parents=True,exist_ok=True)

def wilson(k,n,z=1.959963984540054):
 p=k/n; den=1+z*z/n; centre=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return max(0,centre-half),min(1,centre+half)

run_model(0,.05,60,DEFAULT_PARAMS)
contacts=np.array([.002,.02,.03,.04,.05,.06,.08,.10,.12])
durations=np.array([40,60,80])
N=100
rows=[]; t0=time.time()
for d in durations:
 for c in contacts:
  counts=np.zeros(3,int); first=[]; finals=[]
  seed0=100000+int(d)*1000+int(c*10000)
  for i in range(N):
   r=run_model(seed0+i,float(c),int(d),DEFAULT_PARAMS)
   counts += np.array([r[0],r[1],r[2]],int)
   if r[3]>=0: first.append(r[3]*YEARS_PER_GENERATION)
   finals.append([r[4],r[5],r[6],r[7]])
  finals=np.asarray(finals)
  for idx,crit in enumerate(['permissive','primary','strict']):
   lo,hi=wilson(int(counts[idx]),N)
   rows.append(dict(contact_probability=c,duration_generations=d,duration_years=d*YEARS_PER_GENERATION,
                    criterion=crit,ignitions=int(counts[idx]),replicates=N,ignition_probability=counts[idx]/N,
                    wilson_low=lo,wilson_high=hi,median_first_primary_year_if_any=np.median(first) if first else np.nan,
                    median_final_repertoire=np.median(finals[:,0]),median_final_primary_fraction=np.median(finals[:,1]),
                    median_final_cognition=np.median(finals[:,2]),median_final_infrastructure=np.median(finals[:,3])))
  pd.DataFrame(rows).to_csv(OUT/'dense_phase_grid.csv',index=False)
  print('done',d,c,'primary',counts[1]/N,'elapsed',time.time()-t0,flush=True)
phase=pd.DataFrame(rows)
glm=[]; boundaries=[]
for crit in ['permissive','primary','strict']:
 f=phase[phase.criterion==crit]
 x=f.contact_probability.to_numpy(); ds=f.duration_generations.to_numpy()/60
 X=sm.add_constant(np.c_[x,ds,x*ds]); endog=np.c_[f.ignitions,f.replicates-f.ignitions]
 m=sm.GLM(endog,X,family=sm.families.Binomial()).fit()
 for name,coef,se,pv in zip(['intercept','contact','duration_scaled','interaction'],m.params,m.bse,m.pvalues):
  glm.append(dict(criterion=crit,term=name,coefficient=coef,std_error=se,p_value=pv))
 for d in durations:
  z=d/60; denom=m.params[1]+m.params[3]*z
  c50=-(m.params[0]+m.params[2]*z)/denom
  boundaries.append(dict(criterion=crit,duration_generations=d,duration_years=d*YEARS_PER_GENERATION,estimated_contact_p50=c50))
pd.DataFrame(glm).to_csv(OUT/'phase_glm_coefficients.csv',index=False)
pd.DataFrame(boundaries).to_csv(OUT/'phase_glm_boundaries.csv',index=False)
print('TOTAL',time.time()-t0)
