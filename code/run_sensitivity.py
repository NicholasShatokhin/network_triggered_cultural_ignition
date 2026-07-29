import time
from pathlib import Path
import os
import numpy as np,pandas as pd
from scipy.stats import qmc,spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from stage9_continuous_model import *
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'new_runs')); OUT.mkdir(parents=True,exist_ok=True)
NAMES=['contact_probability','duration_generations','learn_base','learn_cognition','level_difficulty','teaching_max','innovation_rate','non_speciality_factor','skill_fitness_benefit','brain_cost','extra_models_max','recombination_strength','migration_scale','infrastructure_steepness']
LOW=np.array([.005,20,.60,.20,.040,0,.35,1e-4,.03,.30,0,0,0,6.])
HIGH=np.array([.20,100,.70,.42,.080,.40,.80,2e-2,.12,.80,7,9,.10,18.])
N=256; REPS=3
sampler=qmc.LatinHypercube(d=len(NAMES),seed=20260728); u=sampler.random(N); vals=qmc.scale(u,LOW,HIGH)
vals[:,7]=10**(np.log10(LOW[7])+u[:,7]*(np.log10(HIGH[7])-np.log10(LOW[7])))
run_model(0,.05,60,DEFAULT_PARAMS)
rows=[];t0=time.time()
for i,row in enumerate(vals):
 p=DEFAULT_PARAMS.copy(); p[0]=row[2];p[1]=row[3];p[2]=row[4];p[3]=row[5];p[4]=row[6];p[5]=row[7];p[6]=row[8];p[7]=row[9];p[9]=row[10];p[10]=row[11];p[11]=row[12];p[15]=row[13]
 hits=np.zeros(3,int); reps=[]
 for k in range(REPS):
  r=run_model(500000+i*10+k,float(row[0]),int(round(row[1])),p)
  hits+=np.array([r[0],r[1],r[2]],int); reps.append([r[4],r[5]])
 rec={name:float(v) for name,v in zip(NAMES,row)}; a=np.asarray(reps)
 rec.update(primary_ignitions=int(hits[1]),replicates=REPS,primary_probability=hits[1]/REPS,permissive_probability=hits[0]/REPS,strict_probability=hits[2]/REPS,median_final_repertoire=np.median(a[:,0]),median_final_primary_fraction=np.median(a[:,1]))
 rows.append(rec)
 if (i+1)%16==0:
  pd.DataFrame(rows).to_csv(OUT/'global_lhs_samples.csv',index=False); print(i+1,'elapsed',time.time()-t0,flush=True)
sens=pd.DataFrame(rows);sens.to_csv(OUT/'global_lhs_samples.csv',index=False)
X=sens[NAMES].copy();X['non_speciality_factor']=np.log10(X['non_speciality_factor']);y=sens.primary_probability.to_numpy()
rf=RandomForestRegressor(n_estimators=600,min_samples_leaf=4,max_features=.8,random_state=20260728,n_jobs=-1,oob_score=True).fit(X,y)
perm=permutation_importance(rf,X,y,n_repeats=25,random_state=20260728,n_jobs=-1,scoring='r2')
imp=pd.DataFrame({'parameter':NAMES,'rf_impurity_importance':rf.feature_importances_,'permutation_importance_mean':perm.importances_mean,'permutation_importance_sd':perm.importances_std})
r=[]
for col in NAMES:
 xx=np.log10(sens[col]) if col=='non_speciality_factor' else sens[col]
 rho,pv=spearmanr(xx,y);r.append((col,rho,pv))
imp=imp.merge(pd.DataFrame(r,columns=['parameter','spearman_rho','spearman_p_value']),on='parameter').sort_values('permutation_importance_mean',ascending=False)
imp['rf_train_r2']=r2_score(y,rf.predict(X));imp['rf_oob_r2']=rf.oob_score_
imp.to_csv(OUT/'global_sensitivity_importance.csv',index=False)
print(imp.head(10).to_string(index=False));print('elapsed',time.time()-t0)
