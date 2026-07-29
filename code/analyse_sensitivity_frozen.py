from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'reproduced'))
OUT.mkdir(parents=True,exist_ok=True)
NAMES=['contact_probability','duration_generations','learn_base','learn_cognition','level_difficulty','teaching_max','innovation_rate','non_speciality_factor','skill_fitness_benefit','brain_cost','extra_models_max','recombination_strength','migration_scale','infrastructure_steepness']
sens=pd.read_csv(DATA/'global_lhs_samples.csv')
X=sens[NAMES].copy(); X['non_speciality_factor']=np.log10(X['non_speciality_factor']); y=sens.primary_probability.to_numpy()
rf=RandomForestRegressor(n_estimators=600,min_samples_leaf=4,max_features=.8,random_state=20260728,n_jobs=-1,oob_score=True).fit(X,y)
perm=permutation_importance(rf,X,y,n_repeats=25,random_state=20260728,n_jobs=-1,scoring='r2')
imp=pd.DataFrame({'parameter':NAMES,'rf_impurity_importance':rf.feature_importances_,'permutation_importance_mean':perm.importances_mean,'permutation_importance_sd':perm.importances_std})
rows=[]
for col in NAMES:
    xx=np.log10(sens[col]) if col=='non_speciality_factor' else sens[col]
    rho,p=spearmanr(xx,y); rows.append((col,rho,p))
imp=imp.merge(pd.DataFrame(rows,columns=['parameter','spearman_rho','spearman_p_value']),on='parameter').sort_values('permutation_importance_mean',ascending=False)
imp['rf_train_r2']=r2_score(y,rf.predict(X)); imp['rf_oob_r2']=rf.oob_score_
imp.to_csv(OUT/'global_sensitivity_importance.csv',index=False)
plot=imp.head(12).sort_values('permutation_importance_mean')
plt.figure(figsize=(9,5.8)); plt.barh(plot.parameter,plot.permutation_importance_mean,xerr=plot.permutation_importance_sd); plt.xlabel('Permutation importance (decrease in R²)'); plt.title('Latin-hypercube global sensitivity analysis'); plt.tight_layout(); plt.savefig(OUT/'global_sensitivity.png',dpi=200); plt.close()
print(imp.head(12).to_string(index=False))
