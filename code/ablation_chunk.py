from pathlib import Path
import os
import sys,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'new_runs')); OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage9_continuous_model import run_model,DEFAULT_PARAMS,YEARS_PER_GENERATION,ABL_NO_RECOMBINATION,ABL_NO_TEACHING,ABL_NO_GENE_CULTURE,ABL_NO_INFRA_FEEDBACK,ABL_NO_CONTACT_EVENT
MODELS=[('Full model',0),('No recombination',ABL_NO_RECOMBINATION),('No teaching',ABL_NO_TEACHING),('No cultural selection on cognition',ABL_NO_GENE_CULTURE),('No infrastructure feedback',ABL_NO_INFRA_FEEDBACK)]
CONTACTS=[.03,.06,.10,.16];DURATIONS=[40,60,80];REPS=50

def all_tasks():
 out=[]
 for mi,(name,flag) in enumerate(MODELS):
  for d in DURATIONS:
   for c in CONTACTS:
    for r in range(REPS):out.append((name,flag,c,d,1_000_000+mi*100_000+d*1000+int(c*10000)+r))
 for r in range(300):out.append(('No contact event',ABL_NO_CONTACT_EVENT,.10,60,2_000_000+r))
 return out
p=argparse.ArgumentParser();p.add_argument('--start',type=int);p.add_argument('--end',type=int);a=p.parse_args();tasks=all_tasks()[a.start:a.end]
run_model(0,.06,60,DEFAULT_PARAMS,0)
rows=[]
for name,flag,c,d,seed in tasks:
 r=run_model(seed,c,d,DEFAULT_PARAMS,flag)
 rows.append({'model':name,'flag':flag,'contact_probability':c,'duration_generations':d,'duration_years':d*YEARS_PER_GENERATION,'seed':seed,'primary_ignition':int(r[1]),'first_generation':r[3],'final_repertoire':r[4],'final_primary_fraction':r[5],'final_cognition':r[6],'final_infrastructure':r[7]})
pd.DataFrame(rows).to_csv(OUT/f'ablation_raw_{a.start}_{a.end}.csv',index=False)
print(a.start,a.end,len(rows))
