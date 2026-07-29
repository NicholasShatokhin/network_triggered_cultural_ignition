from __future__ import annotations
from pathlib import Path
import os
import sys,argparse,math,json
import numpy as np,pandas as pd,networkx as nx
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(os.environ.get('CULTURAL_IGNITION_OUT', ROOT/'results'/'new_runs')); OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage10_general_model import run_general,DEFAULT_PARAMS,YEARS_PER_GENERATION
CONTACTS=[.02,.04,.06,.08,.12,.18];DURATION=60
STOCH={'Erdos-Renyi','Small-world','Scale-free','Modular bridges','Gravity'}

def metrics(G):
 return {'density':nx.density(G),'clustering':nx.average_clustering(G),'mean_shortest_path':nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,'degree_cv':np.std([d for _,d in G.degree()])/max(np.mean([d for _,d in G.degree()]),1e-9),'modularity':nx.community.modularity(G,nx.community.greedy_modularity_communities(G)) if G.number_of_edges() else 0}
def cdf_graph(G,weights=None):
 n=G.number_of_nodes();W=np.zeros((n,n),float)
 for i,j in G.edges():
  w=1.0 if weights is None else weights.get((i,j),weights.get((j,i),1.0));W[i,j]=W[j,i]=w
 for i in range(n):
  if W[i].sum()==0:W[i]=1;W[i,i]=0
  W[i]/=W[i].sum()
 return np.cumsum(W,axis=1)
def make(name,n,seed):
 rng=np.random.default_rng(seed);weights=None
 if name=='Global':G=nx.complete_graph(n)
 elif name=='Ring':G=nx.cycle_graph(n)
 elif name=='Lattice':
  H=nx.grid_2d_graph(4,4,periodic=True);G=nx.convert_node_labels_to_integers(H)
 elif name=='Wheel':
  G=nx.star_graph(n-1)
  for i in range(1,n):G.add_edge(i,1+(i%(n-1)))
 elif name=='Erdos-Renyi':
  G=nx.erdos_renyi_graph(n,.22,seed=seed)
 elif name=='Small-world':G=nx.watts_strogatz_graph(n,4,.15,seed=seed)
 elif name=='Scale-free':G=nx.barabasi_albert_graph(n,2,seed=seed)
 elif name=='Modular bridges':
  probs=np.full((4,4),.025);np.fill_diagonal(probs,.65);G=nx.stochastic_block_model([4,4,4,4],probs,seed=seed)
 elif name=='Gravity':
  pos=rng.random((n,2));G=nx.complete_graph(n);weights={(i,j):math.exp(-np.linalg.norm(pos[i]-pos[j])/.25)+1e-5 for i,j in G.edges()}
 else:raise ValueError(name)
 if not nx.is_connected(G):
  comps=list(nx.connected_components(G))
  for a,b in zip(comps[:-1],comps[1:]):G.add_edge(next(iter(a)),next(iter(b)))
 return cdf_graph(G,weights),metrics(G)

p=argparse.ArgumentParser();p.add_argument('--topology',required=True);p.add_argument('--contact',type=float,required=True);a=p.parse_args();name=a.topology; CONTACTS=[a.contact]
nreal=20 if name in STOCH else 1; reps=8 if name in STOCH else 120; TOP_INDEX={'Global':0,'Ring':1,'Lattice':2,'Erdos-Renyi':3,'Small-world':4,'Scale-free':5,'Modular bridges':6,'Wheel':7,'Gravity':8}[name]
rows=[]
# warm up
cdf,_=make(name,16,20260728);run_general(0,16,20,cdf,.06,DURATION,DEFAULT_PARAMS)
for c in CONTACTS:
 for realization in range(nreal):
  # Stochastic graph realisations are independent at every contact level,
  # matching the frozen publication ensemble and the manuscript description.
  seed_graph=20_270_000+TOP_INDEX*1_000_000+int(c*10_000)*100+realization
  cdf,m=make(name,16,seed_graph)
  hits=0;first=[];repscores=[]
  for r in range(reps):
   seed=5_000_000+TOP_INDEX*1_000_000+realization*10_000+int(c*10_000)+r
   out=run_general(seed,16,20,cdf,c,DURATION,DEFAULT_PARAMS);hits+=int(out[0]);repscores.append(out[2]);
   if out[1]>=0:first.append(out[1]*YEARS_PER_GENERATION)
  rows.append({'topology':name,'network_realization':realization,'graph_seed':seed_graph,'contact_probability':c,'duration_years':DURATION*YEARS_PER_GENERATION,'ignitions':hits,'agent_replicates':reps,'ignition_probability':hits/reps,'median_final_repertoire':float(np.median(repscores)),'median_first_ignition_year_if_any':float(np.median(first)) if first else np.nan,**m})
out=pd.DataFrame(rows);out.to_csv(OUT/f'topology_extended_{name.replace(" ","_")}_{a.contact:.2f}.csv',index=False)
print(name,'rows',len(out),'realisations',nreal,'agent reps',reps)
