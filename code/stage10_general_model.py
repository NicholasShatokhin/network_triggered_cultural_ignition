from __future__ import annotations
import math
import numpy as np
from numba import njit

N_DOMAINS = 6
MAX_LEVEL = 5
GENERATIONS = 420
YEARS_PER_GENERATION = 25
EVENT_START = 120
BASE_EXTERNAL_MODEL_PROB = 0.002
BASE_MIGRATION_RATE = 0.0005

I_LEARN_BASE=0; I_LEARN_COG=1; I_LEVEL_DIFFICULTY=2; I_TEACHING_MAX=3
I_INNOV_RATE=4; I_NON_SPEC_FACTOR=5; I_SKILL_BENEFIT=6; I_BRAIN_COST=7
I_MUT_SD=8; I_EXTRA_MODELS_MAX=9; I_RECOMB_STRENGTH=10; I_MIGRATION_SCALE=11
I_BASE_MODELS=12; I_TEACH_MID=13; I_TEACH_SCALE=14; I_INFRA_STEEPNESS=15

DEFAULT_PARAMS=np.array([0.65,0.32,0.055,0.32,0.58,0.001,0.070,0.52,0.015,5.0,6.0,0.05,3.0,12.0,2.5,12.0],dtype=np.float64)

@njit(cache=True)
def sigmoid(x):
    if x>60: return 1.0
    if x<-60: return 0.0
    return 1.0/(1.0+math.exp(-x))

@njit(cache=True)
def weighted_choice(weights):
    total=0.0
    for i in range(weights.shape[0]): total += weights[i]
    if total<=0: return np.random.randint(weights.shape[0])
    draw=np.random.random()*total; acc=0.0
    for i in range(weights.shape[0]):
        acc += weights[i]
        if acc>=draw: return i
    return weights.shape[0]-1

@njit(cache=True)
def choose_source_group(cdf, group):
    u=np.random.random()
    for j in range(cdf.shape[1]):
        if u <= cdf[group,j]: return j
    return cdf.shape[1]-1

@njit(cache=True)
def group_metrics(skills,start,group_size,params):
    mean_total=0.0; teacher_pool=0.0; domain_presence=0.0
    for j in range(group_size):
        idx=start+j; total=0.0
        for d in range(N_DOMAINS): total += skills[idx,d]
        mean_total += total
        teacher_pool += sigmoid((total-params[I_TEACH_MID])/params[I_TEACH_SCALE])
    mean_total /= group_size; teacher_pool /= group_size
    for d in range(N_DOMAINS):
        ml=0.0
        for j in range(group_size): ml += skills[start+j,d]
        ml /= group_size
        domain_presence += sigmoid((ml-1.5)/0.45)
    diversity=domain_presence/N_DOMAINS
    complexity=mean_total/(N_DOMAINS*MAX_LEVEL)
    raw=.42*complexity+.38*diversity+.20*teacher_pool
    infra=sigmoid(params[I_INFRA_STEEPNESS]*(raw-.47))
    return diversity,teacher_pool,infra

@njit(cache=True)
def run_general(seed,n_groups,group_size,source_cdf,event_external_probability,event_duration,params,specialisation=True):
    np.random.seed(seed)
    n_agents=n_groups*group_size
    cognition=np.empty(n_agents,dtype=np.float64)
    skills=np.zeros((n_agents,N_DOMAINS),dtype=np.int8)
    for g in range(n_groups):
        speciality=g%N_DOMAINS; start=g*group_size
        for j in range(group_size):
            idx=start+j
            cognition[idx]=min(.80,max(.10,.32+.04*np.random.randn()))
            if np.random.random()<.50: skills[idx,speciality]=1

    primary_good=0; first_primary=-1
    final_mean_rep=0.0; final_primary=0.0; final_cognition=0.0; final_infra=0.0
    mean_rep_trace=np.zeros(GENERATIONS,dtype=np.float64)

    for generation in range(GENERATIONS):
        in_event=EVENT_START<=generation<EVENT_START+event_duration
        ext=event_external_probability if in_event else BASE_EXTERNAL_MODEL_PROB
        if ext>1: ext=1.0
        migration_rate=BASE_MIGRATION_RATE+params[I_MIGRATION_SCALE]*ext
        next_cognition=np.empty_like(cognition); next_skills=np.zeros_like(skills)
        mean_infra=0.0
        for group in range(n_groups):
            start=group*group_size; speciality=group%N_DOMAINS
            diversity,teacher_pool,infra=group_metrics(skills,start,group_size,params)
            mean_infra += infra
            fitness=np.empty(group_size,dtype=np.float64)
            for j in range(group_size):
                idx=start+j; total=0.0
                for d in range(N_DOMAINS): total += skills[idx,d]
                fitness[j]=math.exp(params[I_SKILL_BENEFIT]*total-params[I_BRAIN_COST]*cognition[idx]*cognition[idx])
            expected=params[I_BASE_MODELS]+params[I_EXTRA_MODELS_MAX]*infra
            n_models=int(expected)
            if np.random.random()<expected-n_models: n_models+=1
            if n_models<1: n_models=1
            if n_models>12: n_models=12
            for learner_j in range(group_size):
                parent=start+weighted_choice(fitness)
                lc=min(.95,max(.10,cognition[parent]+params[I_MUT_SD]*np.random.randn()))
                best=np.zeros(N_DOMAINS,dtype=np.int8); bestq=np.zeros(N_DOMAINS,dtype=np.float64)
                observed=np.zeros(N_DOMAINS,dtype=np.int8)
                for _ in range(n_models):
                    if np.random.random()<ext:
                        sg=choose_source_group(source_cdf,group)
                        model=sg*group_size+np.random.randint(group_size)
                    else: model=start+np.random.randint(group_size)
                    mt=0
                    for d in range(N_DOMAINS):
                        mt += skills[model,d]
                        if skills[model,d]>=2: observed[d]=1
                    quality=sigmoid((mt-params[I_TEACH_MID])/params[I_TEACH_SCALE])
                    for d in range(N_DOMAINS):
                        if skills[model,d]>best[d]:
                            best[d]=skills[model,d]; bestq[d]=quality
                observed_div=0.0
                for d in range(N_DOMAINS): observed_div += observed[d]
                observed_div /= N_DOMAINS
                for d in range(N_DOMAINS):
                    copied=0
                    for level in range(1,best[d]+1):
                        teach=params[I_TEACHING_MAX]*bestq[d]*(.35+.65*infra)
                        cp=params[I_LEARN_BASE]+params[I_LEARN_COG]*lc-params[I_LEVEL_DIFFICULTY]*level+teach
                        cp=min(.995,max(.02,cp))
                        if np.random.random()<cp: copied=level
                        else: break
                    next_skills[start+learner_j,d]=copied
                if specialisation:
                    if np.random.random()<.85: domain=speciality
                    else: domain=np.random.randint(N_DOMAINS)
                    sf=1.0 if domain==speciality else params[I_NON_SPEC_FACTOR]
                else:
                    domain=np.random.randint(N_DOMAINS); sf=.875
                current=next_skills[start+learner_j,domain]
                mult=1.0+params[I_RECOMB_STRENGTH]*observed_div*infra
                ip=params[I_INNOV_RATE]*sf*lc*math.exp(-.18*current)*mult
                if ip>.95: ip=.95
                if np.random.random()<ip and current<MAX_LEVEL: next_skills[start+learner_j,domain]+=1
                next_cognition[start+learner_j]=lc
        mean_infra /= n_groups
        n_swaps=int(migration_rate*n_agents/2.0)
        for _ in range(n_swaps):
            ga=np.random.randint(n_groups); gb=choose_source_group(source_cdf,ga)
            a=ga*group_size+np.random.randint(group_size); b=gb*group_size+np.random.randint(group_size)
            tc=next_cognition[a]; next_cognition[a]=next_cognition[b]; next_cognition[b]=tc
            for d in range(N_DOMAINS):
                ts=next_skills[a,d]; next_skills[a,d]=next_skills[b,d]; next_skills[b,d]=ts
        cognition=next_cognition; skills=next_skills
        total_rep=0.0; n_primary=0
        for idx in range(n_agents):
            total=0; domains=0
            for d in range(N_DOMAINS):
                level=skills[idx,d]; total+=level
                if level>=2: domains+=1
            total_rep += total
            if total>=16 and domains>=5: n_primary+=1
        mean_rep=total_rep/n_agents; frac=n_primary/n_agents
        if first_primary<0 and frac>.10: first_primary=generation
        if generation>=GENERATIONS-50 and frac>.50 and mean_rep>18: primary_good+=1
        mean_rep_trace[generation]=mean_rep
        final_mean_rep=mean_rep; final_primary=frac; final_cognition=cognition.mean(); final_infra=mean_infra
    return primary_good>=45,first_primary,final_mean_rep,final_primary,final_cognition,final_infra,mean_rep_trace
