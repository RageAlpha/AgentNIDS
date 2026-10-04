#!/usr/bin/env python
import argparse,os,sys,shlex
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from fixed_core import *
from frozen_functions import attack_probability,gate,choose,tests,boot_metrics,ci,GRID
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run');BASE=ROOT/'outputs/post_p7_diagnostics'
T1=BASE/'ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd'
T2=BASE/'ton_second_refresh_pilot_v1/run_20260929T122552Z_d10acc'
OLD=BASE/'reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr'
BOUNDS=[20680000,22390000,24100000,25810000,27520000]
MAIN=[f'{age}_{m}' for age in ['T1','T2','MIXED'] for m in ['LOCAL','TARGET']]
PAIRS=[(f'MIXED_{m}_{layer}',f'{age}_{m}_{layer}') for m in ['LOCAL','TARGET'] for age in ['T1','T2'] for layer in ['ALL','GATED']]+[(f'MIXED_{m}_ALL','SOURCE_RF_ALL') for m in ['LOCAL','TARGET']]+[('SOURCE_GATE_MIXED',f'SOURCE_GATE_{age}') for age in ['T1','T2']]+[(f'MIXED_{m}_GATED','SOURCE_GATE_MIXED') for m in ['LOCAL','TARGET']]
def condition(name):
    if name.startswith('T1_') or name=='SOURCE_GATE_T1':return 'T1_ONLY'
    if name.startswith('T2_') or name=='SOURCE_GATE_T2':return 'T2_ONLY'
    return 'MIXED' if 'MIXED' in name else 'PUBLIC'
def prepare(run):
    p2=readj(T2/'config/protocol.json');p1=readj(T1/'config/protocol.json');a=Assets(ROOT,'ToN',readj(OLD/'config/protocol_v1.json'))
    frames={age:pd.read_csv(path/'config/recent_sample_roles.csv').sort_values('availability_rank') for age,path in [('T1',T1),('T2',T2)]}
    groups={age+'_'+ro:f.loc[f.role==ro,'availability_rank'].to_numpy(np.int64) for age,f in frames.items() for ro in ['R','C']}
    require(all(len(v)==200 and np.all(np.diff(v)>0) for v in groups.values()),'four groups each200/sorted')
    pool=np.concatenate(list(groups.values()));require(np.unique(pool).size==800 and (pool<BOUNDS[0]).all(),'original groups mutually exclusive and available at T2')
    seeds={'T1_R':20261006,'T2_R':20261007,'T1_C':20261008,'T2_C':20261009};parts=[]
    for group,seed in seeds.items():
        ranks=np.sort(np.random.Generator(np.random.PCG64(seed)).choice(groups[group],100,replace=False));parts.append(pd.DataFrame(dict(availability_rank=ranks,role=group[-1],source=group[:2],source_group=group,seed=seed)))
    mixed=pd.concat(parts).sort_values('availability_rank').reset_index(drop=True);chosen=mixed.availability_rank.to_numpy(np.int64)
    cache=np.load(T2/'tables/evaluation_actions.npz');rank=cache['rank'];ix=np.load(T2/'config/frozen_indices.npz')['rank'];pilot=np.load(OLD/'pilot/ToN_evaluation_sample_ranks.npy');expected=pilot[(pilot>=BOUNDS[0])&(pilot<BOUNDS[-1])]
    require(np.array_equal(rank,ix) and np.array_equal(rank,expected) and len(rank)==68400 and np.all(np.diff(rank)>0),'evaluation identities')
    require(not np.isin(pool,rank).any() and np.bincount((rank-BOUNDS[0])//10000).tolist()==[100]*684,'evaluation disjoint and 684 windows')
    startup=pd.read_csv(OLD/'calibration/sample_roles.csv');startup=startup[startup.domain=='ToN'].availability_rank.to_numpy();require(not np.isin(pool,startup).any(),'startup excluded')
    params=dict(p2['model_parameters']);params['n_jobs']=min(8,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1)
    require(GRID==p2['gate_grid']==p1['gate_grid'] and p1['feature_indices']==p2['feature_indices']==sorted(a.cat+a.num),'fixed grid/features')
    baseline_map={name:dict(memory_condition=condition(name),source_name=name,actions=str(T2/'tables/evaluation_actions.npz'),rules=str(T2/'calibration/selected_rules.json'),model=str(T1/'models/RECENT_TARGET.joblib') if name=='T1_TARGET' else str(T2/'models/T2_TARGET.joblib') if name=='T2_TARGET' else None) for name in ['T1_LOCAL','T1_TARGET','T2_LOCAL','T2_TARGET','SOURCE_GATE_T1','SOURCE_GATE_T2']}
    for path in [T1/'models/RECENT_TARGET.joblib',T2/'models/T2_TARGET.joblib',T1/'calibration/RECENT_reference.npz',T2/'calibration/T2_reference.npz',T2/'calibration/selected_rules.json']:require(path.is_file(),'missing frozen asset '+str(path))
    protocol=dict(experiment='ton_mixed_memory_pilot_v1',created_at=now(),T1_run=str(T1),T2_run=str(T2),original_run=str(OLD),input_assets=a.info,baseline_map=baseline_map,seeds=seeds,generator='numpy.random.Generator(PCG64)',sampling='each sorted original role independently choose100 without replacement; indices only; then sort merged; never exchange R/C',capacity=dict(R=200,C=200,each_source_per_role=100),feature_indices=p2['feature_indices'],distance=p2['distance'],model_parameters=params,gate_grid=GRID,gate=p2['gate'],tie_order=p2['tie_order'],evaluation_range=[BOUNDS[0],BOUNDS[-1]],segment_boundaries=BOUNDS,expected_rows=68400,segment_rows=[17100]*4,segment_windows=[171]*4,methods=MAIN,controls=['SOURCE_GATE_T1','SOURCE_GATE_T2','SOURCE_GATE_MIXED','SOURCE_RF_ALL','ALWAYS_A','ALWAYS_B'],pairs=PAIRS,budget=dict(new_adaptation=0,cumulative_acquired_adaptation=1200,existing_mixing_pool=800,per_local_target=400,per_source_gate=200,evaluation_existing_labels=68400,evaluation_labels_used_for_update=False),bootstrap=dict(p2['bootstrap'],seed=20261010),workpoint=p2['workpoint'],interpretation='previously observed labels and reused development evaluation; not blind, not independent400-budget study, no optimal ratio or drift identification; mixture changes time,class composition,contents',stop='one68400 pilot,no extra labels,no ratio/seed search,no full inference,no representation learning')
    protocol=clean(protocol)  # Normalize tuple pairs to JSON arrays before resume comparison.
    if (run/'config/protocol.json').exists():protocol['created_at']=readj(run/'config/protocol.json')['created_at']
    savej(run/'config/protocol.json',protocol);savecsv(run/'config/mixed_sample_roles.csv',mixed);saven(run/'config/frozen_indices.npz',rank=rank,mixed_rank=chosen,role=mixed.role.to_numpy(dtype='U1'),source=mixed.source.to_numpy(dtype='U2'))
    require((run/'config/protocol.json').exists() and (run/'config/frozen_indices.npz').exists(),'protocol and indices must be saved before labels')
    if not (run/'config/index_freeze.json').exists():writej(run/'config/index_freeze.json',dict(time=now(),sampling_used_indices_only=True,labels_previously_observed=True,roles_frozen_before_build=True))
    # Existing calibration labels only, after mixed indices persist. No new acquisition.
    label_by={};composition=[]
    for age,path in [('T1',T1),('T2',T2)]:
        z=np.load(path/'calibration/recent400.npz');f=frames[age];require(np.array_equal(z['rank'],f.availability_rank),'saved label role identity')
        label_by.update(zip(z['rank'].tolist(),z['y'].tolist()))
        for ro in ['R','C']:
            yy=z['y'][f.role==ro];bc=np.bincount(yy.astype(int),minlength=2).tolist();require(bc=={'T1_R':[128,72],'T1_C':[123,77],'T2_R':[14,186],'T2_C':[15,185]}[age+'_'+ro],'known support mismatch')
            composition.append(dict(group=age+'_'+ro,N=200,true_B=bc[0],true_A=bc[1]))
    y=np.array([label_by[int(v)] for v in chosen],np.int8);r=mixed.role.to_numpy()=='R';c=~r
    require(r.sum()==c.sum()==200 and all((mixed.source_group==g).sum()==100 for g in seeds),'mixed capacity/source counts')
    saven(run/'calibration/mixed400.npz',rank=chosen,y=y)
    for ro,m in [('R',r),('C',c)]:composition.append(dict(group='MIXED_'+ro,N=200,true_B=int((y[m]==0).sum()),true_A=int((y[m]==1).sum())))
    savecsv(run/'calibration/role_composition.csv',composition)
    cols=protocol['feature_indices'];mp=run/'models/MIXED_TARGET.joblib';rp=run/'calibration/selected_rules.json'
    if (run/'tables/actions_complete.json').exists():
        require(mp.exists() and rp.exists(),'completed prepare missing assets');model=joblib.load(mp);require(all(model.get_params()[k]==v for k,v in params.items()),'saved model params');require(readj(rp)['protocol']==protocol,'saved rule protocol')
        z=np.load(run/'tables/evaluation_actions.npz');require(np.array_equal(z['rank'],rank),'saved action ranks');print('prepare complete: protocol, roles, counts and parameters checked; no rebuild');return
    rx=a.features(chosen[r]);saven(run/'calibration/MIXED_reference.npz',rank=chosen[r],x=rx,y=y[r])
    if mp.exists():model=joblib.load(mp);require(all(model.get_params()[k]==v for k,v in params.items()),'saved model parameters')
    else:model=RandomForestClassifier(**params).fit(rx[:,cols],y[r]);joblib.dump(model,mp)
    if not rp.exists():
        cx=a.features(chosen[c]);q,_,_=evidence(cx,rx,y[r],np.ones(200,np.int8),a.cat,a.num,a.scale)
        scores={'MIXED_LOCAL':q,'MIXED_TARGET':attack_probability(model,cx[:,cols]),'SOURCE_GATE_MIXED':a.probs(chosen[c])}
        rules=readj(T2/'calibration/selected_rules.json')['methods'];allrows=[];support=[]
        for name,score in scores.items():
            rule,rows=choose(score,y[c],name);rules[name]=rule;allrows+=rows;savecsv(run/f'calibration/{name}_C.csv',pd.DataFrame(dict(rank=chosen[c],y=y[c],score=score,label=(score>=.5).astype(int),action=gate(score,rule))))
        for age in ['T1','T2','MIXED']:
            for learner in ['LOCAL','TARGET','SOURCE_GATE']:
                name=f'{age}_{learner}' if learner!='SOURCE_GATE' else f'SOURCE_GATE_{age}'
                if age=='MIXED':cf=pd.read_csv(run/f'calibration/{name}_C.csv',float_precision='round_trip')
                elif age=='T1' and learner!='SOURCE_GATE':cf=pd.read_csv(T1/f'calibration/RECENT_{learner}_C.csv',float_precision='round_trip')
                else:cf=pd.read_csv(T2/f'calibration/{name}_C.csv',float_precision='round_trip')
                expected_ranks=chosen[c] if age=='MIXED' else groups[age+'_C'];require(np.array_equal(cf['rank'],expected_ranks),'C source rank alignment')
                expected_y=y[c] if age=='MIXED' else np.array([label_by[int(v)] for v in expected_ranks]);require(np.array_equal(cf.y,expected_y),'C source label alignment')
                rr=rules[name];require(np.array_equal(gate(cf.score.to_numpy(),rr),cf.action),'C frozen action/rule alignment')
                support.append(dict(method=name,memory_condition=condition(name),role='C',N=200,true_B=int((cf.y==0).sum()),true_A=int((cf.y==1).sum()),pred_B=int((cf.label==0).sum()),pred_A=int((cf.label==1).sum()),B_threshold=rr['B']['threshold'],B_accepted=rr['B']['accepted'],B_errors=rr['B']['errors'],A_threshold=rr['A']['threshold'],A_accepted=rr['A']['accepted'],A_errors=rr['A']['errors'],status=rr['status']))
                if learner!='SOURCE_GATE':
                    yy=y[r] if age=='MIXED' else np.array([label_by[int(v)] for v in groups[age+'_R']]);support.append(dict(method=name,memory_condition=condition(name),role='R',N=200,true_B=int((yy==0).sum()),true_A=int((yy==1).sum()),status='R not self-evaluated'))
        savecsv(run/'calibration/support.csv',support);savecsv(run/'calibration/candidates.csv',allrows);writej(rp,dict(freeze_time=now(),protocol=protocol,methods=rules,training_classes=model.classes_.tolist(),new_rule_methods=list(scores),baseline_map=baseline_map))
    fr=readj(rp);require(fr['protocol']==protocol,'frozen rule protocol');rules=fr['methods']
    checks=tests();yy=np.tile([0,1],50);cc=window_counts(yy,np.zeros(100,np.int8),np.full(100,-1,np.int8),1)[0];mm=metrics(cc);require(mm['abstain_B']==mm['abstain_A']==50 and mm['R_B'] is None and mm['R_A'] is None,'OFF conservation')
    _,_,nn=evidence(np.zeros((1,49)),np.zeros((20,49)),np.arange(20)%2,np.ones(1,np.int8),a.cat,a.num,np.ones(49));require(np.array_equal(nn[0],np.arange(15)),'reference tie rule')
    data={k:cache[k] for k in cache.files};require(not any(k in data for k in ['y','truth','label']),'truth in baseline cache')
    for name in ['T1_LOCAL','T1_TARGET','T2_LOCAL','T2_TARGET']:
        require(np.array_equal(gate(data['score_'+name],rules[name]),data['action_'+name+'_GATED']),'baseline gate identity')
    p=data['score_SOURCE_RF_ALL'];require(np.array_equal(data['action_SOURCE_RF_ALL'],(p>=.5).astype(np.int8)),'source original prediction')
    for name in ['SOURCE_GATE_T1','SOURCE_GATE_T2']:require(np.array_equal(gate(p,rules[name]),data['action_'+name]),'source frozen gate')
    for name in ['MIXED_LOCAL','MIXED_TARGET']:data['score_'+name]=np.empty(len(rank))
    for start in range(0,len(rank),3000):
        end=min(start+3000,len(rank));x=a.features(rank[start:end]);q,_,_=evidence(x,rx,y[r],np.ones(end-start,np.int8),a.cat,a.num,a.scale);data['score_MIXED_LOCAL'][start:end]=q;data['score_MIXED_TARGET'][start:end]=attack_probability(model,x[:,cols]);print('actions',end,'/',len(rank),flush=True)
    for name in ['MIXED_LOCAL','MIXED_TARGET']:
        score=data['score_'+name];require(np.isfinite(score).all() and ((score>=0)&(score<=1)).all(),'score range');data['action_'+name+'_ALL']=(score>=.5).astype(np.int8);data['action_'+name+'_GATED']=gate(score,rules[name])
    data['action_SOURCE_GATE_MIXED']=gate(p,rules['SOURCE_GATE_MIXED']);saven(run/'tables/evaluation_actions.npz',**data)
    checks.update(four_original_groups_each200=True,original_counts_verified=True,mixed_each_source100=True,roles_never_exchanged=True,no_new_labels=True,adaptation_evaluation_disjoint=True,baseline_actions_identical=True,synthetic_OFF_conservation=True,synthetic_distance_ties=True)
    savej(run/'config/prepare_checks.json',checks);writej(run/'tables/actions_complete.json',dict(time=now(),rows=68400,windows=684,all_actions_saved_before_truth=True,new_labels=0));print('PREPARE COMPLETE',flush=True)

def savej(p,v):
    if p.exists():require(readj(p)==clean(v),'existing JSON mismatch '+str(p))
    else:writej(p,v)

def saven(p,**data):
    if p.exists():
        with np.load(p) as z:
            require(set(z.files)==set(data) and all(np.array_equal(z[k],v) for k,v in data.items()),'existing NPZ mismatch '+str(p))
    else:np.savez(p,**data)

def savecsv(p,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    text=frame.to_csv(index=False,na_rep='NA')
    if p.exists():require(p.read_text()==text,'existing CSV mismatch '+str(p))
    else:p.write_text(text)
def evaluate(run):
    protocol=readj(run/'config/protocol.json');fr=readj(run/'calibration/selected_rules.json');require(fr['protocol']==protocol,'protocol freeze')
    require((run/'tables/actions_complete.json').exists(),'prepare incomplete')
    a=np.load(run/'tables/evaluation_actions.npz');rank=a['rank'];require(np.array_equal(rank,np.load(run/'config/frozen_indices.npz')['rank']) and len(rank)==68400,'evaluation ranks')
    if (run/'reports/decision.json').exists():print('evaluation already complete; no overwrite');return
    # Evaluation only: no acquisition, training, selection, or action generation.
    ys=[];rs=[]
    if not (run/'tables/evaluation_label_access.json').exists():writej(run/'tables/evaluation_label_access.json',dict(time=now(),rows=68400,source='existing original pilot chunks',actions_complete=readj(run/'tables/actions_complete.json')['time']))
    for path in sorted((OLD/'pilot/ToN').glob('chunk_*.npz')):
        with np.load(path) as z:
            rr=z['rank'];m=(rr>=BOUNDS[0])&(rr<BOUNDS[-1])
            if m.any():rs.append(rr[m]);ys.append(z['y'][m])
    y=np.concatenate(ys);require(np.array_equal(np.concatenate(rs),rank),'truth alignment')
    seg=a['segment'];segw=seg.reshape(684,100)[:,0];require(np.bincount(seg)[1:].tolist()==[17100]*4,'segment counts')
    overall={};segments=[];counts={};windows=[]
    for key in a.files:
        if not key.startswith('action_'):continue
        name=key[7:];pred=a['action_SOURCE_RF_ALL'] if name.startswith('SOURCE') else a['action_'+name.replace('_GATED','_ALL')] if name.endswith('_GATED') else a[key]
        co=window_counts(y,pred,a[key],684);counts[name]=co
        require(np.all(co[:,3]+co[:,5]+co[:,7]==co[:,1]) and np.all(co[:,4]+co[:,6]+co[:,8]==co[:,2]),'denominator and abstention conservation')
        overall[name]=dict(method=name,**metrics(co.sum(0)));overall[name]['coverage']=overall[name]['acceptance'];overall[name]['memory_condition']=condition(name)
        for j in range(1,5):segments.append(dict(method=name,memory_condition=condition(name),segment=j,coverage=metrics(co[segw==j].sum(0))['acceptance'],**metrics(co[segw==j].sum(0))))
        for i,row in enumerate(co):windows.append(dict(method=name,window=i,rank_start=BOUNDS[0]+10000*i,segment=int(segw[i]),**dict(zip(COUNT_KEYS,map(int,row)))))
    blocks=[];strata=[]
    for j in range(1,5):
        ix=np.flatnonzero(segw==j);ids=[]
        for k in range(0,len(ix),10):ids.append(len(blocks));blocks.append(ix[k:k+10])
        strata.append(np.array(ids))
    rng=np.random.Generator(np.random.PCG64(20261010));draw=np.array([np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(1000)])
    weights=np.array([np.bincount(d,minlength=len(blocks)) for d in draw]);saven(run/'tables/bootstrap_draws.npz',indices=draw)
    savecsv(run/'tables/bootstrap_blocks.csv',[dict(block=i,segment=int(segw[b[0]]),first_window=int(b[0]),windows=len(b)) for i,b in enumerate(blocks)])
    boot={n:boot_metrics(weights@np.array([c[b].sum(0) for b in blocks])) for n,c in counts.items()};paired=[]
    for left,right in PAIRS:
        for metric in ['U_bal','U_B','U_A','R_B','R_A']:
            lv=overall[left][metric];rv=overall[right][metric];paired.append(dict(left=left,right=right,metric=metric,delta=lv-rv if lv is not None and rv is not None else None,**ci(boot[left][metric]-boot[right][metric])))
    harm=[]
    for left,right in [(f'MIXED_{m}_ALL',ref) for m in ['LOCAL','TARGET'] for ref in ['SOURCE_RF_ALL',f'T1_{m}_ALL',f'T2_{m}_ALL']]:
        h=a['action_'+left];base=a['action_'+right]
        for j in range(5):
            subset=np.ones(len(y),bool) if j==0 else seg==j
            for direction in ['ALL','B_to_A','A_to_B']:
                mm=subset if direction=='ALL' else subset&(base==(0 if direction=='B_to_A' else 1))&(h==(1 if direction=='B_to_A' else 0))
                fix=int((mm&(base!=y)&(h==y)).sum());hurt=int((mm&(base==y)&(h!=y)).sum());harm.append(dict(method=left,reference=right,scope='overall' if j==0 else f'segment_{j}',direction=direction,wrong_to_right=fix,right_to_wrong=hurt,net_corrections=fix-hurt))
    stable={n:v['target_status']=='JOINT_TARGET_MET' and sum(s['target_status']=='JOINT_TARGET_MET' for s in segments if s['method']==n)>=3 for n,v in overall.items()}
    composition=[]
    for age,rolepath,labelpath in [('T1',T1/'config/recent_sample_roles.csv',T1/'calibration/recent400.npz'),('T2',T2/'config/recent_sample_roles.csv',T2/'calibration/recent400.npz'),('MIXED',run/'config/mixed_sample_roles.csv',run/'calibration/mixed400.npz')]:
        roles=pd.read_csv(rolepath);z=np.load(labelpath);require(np.array_equal(roles.availability_rank,z['rank']),'composition roles')
        for ro in ['R','C']:
            yy=z['y'][roles.role==ro];composition.append(dict(group=age+'_'+ro,N=len(yy),N_B=int((yy==0).sum()),N_A=int((yy==1).sum()),attack_fraction=float(yy.mean())))
    for j in range(1,5):
        yy=y[seg==j];composition.append(dict(group=f'future_{j}',N=len(yy),N_B=int((yy==0).sum()),N_A=int((yy==1).sum()),attack_fraction=float(yy.mean())))
    for file,rows in [('classification_comparison',[v for n,v in overall.items() if n.endswith('_ALL') or n.startswith('ALWAYS')]),('policy_comparison',[v for n,v in overall.items() if n.endswith('_GATED') or n.startswith('SOURCE') or n.startswith('ALWAYS')]),('segment_metrics',segments),('window_counts',windows),('paired_intervals',paired),('correction_harm',harm),('class_composition',composition)]:savecsv(run/f'tables/{file}.csv',rows)
    decision=dict(status='PILOT_COMPLETED',completed_time=now(),rows=68400,windows=684,segment_rows=[17100]*4,budget=protocol['budget'],overall=overall,stable_workpoint=stable,segment_targets={n:[s['target_status'] for s in segments if s['method']==n] for n in overall},next_action='Stop fixed proportional mixing pilot and do not automatically enter further research')
    render(run,fr,decision,segments,paired,harm,composition)
    savej(run/'config/evaluation_checks.json',dict(rows=68400,windows=684,segment_rows=[17100]*4,segment_windows=[171]*4,blocks_per_segment=[18]*4,last_block_windows=[1]*4,shared_replicates=1000,denominator_conservation=True,evaluation_only_frozen_actions=True))
    writej(run/'reports/decision.json',decision)
    print(pd.DataFrame(overall.values())[['method','R_B','R_A','U_B','U_A','U_bal','target_status']].to_string(index=False),flush=True)
def render(run,fr,d,segments,paired,harm,composition):
    pct=lambda x:'NA' if x is None or pd.isna(x) else f'{100*x:.4f}%'
    pp=lambda x:'NA' if x is None or pd.isna(x) else f'{100*x:.4f}'
    lines=['# ToN Fixed Capacity Historical Chords Recent Mixed Memory pilot','', 'Complete a joint assessment of the 68,400 clause, 684 and 10k windows;[20680000,27520000) range, four 17,100 items, 171 window, 20680000/22390000/24100000/25810000/27520000 border.','', 'The new wheel is fitted with labels 0 and has accumulated 1200. Each LOCAL/TARGET condition is R200/C200; Source RF All door controls use only the corresponding C200 and no R. MIXED extracts 400 from observed and developed T1/T2 combined 800 pools, and cannot be called a independent 400 budget experiment. The evaluation is based on the real 68,400 clause, which is not in the construct.','', 'This is a developmental comparison on existing data, not a independent blind test. mixing and changing the time coverage, category composition and sample content; It is not possible to attribute the gains entirely to the category of equilibrium, some kind of drift or the best 50/50 ratio. All adaptations recorded before T2 are available, along the simulated labels reach, without claiming to be true wall clock latency.','', 'Implementation of the ## protocol with freeze','', 'The original roles are sorted by availability_rank, with R1/R2/C1/C2 separately seeded by independent PCG64 and 20261006/20261007/20261008/20261009 without re-selecting 100; After the mixing sequence, the original R/C never switches, starting the 400 does not enter the mixed pool. Sampling is indexed only, without class replacement sampling, retrieval or seed / sample search. Protocols, roles, parameters, time, saving and reconstructing methods; Researchers have previously observed these labels; do not claim new blinded labels.','', 'LOCAL fixed 48 dimension (excluding DNS_QUERY_ID) source filling with scale, prime value representation, mixing distance and intersection, k15 equal weighting, etc. Distance in rank order. MIXED_TARGET only R_M train 200 tree gini deep 5, leaf 5, sqrt, bootstrap, random_state=20260929 small RF, according to classes_ mapping Attack; There is no modulation, class equilibrium, probability calibration or target preprocessing fit. Source RF maintains the original 49 pipeline and p>=0.5 definition.','', 'T1_ONLY mapping T1_LOCAL/T1_TARGET/SOURCE_GATE_T1; T2_ONLY mapping T2_LOCAL/T2_TARGET/SOURCE_GATE_T2All freeze actions are directly from the T2 action file, the original name being retained. SOURCE_GATE_T1 was reconstructed in the T2 experiment, not T1 deployed at the time. baseline source complete see protocol.json of baseline_map; No re-training, no re-election rules.','', 'The door control uses the preserved 0.50 to 1.00 steps 0.05 and OFF; c>=t without epsilon. Non-OFF branches require C acceptance≥10 and 20×errors≤accepted. Maximize U_bal, then min(UB,UA), then minimize FP+FN, then ascending B/A threshold IDs, with OFF last. The C class is missing and the OFF class is double. C experience 5% is not a statistical guarantee. prepare precedes freeze model / rules and all movements; evaluate does not train, does not sample, does not select.','', '## true composition and calibration support','', '| group |N|B|A|A ratio |','|---|---:|---:|---:|---:|']
    for v in composition:lines.append(f"|{v['group']}|{v['N']}|{v['N_B']}|{v['N_A']}|{pct(v['attack_fraction'])}|")
    support=pd.read_csv(run/'calibration/support.csv');lines+=['','R is not trained to self-predict; C is used for selection and cannot be validated as independent. MIXED_TARGET training category:'+str(fr['training_classes'])+'。','','| method | role | True B/A| candidate B/A|B threshold / accepting / error |A threshold / accepting / error | state |','|---|---|---|---|---|---|---|']
    fmt=lambda x:'NA' if pd.isna(x) else str(x)
    for _,v in support.iterrows():lines.append(f"|{v.method}|{v.role}|{v.true_B}/{v.true_A}|{fmt(v.pred_B)}/{fmt(v.pred_A)}|"+'/'.join(fmt(v[k]) for k in ['B_threshold','B_accepted','B_errors'])+'|'+'/'.join(fmt(v[k]) for k in ['A_threshold','A_accepted','A_errors'])+f'|{v.status}|')
    lines+=['','## overall with four results','','RB=FN/(TN+FN),RA=FP/(TP+FP),UB=TN/N_B,UA=TP/N_A; abstention does not count for detection or label correction, the zero factor risk is NA. The operating point dual class experience requires risk for each ≤5% to be used for each ≥10%, with ≥100 bars and ≥10 windows for each action. Stabilize the overall = and at least the 3/4 segment, not the risk assurance.','','| method |RB|RA|UB|UA|U_bal| covers | accepts B/A| overall | four stages | stability |','|---|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for name,v in d['overall'].items():lines.append('|'+name+'|'+'|'.join(pct(v[k]) for k in ['R_B','R_A','U_B','U_A','U_bal','coverage'])+f"|{v['accepted_B']}/{v['accepted_A']}|{v['target_status']}|"+'/'.join('yes' if x=='JOINT_TARGET_MET' else 'no' for x in d['segment_targets'][name])+f"|{d['stable_workpoint'][name]}|")
    lines+=['','| Parts | Methods |N_B/N_A|TN/FN/FP/TP| abstention B/A|RB|RA|UB|UA|U_bal| covers | action B/A| window B/A| operating point / Failure item |','|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for v in segments:lines.append(f"|{v['segment']}|{v['method']}|{v['N_B']}/{v['N_A']}|{v['TN']}/{v['FN']}/{v['FP']}/{v['TP']}|{v['abstain_B']}/{v['abstain_A']}|"+'|'.join(pct(v[k]) for k in ['R_B','R_A','U_B','U_A','U_bal','coverage'])+f"|{v['accepted_B']}/{v['accepted_A']}|{v['windows_B']}/{v['windows_A']}|{v['target_status']}/{v['failure_metrics']}|")
    lines+=['','For complete total confusion, abstention, dividers, number of movement windows, see classification_comparison.csv and policy_comparison.csv.','','## assigned mating intervals','','The table below is left-right, and the difference between the units is a percentage point. The 10 window is blocked in four segments, with the 1 window retained, with 18 blocks with 18 blocks drawn back in each segment. PCG64(20261010) together with 1000, all methods share copies. 2.5%/97.5% percent; A risk difference is invalid if either denominator is 0; intervals are NA when fewer than 90% of replicates are valid. The point-like exploratory range, which has not been repeatedly corrected, does not guarantee an arbitrary drift of effective.','','| Left-right | indicator | difference (pp)|95% range (pp)| effective rate |','|---|---|---:|---|---:|']
    for v in paired:lines.append(f"|{v['left']}−{v['right']}|{v['metric']}|{pp(v['delta'])}|[{pp(v['ci_low'])}, {pp(v['ci_high'])}]|{v['valid_fraction']:.3f}|")
    lines+=['','## All accepted label correction','','The door control refuses to include label correction, with four sections and directions specifying correction_harm.csv.','','| method | see |wrong → right|right → wrong| net label correction |','|---|---|---:|---:|---:|']
    for v in harm:
        if v['scope']=='overall' and v['direction']=='ALL':lines.append(f"|{v['method']}|{v['reference']}|{v['wrong_to_right']}|{v['right_to_wrong']}|{v['net_corrections']}|")
    conclusions=[]
    for name in ['MIXED_LOCAL','MIXED_TARGET','SOURCE_GATE_MIXED']:
        rr=fr['methods'][name];cs=support[(support.method==name)&(support.role=='C')].iloc[0];actionname=name if name.startswith('SOURCE') else name+'_GATED';v=d['overall'][actionname];ss=d['segment_targets'][actionname]
        conclusions.append(f"{name} C candidate B={int(cs.pred_B)}，{'It’s getting to' if cs.pred_B>=10 else 'Not reached'} Minimum support for candidates under 10; B gate control {'Activated' if rr['B']['threshold'] is not None else 'OFF'} A gate control {'Activated' if rr['A']['threshold'] is not None else 'OFF'}. calibration state {rr['status']}. future overall {v['target_status']} Failure {v['failure_metrics'] or 'none'} It’s a good idea.{ss.count('JOINT_TARGET_MET')}/4 and = stable {d['stable_workpoint'][actionname]} and coverage {pct(v['coverage'])}. calibration feasibility is judged separately from future operating point.")
        if rr['B']['threshold'] is not None and v['target_status']!='JOINT_TARGET_MET':conclusions.append(name+'B can be branched on C, but future two-class operating point has not been established; Supporting improvement has not addressed the issue of migration to the assessment area.')
    for v in paired:
        if v['metric']=='U_bal':conclusions.append(f"{v['left']}−{v['right']}：ΔU_bal={pp(v['delta'])} One percentage point, 95% between [{pp(v['ci_low'])}, {pp(v['ci_high'])}]。")
    source=d['overall']['SOURCE_GATE_MIXED'];source_ok=d['stable_workpoint']['SOURCE_GATE_MIXED']
    successful=[]
    for m in ['LOCAL','TARGET']:
        name='MIXED_'+m;source_delta=next(v for v in paired if v['left']==name+'_ALL' and v['right']=='SOURCE_RF_ALL' and v['metric']=='U_bal')
        gains=[v for v in paired if v['left']==name+'_ALL' and v['right'] in [f'T1_{m}_ALL',f'T2_{m}_ALL','SOURCE_RF_ALL'] and v['metric']=='U_bal']
        better=all(v['delta']>0 for v in gains)
        conclusions.append(name+('The total acceptance of U_bal points is estimated to be simultaneously higher than the two single-period conditions and the source RF; It’s a big difference.' if better else 'The total acceptance of U_bal does not simultaneously exceed two single-period conditions and the source RF; It’s not just a relatively poor T2 that can declare success.'))
        if d['stable_workpoint'][name+'_GATED'] and better:successful.append(name)
    if source_ok:nextstep='Source RF Mixed gate control achieves a stable operating point and prioritizes subsequent study of the mixed value and processing of calibration information, without forcing the attribution of the new classifier; It is not yet possible to declare the best ratio.'
    elif successful:nextstep='Under the conditions'+','.join(successful)+'A stable operating point and increased utility can be achieved, with the option of a pre-registered memory retention / update strategy study; It doesn’t prove that 50/50 is the best.'
    else:nextstep='This time, the mixing of fixed proportions did not establish a stable recovery. It is recommended that search be suspended at the current fixed distance to the single mix ratio; If otherwise authorized, priority design controls indicate that the / model intervenes and retains the source RF, freeze memory control. It cannot be promoted as invalid for all neighboring laws or for all memory methods.'
    conclusions.append(nextstep);conclusions.append('Fixed sampling simultaneously changes time, category and sample content, not as a single causal attribution. Stop the current pilot, do not overload, do not change the /k/ ratio from / model / gate control, do not add labels, do not enter the full quantity or follow-up studies.')
    d['evidence_based_conclusions']=conclusions;d['next_action']=nextstep+' The wheel is stopped.'
    lines+=['','## classification, calibration feasibility and operating point rating readings respectively','']+['- '+s for s in conclusions]
    lines+=['','## Assets and Actual Orders','',f' All new assets: `{run}`. baseline raw asset only path reference, not replicated model; Action caching and reuse. Preserving protocols, roles, calibration candidates, / rules, / support, models, movements, matching ranges, label correction, counting and execution logs. Defaultly uncovered assets, restored verification protocol / role / count / parameters.','', '```bash']
    for stage in ['prepare','evaluate']:lines.append(f'{sys.executable} {run}/src/run_pilot.py --stage {stage} --run-dir {run}')
    lines+=['```',''];path=run/'reports/RESULTS_ZH.md'
    audit=run/'reports/execution_audit_ZH.md'
    if audit.exists():lines.append(audit.read_text())
    if path.exists():require(path.read_text()=='\n'.join(lines),'existing report mismatch')
    else:path.write_text('\n'.join(lines))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','evaluate'],required=True);p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args();run=args.run_dir.resolve();require(run.parent==BASE/'ton_mixed_memory_pilot_v1','output scope')
    with (run/'logs/commands.sh').open('a') as f:f.write(shlex.join([sys.executable,*sys.argv])+'\n')
    try:(prepare if args.stage=='prepare' else evaluate)(run)
    except Exception as e:
        with (run/'logs/implementation_events.log').open('a') as f:f.write(now()+' '+repr(e)+' evaluation_labels_opened='+str((run/'tables/evaluation_label_access.json').exists())+'\n')
        writej(run/'reports/blocking_reason.json',dict(time=now(),reason=str(e),stage=args.stage));raise
if __name__=='__main__':main()
