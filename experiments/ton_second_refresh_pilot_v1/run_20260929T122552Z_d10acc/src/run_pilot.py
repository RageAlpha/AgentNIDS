#!/usr/bin/env python
import argparse,os,sys,shlex
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from fixed_core import *
from frozen_functions import attack_probability,gate,choose,tests,boot_metrics,ci,GRID
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
BASE=ROOT/'outputs/post_p7_diagnostics'
T1=BASE/'ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd'
OLD=BASE/'reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr'
BOUNDS=[20680000,22390000,24100000,25810000,27520000]
MAIN=['T1_LOCAL','T1_TARGET','T2_LOCAL','T2_TARGET']
PAIRS=[(f'T2_{m}_{l}',f'T1_{m}_{l}') for m in ['LOCAL','TARGET'] for l in ['ALL','GATED']]+[('SOURCE_GATE_T2','SOURCE_GATE_T1')]+[(f'T2_{m}_ALL','SOURCE_RF_ALL') for m in ['LOCAL','TARGET']]+[(f'T2_{m}_GATED','SOURCE_GATE_T2') for m in ['LOCAL','TARGET']]
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
def prepare(run):
    p1=readj(T1/'config/protocol.json');vp=readj(OLD/'config/protocol_v1.json');a=Assets(ROOT,'ToN',vp)
    startup=pd.read_csv(OLD/'calibration/sample_roles.csv');startup=startup[startup.domain=='ToN'].availability_rank.to_numpy(np.int64)
    r1=pd.read_csv(T1/'config/recent_sample_roles.csv');oldr=r1.availability_rank.to_numpy(np.int64)
    pilot=np.load(OLD/'pilot/ToN_evaluation_sample_ranks.npy');require(len(pilot)==274200 and len(startup)==len(oldr)==400,'input counts')
    w=pd.read_csv(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv');w=w[(w.domain=='ToN')&(w.scheme=='heldout')&(w.window_size==10000)].sort_values('rank_start')
    require(np.array_equal(w.rank_start,np.arange(0,27520000,10000)) and np.array_equal(w.rank_end_exclusive,np.arange(10000,27520001,10000)),'rank coordinate')
    ranks=pilot[(pilot>=BOUNDS[0])&(pilot<BOUNDS[-1])];require(len(ranks)==68400 and np.all(np.diff(ranks)>0),'evaluation rank count/order')
    win=(ranks-BOUNDS[0])//10000;require(np.array_equal(np.bincount(win),np.full(684,100)),'684 windows x 100')
    candidates=np.arange(20570000,20670000,dtype=np.int64);candidates=candidates[~np.isin(candidates,np.r_[startup,oldr,pilot])];require(len(candidates)>=400,'candidate shortage')
    chosen=np.sort(np.random.Generator(np.random.PCG64(20261003)).choice(candidates,400,replace=False));role=np.full(400,'C');role[np.random.Generator(np.random.PCG64(20261004)).permutation(400)[:200]]='R'
    params=dict(p1['model_parameters']);params['n_jobs']=min(8,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1)
    protocol=dict(experiment='ton_second_refresh_pilot_v1',created_at=now(),input_assets=a.info,T1_run=str(T1),original_run=str(OLD),source_model_manifest=str(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json'),T1_model=str(T1/'models/RECENT_TARGET.joblib'),update_rank=20680000,recent_candidate_range=[20570000,20670000],arrival_gap=[20670000,20680000],evaluation_range=[BOUNDS[0],BOUNDS[-1]],segment_boundaries=BOUNDS,expected_rows=68400,segment_rows=[17100]*4,segment_windows=[171]*4,sampling_seed=20261003,split_seed=20261004,generator='PCG64',candidate_count=len(candidates),exclusions='startup400,T1_400,original_pilot274200',distance=p1['distance'],feature_indices=p1['feature_indices'],model_parameters=params,gate_grid=p1['gate_grid'],gate=p1['gate'],tie_order=p1['tie_order'],methods=MAIN,controls=['SOURCE_RF_ALL','SOURCE_GATE_T1','SOURCE_GATE_T2','ALWAYS_A','ALWAYS_B'],source_gate_T1='retrospectively constructed in this run, C1 only',pairs=PAIRS,budget=dict(new_adaptation=400,previous_adaptation=800,cumulative_acquired_adaptation=1200,per_local_target=400,per_source_gate=200,evaluation_existing_labels=68400,evaluation_labels_used_for_update=False),bootstrap=dict(seed=20261005,replicates=1000,block_windows=10,keep_short_final=True,shared_replicates=True,percentiles=[2.5,97.5],minimum_valid_fraction=.9),workpoint=dict(max_R_B=.05,max_R_A=.05,min_U_B=.1,min_U_A=.1,min_action_count=100,min_action_windows=10,stable='overall and at least 3/4 segments'),interpretation='targeted development at observed failing T2; no independent blind test, optimal frequency claim, drift identification or risk guarantee',stop='one 68400-row evaluation; no extra labels, refresh, representation learning or full evaluation')
    require(GRID==p1['gate_grid'],'grid precision mismatch')
    if (run/'config/protocol.json').exists():protocol['created_at']=readj(run/'config/protocol.json')['created_at']
    savej(run/'config/protocol.json',protocol)
    savecsv(run/'config/recent_sample_roles.csv',pd.DataFrame(dict(availability_rank=chosen,role=role)))
    saven(run/'config/frozen_indices.npz',rank=ranks,recent_rank=chosen,role=role)
    if not (run/'config/index_freeze.json').exists():writej(run/'config/index_freeze.json',dict(time=now(),roles_saved_before_labels=True,new_labels=400,candidate_count=len(candidates)))
    require((run/'config/index_freeze.json').exists() and (run/'config/frozen_indices.npz').exists(),'indices must be frozen before labels')
    require(np.unique(chosen).size==400 and not np.isin(chosen,np.r_[startup,oldr,pilot]).any() and (role=='R').sum()==(role=='C').sum()==200,'roles/exclusions')
    if (run/'tables/actions_complete.json').exists():
        z=np.load(run/'tables/evaluation_actions.npz');require(np.array_equal(z['rank'],ranks),'completed action indices');print('prepare already complete; validated protocol/roles/counts');return
    # Only selected 400 new labels may be acquired after the persisted freeze above.
    labelpath=run/'calibration/recent400.npz'
    if labelpath.exists():
        z=np.load(labelpath);require(np.array_equal(z['rank'],chosen),'saved label ranks');y=z['y']
    else:
        labels=np.load(a.paths['labels'],mmap_mode='r');y=np.array(labels[chosen]);require(np.isin(y,[0,1]).all(),'binary labels');saven(labelpath,rank=chosen,y=y)
        writej(run/'calibration/recent_label_access.json',dict(time=now(),rows=400,source=a.paths['labels'],selection='frozen_indices'))
    z1=np.load(T1/'calibration/recent400.npz');require(np.array_equal(z1['rank'],oldr),'T1 labels/roles');y1=z1['y']
    r=role=='R';c=~r;rx=a.features(chosen[r]);cx=a.features(chosen[c]);cols=protocol['feature_indices']
    saven(run/'calibration/T2_reference.npz',rank=chosen[r],x=rx,y=y[r])
    mp=run/'models/T2_TARGET.joblib'
    if mp.exists():model=joblib.load(mp);require(all(model.get_params()[k]==v for k,v in params.items()),'RF params')
    else:model=RandomForestClassifier(**params).fit(rx[:,cols],y[r]);joblib.dump(model,mp)
    q,_,_=evidence(cx,rx,y[r],np.ones(200,np.int8),a.cat,a.num,a.scale)
    scores={'T2_LOCAL':(q,y[c],chosen[c]),'T2_TARGET':(attack_probability(model,cx[:,cols]),y[c],chosen[c]),'SOURCE_GATE_T1':(a.probs(oldr[r1.role=='C']),y1[r1.role=='C'],oldr[r1.role=='C']),'SOURCE_GATE_T2':(a.probs(chosen[c]),y[c],chosen[c])}
    rules={m:readj(T1/'calibration/selected_rules.json')['methods'][m.replace('T1','RECENT')] for m in MAIN if m.startswith('T1')};allrows=[];support=[]
    for name,(score,truth,rr) in scores.items():
        rule,rows=choose(score,truth,name);rules[name]=rule;allrows+=rows
        savecsv(run/f'calibration/{name}_C.csv',pd.DataFrame(dict(rank=rr,y=truth,score=score,label=(score>=.5).astype(int),action=gate(score,rule))))
        support.append(dict(method=name,role='C',N=len(truth),true_B=int((truth==0).sum()),true_A=int((truth==1).sum()),pred_B=int((score<.5).sum()),pred_A=int((score>=.5).sum())))
    for age,ys,roles in [('T1',y1,r1.role.to_numpy()),('T2',y,role)]:
        for learner in ['LOCAL','TARGET']:
            for ro in ['R','C']:
                if age=='T2' and ro=='C':continue
                yy=ys[roles==ro];support.append(dict(method=age+'_'+learner,role=ro,N=len(yy),true_B=int((yy==0).sum()),true_A=int((yy==1).sum()),pred_B=None,pred_A=None))
    savecsv(run/'calibration/support.csv',support);savecsv(run/'calibration/candidates.csv',allrows)
    if not (run/'calibration/selected_rules.json').exists():writej(run/'calibration/selected_rules.json',dict(freeze_time=now(),methods=rules,protocol=protocol,training_classes=model.classes_.tolist()))
    else:require(readj(run/'calibration/selected_rules.json')['methods']==rules,'rules changed')
    checks=tests();checks.update(rank_coordinate_verified=True,exclusions=True,R2=200,C2=200,candidate_count=len(candidates),target_training_classes=model.classes_.tolist(),source_original_rule='p>=0.5 per original protocol; cached predicted_class not authoritative')
    old=np.load(T1/'tables/evaluation_actions.npz');idx=np.searchsorted(old['rank'],ranks);require(np.array_equal(old['rank'][idx],ranks),'T1 cached rank alignment')
    data=dict(rank=ranks,window=win,segment=np.searchsorted(BOUNDS[1:-1],ranks,side='right')+1)
    for method in ['T1_LOCAL','T1_TARGET']:
        prev=method.replace('T1','RECENT');data['score_'+method]=old['score_'+prev][idx]
        for layer in ['ALL','GATED']:data['action_'+method+'_'+layer]=old['action_'+prev+'_'+layer][idx]
        require(np.array_equal(gate(data['score_'+method],rules[method]),data['action_'+method+'_GATED']),'T1 frozen gates')
    p=old['score_SOURCE_RF_ALL'][idx];raw=old['action_SOURCE_RF_ALL'][idx];require(np.array_equal(raw,(p>=.5).astype(np.int8)),'original source prediction differs; preserve original rule and investigate')
    checkidx=np.linspace(0,len(ranks)-1,12,dtype=int);require(np.array_equal(a.probs(ranks[checkidx]),p[checkidx]),'source cached score')
    ref=np.load(T1/'calibration/RECENT_reference.npz');xx=a.features(ranks[checkidx]);qq,_,_=evidence(xx,ref['x'],ref['y'],np.ones(12,np.int8),a.cat,a.num,a.scale)
    require(np.array_equal(qq,data['score_T1_LOCAL'][checkidx]),'T1 LOCAL small recomputation')
    m1=joblib.load(T1/'models/RECENT_TARGET.joblib');require(np.allclose(attack_probability(m1,xx[:,cols]),data['score_T1_TARGET'][checkidx],atol=1e-14,rtol=0),'T1 RF small recomputation')
    checks['frozen_cache_recheck_rows']=12;checks['source_prediction_rule_mismatches']=0
    for method in ['T2_LOCAL','T2_TARGET']:data['score_'+method]=np.empty(len(ranks))
    for start in range(0,len(ranks),3000):
        end=min(start+3000,len(ranks));xx=a.features(ranks[start:end]);qq,_,_=evidence(xx,rx,y[r],np.ones(end-start,np.int8),a.cat,a.num,a.scale)
        data['score_T2_LOCAL'][start:end]=qq;data['score_T2_TARGET'][start:end]=attack_probability(model,xx[:,cols]);print('actions',end,'/',len(ranks),flush=True)
    for method in ['T2_LOCAL','T2_TARGET']:
        score=data['score_'+method];data['action_'+method+'_ALL']=(score>=.5).astype(np.int8);data['action_'+method+'_GATED']=gate(score,rules[method])
    data['score_SOURCE_RF_ALL']=p;data['action_SOURCE_RF_ALL']=raw
    for method in ['SOURCE_GATE_T1','SOURCE_GATE_T2']:data['action_'+method]=gate(p,rules[method])
    data['action_ALWAYS_A']=np.ones(len(ranks),np.int8);data['action_ALWAYS_B']=np.zeros(len(ranks),np.int8)
    saven(run/'tables/evaluation_actions.npz',**data);savej(run/'config/prepare_checks.json',checks)
    writej(run/'tables/actions_complete.json',dict(time=now(),rows=len(ranks),windows=684,segment_rows=[17100]*4,no_truth_in_actions=True,actions_persisted_before_evaluation_labels=True))
    print('PREPARE COMPLETE',flush=True)

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
        overall[name]=dict(method=name,**metrics(co.sum(0)));overall[name]['coverage']=overall[name]['acceptance']
        for j in range(1,5):segments.append(dict(method=name,segment=j,**metrics(co[segw==j].sum(0))))
        for i,row in enumerate(co):windows.append(dict(method=name,window=i,rank_start=BOUNDS[0]+10000*i,segment=int(segw[i]),**dict(zip(COUNT_KEYS,map(int,row)))))
    blocks=[];strata=[]
    for j in range(1,5):
        ix=np.flatnonzero(segw==j);ids=[]
        for k in range(0,len(ix),10):ids.append(len(blocks));blocks.append(ix[k:k+10])
        strata.append(np.array(ids))
    rng=np.random.Generator(np.random.PCG64(20261005));draw=np.array([np.concatenate([rng.choice(s,len(s),replace=True) for s in strata]) for _ in range(1000)])
    weights=np.array([np.bincount(d,minlength=len(blocks)) for d in draw]);saven(run/'tables/bootstrap_draws.npz',indices=draw)
    savecsv(run/'tables/bootstrap_blocks.csv',[dict(block=i,segment=int(segw[b[0]]),first_window=int(b[0]),windows=len(b)) for i,b in enumerate(blocks)])
    boot={n:boot_metrics(weights@np.array([c[b].sum(0) for b in blocks])) for n,c in counts.items()};paired=[]
    for left,right in PAIRS:
        for metric in ['U_bal','U_B','U_A','R_B','R_A']:
            lv=overall[left][metric];rv=overall[right][metric];paired.append(dict(left=left,right=right,metric=metric,delta=lv-rv if lv is not None and rv is not None else None,**ci(boot[left][metric]-boot[right][metric])))
    harm=[]
    for left,right in [(m+'_ALL','SOURCE_RF_ALL') for m in MAIN]+[(f'T2_{m}_ALL',f'T1_{m}_ALL') for m in ['LOCAL','TARGET']]:
        h=a['action_'+left];base=a['action_'+right]
        for j in range(5):
            subset=np.ones(len(y),bool) if j==0 else seg==j
            for direction in ['ALL','B_to_A','A_to_B']:
                mm=subset if direction=='ALL' else subset&(base==(0 if direction=='B_to_A' else 1))&(h==(1 if direction=='B_to_A' else 0))
                fix=int((mm&(base!=y)&(h==y)).sum());hurt=int((mm&(base==y)&(h!=y)).sum());harm.append(dict(method=left,reference=right,scope='overall' if j==0 else f'segment_{j}',direction=direction,wrong_to_right=fix,right_to_wrong=hurt,net_corrections=fix-hurt))
    stable={n:v['target_status']=='JOINT_TARGET_MET' and sum(s['target_status']=='JOINT_TARGET_MET' for s in segments if s['method']==n)>=3 for n,v in overall.items()}
    composition=[]
    for age,rolepath,labelpath in [('T1',T1/'config/recent_sample_roles.csv',T1/'calibration/recent400.npz'),('T2',run/'config/recent_sample_roles.csv',run/'calibration/recent400.npz')]:
        roles=pd.read_csv(rolepath);z=np.load(labelpath);require(np.array_equal(roles.availability_rank,z['rank']),'composition roles')
        for ro in ['R','C']:
            yy=z['y'][roles.role==ro];composition.append(dict(group=age+'_'+ro,N=len(yy),N_B=int((yy==0).sum()),N_A=int((yy==1).sum()),attack_fraction=float(yy.mean())))
    for j in range(1,5):
        yy=y[seg==j];composition.append(dict(group=f'future_{j}',N=len(yy),N_B=int((yy==0).sum()),N_A=int((yy==1).sum()),attack_fraction=float(yy.mean())))
    for file,rows in [('classification_comparison',[v for n,v in overall.items() if n.endswith('_ALL') or n.startswith('ALWAYS')]),('policy_comparison',[v for n,v in overall.items() if n.endswith('_GATED') or n.startswith('SOURCE') or n.startswith('ALWAYS')]),('segment_metrics',segments),('window_counts',windows),('paired_intervals',paired),('correction_harm',harm),('class_composition',composition)]:savecsv(run/f'tables/{file}.csv',rows)
    decision=dict(status='PILOT_COMPLETED',completed_time=now(),rows=68400,windows=684,segment_rows=[17100]*4,budget=protocol['budget'],overall=overall,stable_workpoint=stable,segment_targets={n:[s['target_status'] for s in segments if s['method']==n] for n in overall},next_action='Stop; According to the results of this alternative design follow-up, no refresher or learning is added')
    render(run,fr,decision,segments,paired,harm,composition)
    savej(run/'config/evaluation_checks.json',dict(rows=68400,windows=684,segment_rows=[17100]*4,segment_windows=[171]*4,blocks_per_segment=[18]*4,last_block_windows=[1]*4,shared_replicates=1000,denominator_conservation=True,evaluation_only_frozen_actions=True))
    writej(run/'reports/decision.json',decision)
    print(pd.DataFrame(overall.values())[['method','R_B','R_A','U_B','U_A','U_bal','target_status']].to_string(index=False),flush=True)

def render(run,fr,d,segments,paired,harm,composition):
    pct=lambda x:'NA' if x is None or pd.isna(x) else f'{100*x:.4f}%'
    lines=['# ToN Returns to pilot: T2 Second refresh','', '68,400 has been completed, 684 has been evaluated, 10k has been completed, [20680000,27520000) has been completed, 17,100 has been completed, 171 has been completed. T2=20680000; The [20570000,20670000) queries, followed by the 10,000 line to simulate the labels reaching the interval, which does not represent the real wall clock latency.','', 'This round added new 400 clause, previously 800 clause, cumulative 1200 clause to fit labels. T1/T2 LOCAL and TARGET each use the corresponding R200/C200, not pairing. Each SOURCE_GATE uses only the corresponding C200; The same labels shares, no recurring billing. The 68,400 value is not updated or selected.','', 'T2 is selected from the observed failure stage, which is a targeted development experiment, not a independent blind test. The composition of the sample varies with the time and cannot be attributed entirely to age; A refresh of the unidentified drift type does not prove the optimal frequency or any time point effective.','', '## freeze and Implementation','', 'Protocols, pointers, indexes and roles are saved first, and then the selected 400 line labels is read; Excludes the 400 initiation, T1 addition of 400 and 274,200 of the original pilot index. PCG64 sample 20261003, role 20261004; It’s not a replacement class or a replacement. After training and selection, prepare preserves all the moves, while evaluate only reads freeze moves with a valid rating.','', 'LOCAL repeated 48 dimensional mixing distance, source scale/ filling, cutting, k15 equal weighting stabilizing rank breaking through ties. TARGET with R2 only from the head train fixed 200 tree deep 5 minor RF and take the Attack probability by classes_; There is no target preprocessing fit. Source RF The original 49 pipeline and model remain unchanged, the original p>=0.5 rule, caching checks consistent. T1 models and gate controls are all freeze, with 12 defining the index recalculation check caching.','', 'The SOURCE_GATE_T1 is a control built on the back of the C1 wheel, not the T1 strategy deployed at the time. The source gate control uses only C labels, no trained forecasters, no reverse probability. It is marked with 200 and LOCAL/TARGET as 400, and the training process is not exactly the same.','', 'The gate control uses the original floating-point grid 0.50 to 1.00 steps 0.05 and OFF, containing the equivalent without epsilon; For each activated branch, C accepts ≥10 and 20× accepts ≤ errors. The goal and the ties rule remain the same. The real class C is missing and the dual OFF is missing. C experience 5% does not constitute a statistical assurance of risk.','', '## roles and composition','', '| group |N|B|A|A ratio |','|---|---:|---:|---:|---:|']
    for v in composition:lines.append(f"|{v['group']}|{v['N']}|{v['N_B']}|{v['N_A']}|{pct(v['attack_fraction'])}|")
    lines+=['','Training classes:'+str(fr['training_classes'])+'; R is not a training cluster self-prediction, C is a selective cluster. Full candidates and support see calibration directory.','','| method | calibration state |B threshold / accepts / error |A threshold / accepts / error |','|---|---|---|---|']
    for n,r in fr['methods'].items():lines.append('|'+n+'|'+r['status']+'|'+'|'.join(f"{r[k]['threshold']}/{r[k]['accepted']}/{r[k]['errors']}" for k in ['B','A'])+'|')
    lines+=['','## total and four operating point segments','','RB=FN/(TN+FN),RA=FP/(TP+FP),UB=TN/N_B,UA=TP/N_A. abstention is not detected, OFF risk NAOperating points require both risks≤5%, both utilities≥10%, and each action accepting≥100 records across≥10 windows; Stabilizing the = as a whole and at least the 3/4 point is the operating point experience.','','| method |RB|RA|UB|UA|U_bal| covers | accepts B/A| overall | four stages | stability |','|---|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for n,v in d['overall'].items():lines.append('|'+n+'|'+'|'.join(pct(v[k]) for k in ['R_B','R_A','U_B','U_A','U_bal','coverage'])+f"|{v['accepted_B']}/{v['accepted_A']}|{v['target_status']}|"+'/'.join('yes' if x=='JOINT_TARGET_MET' else 'no' for x in d['segment_targets'][n])+f"|{d['stable_workpoint'][n]}|")
    lines+=['','| Parts | Methods |N_B/N_A|TN/FN/FP/TP| abstention B/A|RB|RA|UB|UA|U_bal| covers | action B/A| action window B/A| operating point and failures |','|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|']
    for v in segments:lines.append(f"|{v['segment']}|{v['method']}|{v['N_B']}/{v['N_A']}|{v['TN']}/{v['FN']}/{v['FP']}/{v['TP']}|{v['abstain_B']}/{v['abstain_A']}|"+'|'.join(pct(v[k]) for k in ['R_B','R_A','U_B','U_A','U_bal','acceptance'])+f"|{v['accepted_B']}/{v['accepted_A']}|{v['windows_B']}/{v['windows_A']}|{v['target_status']}: {v['failure_metrics']}|")
    lines+=['','Overall, the full subset, confusion, abstention and the number of windows see classification_comparison.csv and policy_comparison.csv.','','## predicted pair comparison','','Difference = left-right, unit percentage points; The risk negative value, the positive utility value, represents an improvement in the dimension. The four sections are individually blocked with continuous 10 windows, with the last 1 windows retained, with each 18 block having a retracted 18 block; 1000 times PCG64(20261005), all methods shared. 2.5%/97.5% point exploratory range with no multiple corrections; risk is ineffective if a factor is zero, and effective is <90% in NA. It’s not an arbitrary diversion guarantee.','','| Left-right | indicator | difference (pp)|95% range (pp)| effective rate |','|---|---|---:|---|---:|']
    for v in paired:lines.append(f"|{v['left']}−{v['right']}|{v['metric']}|{pct(v['delta']).replace('%','')}|[{pct(v['ci_low']).replace('%','')}, {pct(v['ci_high']).replace('%','')}]|{v['valid_fraction']:.3f}|")
    lines+=['','## All accepted label correction','','The door controller abstention is not label correction. The four points and directions are shown in the correction_harm.csv detail.','','| method | see |wrong → right|right → wrong| net label correction |','|---|---|---:|---:|---:|']
    for v in harm:
        if v['scope']=='overall' and v['direction']=='ALL':lines.append(f"|{v['method']}|{v['reference']}|{v['wrong_to_right']}|{v['right_to_wrong']}|{v['net_corrections']}|")
    lines+=['','## proof reading and next step','']
    for method in ['T2_LOCAL','T2_TARGET']:
        v=d['overall'][method+'_GATED'];ss=d['segment_targets'][method+'_GATED'];first=next(s for s in segments if s['method']==method+'_GATED' and s['segment']==1)
        lines.append(f"- {method} Overall {v['target_status']} Failure {v['failure_metrics'] or 'none'}; Near end 1 {first['target_status']} Failure {first['failure_metrics'] or 'none'}. 4 points {ss.count('JOINT_TARGET_MET')}/4 , stable operating point ={d['stable_workpoint'][method+'_GATED']}。")
        for right,left in [(method.replace('T2','T1')+'_GATED',method+'_GATED'),('SOURCE_RF_ALL',method+'_ALL'),('SOURCE_GATE_T2',method+'_GATED')]:
            pp=next(p for p in paired if p['left']==left and p['right']==right and p['metric']=='U_bal');lines.append(f"- {left} Relatively {right}：ΔU_bal={pct(pp['delta'])}(percent value), 95% range [{pct(pp['ci_low'])}, {pct(pp['ci_high'])}]; This is a comparison of usefulness and operating point is different.")
        if ss[0]!='JOINT_TARGET_MET':lines.append(f'- {method} Paragraph 1 still fails and cannot be claimed to have been updated in a timely manner; Combining the above calibration support, ALL classification and GATED failure to distinguish the reasons for not introducing features without information.')
        elif ss[-1]!='JOINT_TARGET_MET':lines.append(f'- {method} The near-end recovery and the later phase failed, supporting alternative design refresher frequency studies, this round does not add updates.')
    source=d['overall']['SOURCE_GATE_T2'];lines.append(f"- SOURCE_GATE_T2 as a whole {source['target_status']} and coverage {pct(source['coverage'])}, four sections {d['segment_targets']['SOURCE_GATE_T2']}. Only the control of the source gate is sufficient, should be judged according to the corresponding block operating point and the processing volume, and can not only see the risk decrease.")
    lines+=['','Conclusion: The relative improvement in T1, the relative increase in source RF, and the achievement of two-class operating point are three different problems. Improvements in individual branches or classification indicators are only partial improvements; Failure under the established rule family does not prove that all source RF processing is ineffective.','', 'Stop the current pilot, do not enter the full quantity, do not add labels, do not change the model / from the / gate control, do not add refresh points or indicate learning. Subsequently, the best frequency is determined based on whether the near end is recovered and the source RF controlling study determines the problem, not by this single update.','','## Delivery and Retrieval','','All assets are located at: `'+str(run)+'`. Protocols, roles, fixed input references, checks, models, candidates, rules, actions, bootstrap, complete counts and commands have been saved; The old run is just reading, not hashing.','','```bash']
    py=sys.executable
    for stage in ['prepare','evaluate']:lines.append(f'{py} {run}/src/run_pilot.py --stage {stage} --run-dir {run}')
    lines+=['```',''];
    addendum=run/'reports/interpretation_addendum_ZH.md'
    if addendum.exists():lines.append(addendum.read_text())
    path=run/'reports/RESULTS_ZH.md';require(not path.exists(),'report already exists');path.write_text('\n'.join(lines))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','evaluate'],required=True);p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args();run=args.run_dir.resolve()
    require(str(run.parent)==str(BASE/'ton_second_refresh_pilot_v1'),'output scope')
    with (run/'logs/commands.sh').open('a') as f:f.write(shlex.join([sys.executable,*sys.argv])+'\n')
    try:(prepare if args.stage=='prepare' else evaluate)(run)
    except Exception as e:
        with (run/'logs/implementation_events.log').open('a') as f:f.write(now()+' '+repr(e)+' evaluation_labels_opened='+str((run/'tables/evaluation_label_access.json').exists())+'\n')
        if isinstance(e,ValueError):writej(run/'reports/blocking_reason.json',dict(time=now(),reason=str(e),stage=args.stage))
        raise
if __name__=='__main__':main()
