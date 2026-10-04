#!/usr/bin/env python
"""One fixed ToN recency x learner pilot. No future labels in prepare/action generation."""
import argparse,os,sys,shlex,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn
from fixed_core import Assets,evidence,readj,writej,csv,require,now,window_counts,metrics,COUNT_KEYS,mask
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
OLD=ROOT/'outputs/post_p7_diagnostics/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr'
COR=ROOT/'outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa'
MAIN=['OLD_LOCAL','RECENT_LOCAL','OLD_TARGET','RECENT_TARGET']
BOUNDS=[13820000,17250000,20680000,24100000,27520000]
GRID=[i/100 for i in range(50,101,5)]
PAIRS=[('RECENT_LOCAL','OLD_LOCAL'),('RECENT_TARGET','OLD_TARGET'),('OLD_TARGET','OLD_LOCAL'),('RECENT_TARGET','RECENT_LOCAL')]


def attack_probability(model,x):
    p=model.predict_proba(x);indices=np.flatnonzero(model.classes_==1)
    return p[:,int(indices[0])] if len(indices) else np.zeros(len(x),dtype=np.float64)


def gate(score,rule):
    h=(score>=.5).astype(np.int8);support=np.where(h==1,score,1-score);out=np.full(len(score),-1,dtype=np.int8)
    for cls,key in [(0,'B'),(1,'A')]:
        if rule[key]['threshold'] is not None:out[(h==cls)&(support>=rule[key]['threshold'])]=cls
    return out


def choose(score,y,method):
    h=score>=.5;support=np.where(h,score,1-score);nb=int((y==0).sum());na=int((y==1).sum());branches={};rows=[];best=None
    for cls,key in [(0,'B'),(1,'A')]:
        branches[key]=[]
        for i,t in enumerate(GRID+[None]):
            accepted=(h==cls)&(support>=t) if t is not None else np.zeros(len(y),bool)
            n=int(accepted.sum());err=int((accepted&(y!=cls)).sum());valid=t is None or n>=10 and 20*err<=n
            r=dict(id=f'T{i:03d}',threshold=t,accepted=n,errors=err,correct=n-err,risk=err/n if n else None,feasible=bool(valid))
            branches[key].append(r);rows.append(dict(method=method,record_type='branch',action=key,**r))
    missing=nb==0 or na==0
    for bi,b in enumerate(branches['B']):
        for ai,a in enumerate(branches['A']):
            feasible=b['feasible'] and a['feasible'] and (not missing or bi==11 and ai==11)
            ub=b['correct']/nb if nb else None;ua=a['correct']/na if na else None
            row=dict(method=method,record_type='pair',B_id=b['id'],A_id=a['id'],N_B=nb,N_A=na,TN=b['correct'],FN=b['errors'],FP=a['errors'],TP=a['correct'],U_B=ub,U_A=ua,U_bal=(ub+ua)/2 if not missing else None,feasible=bool(feasible));rows.append(row)
            if feasible:
                key=(-(b['correct']*na+a['correct']*nb),-min(b['correct']*na,a['correct']*nb),b['errors']+a['errors'],bi,ai)
                if best is None or key<best[0]:best=(key,b,a)
    _,b,a=best;on=sum(r['threshold'] is not None for r in [b,a])
    return dict(B=b,A=a,N_B=nb,N_A=na,status='CALIBRATION_CLASS_SUPPORT_MISSING' if missing else ['NO_FEASIBLE_POLICY','PARTIAL_ACTION_POLICY','TWO_ACTION_POLICY'][on],feasible_enabled_candidates={c:sum(v['threshold'] is not None and v['feasible'] for v in ls) for c,ls in branches.items()}),rows


def tests():
    r={'B':{'threshold':.6},'A':{'threshold':.5}}
    assert np.array_equal(gate(np.array([.4,.5,.49,1.]),r),[0,1,-1,1])
    off={'B':{'threshold':None},'A':{'threshold':None}};assert (gate(np.array([0.,1.]),off)==-1).all()
    y=np.r_[np.zeros(19,int),1,np.ones(20,int)];score=np.r_[np.full(20,.1),np.full(20,.9)];rule,_=choose(score,y,'test');assert rule['B']['accepted']==20 and rule['B']['errors']==1
    missing,_=choose(np.full(20,.1),np.zeros(20,int),'test');assert missing['status']=='CALIBRATION_CLASS_SUPPORT_MISSING'
    x=np.arange(20).reshape(-1,1)
    for cls in [0,1]:
        m=RandomForestClassifier(n_estimators=1,random_state=1).fit(x,np.full(20,cls));assert np.all(attack_probability(m,x)==cls)
    v=metrics(np.array([100,50,50,20,5,10,30,20,15,25,75,25,40,1,1]));assert v['R_B']==5/25 and v['R_A']==10/40
    ranks=np.array([13709999,13710000,13809999,13810000]);assert np.array_equal(ranks[(ranks>=13710000)&(ranks<13810000)],[13710000,13809999])
    return dict(half_open_interval=True,score_half_Attack=True,gate_inclusive_no_epsilon=True,OFF=True,integer_risk_boundary=True,single_class_RF_mapping=True,missing_C_class_OFF=True,action_risk_denominators=True)


def prepare(run):
    if (run/'calibration/selected_rules.json').exists():print('prepare complete; skipped');return
    vp=readj(OLD/'config/protocol_v1.json');adapter=Assets(ROOT,'ToN',vp)
    roles=pd.read_csv(OLD/'calibration/sample_roles.csv',float_precision='round_trip');roles=roles[roles.domain=='ToN'].sort_values('availability_rank')
    original=roles.availability_rank.to_numpy(dtype=np.int64);pilot=np.load(OLD/'pilot/ToN_evaluation_sample_ranks.npy')
    require(len(original)==400 and np.unique(original).size==400 and len(pilot)==274200 and np.all(np.diff(pilot)>0),'original index identity')
    windows=pd.read_csv(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv')
    w=windows[(windows.domain=='ToN')&(windows.scheme=='heldout')&(windows.window_size==10000)].sort_values('rank_start')
    require(np.array_equal(w.rank_start,np.arange(0,27520000,10000)) and np.array_equal(w.rank_end_exclusive,np.arange(10000,27520001,10000)),'availability coordinate/window manifest conflict')
    evaluation=pilot[(pilot>=BOUNDS[0])&(pilot<BOUNDS[-1])]
    require(len(evaluation)==137000,'common evaluation sample missing')
    require(np.bincount(np.searchsorted(BOUNDS[1:-1],evaluation,side='right'),minlength=4).tolist()==[34300,34300,34200,34200],'new four segment sample counts')
    n_jobs=min(8,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1)
    params=dict(n_estimators=200,criterion='gini',max_depth=5,min_samples_split=2,min_samples_leaf=5,max_features='sqrt',bootstrap=True,class_weight=None,random_state=20260929,n_jobs=n_jobs)
    protocol=dict(experiment='ton_recency_learner_pilot_v1',version=1,created_at=now(),root=str(ROOT),old_run=str(OLD),correction_run=str(COR),input_assets=adapter.info,original_protocol=str(OLD/'config/protocol_v1.json'),distance=vp['evidence'],feature_indices=sorted(adapter.cat+adapter.num),representation='frozen source fill, existing numeric representation, no target fit or standardization; scale used only in mixed distance',update_rank=13820000,recent_candidate_range=[13710000,13810000],arrival_gap=[13810000,13820000],evaluation_range=[13820000,27520000],segment_boundaries=BOUNDS,segment_windows=[343,343,342,342],expected_rows=137000,recent_sampling=dict(seed=20260930,generator='PCG64',size=400,exclude='all original400 and full original pilot274200',sort_candidate=True,sort_draw_before_split=True),recent_split=dict(seed=20261001,generator='PCG64',R=200,C=200,assignment='permutation of sorted400; first200 R'),gate_grid=GRID,gate='HIGH support for candidate label; fixed threshold or OFF; integer 20*errors<=accepted and accepted>=10',tie_order=['maximum U_bal','maximum min(UB,UA)','minimum FN+FP','ascending B threshold ID','ascending A threshold ID; OFF after1.00'],model_parameters=params,methods=MAIN,reference_methods=['SOURCE_RF_ALL','E1_FROZEN','ALWAYS_A','ALWAYS_B'],budget=dict(original_adaptation=400,new_adaptation=400,per_main_method=400,cumulative_acquired_adaptation=800,evaluation_existing_labels=137000,evaluation_labels_used_for_training=False),bootstrap=dict(seed=20261002,replicates=1000,block_windows=10,stratify='four predetermined subsegments',keep_short_final=True,shared_indices=True,percentiles=[2.5,97.5],minimum_valid_fraction=.9),interpretation='development comparison on common future range after one update; neither independent blind test nor equal cumulative-budget online deployment')
    if (run/'config/protocol.json').exists():
        saved=readj(run/'config/protocol.json');protocol['created_at']=saved['created_at'];require(protocol==saved,'resume protocol mismatch')
    else:writej(run/'config/protocol.json',protocol)
    if (run/'config/recent_sample_roles.csv').exists():
        recent=pd.read_csv(run/'config/recent_sample_roles.csv');chosen=recent.availability_rank.to_numpy(dtype=np.int64)
    else:
        candidates=np.arange(13710000,13810000,dtype=np.int64);candidates=candidates[~np.isin(candidates,np.r_[original,pilot])]
        require(len(candidates)>=400,'recent candidate shortage')
        chosen=np.sort(np.random.Generator(np.random.PCG64(20260930)).choice(candidates,400,replace=False))
        perm=np.random.Generator(np.random.PCG64(20261001)).permutation(400);role=np.full(400,'C');role[perm[:200]]='R'
        recent=pd.DataFrame(dict(availability_rank=chosen,role=role));recent.to_csv(run/'config/recent_sample_roles.csv',index=False)
        np.save(run/'config/recent_sample_ranks.npy',chosen)
    require(len(chosen)==400 and np.unique(chosen).size==400 and np.all(np.diff(chosen)>0) and np.all((chosen>=13710000)&(chosen<13810000)),'recent selected ranks')
    require(not np.isin(chosen,pilot).any() and not np.isin(chosen,original).any() and recent.role.value_counts().to_dict()=={'C':200,'R':200},'recent exclusions/roles')
    if not (run/'config/evaluation_ranks.npy').exists():np.save(run/'config/evaluation_ranks.npy',evaluation)
    else:require(np.array_equal(np.load(run/'config/evaluation_ranks.npy'),evaluation),'saved evaluation indices')
    if not (run/'config/index_freeze.json').exists():writej(run/'config/index_freeze.json',dict(time=now(),selected400=True,roles_saved_before_labels=True,candidate_count=99000,evaluation_rows=len(evaluation)))
    # First new-label access occurs only below, after protocol, indices, roles and time saved.
    if (run/'calibration/recent400.npz').exists():
        z=np.load(run/'calibration/recent400.npz');require(np.array_equal(z['rank'],chosen),'saved recent label index mismatch');recent_y=z['y']
    else:
        label_store=np.load(adapter.paths['labels'],mmap_mode='r');recent_y=np.array(label_store[chosen]);require(np.isin(recent_y,[0,1]).all(),'recent binary labels')
        np.savez(run/'calibration/recent400.npz',rank=chosen,y=recent_y)
        writej(run/'calibration/recent_label_access.json',dict(time=now(),selected_rows=400,index_source='config/recent_sample_roles.csv'))
    checks=tests();groups={};support=[];all_candidates=[];selected={};columns=protocol['feature_indices']
    for age,frame,ys in [('OLD',roles,roles.y.to_numpy(dtype=np.int8)),('RECENT',recent,recent_y)]:
        ranks=frame.availability_rank.to_numpy(dtype=np.int64);rsel=frame.role.to_numpy()=='R';csel=~rsel
        require(not np.isin(ranks[rsel],ranks[csel]).any(),'C in R')
        if age=='OLD':
            require(np.bincount(ys[rsel],minlength=2).tolist()==[177,23] and np.bincount(ys[csel],minlength=2).tolist()==[166,34],'OLD R/C counts differ')
            require(np.array_equal(adapter.orig[ranks],frame.original_row.to_numpy()),'original row alignment')
        rx=adapter.features(ranks[rsel]);cx=adapter.features(ranks[csel]);ry=ys[rsel];cy=ys[csel]
        path=run/f'calibration/{age}_reference.npz'
        if not path.exists():np.savez(path,rank=ranks[rsel],x=rx,y=ry)
        groups[age]=(rx,ry,cx,cy)
        # q is neighbor Attack proportion, independent of a source prediction.
        q,_,nn=evidence(cx,rx,ry,np.ones(len(cx),dtype=np.int8),adapter.cat,adapter.num,adapter.scale)
        require(nn.shape==(200,15) and np.all(np.diff(ranks[rsel])>0),'local neighbor order')
        if age=='OLD':
            old_c=np.load(OLD/'calibration/ToN_selection_evidence.npz');require(np.array_equal(old_c['rank'],ranks[csel]),'old C index order')
            expected=np.where(old_c['raw_p']>=.5,old_c['local_support'],1-old_c['local_support'])
            require(np.allclose(q,expected,atol=1e-15,rtol=0),'old local implementation differs')
        model_path=run/f'models/{age}_TARGET.joblib'
        if model_path.exists():model=joblib.load(model_path);require(all(model.get_params()[k]==v for k,v in params.items()),'model parameter mismatch')
        else:model=RandomForestClassifier(**params).fit(rx[:,columns],ry);joblib.dump(model,model_path)
        ps=attack_probability(model,cx[:,columns])
        for learner,score in [('LOCAL',q),('TARGET',ps)]:
            method=age+'_'+learner;rule,rows=choose(score,cy,method);selected[method]=rule;all_candidates.extend(rows)
            for role,truth,pr in [('R',ry,None),('C',cy,score>=.5)]:support.append(dict(method=method,role=role,N=len(truth),true_B=int((truth==0).sum()),true_A=int((truth==1).sum()),pred_B=int((~pr).sum()) if pr is not None else None,pred_A=int(pr.sum()) if pr is not None else None,prediction_note='not evaluated on training R' if pr is None else 'C candidate labels',training_classes=model.classes_.tolist() if learner=='TARGET' else None))
            pd.DataFrame(dict(rank=ranks[csel],y=cy,score=score,label=(score>=.5).astype(int),action=gate(score,rule))).to_csv(run/f'calibration/{method}_C.csv',index=False)
            print(method,'R',np.bincount(ry,minlength=2).tolist(),'C',np.bincount(cy,minlength=2).tolist(),rule,flush=True)
    csv(run/'calibration/support.csv',support);csv(run/'calibration/candidates.csv',all_candidates)
    e1=readj(COR/'calibration/selected_rules.json')['methods']['E1-local']
    writej(run/'config/checks.json',dict(synthetic=checks,OLD_counts_match=True,recent_disjoint_from_original400_and_pilot=True,C_disjoint_R=True,old_distance_equivalence=True,new_label_access_selected400_only=True))
    writej(run/'calibration/selected_rules.json',dict(freeze_time=now(),protocol=protocol,methods=selected,E1_FROZEN=e1))
    print('PREPARE_COMPLETE',now(),flush=True)


def all_actions(run,fr):
    # This phase must finish and persist all actions before labels are opened for scoring.
    if (run/'tables/actions_complete.json').exists():return
    protocol=fr['protocol'];vp=readj(OLD/'config/protocol_v1.json');a=Assets(ROOT,'ToN',vp);sample=np.load(run/'config/evaluation_ranks.npy');cols=protocol['feature_indices']
    r=np.load(run/'calibration/RECENT_reference.npz');models={age:joblib.load(run/f'models/{age}_TARGET.joblib') for age in ['OLD','RECENT']}
    rs=[];ps=[];qs=[]
    for path in sorted((OLD/'pilot/ToN').glob('chunk_*.npz')):
        with np.load(path) as z:
            rank=z['rank'];m=(rank>=BOUNDS[0])&(rank<BOUNDS[-1])
            if not m.any():continue
            p=z['raw_p'][m];pred=z['raw_pred'][m];ls=z['local_support'][m]
            rs.append(rank[m]);ps.append(p);qs.append(np.where(pred==1,ls,1-ls))
    rank=np.concatenate(rs);p=np.concatenate(ps);oldq=np.concatenate(qs);require(np.array_equal(rank,sample),'original evidence/sample alignment')
    scores={m:np.empty(len(rank),dtype=np.float64) for m in MAIN};scores['OLD_LOCAL'][:]=oldq;t0=time.perf_counter()
    for start in range(0,len(rank),5000):
        end=min(start+5000,len(rank));x=a.features(rank[start:end])
        q,_,_=evidence(x,r['x'],r['y'],np.ones(end-start,dtype=np.int8),a.cat,a.num,a.scale);scores['RECENT_LOCAL'][start:end]=q
        for age in ['OLD','RECENT']:scores[age+'_TARGET'][start:end]=attack_probability(models[age],x[:,cols])
        print('actions',end,'/',len(rank),f'{time.perf_counter()-t0:.1f}s',flush=True)
    data=dict(rank=rank,segment=np.searchsorted(BOUNDS[1:-1],rank,side='right')+1,window=(rank-BOUNDS[0])//10000)
    for method,score in scores.items():
        require(np.isfinite(score).all() and ((score>=0)&(score<=1)).all(),'score range')
        data['score_'+method]=score;data['action_'+method+'_ALL']=(score>=.5).astype(np.int8);data['action_'+method+'_GATED']=gate(score,fr['methods'][method])
    raw=(p>=.5).astype(np.int8);data['score_SOURCE_RF_ALL']=p;data['action_SOURCE_RF_ALL']=raw
    h=(oldq>=.5).astype(np.int8);sup=np.where(h==1,oldq,1-oldq);s=np.maximum(p,1-p);e1=np.full(len(rank),-1,dtype=np.int8)
    for cls,key in [(0,'B'),(1,'A')]:e1[(h==cls)&mask(fr['E1_FROZEN'][key],s,sup)]=cls
    data.update(score_E1_FROZEN=oldq,action_E1_FROZEN=e1,score_ALWAYS_A=np.ones(len(rank)),action_ALWAYS_A=np.ones(len(rank),np.int8),score_ALWAYS_B=np.zeros(len(rank)),action_ALWAYS_B=np.zeros(len(rank),np.int8))
    # Wide NPZ: rank plus explicit method-named score/action columns; no truth column.
    temp=run/'tables/evaluation_actions.tmp.npz';np.savez(temp,**data);temp.replace(run/'tables/evaluation_actions.npz')
    writej(run/'tables/actions_complete.json',dict(time=now(),rows=len(rank),rule_freeze_time=fr['freeze_time'],all_actions_persisted_before_truth_access=True,format='wide NPZ; score_METHOD and action_METHOD_LAYER fields; no label column'))


def boot_metrics(c):
    with np.errstate(divide='ignore',invalid='ignore'):
        ub=c[:,3]/c[:,1];ua=c[:,6]/c[:,2]
        return dict(U_bal=(ub+ua)/2,U_B=ub,U_A=ua,R_B=c[:,4]/(c[:,3]+c[:,4]),R_A=c[:,5]/(c[:,5]+c[:,6]))

def ci(x):
    v=x[np.isfinite(x)];rate=len(v)/1000
    lo,hi=np.percentile(v,[2.5,97.5]) if rate>=.9 else (None,None)
    return dict(ci_low=lo,ci_high=hi,valid_fraction=rate,valid_replicates=len(v))


def evaluate(run):
    if (run/'reports/decision.json').exists():print('evaluate complete; skipped');return
    fr=readj(run/'calibration/selected_rules.json');require(fr['protocol']==readj(run/'config/protocol.json'),'frozen protocol mismatch')
    all_actions(run,fr)
    a=np.load(run/'tables/evaluation_actions.npz');sample=np.load(run/'config/evaluation_ranks.npy');require(np.array_equal(a['rank'],sample),'action rank alignment')
    # Labels loaded only after ALL methods' actions persisted; no fitting below.
    yparts=[];rparts=[]
    for path in sorted((OLD/'pilot/ToN').glob('chunk_*.npz')):
        with np.load(path) as z:
            rank=z['rank'];m=(rank>=BOUNDS[0])&(rank<BOUNDS[-1])
            if m.any():rparts.append(rank[m]);yparts.append(z['y'][m])
    y=np.concatenate(yparts);require(np.array_equal(np.concatenate(rparts),sample) and len(y)==137000,'evaluation labels alignment')
    writej(run/'tables/evaluation_label_access.json',dict(time=now(),rows=len(y),source='existing original pilot evidence only',actions_complete_time=readj(run/'tables/actions_complete.json')['time']))
    nw=1370;seg=a['segment'];segw=seg.reshape(nw,100)[:,0];raw=a['action_SOURCE_RF_ALL'];counts_by={};overall={};segments=[];window_rows=[]
    for key in a.files:
        if not key.startswith('action_'):continue
        name=key[len('action_'):];c=window_counts(y,raw,a[key],nw);counts_by[name]=c
        require(np.all(c[:,3]+c[:,5]+c[:,7]==c[:,1]) and np.all(c[:,4]+c[:,6]+c[:,8]==c[:,2]),'action count conservation')
        family=next((m for m in MAIN if name.startswith(m+'_')),name);layer=name[len(family)+1:] if family!=name else 'REFERENCE'
        overall[name]=dict(method=name,family=family,layer=layer,sampled=True,**metrics(c.sum(axis=0)));overall[name]['coverage']=overall[name]['acceptance']
        segsum=[]
        for j in range(1,5):
            total=c[segw==j].sum(axis=0);segsum.append(total);v=dict(method=name,family=family,layer=layer,segment=j,sampled=True,**metrics(total));v['coverage']=v['acceptance'];segments.append(v)
        require(np.array_equal(np.sum(segsum,axis=0),c.sum(axis=0)),'segment sums')
        for i,row in enumerate(c):window_rows.append(dict(method=name,window=i,rank_start=BOUNDS[0]+i*10000,segment=int(segw[i]),**dict(zip(COUNT_KEYS,map(int,row)))))
    # Nonoverlapping 10-window blocks built separately within each predeclared subsegment.
    blocks=[];strata=[]
    for j in range(1,5):
        idx=np.flatnonzero(segw==j);ids=[]
        for k in range(0,len(idx),10):ids.append(len(blocks));blocks.append(idx[k:k+10])
        strata.append(np.array(ids,dtype=int))
    rng=np.random.Generator(np.random.PCG64(20261002));draw=np.array([np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strata]) for _ in range(1000)])
    weight=np.array([np.bincount(d,minlength=len(blocks)) for d in draw],dtype=np.int64);np.save(run/'tables/bootstrap_block_indices.npy',draw)
    csv(run/'tables/bootstrap_blocks.csv',[dict(block=i,segment=int(segw[b[0]]),first_window=int(b[0]),windows=len(b)) for i,b in enumerate(blocks)])
    boot={name:boot_metrics(weight@np.array([c[b].sum(axis=0) for b in blocks])) for name,c in counts_by.items()}
    paired=[]
    for layer in ['ALL','GATED']:
        for left,right in PAIRS:
            l=left+'_'+layer;r=right+'_'+layer
            for metric in ['U_bal','U_B','U_A','R_B','R_A']:
                lv=overall[l][metric];rv=overall[r][metric]
                paired.append(dict(layer=layer,comparison=left+'-'+right,metric=metric,delta=lv-rv if lv is not None and rv is not None else None,**ci(boot[l][metric]-boot[r][metric])))
    harm=[]
    for method in MAIN:
        h=a['action_'+method+'_ALL']
        for j in range(5):
            subset=np.ones(len(y),bool) if j==0 else seg==j
            for src,dst in [(0,1),(1,0)]:
                m=subset&(raw==src)&(h==dst);fix=int((m&(h==y)).sum());hurt=int((m&(raw==y)).sum());require(fix+hurt==m.sum(),'harm decomposition')
                harm.append(dict(method=method,layer='ALL',scope='overall' if j==0 else f'segment_{j}',direction='B_to_A' if src==0 else 'A_to_B',changed=int(m.sum()),wrong_to_right=fix,right_to_wrong=hurt,net_corrections=fix-hurt))
    target={name:v['target_status']=='JOINT_TARGET_MET' for name,v in overall.items()};stable={name:target[name] and sum(v['target_status']=='JOINT_TARGET_MET' for v in segments if v['method']==name)>=3 for name in overall}
    csv(run/'tables/classification_comparison.csv',[v for k,v in overall.items() if v['layer']=='ALL' or k in ['SOURCE_RF_ALL','ALWAYS_A','ALWAYS_B']]);csv(run/'tables/policy_comparison.csv',[v for k,v in overall.items() if v['layer']=='GATED' or v['layer']=='REFERENCE']);csv(run/'tables/segment_metrics.csv',segments);csv(run/'tables/window_counts.csv',window_rows);csv(run/'tables/correction_harm.csv',harm);csv(run/'tables/paired_intervals.csv',paired)
    deltas={f'{v["layer"]}:{v["comparison"]}':v for v in paired if v['metric']=='U_bal'}
    conclusions=[]
    rl=stable['RECENT_LOCAL_GATED'];rt=stable['RECENT_TARGET_GATED'];ol=stable['OLD_LOCAL_GATED'];ot=stable['OLD_TARGET_GATED']
    dlocal=deltas['GATED:RECENT_LOCAL-OLD_LOCAL']
    if rl and dlocal['delta']>0:conclusions.append('Recently, LOCAL has reached a stable two-class operating point and U_bal is higher than OLD_LOCAL; Support for further research and updating of reference libraries. Difference ranges are reported separately from operating point.')
    if rt and not rl:conclusions.append('Recent TARGET reaches stable operating point and recent LOCAL is not reached; Supporting further study of the target domain model adaptation.')
    if ot and rt and not ol and not rl:conclusions.append('Both groups of TARGET achieved stable operating point and LOCAL did not; The choice of a complete learner may be more important than the current labels timing.')
    for name in MAIN:
        ss=[v for v in segments if v['method']==name+'_GATED']
        if ss[0]['target_status']=='JOINT_TARGET_MET' and ss[-1]['target_status']!='JOINT_TARGET_MET':conclusions.append(name+'Only near end operating point effective, end failure; A setup is not maintained until the end, and the frequency remains as a follow-up issue.')
        net=sum(v['net_corrections'] for v in harm if v['method']==name and v['scope']=='overall')
        if net>0 and not target[name+'_GATED']:conclusions.append(name+f' Acceptance rating net label correction {net} In addition to the above, the door controls do not establish the overall two-class operating point.')
    if not target['RECENT_LOCAL_GATED'] and not target['RECENT_TARGET_GATED']:conclusions.append('Neither of the recent approaches have established an overall two-class operating point; It is necessary to combine R/C with the motion support quantity to distinguish the limited support and the measurement method has not been restored, and it is not possible to infer that features is without information or proof of concept drift.')
    decision=dict(status='PILOT_COMPLETED',completed_time=now(),evaluation_rows=137000,new_labels=400,budget=fr['protocol']['budget'],overall=overall,stable_workpoint=stable,segment_targets={name:{str(v['segment']):v['target_status'] for v in segments if v['method']==name} for name in overall},calibration_status={k:v['status'] for k,v in fr['methods'].items()},utility_deltas=deltas,evidence_based_conclusions=conclusions,method_failures={k:v['failure_metrics'] for k,v in overall.items()},next_action='Stop this pilot; Follow-up of alternate decisions based on the given operating point, four segments, support quantity and mating intervals, without automatic addition of labels or change of configuration.',interpretation='development, one update, common future interval; no independent blind test or risk guarantee')
    render(run,fr,decision,segments,harm,paired)
    checks=readj(run/'config/checks.json');checks.update(actual_rows=137000,segment_rows=np.bincount(seg)[1:].tolist(),window_count=1370,all_actions_saved_before_labels=True,counts_additive=True,bootstrap_stratum_blocks=[len(s) for s in strata],bootstrap_last_block_windows=[len(blocks[s[-1]]) for s in strata],shared_replicates=1000);writej(run/'config/checks.json',checks)
    writej(run/'reports/decision.json',decision)
    print(pd.DataFrame(overall.values())[['method','R_B','R_A','U_B','U_A','U_bal','target_status']].to_string(index=False),flush=True)
    for line in conclusions:print(line,flush=True)


def render(run,fr,d,segments,harm,paired):
    pct=lambda x:'NA' if x is None else f'{100*x:.4f}%'
    lines=['# ToN labels timing × learning machine: controlled pilot','',f" Complete an evaluation of a common future sample under 137,000, range [13820000,27520000). Addition of labels to 400; The original 400 line, each main method using the 400 line, collectively obtained the appropriate labels 800 line. The evaluator also uses an existing 137,000 clause pilot true value, which is not used to train / selection.",'', 'updating the T=13820000; Recent queries range [13710000,13810000), [13810000,13820000) to simulate labels reaching intervals. The candidate shall first exclude the original 400 and all the old pilot indexes; Protocols, indexes, and roles are saved to read the selected 400 true value. The new four windows are 343/343/342/342 and the sample number is 34300/34300/34200/34200.','', 'This is a single update on repeatedly developing data, a common-area comparison, not a comparison with cumulative budget online deployments from start-up, and not a independent blind test.','', '## Role Support and calibration Rules','', 'The two small RF models are trained from scratch on their respective R200 only; the source RF is unchanged. 48 features sequence, source filling and fixed prime values; LOCAL maintains the original mixed distance, source scale, k15 and rank ties rules. TARGET and LOCAL are complete learner comparisons, and the difference cannot be attributed to a certain distance.','']
    support=pd.read_csv(run/'calibration/support.csv');lines+=['| method | role | true B| true A| candidate B| candidate A|','|---|---|---:|---:|---:|---:|']
    for _,v in support.iterrows():lines.append(f"|{v.method}|{v.role}|{v.true_B}|{v.true_A}|{'NA' if pd.isna(v.pred_B) else int(v.pred_B)}|{'NA' if pd.isna(v.pred_A) else int(v.pred_A)}|")
    lines+=['','R does not operate as a training cluster self-prediction, and the prediction class supports NA; C is used for selection and cannot be validated as independent. The four main methods are unified using HIGH gate controls that support the c≥ fixed threshold, 0.50 to 1.00 grid, 0.05 and OFF steps. The OLD_LOCAL classification is the same as the old local majority vote, but the gate control has been changed and cannot be called the original E1.','', '| method | calibration state |B threshold / accepts / error |A threshold / accepts / error |','|---|---|---|---|']
    for m,r in fr['methods'].items():lines.append(f"|{m}|{r['status']}|{r['B']['threshold']}/{r['B']['accepted']}/{r['B']['errors']}|{r['A']['threshold']}/{r['A']['accepted']}/{r['A']['errors']}|")
    lines+=['','## total quadratic with reference','','ALL and GATED respectively; OFF risk is NA and abstention attacks are not detected.','','| method / level |RB|RA|UB|UA|U_bal| coverage | accept B/A|FN/FP| overall operating point | stable operating point |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for name,v in d['overall'].items():lines.append(f"|{name}|{pct(v['R_B'])}|{pct(v['R_A'])}|{pct(v['U_B'])}|{pct(v['U_A'])}|{pct(v['U_bal'])}|{pct(v['coverage'])}|{v['accepted_B']}/{v['accepted_A']}|{v['FN']}/{v['FP']}|{v['target_status']}|{d['stable_workpoint'][name]}|")
    lines+=['','Stabilize the overall operating point = and at least the 3/4 sub-section; It’s not a risk guarantee. Complete real-valued divisors, abstention numbers and movement window numbers see CSV results.','', '## New four sections','','| | Method |RB|RA|UB|UA| Accepted B/A| operating point | Failed |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for v in segments:lines.append(f"|{v['segment']}|{v['method']}|{pct(v['R_B'])}|{pct(v['R_A'])}|{pct(v['U_B'])}|{pct(v['U_A'])}|{v['accepted_B']}/{v['accepted_A']}|{v['target_status']}|{v['failure_metrics'] or 'none'}|")
    lines+=['','## fully accepted label correction classification with additional errors','','the pH of the sample is proportional to SOURCE_RF_ALL and covers the entire common area; Don’t put the abstention on the door when it’s label correction. Four points and directions: see correction_harm.csv in detail.','','| method | label correction | Added error | Net label correction |','|---|---:|---:|---:|']
    for m in MAIN:
        hs=[v for v in harm if v['method']==m and v['scope']=='overall'];fix=sum(v['wrong_to_right'] for v in hs);hurt=sum(v['right_to_wrong'] for v in hs);lines.append(f'|{m}|{fix}|{hurt}|{fix-hurt}|')
    lines+=['','## Four predicted pair comparisons','','Difference = left-right; The risk negative value and the utility positive value respectively represent the corresponding dimensional improvement. The number is a percentage point. 10 window blocks separated into four segments, the end blocks being retained; The same copy of all methods is shared, with each section having the same number of blocks drawn back as the original blocks, combined to recalculate the ratios. 1000 replicated, PCG64(20261002).','','| level | compared to | indicator | difference |95% range | effective rate |','|---|---|---|---:|---|---:|']
    for v in paired:lines.append(f"|{v['layer']}|{v['comparison']}|{v['metric']}|{pct(v['delta'])}|[{pct(v['ci_low'])}, {pct(v['ci_high'])}]|{v['valid_fraction']:.3f}|")
    lines+=['','The exploratory dot-state range, without multiple comparison corrections, is not guaranteed to drift arbitrarily; Zero-sequence duplication is ineffective, effective ratio <90% range NA. The operating point threshold and the difference in efficacy are clearly read separately and cannot be inferred from both methods without the failure of time-efficiency.','', '## evidence and next steps','']+['- '+x for x in d['evidence_based_conclusions']]
    for layer in ['ALL','GATED']:
        for left,right in PAIRS:
            v=d['utility_deltas'][layer+':'+left+'-'+right];lines.append(f"- {layer} {left}−{right}：ΔU_bal={pct(v['delta'])} The [ range {pct(v['ci_low'])}, {pct(v['ci_high'])}]。")
    lines+=['',d['next_action'],'','A 400 line sampling, a single update point cannot represent the other budget / time points; No identification of the type of drift, no declaration of no features information. labels latency is a window simulation.','', '## Delivery and Actual Order','']
    for f in ['config/protocol.json','config/recent_sample_roles.csv','config/checks.json','calibration/support.csv','calibration/candidates.csv','calibration/selected_rules.json','models/OLD_TARGET.joblib','models/RECENT_TARGET.joblib','tables/classification_comparison.csv','tables/policy_comparison.csv','tables/segment_metrics.csv','tables/correction_harm.csv','tables/paired_intervals.csv','tables/evaluation_actions.npz','reports/decision.json','logs/commands.sh','src/run_pilot.py']:lines.append(f'- [{f}]({run/f})')
    lines+=['','The action file uses both a readable NPZ format and a widely used method with ALL/GATED and rank and score, with no real value written into the action file.','', '```bash',(run/'logs/commands.sh').read_text().strip(),'```','', 'After cost and pilot, stop; The old run read-only does not automatically run more labels acquisition, BoT, full quantity, feedback, or demonstration training.']
    (run/'reports/RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','evaluate'],required=True);ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args();run=args.run_dir.resolve()
    require(run.is_relative_to(ROOT/'outputs/post_p7_diagnostics/ton_recency_learner_pilot_v1'),'run outside experiment')
    with (run/'logs/commands.sh').open('a') as f:f.write(shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+'\n')
    writej(run/'config/runtime.json',dict(interpreter=sys.executable,python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__))
    print(now(),args.stage,flush=True)
    try:globals()[args.stage](run)
    except Exception as e:
        with (run/'logs/implementation_events.log').open('a') as f:f.write(now()+' '+type(e).__name__+': '+str(e)+'; evaluation_labels_opened='+str((run/'tables/evaluation_label_access.json').exists())+'\n')
        raise
if __name__=='__main__':main()
