#!/usr/bin/env python
"""ToN action correction repeated pilot. prepare/evaluate/report only."""
import argparse,sys,shlex,time
from pathlib import Path
import numpy as np
import pandas as pd
from fixed_metrics import readj,writej,csv,now,require,metrics,window_counts,COUNT_KEYS
from baselines import q1d,gate_action,tail_action,select_tail,select_local,select_mixed_gate,selftest,counts
METHODS=['RAW','OLD_G2','E1_FIXED','LOCAL_ALL','LOCAL_GATE','P_TAIL_C','P_TAIL_400','P1D_GATE']
COMPARATORS=['P_TAIL_C','P_TAIL_400','P1D_GATE','LOCAL_GATE']
OLD_REL='outputs/post_p7_diagnostics/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr'
CAL_REL='outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa'
SEGMENTS=[100000,6960000,13820000,20670000,27520000]


def prepare(root,run):
    if (run/'calibration/frozen_baselines.json').exists():print('prepare already frozen; skipped');return
    old=root/OLD_REL;cal=root/CAL_REL
    vp=readj(old/'config/protocol_v1.json');cp=readj(cal/'config/protocol.json')
    prev=readj(old/'calibration/frozen_rules.json');e1=readj(cal/'calibration/selected_rules.json')['methods']['E1-local'];oldg2=prev['domains']['ToN']['rules']['G2'];grid=oldg2['grid']
    require(cp['grid']==grid and len(grid)==12,'original grid conflict')
    protocol=dict(experiment='ton_action_correction_pilot_v1',version=1,registered_at=now(),domain='ToN',methods=METHODS,E1_source=str(cal/'calibration/selected_rules.json'),old_run=str(old),calibration_run=str(cal),source_inputs=[str(old/'calibration/sample_roles.csv'),str(old/'calibration/ToN_selection_evidence.npz'),str(cal/'calibration/C_sample_labels.csv'),str(old/'calibration/frozen_rules.json')],R=200,C=200,label_budget=400,RF='frozen rf_All49 raw_probability',distance=vp['evidence'],k=15,original_grid=grid,E1_reselection=False,baselines=dict(LOCAL_GATE='h_L; local support>=u or OFF; no RF s; u=0,.5,.75,1',P_TAIL_C='C only; NORMAL p or REVERSE 1-p; all unique r plus -1,2; b<a; inclusive tails',P_TAIL_400='same family selected on all historical400',P1D_GATE='R200 abs(raw_p-p_R), k15 uniform; ties ascending reference rank; C selects original G2 gate family'),gate_risk_max=.05,gate_minimum_accepted=10,gate_selection_tie_order=vp['gates']['tie_break'],tail_tie_order=['maximum U_bal','maximum min(U_B,U_A)','minimum FN+FP','NORMAL before REVERSE','ascending b index','ascending a index'],local_tie_order=['maximum U_bal','maximum min(U_B,U_A)','minimum FN+FP','minimum u_B','minimum u_A','stable ID'],evaluation=dict(expected_rows=274200,segment_boundaries=SEGMENTS,segment_rows=[68600,68600,68500,68500],window_rows=100,window_rank_width=10000,practical_target=vp['evaluation'],probability_comparators=COMPARATORS[:3],rf_gate_comparator='LOCAL_GATE'),bootstrap=vp['bootstrap'],E1_risk_intervals='same whole-series block multiplicities; mask counts outside each segment to zero; ratios recomputed; no segment re-seeding',research_status='repeated development evaluation: C repeatedly reused and previous pilot labels already viewed',forbidden=['new sampling','new labels','RF training','BoT','full evaluation','feedback','other models','changing E1 or k/distance'])
    writej(run/'config/protocol.json',protocol)
    tests=selftest()
    roles=pd.read_csv(old/'calibration/sample_roles.csv',float_precision='round_trip');roles=roles[roles.domain=='ToN'].sort_values('availability_rank')
    r=roles[roles.role=='R'];c=roles[roles.role=='C']
    require(len(r)==len(c)==200 and roles.availability_rank.nunique()==400 and roles.availability_rank.between(0,99999).all(),'R/C rank/size')
    z=np.load(old/'calibration/ToN_selection_evidence.npz');crank=z['rank'];cy=z['y'];p=z['raw_p'];support=z['local_support']
    require(np.array_equal(crank,c.availability_rank) and np.array_equal(cy,c.y) and np.array_equal(p,c.raw_p),'C evidence alignment')
    q=np.where(p>=.5,support,1-support);h=(q>=.5).astype(np.int8);s=np.maximum(p,1-p)
    rq=r.raw_p.to_numpy(dtype=float);ry=r.y.to_numpy(dtype=np.int8);rr=r.availability_rank.to_numpy(dtype=np.int64)
    require(np.array_equal(np.bincount(ry,minlength=2),[177,23]) and np.array_equal(np.bincount(cy,minlength=2),[166,34]),'historical split counts')
    np.savez(run/'calibration/reference_probabilities.npz',rank=rr,p=rq,y=ry)
    roles.to_csv(run/'calibration/sample_roles.csv',index=False)
    e1action=gate_action(h,s,q,e1);m=counts(cy,e1action)
    require([m[k] for k in ['accepted_B','FN','accepted_A','FP']]==[52,2,13,0],'E1 calibration counts')
    saved=pd.read_csv(cal/'calibration/C_sample_labels.csv',float_precision='round_trip')
    require(np.array_equal(saved.availability_rank,crank) and np.array_equal(saved['action_E1-local'],e1action),'E1 saved actions mismatch')
    require(e1['B']['threshold']==.8577222228050232 and e1['B']['u']==1 and e1['B']['direction']=='LOW' and e1['A']['threshold']==.5 and e1['A']['u']==0 and e1['A']['direction']=='HIGH','E1 fixed definition conflict')
    rules={};candidate_rows=[]
    rules['LOCAL_GATE'],rows=select_local(h,cy,q);candidate_rows.extend(rows)
    for method,pp,yy in [('P_TAIL_C',p,cy),('P_TAIL_400',roles.raw_p.to_numpy(),roles.y.to_numpy())]:
        rules[method],rows=select_tail(pp,yy,method);candidate_rows.extend(rows)
    qp,nn=q1d(p,rq,ry);hp=(qp>=.5).astype(np.int8)
    rules['P1D_GATE'],rows,p1action=select_mixed_gate('P1D_GATE',hp,cy,s,qp,grid);candidate_rows.extend(rows)
    require(nn.shape==(200,15) and np.all(np.diff(rr)>0),'P1D reference order')
    # Scalar implementation check on C only.
    scalar=np.array([np.lexsort((rr,np.abs(float(v)-rq)))[:15] for v in p[:8]])
    require(np.array_equal(nn[:8],scalar),'P1D vector/scalar mismatch')
    csv(run/'calibration/baseline_candidates.csv',candidate_rows)
    csv(run/'calibration/counts.csv',[dict(method='E1_FIXED',**m)]+[dict(method=k,**v['metrics']) for k,v in rules.items()])
    pd.DataFrame(dict(rank=crank,y=cy,raw_p=p,q=q,q_1D=qp,action_E1=e1action,action_P1D=p1action)).to_csv(run/'calibration/C_evidence.csv',index=False)
    writej(run/'checks/basic_checks.json',dict(synthetic=tests,R_C_alignment=True,R_counts=[177,23],C_counts=[166,34],E1_unchanged_actions_and_counts=True,P1D_scalar_stable_equivalence=True,selection_used_only_declared_calibration=True))
    writej(run/'calibration/frozen_baselines.json',dict(freeze_time=now(),protocol=protocol,E1_FIXED=e1,OLD_G2=oldg2,baselines=rules,calibration_metrics_E1=m,stage='PREPARED_FROZEN'))
    print('Frozen',now(),flush=True)
    for k,v in rules.items():print(k,v['status'],v.get('direction'),v.get('b'),v.get('a'),v['metrics'],flush=True)


def evaluate(root,run):
    fr=readj(run/'calibration/frozen_baselines.json');require(fr['protocol']==readj(run/'config/protocol.json'),'protocol freeze conflict')
    if (run/'evaluation/completed.json').exists():print('evaluation already completed; skipped');return
    old=root/OLD_REL;sample=np.load(old/'pilot/ToN_evaluation_sample_ranks.npy')
    require(len(sample)==274200 and np.all(np.diff(sample)>0),'original sample count/order')
    require(np.array_equal((sample.reshape(-1,100)-100000)//10000,np.repeat(np.arange(2742)[:,None],100,axis=1)),'original windows not 100 each')
    r=np.load(run/'calibration/reference_probabilities.npz');cursor=0;parts=0;t0=time.perf_counter()
    files=sorted((old/'pilot/ToN').glob('chunk_*.npz'));require(len(files)>0,'old evidence missing')
    for path in files:
        out=run/f'evaluation/chunk_{parts:04d}.npz'
        # Access only nonlabel fields until all actions for this block are fixed.
        with np.load(path) as z:
            rank=z['rank'];p=z['raw_p'];pred=z['raw_pred'];ls=z['local_support']
            require(np.array_equal(sample[cursor:cursor+len(rank)],rank),'old evidence/sample rank alignment')
            require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.array_equal(pred,p>=.5),'raw probability semantics')
            q=np.where(pred==1,ls,1-ls);require(np.allclose(q*15,np.rint(q*15),rtol=0,atol=1e-12),'q not 15-neighbor proportions')
            h=(q>=.5).astype(np.int8);s=np.maximum(p,1-p);qp,nn=q1d(p,r['p'],r['y']);hp=(qp>=.5).astype(np.int8)
            actions=dict(RAW=pred.astype(np.int8),OLD_G2=gate_action(pred,s,q,fr['OLD_G2']),E1_FIXED=gate_action(h,s,q,fr['E1_FIXED']),LOCAL_ALL=h,LOCAL_GATE=gate_action(h,np.full(len(h),.5),q,fr['baselines']['LOCAL_GATE']),P_TAIL_C=tail_action(p,fr['baselines']['P_TAIL_C']),P_TAIL_400=tail_action(p,fr['baselines']['P_TAIL_400']),P1D_GATE=gate_action(hp,s,qp,fr['baselines']['P1D_GATE']))
            require(np.array_equal(actions['OLD_G2'],z['action_G2']),'OLD_G2 differs from saved actions')
            require(np.array_equal(actions['E1_FIXED']==1,h==1),'E1 all local Attack accepted')
            # Rule freeze exists; selection code no longer runs. Labels used for scoring only.
            y=z['y'];require(np.isin(y,[0,1]).all(),'binary label field')
            segment=np.searchsorted(SEGMENTS[1:-1],rank,side='right')+1;window=(rank-100000)//10000
            if out.exists():
                with np.load(out) as oldpart:require(np.array_equal(oldpart['rank'],rank),'partial output alignment')
            else:
                temp=out.with_suffix('.tmp.npz');np.savez(temp,rank=rank,segment=segment,window=window,y=y,raw_p=p,raw_pred=pred,q=q,q_1D=qp,**{'action_'+k:v for k,v in actions.items()});temp.replace(out)
            cursor+=len(rank);parts+=1
        print(f'evaluate {cursor}/274200 {time.perf_counter()-t0:.2f}s',flush=True)
    require(cursor==274200,'total evidence rows')
    writej(run/'evaluation/completed.json',dict(stage='EVALUATED',rows=cursor,chunks=parts,freeze_time=fr['freeze_time'],completed_time=now(),q_recovered_from_existing_support=True,neighborhood_48D_recomputed=False,P1D_k=15))
    check=readj(run/'checks/basic_checks.json');check.update(pilot_ranks_match_original=True,pilot_rows=cursor,window_count=2742,rows_per_window=100,old_G2_actions_match=True,all_E1_local_Attack_accepted=True,evaluation_labels_used_after_new_rule_freeze=True);writej(run/'checks/basic_checks.json',check)


def bootstrap_metrics(c):
    with np.errstate(divide='ignore',invalid='ignore'):
        ub=c[:,3]/c[:,1];ua=c[:,6]/c[:,2]
        return dict(R_B=c[:,4]/(c[:,3]+c[:,4]),R_A=c[:,5]/(c[:,6]+c[:,5]),U_B=ub,U_A=ua,U_bal=(ub+ua)/2)


def interval(x):
    valid=np.asarray(x)[np.isfinite(x)];fraction=len(valid)/1000
    lo,hi=np.percentile(valid,[2.5,97.5]) if fraction>=.9 else (None,None)
    return dict(ci_low=lo,ci_high=hi,valid_replicates=len(valid),valid_fraction=fraction)


def report(root,run):
    if (run/'reports/decision.json').exists():print('report already complete; skipped');return
    fr=readj(run/'calibration/frozen_baselines.json');require(fr['protocol']==readj(run/'config/protocol.json'),'protocol mismatch')
    complete=readj(run/'evaluation/completed.json');require(complete['rows']==274200,'evaluation not complete')
    parts=[]
    for p in sorted((run/'evaluation').glob('chunk_*.npz')):
        with np.load(p) as z:parts.append({k:z[k] for k in z.files})
    e={k:np.concatenate([z[k] for z in parts]) for k in parts[0]};del parts
    require(len(e['y'])==274200 and np.array_equal(e['rank'],np.load(root/OLD_REL/'pilot/ToN_evaluation_sample_ranks.npy')),'report row alignment')
    require(np.bincount(e['segment'])[1:].tolist()==[68600,68600,68500,68500],'four segment counts')
    nw=2742;nb=(nw+9)//10;rng=np.random.Generator(np.random.PCG64(20261001));draw=rng.integers(0,nb,size=(1000,nb));weights=np.array([np.bincount(x,minlength=nb) for x in draw],dtype=np.int64)
    np.save(run/'tables/bootstrap_block_indices.npy',draw)
    segw=e['segment'].reshape(nw,100)[:,0];overall={};segment_rows=[];window_rows=[];boot={};risk_rows=[];wc={}
    for method in METHODS:
        c=window_counts(e['y'],e['raw_pred'],e['action_'+method],nw);wc[method]=c
        require(np.all(c[:,1]+c[:,2]==c[:,0]) and np.all(c[:,3]+c[:,5]+c[:,7]==c[:,1]) and np.all(c[:,4]+c[:,6]+c[:,8]==c[:,2]),'count conservation')
        overall[method]=dict(method=method,domain='ToN',sampled=True,**metrics(c.sum(axis=0)))
        segsum=[]
        for seg in range(1,5):
            total=c[segw==seg].sum(axis=0);segsum.append(total);segment_rows.append(dict(method=method,domain='ToN',segment=seg,sampled=True,**metrics(total)))
        require(np.array_equal(np.sum(segsum,axis=0),c.sum(axis=0)),'four segment sums')
        for w,row in enumerate(c):window_rows.append(dict(method=method,domain='ToN',window=w,rank_start=100000+w*10000,segment=int(segw[w]),sampled=True,**metrics(row)))
        blocks=np.add.reduceat(c,np.arange(0,nw,10),axis=0);boot[method]=bootstrap_metrics(weights@blocks)
        if method=='E1_FIXED':
            for seg in range(5):
                masked=c if seg==0 else c*(segw==seg)[:,None]
                b=np.add.reduceat(masked,np.arange(0,nw,10),axis=0);ratios=bootstrap_metrics(weights@b);m=metrics(masked.sum(axis=0))
                for metric in ['R_B','R_A']:risk_rows.append(dict(method=method,scope='overall' if seg==0 else f'segment_{seg}',metric=metric,estimate=m[metric],accepted=m['accepted_B' if metric=='R_B' else 'accepted_A'],errors=m['FN' if metric=='R_B' else 'FP'],**interval(ratios[metric])))
    paired=[]
    for comp in COMPARATORS:
        for metric in ['R_B','R_A','U_B','U_A','U_bal']:
            x=overall['E1_FIXED'][metric];y=overall[comp][metric]
            paired.append(dict(comparison='E1_FIXED-'+comp,metric=metric,delta=x-y if x is not None and y is not None else None,**interval(boot['E1_FIXED'][metric]-boot[comp][metric])))
    harms=[];groups=[];local=e['action_LOCAL_ALL'];raw=e['raw_pred'];truth=e['y'];qn=np.rint(e['q']*15).astype(int)
    for seg in range(5):
        subset=np.ones(len(truth),bool) if seg==0 else e['segment']==seg;scope='overall' if seg==0 else f'segment_{seg}'
        for src,dst,name in [(0,1,'B_to_A'),(1,0,'A_to_B')]:
            changed=subset&(raw==src)&(local==dst);fixed=int((changed&(raw!=truth)&(local==truth)).sum());hurt=int((changed&(raw==truth)&(local!=truth)).sum())
            require(fixed+hurt==changed.sum(),'correction partition')
            harms.append(dict(scope=scope,comparison='LOCAL_ALL-RAW',direction=name,N=int(subset.sum()),changed=int(changed.sum()),wrong_to_right=fixed,right_to_wrong=hurt,net_corrections=fixed-hurt))
        for name,lo,hi in [('weak_8_to_11',8,11),('strong_12_to_15',12,15)]:
            m=subset&(local==1)&(qn>=lo)&(qn<=hi);n=int(m.sum());err=int((m&(truth==0)).sum())
            groups.append(dict(scope=scope,group=name,n=n,errors=err,correct=n-err,error_rate=err/n if n else None,windows=len(np.unique(e['window'][m]))))
    require(sum(g['n'] for g in groups if g['scope']=='overall')==overall['E1_FIXED']['accepted_A'],'attack groups count')
    classification_net=sum(h['net_corrections'] for h in harms if h['scope']=='overall')
    require(classification_net==overall['RAW']['FN']+overall['RAW']['FP']-overall['LOCAL_ALL']['FN']-overall['LOCAL_ALL']['FP'],'classification net errors')
    for filename,rows in [('comparison',list(overall.values())),('segment_metrics',segment_rows),('window_counts',window_rows),('correction_harm',harms),('attack_support_groups',groups),('paired_intervals',paired),('E1_risk_intervals',risk_rows)]:csv(run/f'tables/{filename}.csv',rows)
    target={m:v['target_status']=='JOINT_TARGET_MET' for m,v in overall.items()};passing=[m for m in COMPARATORS[:3] if target[m]]
    utility={c:next(v for v in paired if v['comparison']=='E1_FIXED-'+c and v['metric']=='U_bal') for c in COMPARATORS}
    def gain(c):return utility[c]['delta'] is not None and utility[c]['delta']>=.01 and utility[c]['ci_low'] is not None and utility[c]['ci_low']>0
    if not target['E1_FIXED']:
        primary='PROBABILITY_CORRECTION_PREFERRED' if passing else 'CALIBRATION_FEASIBILITY_DID_NOT_TRANSFER'
        local_status='NOT_ESTABLISHED_AT_PILOT';rf_status='NOT_ESTABLISHED_AT_PILOT'
    else:
        primary='LOCAL_CORRECTION_CANDIDATE' if not passing else ('LOCAL_INCREMENT_ESTABLISHED_AT_PILOT' if all(gain(c) for c in passing) else 'NO_ESTABLISHED_LOCAL_INCREMENT')
        local_status=primary;rf_status='NO_ESTABLISHED_RF_GATE_GAIN' if target['LOCAL_GATE'] and not gain('LOCAL_GATE') else 'RF_SELECTION_CANDIDATE'
    temporal={m:('TEMPORALLY_SUPPORTED_PILOT' if sum(x['target_status']=='JOINT_TARGET_MET' for x in segment_rows if x['method']==m)>=3 else 'TEMPORALLY_FRAGILE') if target[m] else None for m in METHODS}
    limitations=['C has been used multiple times in development; The original pilot labels and the results have been examined, and this round is a re-evaluation rather than a new blind test.','E1 is the target labels reference library-supported local subclass and selection; RF offers only the option features, not the pure RF reliability gate control or CP.','The correct A on fixed mixed distance, R200 and 13 clauses C cannot guarantee the transfer or future risk.','Probability controls are infinite for all probability functions; Unreached growth thresholds are not equivalent to statistical equivalence.','The time block space describes only the current sequence, and does not cover calibration sampling, future drift or statistical certification; The maximum risk range is not new through the threshold.','The error number is sampled; abstention attacks are not included in TP.']
    decision=dict(protocol_version=1,stage='ToN_repeated_pilot',status='PILOT_COMPLETED',completed_time=now(),actual_rows=274200,calibration_feasibility='E1_FIXED_TWO_ACTION_POLICY',calibration_E1=fr['calibration_metrics_E1'],new_baseline_calibration={m:v['status'] for m,v in fr['baselines'].items()},transfer_status='JOINT_TARGET_MET' if target['E1_FIXED'] else 'CALIBRATION_FEASIBILITY_DID_NOT_TRANSFER',primary_decision=primary,local_information_increment=local_status,RF_selection_increment=rf_status,joint_target_by_method={m:v['target_status'] for m,v in overall.items()},passing_probability_comparators=passing,segment_target_by_method={m:{str(v['segment']):v['target_status'] for v in segment_rows if v['method']==m} for m in METHODS},temporal_modifier=temporal,E1_failure_metrics=overall['E1_FIXED']['failure_metrics'],E1_segment_failures=[dict(segment=v['segment'],status=v['target_status'],failure_metrics=v['failure_metrics']) for v in segment_rows if v['method']=='E1_FIXED'],classification_net_corrections=classification_net,classification_correction_transferred=classification_net>0,paired_utility_deltas=utility,overall=overall,E1_risk_intervals=risk_rows,recommended_next_action='Stop the full E1 expansion; If there are standard probability rules, it is preferred to discuss simple probability adaptations. For any subsequent experiment, specific instructions are required.' if not target['E1_FIXED'] else 'It is only recommended that the same rules be reviewed for further verification; Combine probability control with LOCAL_GATE to decide which components to retain. This time it’s not running at full capacity.',not_executed=['BoT','full evaluation','feedback','other models','new sampling','new labels'],limitations=limitations)
    checks=readj(run/'checks/basic_checks.json');checks.update(window_segment_overall_count_conservation=True,segment_sample_counts=[68600,68600,68500,68500],correction_harm_partition=True,attack_group_count_conservation=True,bootstrap_blocks=nb,bootstrap_last_block_windows=2,bootstrap_replicates=1000,shared_bootstrap_indices=True,E1_segment_intervals_use_whole_series_indices=True,zero_denominator_null=True,completed=True);writej(run/'checks/basic_checks.json',checks)
    write_summary(run,decision,overall,segment_rows,harms,groups,paired,risk_rows)
    writej(run/'reports/decision.json',decision)
    print(primary,rf_status,'net corrections',classification_net,flush=True)
    print(pd.DataFrame(overall.values())[['method','R_B','R_A','U_B','U_A','U_bal','accepted_B','accepted_A','target_status']].to_string(index=False),flush=True)


def write_summary(run,d,overall,segments,harms,groups,paired,risks):
    pct=lambda x:'NA' if x is None else f'{x*100:.4f}%'
    lines=['# ToN labels Modified pilot: Repeated review','',f" State: PILOT_COMPLETED; The original sample **274,200 is **, four sections are 68,600 / 68,600 / 68,500 / 68,500. Determined: **{d['primary_decision']}**。",f" E1 calibration dual branches are available; Repeat pilot migration status:{d['transfer_status']}. Local information growth:{d['local_information_increment']}; RF selects layer increase:{d['RF_selection_increment']}。",f" It suggests:{d['recommended_next_action']}",'','E1 is directly based on existing rules and not retrieved; Additional LOCAL_GATE, P_TAIL_C, P_TAIL_400, P1D_GATE are selected only on the declared C/400, all of which are evaluated after freeze. q support recovery from the original point without recalculating the 48 neighbourhood dimension; P1D only uses the absolute RF probability of the same R200 distance, k15 and rank to stabilize the ties.','', '## overall risk and abstention','','| method |RB|RA|UB|UA|U_bal| receives B/A|sampled FN/FP| abstention B/A| state |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for m,v in overall.items():lines.append(f"|{m}|{pct(v['R_B'])}|{pct(v['R_A'])}|{pct(v['U_B'])}|{pct(v['U_A'])}|{pct(v['U_bal'])}|{v['accepted_B']}/{v['accepted_A']}|{v['FN']}/{v['FP']}|{v['abstain_B']}/{v['abstain_A']}|{v['target_status']}|")
    lines+=['',f" E1 overall not satisfied:{d['E1_failure_metrics'] or 'none'}. empty acceptance of risk as NA; Acceptance of <100 or coverage window <10 Priority mark EVAL_SUPPORT_LIMITED and risk / Use Failures are retained respectively.",'', '## four-point comparison','','| paragraph | method |RB|RA|UB|UA| accepts B/A| target state | does not meet |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for v in segments:lines.append(f"|{v['segment']}|{v['method']}|{pct(v['R_B'])}|{pct(v['R_A'])}|{pct(v['U_B'])}|{pct(v['U_A'])}|{v['accepted_B']}/{v['accepted_A']}|{v['target_status']}|{v['failure_metrics'] or 'none'}|")
    lines+=['','## classification layer label correction and damage','','The table below is LOCAL_ALL relative to RAW, covering the entire pilot and not being screened with E1; abstention is not detected.','','| range | direction | re-evaluation |wrong → right|right → wrong| net label correction |','|---|---|---:|---:|---:|---:|']
    for h in harms:lines.append(f"|{h['scope']}|{h['direction']}|{h['changed']}|{h['wrong_to_right']}|{h['right_to_wrong']}|{h['net_corrections']}|")
    lines += ['',f" The total net label correction is {d['classification_net_corrections']} Article 4 (label correction reduction of additional errors);{'Additional errors exceed label correction, and the calibration of the classification layer improved the non-migration.' if d['classification_net_corrections']<0 else 'This is a change in the total number of class-layer errors that cannot replace binary risk with utility determination.'}",'', '## E1 Attacks: Weak and Strong Majorities','','| range |q group | sample number | error FP| error rate |','|---|---|---:|---:|---:|']
    for g in groups:lines.append(f"|{g['scope']}|{g['group']}|{g['n']}|{g['errors']}|{pct(g['error_rate'])}|")
    lines+=['','This table is intended only as a default explanation, without modifying the q threshold or adding a 0.75 gate control.','', '## coupled with risk and difference in efficiency','','Δ=E1_FIXED− control; The risk difference-negative value and the utility difference-negative value respectively indicate that the corresponding indicator is more favorable. The 1000 copy shares a continuous 10 window block index, with the final 2 window block retained; effective Rate <90% not reported.','','| controlling | indicator | difference |95% range | effective replicate |','|---|---|---:|---|---:|']
    for v in paired:lines.append(f"|{v['comparison']}|{v['metric']}|{pct(v['delta'])}|[{pct(v['ci_low'])}, {pct(v['ci_high'])}]|{v['valid_replicates']}/1000|")
    lines+=['','The percentage of differences and gaps is expressed as percentage points; Keeping the risk as a weighting tool, no overall acceptance rate or single difference will determine the winner.','', '## E1 self-descriptive range of risk','','The four segments use the same whole-sequence block index to set the extra-sequence counter to zero; No seeds are planted or reproduced.','','| range | risk | estimate |95% range | error / accept | effective replicate |','|---|---|---:|---|---|---:|']
    for v in risks:lines.append(f"|{v['scope']}|{v['metric']}|{pct(v['estimate'])}|[{pct(v['ci_low'])}, {pct(v['ci_high'])}]|{v['errors']}/{v['accepted']}|{v['valid_replicates']}/1000|")
    lines+=['','The risk range is not certified and does not include new screening thresholds with its upper limit.','', '## Interpretation and Execution of Boundaries','']+['- '+x for x in d['limitations']]
    lines+=['', '## delivery file (absolute path)','']
    for f in ['reports/summary_ZH.md','reports/decision.json','tables/comparison.csv','tables/segment_metrics.csv','tables/window_counts.csv','tables/correction_harm.csv','tables/attack_support_groups.csv','tables/paired_intervals.csv','tables/E1_risk_intervals.csv','calibration/frozen_baselines.json','calibration/baseline_candidates.csv','config/protocol.json','checks/basic_checks.json','src/run_experiment.py','logs/commands.sh','logs/execution.log']:lines.append(f'- [{f}]({run/f})')
    lines+=['','Stop when it’s done. The old run, models and caches are read only; No BoT, full quantity, feedback or other models have been run, no labels has been added or re-sampled.']
    (run/'reports/summary_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','evaluate','report'],required=True);args=ap.parse_args();root=args.project_root.resolve();run=args.run_dir.resolve()
    require(run.is_relative_to(root/'outputs/post_p7_diagnostics/ton_action_correction_pilot_v1'),'wrong run root')
    with (run/'logs/commands.sh').open('a') as f:f.write(shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+'\n')
    writej(run/'config/runtime.json',dict(interpreter=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__))
    print(now(),args.stage,sys.executable,flush=True);globals()[args.stage](root,run)
if __name__=='__main__':main()
