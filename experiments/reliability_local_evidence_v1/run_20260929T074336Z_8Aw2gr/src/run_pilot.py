#!/usr/bin/env python
"""S1 prepare -> S2 pilot -> report. No full stage, no legacy imports."""
import argparse,sys,time,shlex,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from core import *


def prepare(root,run,protocol):
    frozen=run/'calibration/frozen_rules.json'
    if frozen.exists(): print('prepare already frozen; skipped',flush=True);return
    tests=selftest();roles=[];supports=[];candidates=[];rules={};assets={};checks={'synthetic':tests,'domains':{}}
    manifest=pd.read_csv(root/protocol['calibration']['manifest'])
    windows=pd.read_csv(root/'outputs/round1_data_prep_v1/window_manifest.csv')
    for domain,dp in protocol['domains'].items():
        try:
            a=Assets(root,domain,protocol);assets[domain]=a.info
            rows=manifest[manifest.domain==domain].sort_values('availability_rank');ranks=rows.availability_rank.to_numpy(dtype=np.int64)
            require(len(ranks)==400 and len(np.unique(ranks))==400 and np.all((ranks>=0)&(ranks<100000)),'historical 400 identity/range')
            check_alignment(a.orig[ranks],rows.original_row.to_numpy())
            y=a.labels(ranks,calibration=True);require(np.array_equal(y,rows.binary_label.to_numpy()),'historical label alignment conflict')
            perm=np.random.Generator(np.random.PCG64(20260929)).permutation(400);ri=np.sort(perm[:200]);ci=np.sort(perm[200:])
            require(len(np.intersect1d(ri,ci))==0 and len(np.union1d(ri,ci))==400,'R/C split overlap')
            p=a.probs(ranks);x=a.features(ranks)
            rc=np.bincount(y[ri],minlength=2);cc=np.bincount(y[ci],minlength=2)
            require(rc.tolist()==list(dp['expected_R_counts'].values()) and cc.tolist()==list(dp['expected_C_counts'].values()),'historical R/C count conflict')
            adequate=min(rc)>=5 and min(cc)>=10
            for name,idx in [('R',ri),('C',ci),('Q400',np.arange(400))]:
                supports.append(dict(domain=domain,role=name,N=len(idx),true_B=int((y[idx]==0).sum()),true_A=int((y[idx]==1).sum()),pred_B=int((p[idx]<.5).sum()),pred_A=int((p[idx]>=.5).sum()),status='ADEQUATE' if adequate else 'LABEL_SUPPORT_LIMITED'))
            for i,r in enumerate(ranks): roles.append(dict(domain=domain,availability_rank=int(r),original_row=int(a.orig[r]),role='R' if i in ri else 'C',y=int(y[i]),raw_p=float(p[i]),raw_pred=int(p[i]>=.5)))
            t0=time.perf_counter();lc,dc,nn=evidence(x[ci],x[ri],y[ri],p[ci]>=.5,a.cat,a.num,a.scale)
            require(nn.min()>=0 and nn.max()<200,'neighbor outside R')
            # Independent scalar reference for exact float64 and stable ordering.
            dd=[]
            for ref in x[ri]:
                total=0.
                for j in sorted(a.cat+a.num):
                    total += float(x[ci[0],j]!=ref[j]) if j in a.cat else min(abs(x[ci[0],j]-ref[j])/a.scale[j],10.)/10.
                dd.append(total/48.)
            ix=np.lexsort((ranks[ri],np.array(dd)))[:15]
            require(np.array_equal(nn[0],ix),'scalar/batched neighbor mismatch')
            require(np.allclose(dc[0],np.array(dd)[ix].mean(),rtol=0,atol=1e-15),'scalar/batched distance mismatch')
            selected={}
            for method in ['G0','G1H','G1F','G2','Q400']:
                if method=='Q400': out,cr=select(p,y,np.zeros(400),method,domain)
                else: out,cr=select(p[ci],y[ci],lc,method,domain)
                selected[method]=out;candidates.extend(cr)
            # Every G1F branch must be present with exactly matching counts in G2 u=0.
            for cls in [0,1]:
                f=branch_candidates(p[ci],y[ci],lc,grid(p[ci]),cls,'G1F')
                g={z['id']:z for z in branch_candidates(p[ci],y[ci],lc,grid(p[ci]),cls,'G2')}
                require(all(z==g[z['id']] for z in f),'G2 u=0 does not contain G1F')
            start,end=dp['evaluation'];w=windows[(windows.domain==domain)&(windows.scheme=='heldout')&(windows.window_size==10000)&(windows.rank_start>=start)&(windows.rank_end_exclusive<=end)].sort_values('rank_start')
            require(np.array_equal(w.rank_start.to_numpy(),np.arange(start,end,10000)) and np.array_equal(w.rank_end_exclusive.to_numpy(),np.arange(start+10000,end+1,10000)),'10k window range/alignment conflict')
            # Index generation uses only fixed window starts and RNG, not labels/predictions.
            rng=np.random.Generator(np.random.PCG64(dp['pilot_seed']))
            sample=np.concatenate([np.sort(rng.choice(10000,100,replace=False))+s for s in w.rank_start.to_numpy()])
            require(len(sample)==dp['expected_pilot_rows'] and np.all(np.diff(sample)>0),'pilot count/unique ranks')
            np.save(run/f'pilot/{domain}_evaluation_sample_ranks.npy',sample)
            w[['window_id','rank_start','rank_end_exclusive']].to_csv(run/f'config/{domain}_windows.csv',index=False)
            # Small unlabeled future feature batch; no future label access.
            tx=a.features(sample[:256]);tp=a.probs(sample[:256]);t1=time.perf_counter()
            evidence(tx,x[ri],y[ri],tp>=.5,a.cat,a.num,a.scale);secs=time.perf_counter()-t1
            np.savez(run/f'calibration/{domain}_reference.npz',rank=ranks[ri],x=x[ri],y=y[ri])
            np.savez(run/f'calibration/{domain}_selection_evidence.npz',rank=ranks[ci],y=y[ci],raw_p=p[ci],local_support=lc,mean_neighbor_distance=dc,neighbor_ranks=ranks[ri][nn])
            rules[domain]=dict(input_status='READY',calibration_support_status='ADEQUATE' if adequate else 'LABEL_SUPPORT_LIMITED',rules=selected)
            checks['domains'][domain]=dict(historical_400_unique=True,rank_and_label_alignment=True,R_C_disjoint=True,R_counts=rc.tolist(),C_counts=cc.tolist(),source_scale_positive=True,raw_probability_range=True,neighbor_R_only=True,scalar_distance_equivalence=True,g2_u0_nested=True,pilot_indices_frozen=True,pilot_rows=len(sample),window_count=len(w),unlabeled_probe_rows=256,unlabeled_probe_seconds=secs,prepare_seconds=time.perf_counter()-t0)
            print(domain,'prepared',rc.tolist(),cc.tolist(),'probe seconds',secs,flush=True)
        except (ValueError,FileNotFoundError,KeyError) as e:
            rules[domain]=dict(input_status='INPUT_BLOCKED',reason=str(e),calibration_support_status='UNKNOWN');print(domain,'INPUT_BLOCKED',e,flush=True)
    csv(run/'calibration/sample_roles.csv',roles);csv(run/'calibration/support.csv',supports);csv(run/'calibration/candidates.csv',candidates)
    writej(run/'config/resolved_assets.json',assets)
    writej(run/'config/metric_definitions.json',dict(count_columns=COUNT_KEYS,actions={'-1':'ABSTAIN','0':'B','1':'A'},R_B='FN/(TN+FN)',R_A='FP/(TP+FP)',U_B='TN/N_B',U_A='TP/N_A',U_bal='0.5*(U_B+U_A)',absolute_errors='sampled; never scaled by 100',zero_denominator=None,candidate_id='B_<direction>_T<ascending grid index:03>_U<u index:03>__A_...; OFF_T999_U000; lexicographic final tie; G0 both enabled must share T',tie_order=protocol['gates']['tie_break']))
    writej(run/'checks/basic_checks.json',checks)
    writej(frozen,dict(protocol_version=1,freeze_time=now(),protocol=protocol,domains=rules,stage='S1_FROZEN',evaluation_labels_read=False))
    print('S1_FROZEN',now(),flush=True)


def pilot(root,run,protocol):
    frozen=run/'calibration/frozen_rules.json';fr=readj(frozen);require(fr['protocol']==protocol,'protocol differs from freeze')
    checks=readj(run/'checks/basic_checks.json');completed={}
    for domain,dp in protocol['domains'].items():
        if fr['domains'][domain]['input_status']!='READY': continue
        done=run/f'pilot/{domain}_completed.json'
        if done.exists(): completed[domain]=readj(done);print(domain,'pilot already complete; skipped',flush=True);continue
        a=Assets(root,domain,protocol);r=np.load(run/f'calibration/{domain}_reference.npz');sample=np.load(run/f'pilot/{domain}_evaluation_sample_ranks.npy');require(len(sample)==dp['expected_pilot_rows'],'pilot indices count')
        folder=run/f'pilot/{domain}';folder.mkdir(exist_ok=True);batch=10000;t0=time.perf_counter();parts=0
        for start in range(0,len(sample),batch):
            stop=min(start+batch,len(sample));out=folder/f'chunk_{parts:04d}.npz';rr=sample[start:stop]
            if out.exists():
                z=np.load(out);require(np.array_equal(z['rank'],rr) and len(z['y'])==len(rr),'completed chunk rank/count conflict');parts+=1;continue
            p=a.probs(rr);pred=(p>=.5).astype(np.int8);x=a.features(rr)
            local,dist,nn=evidence(x,r['x'],r['y'],pred,a.cat,a.num,a.scale)
            acts=actions(p,local,fr['domains'][domain]['rules'])
            # Sole future label access: after global S1 freeze and after fixed actions.
            y=a.labels(rr,frozen=frozen)
            segment=np.searchsorted(dp['segments'][1:-1],rr,side='right')+1;window=(rr-dp['evaluation'][0])//10000
            data=dict(domain=np.full(len(rr),domain),rank=rr,segment=segment,window=window,y=y,raw_p=p,raw_pred=pred,confidence=np.maximum(p,1-p),local_support=local,mean_neighbor_distance=dist,**{'action_'+k:v for k,v in acts.items()})
            tmp=out.with_suffix('.tmp.npz');np.savez(tmp,**data);tmp.replace(out);parts+=1
            print(domain,f'{stop}/{len(sample)} rows',f'{time.perf_counter()-t0:.1f}s',flush=True)
        completed[domain]=dict(domain=domain,stage='S2_PILOT',rows=len(sample),chunks=parts,completed_time=now(),freeze_time=fr['freeze_time'],batch_rows=batch)
        writej(done,completed[domain]);checks['domains'][domain].update(pilot_completed=True,actual_pilot_rows=len(sample),raw_label_preserved=True)
    writej(run/'pilot/completed.json',completed);writej(run/'checks/basic_checks.json',checks)


def load_evidence(run,domain,expected):
    parts=[]
    for p in sorted((run/f'pilot/{domain}').glob('chunk_*.npz')):
        with np.load(p) as z: parts.append({k:z[k] for k in z.files})
    out={k:np.concatenate([z[k] for z in parts]) for k in parts[0]}
    require(len(out['rank'])==expected and np.array_equal(out['rank'],np.load(run/f'pilot/{domain}_evaluation_sample_ranks.npy')),'report rows alignment')
    return out


def ci(values):
    v=np.array(values,dtype=float);v=v[np.isfinite(v)];rate=len(v)/1000
    lo,hi=np.percentile(v,[2.5,97.5]) if rate>=.9 else (None,None)
    return dict(ci_low=lo,ci_high=hi,valid_replicates=len(v),valid_fraction=rate)


def bootstrap_ratios(c):
    # c columns follow COUNT_KEYS. All ratios are recomputed from sampled totals.
    with np.errstate(divide='ignore',invalid='ignore'):
        ub=c[:,3]/c[:,1];ua=c[:,6]/c[:,2]
        return dict(R_B=c[:,4]/(c[:,3]+c[:,4]),R_A=c[:,5]/(c[:,6]+c[:,5]),U_B=ub,U_A=ua,U_bal=(ub+ua)/2,acceptance=(c[:,3:7].sum(axis=1))/c[:,0])


def analyze_domain(run,domain,dp,fr):
    e=load_evidence(run,domain,dp['expected_pilot_rows']);nw=len(e['y'])//100;nb=(nw+9)//10
    rng=np.random.Generator(np.random.PCG64(20261001));draw=rng.integers(0,nb,size=(1000,nb))
    # Block multiplicities shared for policy differences and every local-signal stratum.
    weight=np.array([np.bincount(z,minlength=nb) for z in draw],dtype=np.int64)
    segment_w=e['segment'].reshape(nw,100)[:,0];overall={};segments=[];winrows=[];boot={};intervals=[]
    for method in METHODS:
        c=window_counts(e['y'],e['raw_pred'],e['action_'+method],nw)
        require(np.all(c[:,1]+c[:,2]==c[:,0]) and np.all(c[:,3]+c[:,5]+c[:,7]==c[:,1]) and np.all(c[:,4]+c[:,6]+c[:,8]==c[:,2]),'count conservation')
        overall[method]=dict(domain=domain,method=method,sampled=True,**metrics(c.sum(axis=0)))
        sums=[]
        for seg in range(1,5):
            total=c[segment_w==seg].sum(axis=0);sums.append(total);segments.append(dict(domain=domain,segment=seg,method=method,sampled=True,**metrics(total)))
        require(np.array_equal(np.sum(sums,axis=0),c.sum(axis=0)),'segments do not sum to total')
        for w,counts in enumerate(c): winrows.append(dict(domain=domain,window=w,rank_start=dp['evaluation'][0]+w*10000,segment=int(segment_w[w]),method=method,sampled=True,**metrics(counts)))
        blocks=np.add.reduceat(c,np.arange(0,nw,10),axis=0);boot[method]=bootstrap_ratios(weight@blocks)
    for comp in ['G1F','Q400']:
        for metric in ['R_B','R_A','U_B','U_A','U_bal','acceptance']:
            x=overall['G2'][metric];y=overall[comp][metric]
            intervals.append(dict(domain=domain,comparison='G2-'+comp,metric=metric,delta=x-y if x is not None and y is not None else None,**ci(boot['G2'][metric]-boot[comp][metric])))
    signal=[];byclass={};edges=np.array([.5,.6,.7,.8,.9,.95,.99,1.]);bins=np.minimum(np.searchsorted(edges,e['confidence'],side='right')-1,6);block=e['window']//10
    err=e['raw_pred']!=e['y']
    for cls in [0,1]:
        eligible=[];positive_segments=set()
        for seg in range(1,5):
            for b in range(7):
                base=(e['segment']==seg)&(e['raw_pred']==cls)&(bins==b);low=base&(e['local_support']<=.5);high=base&(e['local_support']>=.75)
                nl=int(low.sum());nh=int(high.sum());el=int((low&err).sum());eh=int((high&err).sum());wl=len(np.unique(e['window'][low]));wh=len(np.unique(e['window'][high]));valid=nl>=30 and nh>=30 and wl>=5 and wh>=5
                difference=el/nl-eh/nh if nl and nh else None
                row=dict(domain=domain,predicted_class='B' if cls==0 else 'A',segment=seg,confidence_low=edges[b],confidence_high=edges[b+1],last_includes_one=b==6,n_low=nl,n_high=nh,n_middle=int(base.sum())-nl-nh,errors_low=el,errors_high=eh,windows_low=wl,windows_high=wh,error_low=ratio(el,nl),error_high=ratio(eh,nh),error_low_minus_high=difference,eligible=valid,fixed_weight=min(nl,nh) if valid else 0,status='ELIGIBLE' if valid else 'INSUFFICIENT_OVERLAP')
                signal.append(row)
                if valid:
                    def bc(m): return np.bincount(block[m],minlength=nb)
                    eligible.append((row,np.column_stack([bc(low),bc(low&err),bc(high),bc(high&err)])))
                    if difference>0: positive_segments.add(seg)
        if eligible:
            totalw=sum(row['fixed_weight'] for row,_ in eligible);estimate=sum(row['fixed_weight']*row['error_low_minus_high'] for row,_ in eligible)/totalw
            values=np.zeros(1000);valid=np.ones(1000,dtype=bool)
            for row,counts in eligible:
                z=weight@counts;valid &= (z[:,0]>0)&(z[:,2]>0)
                with np.errstate(divide='ignore',invalid='ignore'): values+=row['fixed_weight']*(z[:,1]/z[:,0]-z[:,3]/z[:,2])/totalw
            values[~valid]=np.nan;interval=ci(values)
            if estimate>=.02 and interval['ci_low'] is not None and interval['ci_low']>0 and len(positive_segments)>=2: status='LOCAL_SIGNAL_POSITIVE'
            elif interval['ci_high'] is not None and interval['ci_high']<.02: status='NO_MATERIAL_SIGNAL_AT_THIS_RESOLUTION'
            else: status='INCONCLUSIVE'
        else: estimate=None;interval=dict(ci_low=None,ci_high=None,valid_replicates=0,valid_fraction=0.);status='INCONCLUSIVE';totalw=0
        byclass['B' if cls==0 else 'A']=dict(status=status,weighted_error_low_minus_high=estimate,eligible_strata=len(eligible),fixed_weight_total=totalw,positive_segments=sorted(positive_segments),**interval)
    return overall,segments,winrows,signal,intervals,byclass


def report(root,run,protocol):
    if (run/'reports/decision.json').exists(): print('report already complete; skipped',flush=True);return
    fr=readj(run/'calibration/frozen_rules.json');require(fr['protocol']==protocol,'protocol differs from freeze')
    alloverall=[];allsegments=[];allwin=[];allsignal=[];allci=[];decisions=[];checks=readj(run/'checks/basic_checks.json')
    limitations=['Under controlled exploration, target domain has already been exposed, not a new blind test.','The 400 is divided into single sections; In particular, BoT positive support is limited; 5% is an experimental engineering operating point, not a risk certification.','Fixed mixed distance and category equal weighting may not represent the attack semantics; local_support is not a probability.','The substrate is relatively correlated and may still be affected by uncontrolled features; Without effective overlap is not the same as without a signal.','Block bootstrap only carries the current time sequence and does not cover calibration sampling, model seed or future arbitrary drift.','S2 for window 1% estimate; The absolute number of errors is sampled; S3 contains S2, not independent reproduction.']
    for domain,dp in protocol['domains'].items():
        f=fr['domains'][domain]
        if f['input_status']!='READY':
            decisions.append(dict(protocol_version=1,stage='S2_PILOT',domain=domain,input_status='INPUT_BLOCKED',calibration_support_status=f['calibration_support_status'],selected_policy_status=None,evaluation_support_status=None,local_signal_by_predicted_class=None,joint_target_by_method=None,segment_target_by_method=None,paired_utility_deltas=None,primary_decision='INPUT_BLOCKED',temporal_modifier=None,reason_counts_and_metrics=f['reason'],recommended_next_action='Only repair the adaptation of the affected domain data; It’s not hard, it’s not training.',not_authorized_next_actions=protocol['prohibited'],limitations=limitations));continue
        require((run/f'pilot/{domain}_completed.json').exists(),'pilot not completed')
        overall,segments,win,signal,intervals,byclass=analyze_domain(run,domain,dp,fr)
        alloverall.extend(overall.values());allsegments.extend(segments);allwin.extend(win);allsignal.extend(signal);allci.extend(intervals)
        target={k:v['target_status']=='JOINT_TARGET_MET' for k,v in overall.items()}
        segtarget={m:{str(s['segment']):s['target_status'] for s in segments if s['method']==m} for m in METHODS}
        deltas={c:next(r for r in intervals if r['comparison']=='G2-'+c and r['metric']=='U_bal') for c in ['G1F','Q400']}
        passing=[c for c in ['G1F','Q400'] if target[c]]
        if target['G2']:
            gain=all(deltas[c]['delta']>=.01 and deltas[c]['ci_low'] is not None and deltas[c]['ci_low']>0 for c in passing)
            decision='LOCAL_POLICY_CANDIDATE' if gain else 'NO_ESTABLISHED_LOCAL_POLICY_GAIN'
        elif passing: decision='SIMPLE_POLICY_PREFERRED'
        elif any(v['status']=='LOCAL_SIGNAL_POSITIVE' for v in byclass.values()): decision='SIGNAL_WITHOUT_USEFUL_POLICY'
        elif f['calibration_support_status']=='ADEQUATE' and all(v['status']=='NO_MATERIAL_SIGNAL_AT_THIS_RESOLUTION' for v in byclass.values()): decision='NO_MATERIAL_LOCAL_GAIN_THIS_SETUP'
        else: decision='INCONCLUSIVE'
        temporal={m:('TEMPORALLY_SUPPORTED_PILOT' if sum(v=='JOINT_TARGET_MET' for v in segtarget[m].values())>=3 else 'TEMPORALLY_FRAGILE') if target[m] else None for m in METHODS}
        nextaction={'LOCAL_POLICY_CANDIDATE':'Recommends alternative authorization of the same rule as S3; First, check out the four weaknesses.','NO_ESTABLISHED_LOCAL_POLICY_GAIN':'Simplified rules of probability that prioritize keeping a score; If full verification is required, an alternative authorization is required.','SIMPLE_POLICY_PREFERRED':'It is recommended that an alternative authorization of simple probability rules for validation of S3 be made without the use of local components for the time being.','SIGNAL_WITHOUT_USEFUL_POLICY':'Reporting signal disconnection from operating point; Alternatively, single-factor testing with gate control or data support does not automatically expand.','NO_MATERIAL_LOCAL_GAIN_THIS_SETUP':'Stop expanding the current +k+400 distance scheme; Afterwards, only one of the budgets will be discussed or made public.','INCONCLUSIVE':'Indicates layer overlap, labels or movement support gaps; Additional instructions are required for complementary design.'}[decision]
        decisions.append(dict(protocol_version=1,stage='S2_PILOT',domain=domain,input_status='READY',actual_rows=dp['expected_pilot_rows'],calibration_support_status=f['calibration_support_status'],selected_policy_status={m:r['status'] for m,r in f['rules'].items()},unavailable_action_branches={m:[c for c in ['B','A'] if r[c]['direction']=='OFF'] for m,r in f['rules'].items()},evaluation_support_status={m:('EVAL_SUPPORT_LIMITED' if v['target_status']=='EVAL_SUPPORT_LIMITED' else 'ADEQUATE') for m,v in overall.items()},local_signal_by_predicted_class=byclass,joint_target_by_method={m:v['target_status'] for m,v in overall.items()},segment_target_by_method=segtarget,paired_utility_deltas=deltas,primary_decision=decision,temporal_modifier=temporal,degenerate_or_partial={m:r['status']!='TWO_ACTION_POLICY' for m,r in f['rules'].items()},reason_counts_and_metrics=dict(overall=overall,segments=segments),recommended_next_action=nextaction,not_authorized_next_actions=['S3 full','G3','other models','feedback','coordination','retraining'],limitations=limitations))
        checks['domains'][domain].update(report_count_conservation=True,window_segment_total_agree=True,bootstrap_replicates=1000,bootstrap_blocks=(dp['expected_pilot_rows']//100+9)//10,shared_bootstrap_indices=True)
        print(domain,decision,byclass,flush=True)
    csv(run/'tables/comparison.csv',alloverall);csv(run/'tables/segment_metrics.csv',allsegments);csv(run/'tables/window_counts.csv',allwin);csv(run/'tables/local_signal.csv',allsignal);csv(run/'tables/paired_intervals.csv',allci)
    writej(run/'tables/local_signal_summary.json',{d['domain']:d['local_signal_by_predicted_class'] for d in decisions})
    status='PILOT_COMPLETED' if all(d['input_status']=='READY' for d in decisions) else 'PARTIAL_BLOCKED'
    checks['status']=status;checks['actual_total_rows']=sum(d.get('actual_rows',0) for d in decisions)
    if status=='PILOT_COMPLETED': require(checks['actual_total_rows']==442500,'total pilot rows')
    writej(run/'checks/basic_checks.json',checks)
    lines=[f'# S2 pilot：{status}',f" Practical evaluation {checks['actual_total_rows']:,} the article; Fixed by 10k window 100 line (1%), not full. S1 freeze:{fr['freeze_time']}。",'']
    def fmt(v): return 'NA' if v is None else f'{v:.6f}'
    for d in decisions:
        lines += [f"## {d['domain']}：{d['primary_decision']}"]
        if d['input_status']!='READY': lines += [str(d['reason_counts_and_metrics'])];continue
        lines += [f" The actual number of rows {d['actual_rows']:,}. It is not possible to use a branch of action:{json.dumps(d['unavailable_action_branches'],ensure_ascii=False)}。",f" G2 relative to G1F / Q400 efficiency increase:"+'；'.join(f"{c} ΔU_bal={fmt(v['delta'])}, 95% CI=[{fmt(v['ci_low'])}, {fmt(v['ci_high'])}]" for c,v in d['paired_utility_deltas'].items())+'。',f" The 02 table determines:{d['primary_decision']}. It suggests:{d['recommended_next_action']}",'','| method |RB|RA|UB|UA|U_bal| accepts B/A|sampled FN/FP| overall goal |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for m,v in d['reason_counts_and_metrics']['overall'].items(): lines.append(f"|{m}|{fmt(v['R_B'])}|{fmt(v['R_A'])}|{fmt(v['U_B'])}|{fmt(v['U_A'])}|{fmt(v['U_bal'])}|{v['accepted_B']}/{v['accepted_A']}|{v['FN']}/{v['FP']}|{v['target_status']}|")
        lines += ['','Four comparisons (see segment_metrics.csv for all methods and their subdivisions):','']
        for m in ['G1F','G2','Q400']:
            for s in d['reason_counts_and_metrics']['segments']:
                if s['method']==m: lines.append(f"- {m} Section 1 {s['segment']}: RB={fmt(s['R_B'])}, RA={fmt(s['R_A'])}, UB={fmt(s['U_B'])}, UA={fmt(s['U_A'])}; {s['target_status']}; does not meet ={s['failure_metrics'] or 'none'}。")
        lines += ['', 'Local signals (all 56 substrate / domains are retained in local_signal.csv):']
        for cls,v in d['local_signal_by_predicted_class'].items(): lines.append(f"- forecast {cls}: {v['status']}; effective layer {v['eligible_strata']}; Acceleration difference ={fmt(v['weighted_error_low_minus_high'])}；CI=[{fmt(v['ci_low'])}, {fmt(v['ci_high'])}]; effective Replicate rate ={v['valid_fraction']:.3f};= to the time period {v['positive_segments']}。")
        lines += [f" The timing is modified:{json.dumps(d['temporal_modifier'],ensure_ascii=False)}。",'']
    lines += ['## explains the boundaries']+['- '+s for s in limitations]+['','The old Variant B is only historical in context, its sampling is different from that which could be re-judged, it has not been re-run and is not involved in the main comparison.','This time, stop at the S2 report. Recommendation is not equivalent to the execution of authorization; No S3, other models, G3, continuous feedback or coordination.']
    (run/'reports/summary_ZH.md').write_text('\n'.join(lines)+'\n')
    writej(run/'reports/decision.json',dict(protocol_version=1,stage='S2_PILOT',status=status,completed_time=now(),actual_total_rows=checks['actual_total_rows'],domains=decisions))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--stage',choices=['prepare','pilot','report'],required=True);args=ap.parse_args()
    root=args.project_root.resolve();run=args.run_dir.resolve();protocol=readj(args.protocol)
    require(run.is_relative_to(root/protocol['output_base']),'run outside experiment root')
    with (run/'logs/commands.sh').open('a') as f: f.write(shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+'\n')
    writej(run/'config/runtime.json',dict(interpreter=sys.executable,python=sys.version,numpy=np.__version__,pandas=pd.__version__,run_dir=str(run)))
    print(now(),args.stage,sys.executable,flush=True)
    globals()[args.stage](root,run,protocol)
if __name__=='__main__': main()
