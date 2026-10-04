import cupy as cp
import numpy as np
from nids.replay.predictions import write_json


def draws_for_domains(data,seed):
    rng=cp.random.RandomState(seed);out=[]
    for d in data:
        blocks=(d.n+99999)//100000;samples=rng.randint(blocks,size=(2000,blocks));counts=cp.bincount((samples+cp.arange(2000)[:,None]*blocks).ravel(),minlength=2000*blocks).reshape(2000,blocks)
        sizes=cp.minimum(100000,d.n-cp.arange(blocks)*100000);out.append((counts,sizes))
    return out


def bounds(stats,draws,populations):
    accepted=[];errors=[];pop=[];coverage=[];risk=[]
    for (a,e),(mult,sizes),population in zip(stats,draws,populations):
        denom=mult@sizes
        accepted.append((mult@a)/denom);errors.append((mult@e)/denom);pop.append((mult@population)/denom)
        coverage.append(float(a.sum()/sizes.sum()));risk.append(float(e.sum()/sizes.sum()))
    accepted=sum(accepted)/2;errors=sum(errors)/2;population=sum(pop)/2
    rep=cp.where(accepted>0,errors/cp.where(accepted>0,accepted,1),1.)
    upper=float(cp.percentile(rep,95));point_a=sum(coverage)/2;point_e=sum(risk)/2
    pop_point=sum(float(v.sum()/s.sum()) for v,(_,s) in zip(populations,draws))/2
    return dict(coverage=point_a/pop_point if pop_point else 0,accepted_mass=point_a,error_mass=point_e,risk=point_e/point_a if point_a else None,upper95=upper)


def choose(rows):
    feasible=[r for r in rows if r['coverage']>=.1 and r['upper95']<=.05]
    return min(feasible,key=lambda r:(-r['coverage'],r['risk'],-r['threshold'])) if feasible else None


def select_thresholds(data,paths,cap,draws,output,allow_stop=True):
    arrays=[]
    for d,path in zip(data,paths):
        tr=np.load(path,mmap_mode='r');arrays.append(dict(p=cp.asarray(tr['fused'][:,:cap]),r=cp.asarray(tr['reliability'][:,:cap]),y=d.labels(),block=cp.arange(d.n)//100000,calls=cp.asarray(tr['calls'][:,:cap]),cost=cp.asarray(tr['cost'][:,:cap])))
    stop_rows=[]
    for k in range(101):
        tau=k/100;stats=[]
        for v,(_,sizes) in zip(arrays,draws):
            crossing=v['r'][:,:cap-1]>=tau;has=crossing.any(1) if cap>1 else cp.zeros(len(v['y']),dtype=bool)
            stage=crossing.argmax(1) if cap>1 else cp.zeros(len(v['y']),dtype=int);p=v['p'][cp.arange(len(stage)),stage];wrong=(p>=.5)!=v['y']
            stats.append((cp.bincount(v['block'],weights=has,minlength=len(sizes)),cp.bincount(v['block'],weights=has&wrong,minlength=len(sizes))))
        b=bounds(stats,draws,[s for _,s in draws]);stop_rows.append(dict(threshold=tau,**b))
    selected_stop=choose(stop_rows) if cap>1 and allow_stop else None;final_rows=[];terminal=[];early=[]
    for v,(_,sizes) in zip(arrays,draws):
        crossing=v['r'][:,:cap-1]>=selected_stop['threshold'] if selected_stop else cp.zeros((len(v['y']),cap-1),dtype=bool)
        has=crossing.any(1);stage=cp.where(has,crossing.argmax(1),cap-1) if cap>1 else cp.zeros(len(v['y']),dtype=int)
        early.append((has,stage));terminal.append(cp.bincount(v['block'],weights=~has,minlength=len(sizes)))
    for k in range(101):
        stats=[]
        for v,(has,stage),(_,sizes) in zip(arrays,early,draws):
            accept=(~has)&(v['r'][:,-1]>=k/100);wrong=(v['p'][:,-1]>=.5)!=v['y'];stats.append((cp.bincount(v['block'],weights=accept,minlength=len(sizes)),cp.bincount(v['block'],weights=accept&wrong,minlength=len(sizes))))
        final_rows.append(dict(threshold=k/100,**bounds(stats,draws,terminal)))
    selected_final=choose(final_rows);domain_metrics=[]
    for d,v,(has,stage),path in zip(data,arrays,early,paths):
        idx=cp.arange(d.n);accepted=has|((~has)&(v['r'][:,-1]>=selected_final['threshold'])) if selected_final else has;wrong=(v['p'][idx,stage]>=.5)!=v['y'];consult=v['calls'][idx,stage]>1;initialwrong=(v['p'][:,0]>=.5)!=v['y']
        m=dict(domain=d.domain,rows=d.n,coverage=float(accepted.mean()),accepted_error_mass=float((accepted&wrong).mean()),full_error=float(wrong.mean()),mean_calls=float(v['calls'][idx,stage].mean()),normalized_cost=float(v['cost'][idx,stage].mean()),consultation_rate=float(consult.mean()),corrective_yield=float((consult&initialwrong&~wrong).sum()/cp.maximum(consult.sum(),1)),consultation_harm=float((consult&~initialwrong&wrong).sum()/cp.maximum(consult.sum(),1)))
        domain_metrics.append(m)
        np.savez(output/f'{d.domain}_cap{cap}_decisions.npz',stage=cp.asnumpy(stage).astype(np.uint8),accepted=cp.asnumpy(accepted),early_stop=cp.asnumpy(has),reason=cp.asnumpy(cp.where(has,0,cp.where(accepted,1,2 if selected_final else 3))).astype(np.uint8))
    summary={key:sum(m[key] for m in domain_metrics)/2 for key in ['coverage','accepted_error_mass','full_error','mean_calls','normalized_cost','consultation_rate','corrective_yield','consultation_harm']};summary['risk']=summary['accepted_error_mass']/summary['coverage'] if summary['coverage'] else None
    stop_status='DISABLED_BY_CAP' if cap==1 else 'DISABLED_BY_METHOD' if not allow_stop else 'ENABLED' if selected_stop else 'DISABLED_NO_FEASIBLE_THRESHOLD'
    result=dict(cap=cap,tau_stop=selected_stop['threshold'] if selected_stop else None,tau_final=selected_final['threshold'] if selected_final else None,early_stop_status=stop_status,final_status='ENABLED' if selected_final else 'ABSTAIN_NO_FEASIBLE_THRESHOLD',stop_candidates=stop_rows,final_candidates=final_rows,domain_metrics=domain_metrics,summary=summary)
    write_json(output/f'cap{cap}_selection.json',result);return result
