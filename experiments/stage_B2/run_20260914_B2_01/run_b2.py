"""Post-unblinding B2: 18 All49 entropy-only correctness models. No detector fitting."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import sys
import time
import gc
import numpy as np
import cupy as cp
from cupyx.scipy.special import xlogy
import torch
from torch.nn import functional as F
import pyarrow.parquet as pq

ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT=Path(__file__).resolve().parent
A=ROOT/'outputs/post_p7_diagnostics/stage_A/run_20260914_A1'
B1=ROOT/'outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01'
sys.path.insert(0,str(ROOT/'src'))
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import checked_windows
from nids.evaluation.round1_metrics import ranking_groups, grouped_aurc, harmonic_numbers, BlockRanking
from nids.evaluation.round1_bootstrap import percentile_ci

MODELS=['rf_All49','xgb_All49','mlp_All49']
PCTS=[1,5,10,20,40,60,80,100]
CHUNK=262144
def read(p): return json.loads(Path(p).read_text())
def rows(p): return list(csv.DictReader(Path(p).open()))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(name,rs):
    if not rs: return
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:'NA' if r.get(k) is None else r[k] for k in fields} for r in rs)
def ratio(a,b): return a/b if b else None
def release():
    gc.collect();cp.get_default_memory_pool().free_all_blocks();torch.cuda.empty_cache()
def entropy(p): return -(xlogy(p,p)+xlogy(1-p,1-p))/cp.log(2.)

CFG=read(ROOT/'configs/execution_v2.json')

def inventory():
    roots=[ROOT/'configs',ROOT/'docs',ROOT/'src/nids',ROOT/'outputs/source_v2',
        ROOT/'outputs/prediction_cache_v2',ROOT/'outputs/round1_data_prep_v1',ROOT/'outputs/round1_v2',
        ROOT/'outputs/outer_v2',ROOT/'outputs/round2_v2',ROOT/'outputs/shift_v2',A,B1]
    result={}
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                s=p.stat();result[str(p)]=[s.st_size,s.st_mtime_ns]
    return result

def scopes(prep):
    result={}
    for d in ['CIC','BoT','ToN']:
        for partname in ['R_fit','R_cal','R_select','T_heldout']:
            scheme='heldout' if partname=='T_heldout' else 'development'
            part=prep.partition(d,scheme,partname)
            if scheme=='heldout':
                _,ws=checked_windows(prep,d,pq.read_table(ROOT/f'outputs/shift_v2/10k/{d}_windows.parquet').to_pylist())
            else:
                ws=sorted([w for w in prep.windows if w['domain']==d and w['scheme']==scheme and w['partition']==partname and int(w['window_size'])==10000],key=lambda w:int(w['rank_start']))
            a=part.rank_start;cursor=a
            order=prep.sorted_rows(d)
            for w in ws:
                x,z=int(w['rank_start']),int(w['rank_end_exclusive'])
                assert x==cursor and z-x==10000
                assert int(order[x])==int(w['first_original_row']) and int(order[z-1])==int(w['last_original_row'])
                cursor=z
            assert 0<=part.rank_end_exclusive-cursor<10000
            y=np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')[a:cursor]
            result[d,partname]=dict(domain=d,partition=partname,rank_start=a,rank_end_exclusive=cursor,rows=cursor-a,
                windows=len(ws),excluded_tail=part.rank_end_exclusive-cursor,benign=int((y==0).sum()),attack=int((y==1).sum()),
                initialization_rows=sum(str(w['initialization']).lower()=='true' for w in ws)*10000)
    return result

def load_data(d,m,o,scope):
    a,b=scope['rank_start'],scope['rank_end_exclusive'];n=b-a
    cachepath=ROOT/f'outputs/prediction_cache_v2/{d}/{m}.npy'
    manifest=read(cachepath.with_name(m+'_manifest.json'))
    train=read(ROOT/f'outputs/source_v2/models/{m}/training_manifest.json')
    assert manifest['model_id']==m and manifest['domain']==d and manifest['feature_indices']==train['feature_indices']
    cache=np.load(cachepath,mmap_mode='r')
    u=cp.empty(n,dtype=cp.float64);pred=cp.empty(n,dtype=cp.bool_)
    y=cp.asarray(np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')[a:b])
    for lo in range(0,n,CHUNK):
        hi=min(n,lo+CHUNK)
        p=cp.asarray(cache['raw_probability' if o=='Raw' else 'probability'][a+lo:a+hi])
        assert bool(cp.isfinite(p).all()) and bool(((p>=0)&(p<=1)).all())
        uu=entropy(p)
        assert bool(cp.isfinite(uu).all()) and bool(((uu>=0)&(uu<=1)).all())
        assert bool((uu[(p==0)|(p==1)]==0).all())
        u[lo:hi]=uu;pred[lo:hi]=p>=.5
    assert int(y.sum())==scope['attack']
    return u,pred,y

def fit_u(us,corrects,audits):
    """P4 fit_cre_family u-only extraction: same arithmetic, weights, objective, solver."""
    counts=[len(u) for u in us];total=sum(counts)
    x=cp.concatenate(us).astype(cp.float64);y=cp.concatenate(corrects).astype(cp.float64)
    w=cp.concatenate([cp.full(n,total/(2*n),dtype=cp.float64) for n in counts])
    anchor=us[0][0].copy()
    offset=sum((v-anchor).mean() for v in us)/2
    variance=sum(((v-anchor-offset)**2).mean() for v in us)/2
    scale=cp.sqrt(variance);constant=bool(scale==0)
    scale=cp.where(scale==0,CFG['cre']['constant_scale'],scale)
    z=(x-anchor-offset)/scale
    prior=float(cp.dot(w,y)/total)
    assert abs(float(w.sum())-total)<1e-6
    assert abs(float(cp.dot(w,z)/total))<1e-10
    record=dict(rows=total,correct=int(y.sum()),incorrect=total-int(y.sum()),domain_rows=counts,
        domain_weights=[total/(2*n) for n in counts],domain_total_weight=[total/2,total/2],weight_sum=total,
        anchor=float(anchor),mean_offset=float(offset),mean=float(anchor+offset),scale=float(scale),variance=float(variance),
        constant_input=constant,prior=prior,training_audits=audits,fit_partition='R_fit',
        objective='weighted BCE / N + beta1^2/(2*C*N); intercept unpenalized',C=CFG['cre']['C'],
        solver=CFG['logistic_solver'],closure_rows=CFG['batches']['logistic_closure_rows'],true_class_reweighting=False,
        scaler_arithmetic='P4 translation-centered equal-domain weighted population variance',beta0=None,beta1=None)
    if int(y.sum()) in [0,total]:
        return dict(record,status='NOT_FITTABLE_SINGLE_CORRECTNESS_CLASS',converged_gradient=False)
    sol=CFG['logistic_solver']
    tz=torch.from_dlpack(cp.ascontiguousarray(z[:,None]));ty=torch.from_dlpack(y);tw=torch.from_dlpack(w)
    theta=torch.full((2,),float(sol['initial_parameters']),dtype=torch.float64,device=sol['device'],requires_grad=True)
    optimizer=torch.optim.LBFGS([theta],max_iter=sol['max_iter'],tolerance_grad=sol['tolerance_grad'],
        tolerance_change=sol['tolerance_change'],history_size=sol['history_size'],line_search_fn=sol['line_search_fn'])
    calls=0;start=time.time()
    def closure():
        nonlocal calls
        calls+=1;optimizer.zero_grad(set_to_none=True)
        objective=torch.zeros((),device='cuda',dtype=torch.float64)
        for a in range(0,total,CFG['batches']['logistic_closure_rows']):
            b=min(a+CFG['batches']['logistic_closure_rows'],total)
            loss=(F.binary_cross_entropy_with_logits(tz[a:b]@theta[:-1]+theta[-1],ty[a:b],reduction='none')*tw[a:b]).sum()/total
            loss.backward();objective+=loss.detach()
        penalty=(theta[:-1]**2).sum()/(2*CFG['cre']['C']*total)
        penalty.backward()
        return objective+penalty.detach()
    optimizer.step(closure);objective=float(closure());grad=float(theta.grad.abs().max())
    assert bool(torch.isfinite(theta).all()) and np.isfinite(objective)
    state=optimizer.state[theta];converged=grad<=sol['tolerance_grad']
    return dict(record,status='COMPLETE' if converged else 'SOLVER_COMPLETION_CRITERION_NOT_MET',
        beta1=float(theta[0].detach()),beta0=float(theta[1].detach()),objective_value=objective,gradient_inf=grad,
        gradient_l2=float(torch.linalg.vector_norm(theta.grad)),converged_gradient=converged,iterations=state['n_iter'],
        function_evaluations=state['func_evals'],closure_calls=calls,seconds=time.time()-start)

def predict(u,fit):
    z=cp.ascontiguousarray(((u-fit['anchor']-fit['mean_offset'])/fit['scale'])[:,None])
    with torch.inference_mode():
        logits=torch.from_dlpack(z)@torch.tensor([fit['beta1']],device='cuda',dtype=torch.float64)+fit['beta0']
        p=cp.from_dlpack(torch.sigmoid(logits))
    assert bool(cp.isfinite(p).all()) and bool(((p>=0)&(p<=1)).all())
    return p

def grouped(score,correct,constant=False):
    if constant:
        order=None;starts=cp.asarray([0]);count=cp.asarray([len(score)],dtype=cp.int64)
        pos=cp.asarray([int(correct.sum())],dtype=cp.int64);levels=score[:1]
    else:
        order,starts,count,pos=ranking_groups(score,correct)
        levels=score[order[starts]]
    error=(count-pos).astype(cp.int64);cum=cp.cumsum(count);ce=cp.cumsum(error)
    return dict(order=order,starts=starts,count=count,error=error,cum=cum,ce=ce,levels=levels,
        AURC=float(grouped_aurc(count,error,harmonic_numbers(len(score)))))

def direction(u,baseline,fu,fit,key):
    order=cp.argsort(u)
    ss=baseline[order];ff=fu[order]
    db=ss[1:]-ss[:-1];df=ff[1:]-ff[:-1]
    bt=ss[1:]==ss[:-1];ft=ff[1:]==ff[:-1]
    beta=fit['beta1']
    violations=int((df<0).sum()) if beta>0 else int((df>0).sum()) if beta<0 else int((df!=0).sum())
    assert int((db>0).sum())==0 and violations==0
    added=int((~bt & ft).sum());split=int((bt & ~ft).sum())
    identity=bool(beta<0 and added==0 and split==0)
    return dict(**key,beta1=beta,theoretical_direction='same' if beta<0 else 'reverse' if beta>0 else 'constant',
        baseline_groups=int((~bt).sum())+1,fu_groups=int((~ft).sum())+1,
        added_tie_boundaries=added,split_baseline_tie_boundaries=split,monotonicity_violations=violations,
        exact_same_order_and_record_tie_partition=identity,
        fu_zero_count=int((fu==0).sum()),fu_one_count=int((fu==1).sum()),fu_min=float(fu.min()),fu_max=float(fu.max()),
        audit_scope='all records sorted by entropy; adjacent boundary equality and sign checked, not a sample')

def evaluate(d,m,o,fold,part,scope,fit,b1map,b1ops,amap,acats):
    key=dict(fold=fold,domain=d,partition=part,model=m,output=o)
    u,pred,y=load_data(d,m,o,scope);n=len(y);correct=pred==y
    base=1-u;fu=predict(u,fit) if fit['status']=='COMPLETE' else None
    scores={'one_minus_u':base,'prior':cp.full(n,fit['prior'],dtype=cp.float64)}
    if fu is not None: scores['fu']=fu
    audit=direction(u,base,fu,fit,key) if fu is not None else dict(**key,status=fit['status'])
    metrics=[];ops=[];cats=[];commons=[];curves={};regions=[]
    code=cp.asarray(np.load(ROOT/f'outputs/feature_store_v2/{d}_attack_code.npy',mmap_mode='r')[scope['rank_start']:scope['rank_end_exclusive']])
    book=read(ROOT/f'outputs/feature_store_v2/{d}_manifest.json')['attack_codebook']
    totals=cp.asnumpy(cp.bincount(code.astype(cp.int32),minlength=len(book)))
    attackcodes=[]
    for k,name in book.items():
        original=next(r for r in acats if r['domain']==d and r['original_category']==name)
        truth=int(original['binary_label'])
        assert bool((y[code==int(k)]==truth).all())
        if truth==1: attackcodes.append(int(k))
    assert sum(totals[c] for c in attackcodes)==int(y.sum())
    bn=int((y==0).sum());an=int(y.sum())
    for method,score in scores.items():
        group=None
        for stratum,mask in [('all',None),('Benign',y==0),('Attack',y==1)]:
            sc=score if mask is None else score[mask];cc=correct if mask is None else correct[mask]
            ns=len(cc);nc=int(cc.sum())
            if ns:
                gr=grouped(sc,cc,method=='prior');aurc=gr['AURC'];ng=len(gr['count'])
                if stratum=='all': group=gr
                else: del gr
            else: aurc=None;ng=0
            brier=nll=None
            if method!='one_minus_u' and ns:
                pc=cp.clip(sc,CFG['metrics']['nll_epsilon'],1-CFG['metrics']['nll_epsilon'])
                brier=float(cp.mean((sc-cc)**2));nll=float(-cp.mean(cc*cp.log(pc)+(1-cc)*cp.log1p(-pc)))
            metrics.append(dict(**key,method=method,stratum=stratum,rows=ns,correct=nc,incorrect=ns-nc,
                full_accept_error_rate=ratio(ns-nc,ns),AURC=aurc,Brier=brier,NLL=nll,score_tie_groups=ng,
                probability_target='correctness' if method!='one_minus_u' else 'ranking_only',
                undefined_reason='empty stratum' if not ns else ('not a correctness probability' if method=='one_minus_u' else None)))
        g=group;cum=g['cum'];ce=g['ce'];cnt=g['count'];er=g['error'];levels=g['levels']
        before=cum-cnt;eb=ce-er
        grid=cp.maximum(cp.arange(1,1001,dtype=cp.int64)*n//1000,1)
        j=cp.searchsorted(cum,grid,side='left');risk=(eb[j]+(grid-before[j])*er[j]/cnt[j])/grid
        curves[method+'_coverage']=cp.asnumpy(grid/n);curves[method+'_risk']=cp.asnumpy(risk)
        expected={}
        prior_accept=cp.zeros(n,dtype=cp.bool_)
        enc=code.astype(cp.int32)*2+pred.astype(cp.int32)
        for pct in PCTS:
            cap=n*pct//100;ix=int(cp.searchsorted(cum,cap,side='right'))-1
            cutoff=float(levels[ix]) if ix>=0 else None
            acc=score>=cutoff if cutoff is not None else cp.zeros(n,dtype=cp.bool_)
            count=int(acc.sum());assert count==(int(cum[ix]) if ix>=0 else 0) and count<=cap
            assert ix==len(cum)-1 or int(cum[ix+1])>cap
            assert not bool((prior_accept & ~acc).any());prior_accept=acc
            tn=int((acc & (y==0)&~pred).sum());fp=int((acc & (y==0)&pred).sum())
            fn=int((acc & (y==1)&~pred).sum());tp=int((acc & (y==1)&pred).sum())
            assert tn+fp+fn+tp==count and fp+fn==(int(ce[ix]) if ix>=0 else 0)
            op=dict(**key,method=method,requested_coverage=pct/100,accepted_count=count,actual_coverage=count/n,
                selective_risk=ratio(fp+fn,count),accepted_errors=fp+fn,TN=tn,FP=fp,FN=fn,TP=tp,
                benign_total=bn,benign_accepted=tn+fp,benign_coverage=ratio(tn+fp,bn),accepted_benign_fpr=ratio(fp,tn+fp),
                attack_total=an,attack_accepted=tp+fn,attack_coverage=ratio(tp+fn,an),accepted_attack_fnr=ratio(fn,tp+fn),
                effective_attack_detection=ratio(tp,an),wrong_accepted_attack_fraction=ratio(fn,an),
                attack_abstained=an-tp-fn,attack_abstention_fraction=ratio(an-tp-fn,an),
                attack_fraction_in_accepted=ratio(tp+fn,count),score_cutoff=cutoff,
                undefined_reason='empty accepted set' if count==0 else None)
            if part=='T_heldout':
                if pct==100:
                    old=amap[d,m,o]
                    assert [tn,fp,fn,tp]==[int(old[k]) for k in ['TN','FP','FN','TP']]
                if method=='one_minus_u':
                    old=b1ops[d,m,o,pct]
                    for k in ['accepted_count','TN','FP','FN','TP']:
                        assert int(op[k])==int(old[k])
                    assert op['selective_risk']==(None if old['selective_risk']=='NA' else float(old['selective_risk']))
            ops.append(op)
            bins=cp.asnumpy(cp.bincount(enc,weights=acc.astype(cp.int64),minlength=2*len(book))).astype(np.int64).reshape(-1,2)
            sumtp=sumfn=0
            for c in attackcodes:
                total=int(totals[c]);wrong,right=map(int,bins[c]);accepted=wrong+right
                assert accepted<=total;sumtp+=right;sumfn+=wrong
                cats.append(dict(**key,method=method,requested_coverage=pct/100,actual_overall_coverage=count/n,
                    original_attack_category=book[str(c)],category_total=total,accepted_count=accepted,category_coverage=ratio(accepted,total),
                    correct_accepted_attack_count=right,effective_category_detection=ratio(right,total),
                    wrong_accepted_benign_count=wrong,wrong_accepted_fraction=ratio(wrong,total),
                    abstained_count=total-accepted,abstained_fraction=ratio(total-accepted,total),
                    undefined_reason='zero category support' if total==0 else None))
            assert sumtp==tp and sumfn==fn
            gi=int(cp.searchsorted(cum,cap,side='left'))
            expected[pct]=float((eb[gi]+(cap-before[gi])*er[gi]/cnt[gi])/cap)
            commons.append(dict(**key,method=method,requested_coverage=pct/100,common_prefix_count=cap,common_coverage=cap/n,
                expected_risk=expected[pct],whole_group_coverage=count/n,whole_group_risk=op['selective_risk']))
        if part=='T_heldout' and method=='one_minus_u':
            assert abs(g['AURC']-float(b1map[d,m,o]['AURC']))<1e-12
        # Full attainable-endpoint region inventory, never a deployed threshold choice.
        if part=='T_heldout':
            if method=='prior':
                ca=cp.asarray([an]);ct=cp.asarray([int((pred & (y==1)).sum())])
            else:
                ca=cp.cumsum(cp.add.reduceat(y[g['order']].astype(cp.int64),g['starts']))
                ct=cp.cumsum(cp.add.reduceat((y[g['order']]&pred[g['order']]).astype(cp.int64),g['starts']))
            ok=(cum*10>=n)&(ce*20<=cum)
            starts=cp.asnumpy(cp.flatnonzero(ok&cp.concatenate((cp.asarray([True]),~ok[:-1]))))
            ends=cp.asnumpy(cp.flatnonzero(ok&cp.concatenate((~ok[1:],cp.asarray([True])))))
            for rid,(l,h) in enumerate(zip(starts,ends)):
                rr=dict(**key,method=method,region=rid+1,attainable_endpoints=int(h-l+1))
                for label,ix in [('start',int(l)),('end',int(h))]:
                    nc,ne,na,nt=map(int,[cum[ix],ce[ix],ca[ix],ct[ix]])
                    rr.update({label+'_coverage':nc/n,label+'_risk':ne/nc,label+'_effective_attack_detection':ratio(nt,an),
                        label+'_wrong_accepted_attack_fraction':ratio(na-nt,an),label+'_attack_coverage':ratio(na,an),
                        label+'_benign_coverage':ratio(nc-na,bn),label+'_accepted_benign_fpr':ratio(ne-(na-nt),nc-na)})
                regions.append(rr)
        del group,g
    if fu is not None:
        allm={r['method']:r for r in metrics if r['stratum']=='all'}
        audit['AURC_delta_fu_minus_fixed']=allm['fu']['AURC']-allm['one_minus_u']['AURC']
        if audit['exact_same_order_and_record_tie_partition']:
            assert abs(audit['AURC_delta_fu_minus_fixed'])<1e-12
    name=f'{fold}__{d}__{part}__{m}__{o}'
    np.savez_compressed(OUT/'curves'/f'{name}.npz',**curves)
    return metrics,ops,cats,commons,regions,audit

def main():
    if (OUT/'fit_manifest.csv').exists(): raise RuntimeError('Use a fresh run directory')
    start=time.time();before=inventory();write(OUT/'upstream_inventory.json',before)
    write(OUT/'fit_config.json',dict(original_config=str(ROOT/'configs/execution_v2.json'),
        logistic_solver=CFG['logistic_solver'],cre=CFG['cre'],batches=CFG['batches'],bootstrap=CFG['bootstrap'],metrics=CFG['metrics'],
        scope='B2 All49 u-only; post-P7 exploratory; do not invoke P4 family/hu/hus training'))
    prep=PreparedData(ROOT/'outputs/round1_data_prep_v1');sc=scopes(prep)
    save('evaluation_scope.csv',list(sc.values()))
    assert np.array_equal(cp.asnumpy(entropy(cp.asarray([0.,.5,1.]))),[0.,1.,0.])
    b1map={(r['domain'],r['model'],r['output']):r for r in rows(B1/'overall_selective_metrics.csv')}
    b1ops={(r['domain'],r['model'],r['output'],round(float(r['requested_coverage'])*100)):r for r in rows(B1/'operating_points.csv')}
    amap={(r['domain'],r['model'],r['output']):r for r in rows(A/'detector_raw_platt_metrics.csv')}
    acats=rows(A/'attack_category_recall.csv')
    fits={};manifest=[]
    # Freeze every fit before starting any R_cal/R_select/target f(u) performance evaluation.
    for fold,info in CFG['folds'].items():
        for m in MODELS:
            for o in ['Raw','Platt']:
                us=[];cs=[];audits=[]
                for d in info['development']:
                    u,pred,y=load_data(d,m,o,sc[d,'R_fit']);c=pred==y
                    us.append(u);cs.append(c)
                    audits.append(dict(sc[d,'R_fit'],correct=int(c.sum()),incorrect=int((~c).sum()),
                        benign_correct=int((c&(y==0)).sum()),attack_correct=int((c&(y==1)).sum())))
                print(f'FIT {fold}/{m}/{o} domains={info["development"]}',flush=True)
                f=fit_u(us,cs,audits);f.update(fold=fold,model=m,output=o,development=info['development'],target=info['heldout'],
                    saved_at_utc=datetime.now(timezone.utc).isoformat(),probability_target='base classification correctness')
                fits[fold,m,o]=f;write(OUT/'models'/f'{fold}__{m}__{o}.json',f)
                manifest.append(dict(fold=fold,model=m,output=o,target=info['heldout'],status=f['status'],rows=f['rows'],
                    correct=f['correct'],incorrect=f['incorrect'],prior=f['prior'],beta0=f['beta0'],beta1=f['beta1'],
                    anchor=f['anchor'],mean_offset=f['mean_offset'],mean=f['mean'],scale=f['scale'],constant_input=f['constant_input'],
                    domain_rows=json.dumps(f['domain_rows']),domain_weights=json.dumps(f['domain_weights']),
                    development=json.dumps(info['development']),training_audits=json.dumps(audits),
                    gradient_inf=f.get('gradient_inf'),converged_gradient=f['converged_gradient'],iterations=f.get('iterations'),
                    objective=f.get('objective_value'),C=f['C'],model_file=f'models/{fold}__{m}__{o}.json'))
                save('fit_manifest.csv',manifest)
                print(f'  {f["status"]} beta1={f["beta1"]} gradient={f.get("gradient_inf")}',flush=True)
                del us,cs,u,pred,y,c;release()
    write(OUT/'fits_frozen.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),models=len(fits),
        status='ALL_FITS_SAVED_BEFORE_ANY_B2_EVALUATION',source='only each folds R_fit; R_cal/R_select/target not used for fitting'))
    metrics=[];ops=[];cats=[];common=[];regions=[];directions=[]
    for (fold,m,o),fit in fits.items():
        tasks=[(d,p) for p in ['R_cal','R_select'] for d in fit['development']]+[(fit['target'],'T_heldout')]
        for d,part in tasks:
            print(f'EVAL {fold}/{m}/{o}/{d}/{part}',flush=True)
            me,op,ca,co,re,di=evaluate(d,m,o,fold,part,sc[d,part],fit,b1map,b1ops,amap,acats)
            metrics.extend(me);ops.extend(op);cats.extend(ca);common.extend(co);regions.extend(re);directions.append(di)
            save('development_target_metrics.csv',metrics);save('operating_points.csv',ops);save('attack_category_selective_metrics.csv',cats)
            save('common_coverage_risks.csv',common);save('entropy_direction_audit.csv',directions);save('diagnostic_regions.csv',regions)
            release()
    after=inventory();assert before==after,'Frozen upstream size/mtime changed'
    write(OUT/'point_validation.json',dict(status='POINT_ESTIMATES_COMPLETE',models=len(fits),
        model_statuses={s:sum(f['status']==s for f in fits.values()) for s in set(f['status'] for f in fits.values())},
        evaluations=len(directions),metric_rows=len(metrics),operating_points=len(ops),category_rows=len(cats),
        A_full_confusion=True,B1_fixed_AURC_and_operating_counts=True,category_balance=True,nested_acceptance=True,
        endpoints_and_ranges=True,domain_equal_weight=True,all_fits_before_evaluation=True,upstream_size_mtime_unchanged=True,
        upstream_files=len(before),seconds=time.time()-start,gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cupy=cp.__version__,torch=torch.__version__,numpy=np.__version__,bootstrap_pending=True))
    print('POINT ESTIMATES COMPLETE; bootstrap is a separate next step',flush=True)

if __name__=='__main__':main()
