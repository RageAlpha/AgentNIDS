from __future__ import annotations

import cupy as cp
import numpy as np


def harmonic_numbers(n):
    return cp.concatenate((cp.zeros(1,dtype=cp.float64),cp.cumsum(1/cp.arange(1,n+1,dtype=cp.float64))))


def grouped_aurc(count,error,harmonic):
    cumulative=cp.cumsum(count,dtype=cp.int64)
    before=cumulative-count
    errors_before=cp.cumsum(error,dtype=cp.float64)-error
    rate=error/cp.maximum(count,1)
    contribution=error+(errors_before-before*rate)*(harmonic[cumulative]-harmonic[before])
    return contribution.sum()/cumulative[-1]


def ranking_groups(score,positive):
    order=cp.argsort(-score)
    sorted_score=score[order]
    starts=cp.concatenate((cp.zeros(1,dtype=cp.int64),cp.flatnonzero(sorted_score[1:]!=sorted_score[:-1])+1))
    count=cp.diff(cp.concatenate((starts,cp.asarray([len(score)])))).astype(cp.int64)
    pos=cp.add.reduceat(positive[order].astype(cp.float64),starts)
    return order,starts,count,pos


def probability_metrics(positive,score,config,ranking_only=False):
    n=len(score)
    if n==0:
        return dict(rows=0,status='EMPTY_STRATUM')
    positive=positive.astype(cp.float64)
    order,starts,count,pos=ranking_groups(score,positive)
    total_pos=float(pos.sum()); total_neg=n-total_pos
    cumpos=cp.cumsum(pos); cumcount=cp.cumsum(count)
    auroc=None
    if total_pos>0 and total_neg>0:
        neg=count-pos
        auroc=float(cp.sum(neg*(cumpos-pos/2))/(total_pos*total_neg))
    ap=float(cp.sum((pos/total_pos)*(cumpos/cumcount))) if total_pos else None
    h=harmonic_numbers(n)
    aurc=float(grouped_aurc(count,count-pos,h))
    accepted=cp.maximum(cp.rint(cp.linspace(.01,1,100)*n).astype(cp.int64),1)
    group=cp.searchsorted(cumcount,accepted)
    before=cumcount-count
    error=count-pos
    errors_before=cp.cumsum(error)-error
    risks=(errors_before[group]+(accepted-before[group])*error[group]/count[group])/accepted
    result=dict(rows=n,positive=int(total_pos),negative=int(total_neg),AURC=aurc,AUROC=auroc,AUPRC=ap,
        score_tie_groups=len(count),risk_coverage=dict(accepted=cp.asnumpy(accepted).tolist(),risk=cp.asnumpy(risks).tolist()),
        undefined_metrics_reason='single_target_class' if not(total_pos>0 and total_neg>0) else None)
    if not ranking_only:
        eps=config['metrics']['nll_epsilon']; bins=config['metrics']['ece_bins']
        p=cp.clip(score,eps,1-eps)
        idx=cp.minimum((score*bins).astype(cp.int32),bins-1)
        bcount=cp.bincount(idx,minlength=bins)
        bpos=cp.bincount(idx,weights=positive,minlength=bins)
        bprob=cp.bincount(idx,weights=score,minlength=bins)
        result.update(Brier=float(cp.mean((score-positive)**2)),NLL=float(-cp.mean(positive*cp.log(p)+(1-positive)*cp.log1p(-p))),
                      ECE=float(cp.abs(bpos-bprob).sum()/n),ece_bins=bins,
                      calibration_bins=dict(count=cp.asnumpy(bcount).tolist(),positive_sum=cp.asnumpy(bpos).tolist(),probability_sum=cp.asnumpy(bprob).tolist()))
    return result


def detection_metrics(labels,probability,config):
    pred=probability>=config['competence']['attack_threshold']
    correct=pred==labels
    tp=int(cp.sum(pred & (labels==1))); fp=int(cp.sum(pred & (labels==0)))
    fn=int(cp.sum(~pred & (labels==1))); tn=int(cp.sum(~pred & (labels==0)))
    p=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    tnr=tn/(tn+fp) if tn+fp else None
    result=probability_metrics(labels,probability,config)
    result.pop('AURC');result.pop('risk_coverage')
    result.update(TP=tp,FP=fp,FN=fn,TN=tn,accuracy=float(correct.mean()),error_rate=float((~correct).mean()),
        precision=p,recall=recall,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        balanced_accuracy=(recall+tnr)/2 if recall is not None and tnr is not None else None)
    return result


def complementarity(errors,model_ids):
    n=len(errors)
    if n==0:
        return dict(rows=0,status='EMPTY_STRATUM')
    rates=errors.mean(axis=0)
    all_wrong=errors.all(axis=1)
    best=int(rates.argmin())
    oracle=float(all_wrong.mean())
    pairs=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        a=errors[:,i].astype(cp.float64);b=errors[:,j].astype(cp.float64)
        va=float(a.var());vb=float(b.var())
        pairs.append(dict(models=[model_ids[i],model_ids[j]],disagreement=float((a!=b).mean()),
            double_fault=float((a*b).mean()),error_correlation=float(((a-a.mean())*(b-b.mean())).mean())/(va*vb)**.5 if va*vb>0 else None))
    return dict(rows=n,error_rates=cp.asnumpy(rates).tolist(),best_single=model_ids[best],best_single_error=float(rates[best]),
        all_three_wrong_rate=oracle,all_three_wrong_count=int(all_wrong.sum()),oracle_accuracy=1-oracle,
        oracle_headroom=float(rates[best])-oracle,disagreement_any=float((errors.any(axis=1)&~all_wrong).mean()),
        recoverable_error_counts=[int((errors[:,i]&~all_wrong).sum()) for i in range(3)],
        recoverable_error_rates=[float((errors[:,i]&~all_wrong).mean()) for i in range(3)],pairs=pairs)


class BlockRanking:
    """Compress identical (score tie group, original block) cells without approximation."""
    def __init__(self,score,correct,block_id,blocks):
        order,starts,count,pos=ranking_groups(score,correct)
        group=cp.repeat(cp.arange(len(count),dtype=cp.int64),count)
        encoded=group*blocks+block_id[order]
        keys,inverse,counts=cp.unique(encoded,return_inverse=True,return_counts=True)
        errors=cp.bincount(inverse,weights=(1-correct[order]).astype(cp.float64),minlength=len(keys))
        groups=keys//blocks
        self.block=(keys%blocks).astype(cp.int32)
        self.count=counts.astype(cp.int64)
        self.error=errors
        self.starts=cp.concatenate((cp.zeros(1,dtype=cp.int64),cp.flatnonzero(groups[1:]!=groups[:-1])+1))
        self.groups=len(count)
        self.entries=len(keys)

    def aurc(self,multiplicity,harmonic):
        weights=multiplicity[self.block]
        counts=cp.add.reduceat(weights*self.count,self.starts)
        errors=cp.add.reduceat(weights*self.error,self.starts)
        return grouped_aurc(counts,errors,harmonic)
