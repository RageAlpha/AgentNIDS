"""Exact mathematical checks: ties, duplicated blocks, unequal block length and metric semantics."""
import itertools
import json
import numpy as np
import cupy as cp
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from nids.evaluation.round1_metrics import probability_metrics,BlockRanking,harmonic_numbers
from nids.evaluation.round1_bootstrap import sampled_headroom,gate

config=json.load(open('configs/execution_v2.json'))

def exact_aurc(scores,correct):
    groups=[np.flatnonzero(scores==s).tolist() for s in sorted(set(scores),reverse=True)]
    risks=[]
    for order_groups in itertools.product(*[list(itertools.permutations(g)) for g in groups]):
        order=list(itertools.chain.from_iterable(order_groups))
        risks.append(np.mean(np.cumsum(1-correct[order])/np.arange(1,len(order)+1)))
    return np.mean(risks)

checks=[]
for scores,correct in [(np.array([.9,.9,.2,.2]),np.array([1,0,1,0])),(np.ones(4)*.3,np.array([1,1,0,0])),(np.array([.8,.4,.1]),np.array([1,0,1]))]:
    m=probability_metrics(cp.asarray(correct),cp.asarray(scores),config)
    np.testing.assert_allclose(m['AURC'],exact_aurc(scores,correct),atol=2e-14)
    np.testing.assert_allclose(m['AUROC'],roc_auc_score(correct,scores),atol=2e-14)
    np.testing.assert_allclose(m['AUPRC'],average_precision_score(correct,scores),atol=2e-14)
    np.testing.assert_allclose(m['NLL'],log_loss(correct,scores),atol=2e-14)
    np.testing.assert_allclose(m['Brier'],brier_score_loss(correct,scores),atol=2e-14)
checks.append('tie_expected_AURC_matches_exhaustive_permutations_and_standard_metrics')
scores=np.array([.9,.5,.5]);correct=np.array([0,1,0]);blocks=np.array([0,0,1])
r=BlockRanking(cp.asarray(scores),cp.asarray(correct),cp.asarray(blocks),2)
for multiplicity in [np.array([2,0]),np.array([0,2]),np.array([1,1])]:
    expanded=np.repeat(np.arange(3),multiplicity[blocks])
    actual=float(r.aurc(cp.asarray(multiplicity),harmonic_numbers(4)))
    np.testing.assert_allclose(actual,exact_aurc(scores[expanded],correct[expanded]),atol=2e-14)
checks.append('block_multiplicity_and_short_final_block_exact_expansion')
for counts in [[0,10,0,0,0,0,0,0],[0,0,10,0,0,0,0,0],[0,5,4,0,3,0,0,2]]:
    c=np.asarray(counts);errors=np.array([sum(c[k] for k in range(8) if k&(1<<i)) for i in range(3)])
    np.testing.assert_allclose(float(sampled_headroom(cp.asarray(c,dtype=cp.float64))),(errors.min()-c[7])/c.sum())
checks.append('G0_reselects_best_single_with_each_resample')
assert gate(.02,[.001,.03],.01,True)=='PASS'
assert gate(.002,[-.001,.009],.01,True)=='BELOW_MINIMUM_EFFECT'
assert gate(.02,[-.001,.03],.01,True)=='INCONCLUSIVE'
assert gate(.02,[.001,.03],.01,False)=='INCONCLUSIVE'
checks.append('frozen_gate_rules')
print(json.dumps(dict(status='PASS',checks=checks),indent=2))
