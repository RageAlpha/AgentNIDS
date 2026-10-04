from __future__ import annotations

import time
import cupy as cp
import numpy as np
import torch
from torch.nn import functional as F

FEATURES = {'u':[1], 'hu':[0,1], 'hus':[0,1,2]}


def fit_cre_family(domain_x, domain_correct, config):
    """One immutable two-domain row/weight contract and scaler for all three CREs."""
    started = time.perf_counter()
    counts = [len(x) for x in domain_x]
    total = sum(counts)
    x = cp.concatenate(domain_x).astype(cp.float64)
    y = cp.concatenate(domain_correct).astype(cp.float64)
    weights = cp.concatenate([cp.full(n,total/(2*n),dtype=cp.float64) for n in counts])
    anchor = domain_x[0][0].copy()
    mean_offset = sum((v-anchor).mean(axis=0) for v in domain_x)/2
    mean = anchor + mean_offset
    variance = sum(((v-anchor-mean_offset)**2).mean(axis=0) for v in domain_x)/2
    scale = cp.sqrt(variance)
    constant = scale==0
    scale = cp.where(constant,config['cre']['constant_scale'],scale)
    standardized = (x-anchor-mean_offset)/scale
    prior = float(cp.dot(weights,y)/total)
    assert bool(cp.isfinite(standardized).all())
    assert abs(float(weights.sum())-total)<1e-6
    assert np.allclose(cp.asnumpy((standardized*weights[:,None]).sum(axis=0)/total),0,atol=1e-10)
    models = {}
    positive = int(y.sum())
    if positive in (0,total):
        return dict(status='NOT_FITTABLE_SINGLE_CORRECTNESS_CLASS',rows=total,positive=positive,prior=prior,models={})
    solver = config['logistic_solver']
    torch_y = torch.from_dlpack(y)
    torch_w = torch.from_dlpack(weights)
    for name,indices in FEATURES.items():
        fit_start = time.perf_counter()
        z = torch.from_dlpack(cp.ascontiguousarray(standardized[:,indices]))
        theta = torch.full((len(indices)+1,),float(solver['initial_parameters']),dtype=torch.float64,device=solver['device'],requires_grad=True)
        optimizer = torch.optim.LBFGS([theta],max_iter=solver['max_iter'],tolerance_grad=solver['tolerance_grad'],
            tolerance_change=solver['tolerance_change'],history_size=solver['history_size'],line_search_fn=solver['line_search_fn'])
        calls = 0
        chunk = config['batches']['logistic_closure_rows']
        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad(set_to_none=True)
            objective = torch.zeros((),device='cuda',dtype=torch.float64)
            for a in range(0,total,chunk):
                b=min(a+chunk,total)
                loss=(F.binary_cross_entropy_with_logits(z[a:b]@theta[:-1]+theta[-1],torch_y[a:b],reduction='none')*torch_w[a:b]).sum()/total
                loss.backward()
                objective += loss.detach()
            penalty=(theta[:-1]**2).sum()/(2*config['cre']['C']*total)
            penalty.backward()
            return objective+penalty.detach()
        optimizer.step(closure)
        objective=float(closure())
        gradient=theta.grad.detach()
        grad_inf=float(gradient.abs().max())
        assert bool(torch.isfinite(theta).all()) and np.isfinite(objective)
        state=optimizer.state[theta]
        models[name]=dict(coefficients=theta[:-1].detach().cpu().tolist(),intercept=float(theta[-1].detach()),
            indices=indices,iterations=state['n_iter'],function_evaluations=state['func_evals'],closure_calls=calls,
            objective=objective,gradient_inf=grad_inf,gradient_l2=float(torch.linalg.vector_norm(gradient)),
            converged_gradient=grad_inf<=solver['tolerance_grad'],device=str(theta.device),seconds=time.perf_counter()-fit_start,
            rows=total,weight_sum=float(weights.sum()),domain_rows=counts,solver=dict(solver),C=config['cre']['C'])
        print(f"CRE {name}: rows={total:,}, iter={state['n_iter']}, objective={objective:.9g}, gradient_inf={grad_inf:.3g}",flush=True)
        del optimizer,theta,z
    assert models['hu']['rows']==models['hus']['rows'] and models['hu']['solver']==models['hus']['solver']
    return dict(status='COMPLETE',rows=total,correct=positive,incorrect=total-positive,domain_rows=counts,
        domain_weight=[total/(2*n) for n in counts],weight_sum=total,prior=prior,
        feature_order=['h','u','s'],mean=cp.asnumpy(mean).tolist(),scale=cp.asnumpy(scale).tolist(),
        anchor=cp.asnumpy(anchor).tolist(),mean_offset=cp.asnumpy(mean_offset).tolist(),
        scaler_arithmetic='translation_centered_equal_domain_weighted_mean_and_population_variance',
        constant_features=np.asarray(['h','u','s'])[cp.asnumpy(constant)].tolist(),models=models,
        paired_rows_weights_and_scaler='shared_identical_objects_and_column_subsets',seconds=time.perf_counter()-started)


def predict_cre(x, fitted):
    normalized=(x-cp.asarray(fitted['anchor'])-cp.asarray(fitted['mean_offset']))/cp.asarray(fitted['scale'])
    outputs={'prior':cp.full(len(x),fitted['prior']),'raw_h':x[:,0],'one_minus_u':1-x[:,1]}
    for name,model in fitted['models'].items():
        z=cp.ascontiguousarray(normalized[:,model['indices']])
        with torch.inference_mode():
            logits=torch.from_dlpack(z)@torch.tensor(model['coefficients'],device='cuda',dtype=torch.float64)+model['intercept']
            outputs[name]=cp.from_dlpack(torch.sigmoid(logits))
    for p in outputs.values():
        assert bool(cp.isfinite(p).all()) and float(p.min())>=0 and float(p.max())<=1
    return outputs
