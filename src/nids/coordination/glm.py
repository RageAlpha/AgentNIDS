import time
import cupy as cp
import numpy as np
import torch


def fit(factory,features,total_weight,config,name,soft=False):
    started=time.perf_counter();solver=config['logistic_solver'];d=len(features)
    anchor=None;offset=cp.zeros(d,dtype=cp.float64);positive=0.;rows=0
    for x,y,w in factory():
        if anchor is None:anchor=x[0].copy()
        offset+=w*(x-anchor).sum(0);positive+=float(w*y.sum());rows+=len(x)
    offset/=total_weight;variance=cp.zeros(d,dtype=cp.float64)
    for x,y,w in factory():variance+=w*((x-anchor-offset)**2).sum(0)
    scale=cp.sqrt(variance/total_weight);scale=cp.where(scale==0,1,scale)
    base=dict(name=name,features=features,rows=rows,weight_sum=total_weight,weighted_positive=positive,
        anchor=cp.asnumpy(anchor).tolist(),offset=cp.asnumpy(offset).tolist(),scale=cp.asnumpy(scale).tolist(),device='cuda:0',solver=solver,C=1)
    if positive==0 or positive==total_weight:
        base.update(status='CONSTANT_SINGLE_LABEL',constant=(positive+1)/(total_weight+2),seconds=time.perf_counter()-started);return base
    shape=(3,d+1) if soft else (d+1,)
    theta=torch.zeros(shape,dtype=torch.float64,device='cuda',requires_grad=True)
    opt=torch.optim.LBFGS([theta],max_iter=solver['max_iter'],tolerance_grad=solver['tolerance_grad'],tolerance_change=solver['tolerance_change'],history_size=solver['history_size'],line_search_fn=solver['line_search_fn'])
    calls=0
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad(set_to_none=True);objective=cp.zeros((),dtype=cp.float64)
        if soft:
            for x,y,w in factory():
                z=torch.from_dlpack(cp.ascontiguousarray((x-anchor-offset)/scale));mask=torch.from_dlpack(x[:,:3]>0);prob=torch.from_dlpack(cp.ascontiguousarray(x[:,3:6]));target=torch.from_dlpack(y)
                logits=z@theta[:,:-1].T+theta[:,-1];logits=logits.masked_fill(~mask,-torch.inf)
                fused=(torch.softmax(logits,1)*prob).sum(1).clamp(1e-15,1-1e-15)
                loss=(-(target*torch.log(fused)+(1-target)*torch.log1p(-fused))).sum()*w/total_weight;loss.backward();objective+=cp.from_dlpack(loss.detach())
            penalty=(theta[:,:-1]**2).sum()/(2*total_weight);penalty.backward();value=torch.from_dlpack(objective)+penalty.detach()
        else:
            t=cp.from_dlpack(theta.detach());gradient=cp.zeros(d+1,dtype=cp.float64)
            for x,y,w in factory():
                z=(x-anchor-offset)/scale;logits=z@t[:-1]+t[-1];prob=1/(1+cp.exp(-logits));res=(prob-y)*w/total_weight
                objective+=w*(cp.logaddexp(0,logits)-y*logits).sum()/total_weight
                gradient[:-1]+=z.T@res;gradient[-1]+=res.sum()
            objective+=(t[:-1]**2).sum()/(2*total_weight);gradient[:-1]+=t[:-1]/total_weight
            theta.grad=torch.from_dlpack(gradient);value=torch.from_dlpack(objective)
        return value
    opt.step(closure);loss=float(closure());grad=float(theta.grad.abs().max())
    base.update(status='CONVERGED' if grad<=solver['tolerance_grad'] else 'GRADIENT_CRITERION_NOT_MET',coefficients=theta.detach().cpu().tolist(),objective=loss,gradient_inf=grad,iterations=opt.state[theta]['n_iter'],closure_calls=calls,seconds=time.perf_counter()-started,soft=soft)
    print(f'{name}: {base["status"]}, rows={rows:,}, iter={base["iterations"]}, grad={grad:.3g}, {base["seconds"]:.2f}s',flush=True)
    assert base['status']=='CONVERGED'
    return base


_GPU_MODELS={}


def prepare(m):
    key=id(m)
    if key not in _GPU_MODELS:
        _GPU_MODELS[key]=(m,cp.asarray(m['anchor']),cp.asarray(m['offset']),cp.asarray(m['scale']),cp.asarray(m['coefficients']))
    return _GPU_MODELS[key][1:]


def predict(x,m):
    if m['status']=='CONSTANT_SINGLE_LABEL':return cp.full(len(x),m['constant'])
    anchor,offset,scale,t=prepare(m);z=(x-anchor-offset)/scale
    if m.get('soft',False):
        logits=z@t[:,:-1].T+t[:,-1];logits=cp.where(x[:,:3]>0,logits,-cp.inf);v=cp.exp(logits-logits.max(1,keepdims=True));v/=v.sum(1,keepdims=True)
        return (v*x[:,3:6]).sum(1)
    return 1/(1+cp.exp(-(z@t[:-1]+t[-1])))
