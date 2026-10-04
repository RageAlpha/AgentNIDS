import json
import time
from pathlib import Path
import cupy as cp
import numpy as np
from nids.models.base_models import MODEL_SPECS,load_model
from nids.reliability.round1_data import PRIMARY
from nids.replay.predictions import load_preprocessor,gpu_raw_probability,gpu_evidence,write_json
from nids.reliability.cre import predict_cre
from .evidence import post_features,primary_pre
from .glm import predict


def timing(fn,warmup=30,iterations=1000):
    for i in range(warmup):fn(i)
    values=[]
    for i in range(iterations):
        cp.cuda.Stream.null.synchronize();t=time.perf_counter_ns();fn(i);cp.cuda.Stream.null.synchronize();values.append((time.perf_counter_ns()-t)/1e6)
    return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples_ms=values)


def benchmark(c,data,output):
    root=Path(c['project_root']);source=root/c['outputs']['source'];pre=load_preprocessor(c)
    selected=[];identities=[]
    for d in data:
        rank=np.linspace(d.a,d.b-1,500,dtype=np.int64);x=np.load(root/c['outputs']['feature_store']/f'{d.domain}_X.npy',mmap_mode='r');selected.append(x[rank]);identities.extend(dict(domain=d.domain,rank=int(r)) for r in rank)
    x=np.concatenate(selected);result=dict(device='cuda:0',records=identities,models={})
    for mid in PRIMARY:
        spec=MODEL_SPECS[mid];folder=source/'models'/mid;m=json.loads((folder/'training_manifest.json').read_text());model=load_model(spec,folder/m['artifact']);cal=json.loads((source/'calibrators'/f'{mid}.json').read_text());h=cp.asarray([r['h'] for r in json.loads((source/'competence'/f'{mid}.json').read_text())['table']])
        def query(i):
            z=cp.ascontiguousarray(pre.transform(x[i:i+1],spec.standardized)[:,m['feature_indices']]);raw=gpu_raw_probability(model,spec,z);return [cp.asnumpy(v) for v in gpu_evidence(raw,cal,h,.5)]
        latency=timing(query)
        z=cp.ascontiguousarray(pre.transform(x,spec.standardized)[:,m['feature_indices']]);cp.cuda.Stream.null.synchronize();t=time.perf_counter();gpu_raw_probability(model,spec,z);cp.cuda.Stream.null.synchronize();batch=time.perf_counter()-t
        latency.update(batch_rows=1000,batch_detector_seconds=batch,batch_detector_flows_per_second=1000/batch)
        result['models'][mid]=latency;print(f'{mid} query median={latency["median_ms"]:.6f}ms',flush=True)
        del model
    medians=np.array([result['models'][m]['median_ms'] for m in PRIMARY]);result['normalized_cost']=(medians/np.median(medians)).tolist();result['query_median_ms']=medians.tolist()
    cached=data[0].cache[0]
    result['cache_read']=timing(lambda i:cached[data[0].a+i].copy())
    write_json(output/'latency.json',result);return result


def benchmark_coordinator(fitted,data,output):
    v=data[0].accessor(0,1);rows=cp.arange(1)
    for i in range(3):v.query(rows,i)
    def call(i):
        current=v.current(rows,cp.zeros(1,dtype=cp.int64));predict(current,fitted['corrective']['0_1']);predict(current,fitted['corrective']['0_2']);features=post_features(v.mask,v.p,v.r,v.s);predict(features,fitted['post'])
    result=timing(call);write_json(output/'coordinator_latency.json',result);return result


def benchmark_method_overheads(fitted,data,initial,latency,output):
    from .replay import route
    from .evidence import entropy
    kinds=['coordinator','learned','random','cheapest','uncertainty','best','uniform','static','confidence','soft','hard','fixed'];result={}
    d=data[0];v=d.accessor(0,1);rows=cp.arange(1)
    for j in range(3):v.query(rows,j)
    order=initial['order'];current=cp.asarray([order[0]]);all_p=v.p.copy();all_r=v.r.copy();random=cp.full((1,2),.5)
    for kind in kinds:
        stages=[]
        for stage in range(3):
            v.mask[:]=False;v.mask[:,order[:stage+1]]=True;current[:]=order[stage]
            def call(i):
                j=order[stage];r=primary_pre(cp.column_stack((v.h[:,j],v.u[:,j],v.s)),d.pre[j]);v.r[:,j]=r
                x=post_features(v.mask,v.p,v.r,v.s)
                if kind=='soft':p=predict(cp.column_stack((x[:,:9],x[:,-1])),fitted['soft_moe']);entropy(p)
                elif kind in ['uniform','static','confidence']:
                    weights=v.mask*(cp.ones(3) if kind=='uniform' else cp.asarray(fitted['static_weights']) if kind=='static' else 1-v.u);p=(weights*v.p).sum(1)/weights.sum(1);entropy(p)
                elif kind not in ['best','hard']:predict(x,fitted['post'])
                if stage<2 and kind not in ['best','uniform','static','confidence','soft'] and (kind!='hard' or stage==0):route(v,rows,current,kind,fitted,latency['normalized_cost'],stage,order,random)
            m=timing(call);stages.append(m)
        result[kind]=stages
    write_json(output/'method_overhead_latency.json',result)
    median_query=float(np.median(latency['query_median_ms']));latency['overhead_normalized']={k:[v['median_ms']/median_query for v in values] for k,values in result.items()}
    latency['normalization_ms']=median_query
    latency['budget_reference_full_cost']=sum(latency['normalized_cost'])+max(sum(v) for v in latency['overhead_normalized'].values())
    write_json(output/'cost_model.json',latency);return latency
