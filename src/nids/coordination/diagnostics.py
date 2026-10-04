import cupy as cp
from .glm import predict
from .evidence import post_features
from nids.evaluation.round1_metrics import probability_metrics
from nids.replay.predictions import write_json


def model_diagnostics(data,fitted,c,output):
    results={}
    for d in data:
        v=d.training_arrays();metrics={}
        for family in ['corrective','uncertainty','hard']:
            for pair,m in fitted[family].items():
                i,j=map(int,pair.split('_'));x=cp.column_stack((v['h'][:,i],v['u'][:,i],v['s'],v['p'][:,i]>=.5));correct=(v['p'][:,j]>=.5)==v['y'];y=correct if family=='hard' else correct&((v['p'][:,i]>=.5)!=v['y'])
                p=predict(x[:,1:2] if family=='uncertainty' else x,m);metrics[f'{family}/{pair}']=probability_metrics(y.astype(cp.uint8),p,c)
        for subset in range(1,8):
            mask=cp.broadcast_to(cp.asarray([(subset>>i)&1 for i in range(3)],dtype=bool),(d.n,3));x=post_features(mask,v['p'],v['r'],v['s']);p=predict(x,fitted['post']);y=(x[:,9]>=.5)==v['y'];metrics[f'post/subset_{subset}']=probability_metrics(y.astype(cp.uint8),p,c)
        results[d.domain]=metrics;write_json(output/f'{d.domain}_model_R_cal_diagnostics.json',metrics)
        del v;cp.get_default_memory_pool().free_all_blocks()
    return results


def query_diagnostics(data,folder,policies):
    import numpy as np
    results={}
    for name,points in policies.items():
        for point in points:
            domains=[];cap=point['cap'];policy=folder/'policies'/name
            for d in data:
                trace=np.load(policy/f'{d.domain}_cap{cap}_actual_query_trace.npy',mmap_mode='r');query=cp.asarray(trace['query']);prob=cp.asarray(trace['p']);y=d.labels();all_correct=cp.column_stack([cp.asarray(cache['predicted_class'][d.a:d.b])==y for cache in d.cache]);seen=cp.zeros((d.n,3),dtype=bool);rows=cp.arange(d.n);counts=[]
                for step in range(3):
                    executed=query[:,step]>=0
                    if step:
                        current_wrong=(prob[:,step-1]>=.5)!=y;new_correct=(prob[:,step]>=.5)==y;opportunity=current_wrong&(all_correct&~seen).any(1);hit=executed&current_wrong&new_correct
                        counts.append(dict(step=step+1,consulted=int(executed.sum()),corrective_hits=int(hit.sum()),oracle_opportunities=int((executed&opportunity).sum())))
                    seen[rows[executed],query[executed,step]]=True
                domains.append(dict(domain=d.domain,rows=d.n,steps=counts))
            results[f'{name}/cap{cap}']=dict(domains=domains,oracle_access='offline_diagnostic_after_policy_and_actual_query_trace_frozen')
    write_json(folder/'G2_query_diagnostics.json',results)
