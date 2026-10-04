import time
import cupy as cp
import numpy as np
from .glm import predict
from .evidence import post_features,entropy

TRACE=np.dtype([('query', 'i1',(3,)),('p','f8',(3,)),('u','f8',(3,)),('h','f8',(3,)),('pre','f8',(3,)),('fused','f8',(3,)),('reliability','f8',(3,)),('calls','u1',(3,)),('cost','f8',(3,)),('utility','f8',(2,3))])


def route(v,rows,current,kind,fitted,costs,step,order,random_values):
    n=len(rows);scores=cp.full((n,3),-cp.inf);seen=v.mask[rows];features=v.current(rows,current)
    if kind in ['coordinator','learned','uncertainty','hard']:
        family={'coordinator':'corrective','learned':'corrective','uncertainty':'uncertainty','hard':'hard'}[kind]
        for i in range(3):
            local=cp.where(current==i)[0]
            if not len(local):continue
            for j in range(3):
                if i==j:
                    if kind=='hard':scores[local,j]=v.r[rows[local],j]
                    continue
                x=features[local,1:2] if kind=='uncertainty' else features[local]
                scores[local,j]=predict(x,fitted[family][f'{i}_{j}'])/(costs[j] if kind=='coordinator' else 1)
    elif kind=='cheapest':scores=cp.broadcast_to(-cp.asarray(costs),(n,3)).copy()
    elif kind=='random':
        available=cp.where(~seen,cp.arange(3),3);available=cp.sort(available,axis=1);k=(random_values[rows,step]*(~seen).sum(1)).astype(cp.int64);choice=available[cp.arange(n),k];scores[cp.arange(n),choice]=1
    else:
        for position,j in enumerate(order):scores[:,j]=-position
    if kind!='hard':scores=cp.where(seen,-cp.inf,scores)
    return scores.argmax(1),scores


def trajectory(data,fitted,latency,initial,kind,output,order=None,chunk=262144,cap=3,tau_stop=None,record_sink=None):
    started=time.perf_counter();order=list(order) if order is not None else initial['order'];first=order[0] if kind=='fixed' else initial['selected'];costs=latency['normalized_cost']
    trace=np.lib.format.open_memmap(output,mode='w+',dtype=TRACE,shape=(data.n,)) if record_sink is None else None
    rng=cp.random.RandomState(20260905+['CIC','BoT','ToN'].index(data.domain));audit_reads=0
    for start in range(0,data.n,chunk):
        end=min(start+chunk,data.n);v=data.accessor(start,end);n=end-start;rows=cp.arange(n);current=cp.full(n,first,dtype=cp.int64);active=rows;random_values=rng.random_sample((n,2));fields={name:cp.zeros((n,)+TRACE.fields[name][0].shape,dtype=TRACE.fields[name][0].base) for name in TRACE.names};fields['query'][:]=-1
        for step in range(3):
            for j in range(3):
                selected=active[current[active]==j]
                if len(selected):v.query(selected,j)
            for key,array in [('p',v.p),('u',v.u),('h',v.h),('pre',v.r)]:fields[key][active,step]=array[active,current[active]]
            fields['query'][active,step]=current[active]
            features=post_features(v.mask,v.p,v.r,v.s);fused=features[:,9]
            if kind in ['uniform','static','confidence','soft']:
                if kind=='soft':fused=predict(cp.column_stack((features[:,:9],features[:,-1])),fitted['soft_moe'])
                else:
                    w=v.mask*(cp.ones(3) if kind=='uniform' else cp.asarray(fitted['static_weights']) if kind=='static' else 1-v.u);denom=w.sum(1);fused=cp.where(denom>0,(w*v.p).sum(1)/cp.where(denom>0,denom,1),(v.mask*v.p).sum(1)/v.mask.sum(1))
                reliability=1-entropy(fused)
            elif kind in ['best','hard']:
                fused=v.p[rows,current];reliability=v.r[rows,current]
            else:reliability=predict(features,fitted['post'])
            fields['fused'][:,step]=fused;fields['reliability'][:,step]=reliability;fields['calls'][:,step]=v.mask.sum(1);fields['cost'][:,step]=(v.mask*cp.asarray(costs)).sum(1)
            overhead=cp.asarray(latency['overhead_normalized'][kind]);calls=v.mask.sum(1).astype(cp.int64)
            fields['cost'][:,step]+=cp.cumsum(overhead)[calls-1]+latency['shift_normalized'][data.domain]
            if step<2:
                if tau_stop is not None and step<cap-1:
                    active=active[reliability[active]<tau_stop]
                if step>=cap-1:active=cp.empty(0,dtype=cp.int64)
                if kind=='best' or (kind=='hard' and step==1):active=cp.empty(0,dtype=cp.int64)
                elif len(active):
                    chosen,utilities=route(v,active,current[active],kind,fitted,costs,step,order,random_values)
                    fields['utility'][active,step]=cp.where(cp.isfinite(utilities),utilities,0)
                    if kind=='hard':
                        keep=~v.mask[active,chosen];active=active[keep];chosen=chosen[keep]
                    current[active]=chosen
            assert bool(cp.isfinite(fused).all()) and bool(cp.isfinite(reliability).all())
        if record_sink is None:
            for key,value in fields.items():trace[key][start:end]=cp.asnumpy(value)
        else:
            record_sink(start,end,{key:cp.asnumpy(value) for key,value in fields.items()})
        audit_reads+=sum(n for _,n in v.reads)
    if trace is not None:trace.flush()
    return dict(rows=data.n,seconds=time.perf_counter()-started,detector_record_reads=audit_reads,visibility='query_only_accessor',cap=cap,tau_stop=tau_stop,path=str(output))
