import itertools
import cupy as cp
from .glm import fit
from .evidence import POST_FEATURES,GATE_FEATURES,post_features
from nids.replay.predictions import write_json


def train_models(data,c,output):
    arrays=[d.training_arrays() for d in data];counts=[d.n for d in data];total=sum(counts);chunk=c['batches']['logistic_closure_rows']
    fitted={'corrective':{},'uncertainty':{},'hard':{}}
    for family in fitted:
        for i,j in itertools.permutations(range(3),2):
            def batches():
                for n,v in zip(counts,arrays):
                    for a in range(0,n,chunk):
                        b=min(a+chunk,n);x=cp.column_stack((v['h'][a:b,i],v['u'][a:b,i],v['s'][a:b],v['p'][a:b,i]>=.5))
                        correct_j=(v['p'][a:b,j]>=.5)==v['y'][a:b]
                        y=correct_j if family=='hard' else correct_j&((v['p'][a:b,i]>=.5)!=v['y'][a:b])
                        yield x[:,1:2] if family=='uncertainty' else x,y.astype(cp.float64),total/(2*n)
            name=f'{family}_{i}_{j}';m=fit(batches,['u_i'] if family=='uncertainty' else ['h_i','u_i','s','yhat_i'],total,c,name)
            fitted[family][f'{i}_{j}']=m;write_json(output/'models'/f'{name}.json',m)
    for soft,name in [(False,'post'),(True,'soft_moe')]:
        def batches():
            for n,v in zip(counts,arrays):
                for subset in range(1,8):
                    bits=cp.asarray([(subset>>i)&1 for i in range(3)],dtype=bool)
                    for a in range(0,n,chunk):
                        b=min(a+chunk,n);mask=cp.broadcast_to(bits,(b-a,3));x=post_features(mask,v['p'][a:b],v['r'][a:b],v['s'][a:b])
                        if soft:yield cp.column_stack((x[:,:9],x[:,-1])),v['y'][a:b].astype(cp.float64),total/(2*n)
                        else:yield x,((x[:,9]>=.5)==v['y'][a:b]).astype(cp.float64),total/(2*n)
        m=fit(batches,GATE_FEATURES if soft else POST_FEATURES,7*total,c,name,soft=soft);fitted[name]=m;write_json(output/'models'/f'{name}.json',m)
    fitted['static_weights']=[float(sum(((v['p'][:,i]>=.5)==v['y']).mean() for v in arrays)/2) for i in range(3)]
    write_json(output/'models/fitted.json',fitted)
    return fitted
