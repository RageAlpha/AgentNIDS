from pathlib import Path
import json
import time
import cupy as cp
import xgboost as xgb
from .round1_data import PRIMARY

METHODS=['prior','raw_h','hu','hus','one_minus_u']


def nonlinear_fitter(output):
    detectors=iter(PRIMARY)
    def fit(domain_x,domain_correct,config):
        started=time.perf_counter();mid=next(detectors);folder=Path(output)/'models';folder.mkdir(parents=True,exist_ok=True)
        counts=[len(x) for x in domain_x];total=sum(counts)
        x=cp.concatenate(domain_x).astype(cp.float32);y=cp.concatenate(domain_correct).astype(cp.float32)
        weights=cp.concatenate([cp.full(n,total/(2*n),dtype=cp.float32) for n in counts])
        positive=int(y.sum(dtype=cp.float64));assert 0<positive<total
        prior=float(sum(c.mean(dtype=cp.float64) for c in domain_correct)/2)
        base=config['models']['xgb'];setting=config['cre']['nonlinear_sensitivity']
        params={k:v for k,v in base.items() if k not in ['backend','views','n_estimators']}
        params.update(max_depth=setting['max_depth'],learning_rate=setting['learning_rate'],seed=config['seed'],eval_metric='logloss')
        models={}
        for method,indices in [('hu',[0,1]),('hus',[0,1,2])]:
            t=time.perf_counter();z=cp.ascontiguousarray(x[:,indices])
            matrix=xgb.QuantileDMatrix(z,label=y,weight=weights,max_bin=params['max_bin'])
            history={};booster=xgb.train(params,matrix,num_boost_round=setting['n_estimators'],evals=[(matrix,'R_fit')],evals_result=history,verbose_eval=False)
            actual=json.loads(booster.save_config())
            assert actual['learner']['generic_param']['device']=='cuda:0'
            assert booster.num_boosted_rounds()==setting['n_estimators']
            path=folder/f'{mid}_{method}.ubj';booster.save_model(path)
            models[method]=dict(path=str(path.resolve()),indices=indices,parameters=params,rounds=booster.num_boosted_rounds(),training_complete=True,
                device='cuda:0',rows=total,domain_rows=counts,weight_sum=float(weights.sum(dtype=cp.float64)),history=history,seconds=time.perf_counter()-t,actual_config=actual)
            print(f'{mid}/{method}: CUDA XGBoost {booster.num_boosted_rounds()} rounds, {time.perf_counter()-t:.2f}s',flush=True)
            del matrix,booster,z
        return dict(status='COMPLETE',rows=total,correct=positive,incorrect=total-positive,domain_rows=counts,domain_weight=[total/(2*n) for n in counts],
            weight_dtype='float32',weight_sum=float(weights.sum(dtype=cp.float64)),prior=prior,models=models,seconds=time.perf_counter()-started,
            paired_rows_weights_and_features='shared_identical_GPU_arrays_and_column_subsets',scaler='tree_inputs_original_h_u_shift')
    return fit


def predict_nonlinear(x,fitted):
    result={'prior':cp.full(len(x),fitted['prior']),'raw_h':x[:,0],'one_minus_u':1-x[:,1]}
    for method,m in fitted['models'].items():
        booster=xgb.Booster();booster.load_model(m['path']);booster.set_param({'device':'cuda:0'})
        result[method]=booster.inplace_predict(cp.ascontiguousarray(x[:,m['indices']],dtype=cp.float32)).astype(cp.float64)
    for v in result.values():assert bool(cp.isfinite(v).all()) and float(v.min())>=0 and float(v.max())<=1
    return result
