"""Resolve P2/P3 source MLP numerical differences without fitting or cache writes."""
from run_stage_a import ROOT, OUT, read, save
import numpy as np
import cupy as cp
from nids.data.partitions import PreparedData
from nids.p2 import SourceModelData
from nids.models.base_models import MODEL_SPECS, load_model
from nids.reliability.calibration import apply_platt
from nids.evaluation.source import binary_probability_metrics
from nids.replay.predictions import gpu_raw_probability, load_preprocessor

cfg=read('configs/execution_v2.json')
prep=PreparedData(ROOT/'outputs/round1_data_prep_v1')
data=SourceModelData(cfg,prep)
part=prep.partition('UNSW','source','S_source_test')
rows=[]
for m in ['mlp_Behavior','mlp_All49']:
    spec=MODEL_SPECS[m]
    manifest=read(f'outputs/source_v2/models/{m}/training_manifest.json')
    model=load_model(spec,ROOT/f'outputs/source_v2/models/{m}'/manifest['artifact'])
    indices=manifest['feature_indices']
    # Exactly the original partition-relative P2 batch size, preprocessing and layout.
    raw=data.predict_partition(model,spec,'S_source_test',indices)
    cal=read(f'outputs/source_v2/calibrators/{m}.json')
    calibrated=apply_platt(raw,cal)
    cached=np.load(ROOT/f'outputs/prediction_cache_v2/UNSW/{m}.npy',mmap_mode='r')[part.rank_start:part.rank_end_exclusive]
    historical=read(f'outputs/source_v2/source_metrics/{m}.json')
    for condition,p,field,key in [('Raw',raw,'raw_probability','raw'),('Platt',calibrated,'probability','calibrated')]:
        z=binary_probability_metrics(data.partition_labels('S_source_test'),p,.5,1e-15,15)
        for metric in ['auroc','average_precision','accuracy','balanced_accuracy','ece','brier']:
            rows.append(dict(model=m,output=condition,metric=metric,p2_reinference=z[metric],p2_saved=historical[key][metric],
                delta=z[metric]-historical[key][metric],p2_vs_p3_probability_max_abs=float(np.max(np.abs(p-cached[field]))),
                p2_vs_p3_different_probability_rows=int(np.count_nonzero(p!=cached[field])),
                p2_vs_p3_class_flips=int(np.count_nonzero((p>=.5)!=(cached[field]>=.5)))))
        assert z['confusion']==historical[key]['confusion']
    print(m,rows[-1],flush=True)
save('source_p2_inference_audit.csv',rows)

# Count clipped tails by true class; no alternate decision thresholds evaluated.
tails=[]
for scope in __import__('csv').DictReader((OUT/'evaluation_scope.csv').open()):
    d=scope['domain']; a=int(scope['rank_start']);b=int(scope['rank_end_exclusive'])
    y=np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
    for m in cfg['models']['schedule']:
        cache=np.load(ROOT/f'outputs/prediction_cache_v2/{d}/{m}.npy',mmap_mode='r')
        eps=read(f'outputs/source_v2/calibrators/{m}.json')['input_probability_epsilon']
        counts=np.zeros((2,3),dtype=np.int64)
        for lo in range(a,b,262144):
            hi=min(b,lo+262144); p=cache['raw_probability'][lo:hi]; yy=y[lo:hi]
            for t in [0,1]:
                counts[t]+=[np.count_nonzero(yy==t),np.count_nonzero((yy==t)&(p<eps)),np.count_nonzero((yy==t)&(p>1-eps))]
        for t in [0,1]:
            tails.append(dict(domain=d,model=m,true_class=['Benign','Attack'][t],rows=int(counts[t,0]),
                raw_below_epsilon=int(counts[t,1]),raw_above_one_minus_epsilon=int(counts[t,2]),epsilon=eps))
save('platt_clipped_tail_counts.csv',tails)
