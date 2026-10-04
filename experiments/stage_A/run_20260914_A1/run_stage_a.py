"""Post-unblinding Stage A. Read frozen assets, write only this run directory."""
from pathlib import Path
import csv
import json
import gc
import time
import sys
import numpy as np
import cupy as cp
import torch
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import checked_windows
from nids.evaluation.round1_metrics import detection_metrics
from nids.models.base_models import MODEL_SPECS, load_model

def read(p):
    return json.loads((ROOT/p).read_text())

def save(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: 'NA' if v is None else v for k,v in r.items()} for r in rows])

def main():
    start = time.time()
    cfg = read('configs/execution_v2.json')
    prep = PreparedData(Path(cfg['prepared_run']))
    models = cfg['models']['schedule']
    store = ROOT/cfg['outputs']['feature_store']
    source = ROOT/cfg['outputs']['source']
    p3 = read('outputs/round1_v2/P3/P3_manifest.json')
    feature_order = read(store/'feature_store_manifest.json')['feature_order']
    metrics, categories, flips, audits, scopes, comparisons, histrows, assets = [], [], [], [], [], [], [], []
    histograms = {}
    for m in models:
        manifest = read(source/'models'/m/'training_manifest.json')
        spec = MODEL_SPECS[m]
        assert manifest['parameters'] == cfg['models'][spec.family]
        assert manifest['feature_names'] == [feature_order[i] for i in manifest['feature_indices']]
        model = load_model(spec, source/'models'/m/manifest['artifact'])
        classes = cp.asnumpy(cp.asarray(model.classes_)).tolist() if spec.family != 'mlp' else None
        assert classes is None or classes == [0,1]
        cal = read(source/'calibrators'/f'{m}.json')
        assert cal['input'] == 'logit_raw_attack_probability'
        assets.append(dict(model=m, view=spec.view, feature_count=len(manifest['feature_indices']),
            classes=classes if classes is not None else 'NA: single BCE logit, sigmoid gives P(Label=1)',
            artifact=str(source/'models'/m/manifest['artifact']), calibrator=str(source/'calibrators'/f'{m}.json'),
            coefficient=cal['coefficient'], intercept=cal['intercept'],
            raw_boundary_for_platt_half=float(1/(1+np.exp(cal['intercept']/cal['coefficient']))),
            native_decision='Not evaluated; historical P2/P7 explicitly use probability >= 0.5'))
        del model
    save('model_asset_audit.csv', assets)
    for domain, fold in [('UNSW',None),('ToN','F1'),('CIC','F3'),('BoT','F2')]:
        if fold:
            shiftrows = pq.read_table(ROOT/cfg['outputs']['shift']/'10k'/f'{domain}_windows.parquet').to_pylist()
            part, windows = checked_windows(prep,domain,shiftrows)
            a,b = part.rank_start,windows[-1]['rank_end_exclusive']
            old = read(f'outputs/outer_v2/P7/{fold}/round1/metrics.json')
            assert old['audit']['rows'] == b-a
            cursor = a
            order = prep.sorted_rows(domain)
            for batch in pq.ParquetFile(ROOT/f'outputs/outer_v2/P7/{fold}/identities.parquet').iter_batches(
                    batch_size=262144, columns=['rank','original_row','record_id']):
                end = cursor+batch.num_rows
                assert np.array_equal(batch.column('rank').to_numpy(), np.arange(cursor,end))
                assert np.array_equal(batch.column('original_row').to_numpy(), order[cursor:end])
                expected = pc.binary_join_element_wise(pc.cast(batch.column('original_row'),'string'), '', '')
                expected = pc.binary_join_element_wise(domain+':',expected,'')
                assert pc.all(pc.equal(expected,batch.column('record_id'))).as_py()
                cursor = end
            assert cursor == b
        else:
            part = prep.partition(domain,'source','S_source_test')
            a,b = part.rank_start,part.rank_end_exclusive
            windows = []
        n = b-a
        print(f'{domain}: scope ranks [{a},{b}), {n:,} rows; identities checked',flush=True)
        ycpu = np.load(store/f'{domain}_label.npy',mmap_mode='r')
        codecpu = np.load(store/f'{domain}_attack_code.npy',mmap_mode='r')
        y = cp.asarray(ycpu[a:b]); code = cp.asarray(codecpu[a:b])
        book = read(store/f'{domain}_manifest.json')['attack_codebook']
        assert bool(cp.all((y==0)|(y==1)))
        assert bool(cp.all((code>=0)&(code<len(book))))
        category_binary = []
        for k,name in book.items():
            yy = y[code==int(k)]
            values = cp.asnumpy(cp.unique(yy)).tolist()
            assert len(values) <= 1
            if len(values):
                category_binary.append((int(k),name,values[0],len(yy)))
            else:
                original_values=np.unique(ycpu[codecpu==int(k)]).tolist()
                assert len(original_values)==1
                category_binary.append((int(k),name,original_values[0],0))
        assert sum(z[3] for z in category_binary if z[2]==1) == int(y.sum())
        assert sum(z[3] for z in category_binary if z[2]==0) == int((y==0).sum())
        if fold:
            assert old['audit']['attack'] == int(y.sum())
            assert old['audit']['benign'] == int((y==0).sum())
        scopes.append(dict(domain=domain,partition=part.name,fold=fold or 'source',rank_start=a,rank_end_exclusive=b,
            rows=n,benign=int((y==0).sum()),attack=int(y.sum()),complete_windows=len(windows) if fold else 'NA',
            excluded_tail=part.rank_end_exclusive-b,source_tail_included=n%10000 if not fold else 0,
            identity_check='all P7 rank/original_row/record_id' if fold else 'P2 source interval + P1/P3 rank contract'))
        domain_manifest = read(ROOT/cfg['outputs']['base_predictions']/domain/'manifest.json')
        assert domain_manifest['row_order']=='availability_rank'
        assert Path(domain_manifest['original_row_index']) == prep.root/f'{domain}_sorted_original_rows.npy'
        for m in models:
            print(f'{domain}/{m}: computing Raw + Platt',flush=True)
            train = read(source/'models'/m/'training_manifest.json')
            cachepath = ROOT/cfg['outputs']['base_predictions']/domain/f'{m}.npy'
            cache = np.load(cachepath,mmap_mode='r')
            manifest = read(cachepath.with_name(m+'_manifest.json'))
            assert manifest['model_id']==m and manifest['domain']==domain
            assert manifest['feature_indices']==train['feature_indices']
            assert manifest['feature_names']==train['feature_names']
            assert len(cache)==len(ycpu)==domain_manifest['rows']==manifest['rows']
            assert cache.dtype.names==('raw_probability','probability','predicted_class','u','h')
            assert manifest in domain_manifest['models']
            cal = read(source/'calibrators'/f'{m}.json')
            raw,platt = cp.empty(n,dtype=cp.float64),cp.empty(n,dtype=cp.float64)
            maxerr = 0.
            old_class_mismatch = 0
            for lo in range(0,n,262144):
                hi=min(n,lo+262144)
                raw[lo:hi]=cp.asarray(cache['raw_probability'][a+lo:a+hi])
                platt[lo:hi]=cp.asarray(cache['probability'][a+lo:a+hi])
                r=cp.clip(raw[lo:hi],cal['input_probability_epsilon'],1-cal['input_probability_epsilon'])
                logit=cp.log(r)-cp.log1p(-r)
                with torch.inference_mode():
                    reconstructed=cp.from_dlpack(torch.sigmoid(torch.from_dlpack(logit)*cal['coefficient']+cal['intercept']))
                maxerr=max(maxerr,float(cp.max(cp.abs(reconstructed-platt[lo:hi]))))
                old_class_mismatch += int(cp.count_nonzero((platt[lo:hi]>=.5)!=cp.asarray(cache['predicted_class'][a+lo:a+hi])))
            assert maxerr < 1e-12 and old_class_mismatch==0
            assert bool(cp.isfinite(raw).all() & cp.isfinite(platt).all())
            assert 0<=float(raw.min())<=float(raw.max())<=1 and 0<=float(platt.min())<=float(platt.max())<=1
            audits.append(dict(domain=domain,model=m,rows=n,platt_reconstruction_max_error=maxerr,
                cached_class_mismatches=old_class_mismatch,raw_min=float(raw.min()),raw_max=float(raw.max()),
                platt_min=float(platt.min()),platt_max=float(platt.max()),raw_equals_half=int((raw==.5).sum()),
                platt_equals_half=int((platt==.5).sum()),range_finite_mapping='PASS'))
            for truth,name in [(0,'Benign'),(1,'Attack')]:
                mask=y==truth
                good=(raw>=.5)==y; after=(platt>=.5)==y
                count=int(mask.sum())
                fixed=int((mask & ~good & after).sum()); broken=int((mask & good & ~after).sum())
                flips.append(dict(domain=domain,model=m,true_class=name,rows=count,raw_wrong_platt_right=fixed,
                    raw_wrong_platt_right_rate=fixed/count if count else None,raw_right_platt_wrong=broken,
                    raw_right_platt_wrong_rate=broken/count if count else None,denominator='all records in true class'))
            for output,p in [('Raw',raw),('Platt',platt)]:
                z=detection_metrics(y,p,cfg)
                assert sum(z[k] for k in ['TN','FP','FN','TP'])==n
                row=dict(domain=domain,model=m,feature_view=train['view'],feature_count=train['feature_count'],
                    output=output,threshold=.5,rows=n,benign=int((y==0).sum()),attack=int(y.sum()),
                    **{k:z[k] for k in ['TN','FP','FN','TP']},FPR=z['FP']/(z['FP']+z['TN']) if z['FP']+z['TN'] else None,
                    Recall=z['recall'],Accuracy=z['accuracy'],Balanced_accuracy=z['balanced_accuracy'],
                    AUROC=z['AUROC'],Average_precision=z['AUPRC'],ECE=z['ECE'],Brier=z['Brier'],
                    undefined_reason=z['undefined_metrics_reason'])
                metrics.append(row)
                for k,name,truth,count in category_binary:
                    pred=int(((code==k)&(p>=.5)).sum())
                    categories.append(dict(domain=domain,model=m,output=output,original_category=name,binary_label=truth,
                        rows=count,predicted_attack=pred,attack_detection_rate=pred/count if count else None,
                        attack_category_recall=pred/count if truth==1 and count else None,
                        note='NA: zero records in evaluation scope' if not count else ('binary attack recall, not family classification' if truth==1 else 'Benign: attack rate is FPR; recall NA')))
                bins=z['calibration_bins']
                for j in range(15):
                    histrows.append(dict(domain=domain,model=m,output=output,bin=j,left=j/15,right=(j+1)/15,
                        right_closed=j==14,count=bins['count'][j],attack_sum=bins['positive_sum'][j],probability_sum=bins['probability_sum'][j]))
                if domain=='ToN' and m.startswith(('rf','xgb')):
                    histograms[(m,output)] = [cp.asnumpy(cp.histogram(p[y==t],bins=cp.linspace(0,1,51))[0]) for t in [0,1]]
                if fold and output=='Platt':
                    historical=old['detectors']['all'][m]
                    mapping={k:k for k in ['TN','FP','FN','TP','AUROC','ECE','Brier']}
                    mapping.update(Recall='recall',Accuracy='accuracy',Balanced_accuracy='balanced_accuracy',Average_precision='AUPRC')
                    for newkey,oldkey in mapping.items():
                        diff=row[newkey]-historical[oldkey]
                        comparisons.append(dict(domain=domain,model=m,output=output,metric=newkey,current=row[newkey],historical=historical[oldkey],delta=diff,source='P7',match=abs(diff)<1e-12))
                        assert abs(diff)<1e-12
                elif not fold:
                    historical=read(source/'source_metrics'/f'{m}.json')['raw' if output=='Raw' else 'calibrated']
                    mapping=dict(Recall='recall_attack',Accuracy='accuracy',Balanced_accuracy='balanced_accuracy',AUROC='auroc',Average_precision='average_precision',ECE='ece',Brier='brier')
                    for key in ['TN','FP','FN','TP']:
                        hv=historical['confusion'][key.lower()]
                        comparisons.append(dict(domain=domain,model=m,output=output,metric=key,current=row[key],historical=hv,delta=row[key]-hv,source='P2',match=row[key]==hv))
                    for nk,ok in mapping.items():
                        hv=historical[ok]
                        comparisons.append(dict(domain=domain,model=m,output=output,metric=nk,current=row[nk],historical=hv,delta=row[nk]-hv,source='P2',match=abs(row[nk]-hv)<1e-12))
            save('detector_raw_platt_metrics.csv',metrics)
            save('attack_category_recall.csv',categories)
            save('prediction_flip_summary.csv',flips)
            save('probability_validation.csv',audits)
            save('historical_metric_comparison.csv',comparisons)
            del raw,platt,p,r,logit,reconstructed,good,after,mask
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        del y,code
    save('evaluation_scope.csv',scopes)
    save('ece_bins.csv',histrows)
    deltas=[]
    lookup={(r['domain'],r['model'],r['output']):r for r in metrics}
    keys=['FPR','Recall','Accuracy','Balanced_accuracy','AUROC','Average_precision','ECE','Brier']
    def delta(kind,domain,left,right,condition):
        for key in keys:
            pp=key in keys[:4]
            deltas.append(dict(effect=kind,domain=domain,from_model=left['model'],to_model=right['model'],condition=condition,
                metric=key,from_value=left[key],to_value=right[key],delta=(right[key]-left[key])*(100 if pp else 1),
                unit='percentage_points' if pp else 'original_scale',improvement_direction='lower' if key in ['FPR','ECE','Brier'] else 'higher'))
    for domain in ['UNSW','ToN','CIC','BoT']:
        for split in ['rf_Traffic','xgb_Protocol','mlp_Behavior']:
            all49=split.split('_')[0]+'_All49'
            for output in ['Raw','Platt']:
                delta('All49_minus_split',domain,lookup[domain,split,output],lookup[domain,all49,output],output)
            for m in [split,all49]:
                delta('Platt_minus_Raw',domain,lookup[domain,m,'Raw'],lookup[domain,m,'Platt'],m)
    save('feature_calibration_deltas.csv',deltas)
    np.savez(OUT/'probability_histograms.npz', **{m+'__'+o: np.array(v) for (m,o),v in histograms.items()})
    result=dict(status='COMPLETE',scope='post-unblinding exploratory Stage A only',conditions=len(metrics),
        outer_rows=sum(s['rows'] for s in scopes if s['domain']!='UNSW'),source_rows=scopes[0]['rows'],
        outer_windows=sum(s['complete_windows'] for s in scopes if s['domain']!='UNSW'),
        excluded_outer_tail=sum(s['excluded_tail'] for s in scopes),elapsed_seconds=time.time()-start,
        max_platt_reconstruction_error=max(r['platt_reconstruction_max_error'] for r in audits),
        p7_all_metrics_match=all(r['match'] for r in comparisons if r['source']=='P7'),
        p2_nonmatching_metrics=[r for r in comparisons if r['source']=='P2' and not r['match']],
        gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),numpy=np.__version__,cupy=cp.__version__,torch=torch.__version__,
        no_fitting=True,no_new_inference=True,no_threshold_search=True)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
