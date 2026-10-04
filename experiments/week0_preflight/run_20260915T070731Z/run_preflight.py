import csv, json, hashlib, os, platform, sys, re
from pathlib import Path
import numpy as np
import pyarrow.csv as pcsv
import pyarrow as pa
import pyarrow.parquet as pq
from implementation import synthetic_tests, rank_k
R=Path('/home/karri/projects/NIDSProject_Professor_Run'); O=Path(__file__).resolve().parent

def js(p): return json.loads((R/p).read_text())
def save(name,v): (O/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p): return list(csv.DictReader((R/p).open()))
def table(name,data):
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def inventory():
    return {str(p.relative_to(R)):[p.stat().st_size,p.stat().st_mtime_ns] for base in ['outputs','configs','docs','src','datasets','logs','scripts'] for p in (R/base).rglob('*') if p.is_file() and not p.is_relative_to(O.parent)}
before=inventory(); save('upstream_inventory_before.json',before)
# Entire protocol texts are read, with versions and hashes captured below.
paths=set()
for pattern in ['docs/*.md','configs/*.json','configs/*.csv','outputs/round1_v2/config_snapshot/*','outputs/feature_store_v2/*manifest.json','outputs/feature_store_v2/*validation.json','outputs/source_v2/P2/*.json','outputs/source_v2/models/*/training_manifest.json','outputs/source_v2/preprocessing/*','outputs/round1_v2/P3/*.json','outputs/round1_v2/P3/*.md','outputs/outer_v2/P7/*manifest.json','outputs/outer_v2/P7/stage_report.md','outputs/post_p7_diagnostics/stage_*/*/*.md','outputs/post_p7_diagnostics/strict_k_reply4/*/stage*report.md','outputs/post_p7_diagnostics/strict_k_reply4/*/protocol*.json','outputs/post_p7_diagnostics/strict_k_reply4/*/preflight/*.md','outputs/post_p7_diagnostics/strict_k_reply4/*/preflight/protocol_draft.json']:
    paths.update(p for p in R.glob(pattern) if p.is_file())
for p in ['outputs/round1_data_prep_v1/split_manifest.csv','outputs/round1_data_prep_v1/window_manifest.csv','outputs/round1_data_prep_v1/tail_manifest.csv','outputs/round1_data_prep_v1/fold_manifest.json','outputs/source_v2/models/rf_All49/model.pkl','outputs/source_v2/calibrators/rf_All49.json','outputs/post_p7_diagnostics/stage_A/run_20260914_A1/model_asset_audit.csv','outputs/outer_v2/P7/F1/data_audit.json','outputs/outer_v2/P7/F1/identities.parquet','requirements-runtime.txt','src/nids/p2.py','src/nids/models/base_models.py','src/nids/data/preprocessing.py','src/nids/data/feature_store.py','src/nids/data/partitions.py','src/nids/replay/predictions.py','src/nids/reliability/calibration.py','outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/calibration_support.csv']:
    paths.add(R/p)
for d in ['UNSW','ToN']:
    for s in ['manifest.json','rf_All49_manifest.json','rf_All49.npy']: paths.add(R/f'outputs/prediction_cache_v2/{d}/{s}')
    for s in ['sorted_original_rows.npy','sorted_end_ms.npy']: paths.add(R/f'outputs/round1_data_prep_v1/{d}_{s}')
    for s in ['label.npy','attack_code.npy']: paths.add(R/f'outputs/feature_store_v2/{d}_{s}')
paths.add(R/'outputs/feature_store_v2/UNSW_X.npy')
audit=[]
for p in sorted(paths):
    if not p.exists():
        audit.append(dict(path=str(p),version='MISSING',sha256='',bytes=0,mtime_ns=0,role='missing'));continue
    version='unversioned; SHA256 identifies exact snapshot'
    if p.suffix in ['.md','.py','.txt','.json','.csv']:
        text=p.read_text(encoding='utf-8-sig')
        if p.suffix=='.json':
            try:
                obj=json.loads(text)
                if isinstance(obj,dict): version=str(obj.get('version',obj.get('completed_at_utc',version)))
            except ValueError: pass
        elif p.suffix=='.md': version=' | '.join(text.splitlines()[:5])[:500]
    audit.append(dict(path=str(p),version=version,sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,role='roadmap' if 'Research_Roadmap' in str(p) else 'existing_project'))
table('source_hashes.csv',audit); print('Hashed sources/assets',len(audit),flush=True)
manifest=js('outputs/source_v2/models/rf_All49/training_manifest.json')
pre=js('outputs/source_v2/preprocessing/preprocessing_manifest.json')
feature=js('outputs/feature_store_v2/feature_store_manifest.json')
config=js('configs/execution_v2.json')
assert manifest['feature_count']==49 and manifest['feature_indices']==list(range(49))
assert manifest['feature_names']==pre['feature_order']==feature['feature_order']
assert manifest['parameters']==config['models']['rf']
assert manifest['training_partition']==pre['fit_partition']=='S_train'
assert manifest['training_rank_end_exclusive']==pre['fit_rank_end_exclusive']==1420000
assert sha(R/'configs/execution_v2.json')==sha(R/'outputs/round1_v2/config_snapshot/execution_v2.json')
npz=np.load(R/'outputs/source_v2/preprocessing/preprocessing_stats.npz')
assert all(np.isfinite(npz[k]).all() and npz[k].shape==(49,) for k in ['fill','mean','scale'])
assert (npz['scale']>0).all()
save('feature_contract.json',dict(feature_count=49,feature_names=manifest['feature_names'],feature_indices=manifest['feature_indices'],storage_dtype='float32',preprocessing_arithmetic='float64 then float32',input_dtype='float32',standardized=False,missing='nonfinite -> NaN -> frozen S_train fill; numeric finite median, categorical mode smallest tie; all missing 0',classes=[0,1],probability_columns={'0':'Benign','1':'Attack'},raw_probability='predict_proba[:,1] converted from float32 to float64; no softmax',forbidden=['probability','predicted_class','f(u)','C1 threshold','C2 threshold']))
splits=[r for r in rows('outputs/round1_data_prep_v1/split_manifest.csv') if r['domain']=='UNSW' and r['scheme']=='source']
order=np.load(R/'outputs/round1_data_prep_v1/UNSW_sorted_original_rows.npy',mmap_mode='r')
ends=np.load(R/'outputs/round1_data_prep_v1/UNSW_sorted_end_ms.npy',mmap_mode='r')
assert np.array_equal(np.sort(order),np.arange(len(order)))
assert (ends[1:]>=ends[:-1]).all()
assert (order[1:][ends[1:]==ends[:-1]]>order[:-1][ends[1:]==ends[:-1]]).all()
y=np.load(R/'outputs/feature_store_v2/UNSW_label.npy',mmap_mode='r')
x=np.load(R/'outputs/feature_store_v2/UNSW_X.npy',mmap_mode='r')
cache=np.load(R/'outputs/prediction_cache_v2/UNSW/rf_All49.npy',mmap_mode='r')
assert len(y)==len(order)==len(cache)==len(x)==2365424 and x.shape[1]==49 and x.dtype==np.float32
sa=[]
for r in splits:
    a,b=int(r['rank_start']),int(r['rank_end_exclusive']); yy=y[a:b]
    assert np.isin(yy,[0,1]).all() and int((yy==1).sum())==int(r['attack']) and int((yy==0).sum())==int(r['benign'])
    sa.append(dict(partition=r['partition'],rank_start=a,rank_end_exclusive=b,rows=b-a,benign=int((yy==0).sum()),attack=int((yy==1).sum()),identity_rule='UNSW:original_row; zero based CSV row excluding header',source=str(R/'outputs/round1_data_prep_v1/split_manifest.csv'),identity_sha256=hashlib.sha256(np.asarray(order[a:b]).tobytes()).hexdigest(),status='PASS'))
table('source_partition_audit.csv',sa)
a,b=1420000,1780000
raw=cache['raw_probability'][a:b]
assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
assert np.array_equal(raw,raw.astype(np.float32).astype(np.float64))
# Independent CSV label gather validates every calibration label, not a performance evaluation.
original_labels=[]
reader=pcsv.open_csv(R/'datasets/NF-UNSW-NB15-v3.csv',read_options=pcsv.ReadOptions(block_size=16*1024*1024),convert_options=pcsv.ConvertOptions(include_columns=['Label'],column_types={'Label':pa.int8()}))
for batch in reader: original_labels.append(batch.column(0).to_numpy())
original_labels=np.concatenate(original_labels)
assert len(original_labels)==len(order) and np.array_equal(original_labels[order[a:b]],y[a:b])
save('calibration_data_check.json',dict(status='PASS',rows=b-a,benign=int((y[a:b]==0).sum()),attack=int((y[a:b]==1).sum()),finite=True,within_unit_interval=True,raw_dtype=str(raw.dtype),raw_field_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),labels_sha256=hashlib.sha256(y[a:b].tobytes()).hexdigest(),identity_alignment='All 360000 labels checked against original CSV Label via original_row; Raw row index contract checked in P3 writer, source manifests and full-domain shape. Raw cache has no embedded IDs; no new inference equivalence claim.',disjoint_train=True,disjoint_source_test=True,all_rows_no_resampling=True,threshold_computed=False,k_one_based=rank_k(b-a),k_zero_based=rank_k(b-a)-1))
print('Source partitions and all calibration labels/probabilities validated; no q',flush=True)
# ToN: header + identity + time/window values only. Cache bytes were hashed, never decoded as probabilities.
ton=np.load(R/'outputs/prediction_cache_v2/ToN/rf_All49.npy',mmap_mode='r')
assert ton.shape==(27520260,) and ton.dtype==cache.dtype
order=np.load(R/'outputs/round1_data_prep_v1/ToN_sorted_original_rows.npy',mmap_mode='r')
ends=np.load(R/'outputs/round1_data_prep_v1/ToN_sorted_end_ms.npy',mmap_mode='r')
assert np.array_equal(np.sort(order),np.arange(27520260))
assert (ends[1:]>=ends[:-1]).all()
assert (order[1:][ends[1:]==ends[:-1]]>order[:-1][ends[1:]==ends[:-1]]).all()
win=[r for r in rows('outputs/round1_data_prep_v1/window_manifest.csv') if r['domain']=='ToN' and r['partition']=='T_heldout' and r['scheme']=='heldout' and r['window_size']=='10000']
assert len(win)==2752
for i,w in enumerate(win):
    lo,hi=int(w['rank_start']),int(w['rank_end_exclusive'])
    assert (lo,hi)==(i*10000,(i+1)*10000)
    assert int(w['first_original_row'])==order[lo] and int(w['last_original_row'])==order[hi-1]
    assert int(w['first_available_ms'])==ends[lo] and int(w['last_available_ms'])==ends[hi-1]
    assert w['predecessor_window_id']==('' if i==0 else win[i-1]['window_id'])
table('evaluation_window_references.csv',win)
pf=pq.ParquetFile(R/'outputs/outer_v2/P7/F1/identities.parquet')
print('P7 identity schema:',pf.schema_arrow,flush=True)
count=0
for batch in pf.iter_batches(batch_size=262144, columns=['rank','original_row','record_id','availability_ms','window_id']):
    d=batch.to_pydict(); n=batch.num_rows; ranks=np.arange(count,count+n)
    rank_key='availability_rank' if 'availability_rank' in d else 'rank'
    assert np.array_equal(d[rank_key],ranks)
    assert np.array_equal(d['original_row'],order[count:count+n])
    assert d['record_id']==['ToN:'+str(v) for v in order[count:count+n]]
    assert np.array_equal(d['availability_ms'],ends[count:count+n])
    assert d['window_id']==[win[int(r)//10000]['window_id'] for r in ranks]
    count+=n
assert count==27520000
p7=js('outputs/outer_v2/P7/F1/data_audit.json')
assert (p7['rows'],p7['windows'],p7['excluded_tail'])==(count,len(win),260)
table('evaluation_scope.csv',[dict(domain='UNSW',partition='S_source_test',rank_start=2130000,rank_end_exclusive=2365424,rows=235424,full_10k_windows=23,retained_partial_rows=5424,excluded_tail=0,exclude_startup_rows=0,source=str(R/'outputs/round1_data_prep_v1/split_manifest.csv')),dict(domain='ToN',partition='T_heldout',rank_start=0,rank_end_exclusive=27520000,rows=27520000,full_10k_windows=2752,retained_partial_rows=0,excluded_tail=260,exclude_startup_rows=0,source=str(R/'outputs/outer_v2/P7/F1/data_audit.json'))])
save('target_identity_check.json',dict(status='PASS',rows=count,windows=len(win),cache_shape=list(ton.shape),cache_dtype=ton.dtype.descr,probability_values_accessed=False,hash_only_byte_reads=True,time_order='FLOW_END_MILLISECONDS ascending, original_row ascending ties',all_p7_identity_rows_checked=True,four_segments_windows=[688]*4))
table('asset_audit.csv',[r for r in audit if not r['path'].endswith('.md')])
save('implementation_check.json',synthetic_tests())
after=inventory(); save('upstream_inventory_after.json',after)
assert before==after
save('validation_checks.json',dict(status='PASS',upstream_files_checked=len(before),upstream_size_mtime_unchanged=True,source_and_model_contract='PASS',source_calibration='PASS',target_metadata_identity_windows='PASS',synthetic_tests='PASS',real_q_computed=False,target_sets_computed=False,pilot_performance_computed=False,model_load=js(str(O.relative_to(R)/'model_load_result.json'))['status']))
print('PASS: read-only audits and synthetic tests; stopped before pilot',flush=True)
