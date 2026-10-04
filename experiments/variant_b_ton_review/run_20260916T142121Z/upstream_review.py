"""Independent full raw-record audit and frozen RF inference; never imports pilot.py.
Only compact aggregates and first differences are persisted. Raw arrays stay in RAM.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import csv,json,pickle,sys,time,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.compute as pc
R=Path('/home/karri/projects/NIDSProject_Professor_Run'); O=Path(__file__).resolve().parent
OLD=R/'outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z'
def loadj(p):return json.loads(Path(p).read_text())
def csvrows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def outj(name,v):
 with (O/name).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def outcsv(name,rows):
 if not rows:rows=[dict(status='NO_DIFFERENCES')]
 with (O/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def snapshot():return {str(p.relative_to(OLD)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in OLD.rglob('*') if p.is_file()}
def state(p,qb,qa):
 # Independent bit encoding: 0 empty, 1 Benign, 2 Attack, 3 both. No labels.
 return np.less_equal(p,qb).astype(np.int64)+2*np.less_equal(np.subtract(1.,p),qa)
def histogram(ranks,p,y,domain,qb,qa):
 base=100000 if domain=='ToN' else 2130000
 nw=2742 if domain=='ToN' else 24
 bucket=(ranks-base)//10000
 z=state(p,qb,qa)*4+y.astype(np.int64)*2+np.greater_equal(p,.5)
 return np.bincount(bucket*16+z,minlength=nw*16).reshape(nw,16)

def main():
 outj('old_inventory_before.json',snapshot())
 # Read all requested historical CSV records, not just summaries.
 reads=[]
 for p in sorted(OLD.glob('*.csv')):
  rr=csvrows(p); reads.append(dict(file=str(p),rows=len(rr),fields=list(rr[0]) if rr else []))
 outj('original_csv_inventory.json',reads)
 cfg=loadj(R/'configs/execution_v2.json'); train=loadj(R/'outputs/source_v2/models/rf_All49/training_manifest.json')
 fm=loadj(R/'outputs/feature_store_v2/feature_store_manifest.json'); prep=loadj(R/'outputs/source_v2/preprocessing/preprocessing_manifest.json')
 features=[r['feature'] for r in csvrows(R/'configs/feature_views.csv')]
 assert len(features)==49 and features==fm['feature_order']==train['feature_names']==prep['feature_order']
 assert train['parameters']==cfg['models']['rf'] and train['feature_indices']==list(range(49))
 fill=np.load(R/'outputs/source_v2/preprocessing/preprocessing_stats.npz')['fill']
 assert fill.shape==(49,) and np.isfinite(fill).all()
 outcsv('frozen_preprocessing_reference.csv',[dict(index=i,feature=f,fill=float(fill[i]),representation='CSV float64 -> nonfinite NaN -> float32 store -> float64 impute saved S_train fill -> float32 RF; no unit conversion or standardization') for i,f in enumerate(features)])
 frozen=loadj(OLD/'threshold_freeze.json'); qb,qa=frozen['q_Benign'],frozen['q_Attack']
 cal=csvrows(OLD/'calibration_sample_manifest.csv'); calr=np.array([int(x['rank']) for x in cal]); cal_index={int(r):i for i,r in enumerate(calr)}
 calref=[None]*400
 model=None; env=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,pyarrow=pa.__version__,status='NOT_VERIFIED')
 try:
  import cupy as cp,cuml
  with (R/'outputs/source_v2/models/rf_All49/model.pkl').open('rb') as f:model=pickle.load(f)
  classes=cp.asnumpy(model.classes_).tolist(); assert classes==[0,1] and int(model.n_features_in_)==49
  assert model.get_params()['output_type']=='cupy'
  env.update(status='PASS',cupy=cp.__version__,cuml=cuml.__version__,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
   runtime_version=cp.cuda.runtime.runtimeGetVersion(),driver_version=cp.cuda.runtime.driverGetVersion(),classes=classes,n_features_in=int(model.n_features_in_),parameters=model.get_params(),
   batch_policy='stream original CSV order; select requested ranks; <=131072 rows per predict_proba call; original inference batch limit also 131072; compare every output bit exactly',attack_column=1)
 except Exception as exc:env['error']=repr(exc);model=None
 outj('environment.json',env);log('Model environment: '+env['status'])
 alignment=[]; comparisons=[]; first=[]; boundary=[]; npz={}; batches=[]
 for d,n,filename in [('ToN',27520260,'NF-ToN-IoT-v3.csv'),('UNSW',2365424,'NF-UNSW-NB15-v3.csv')]:
  order=np.load(R/f'outputs/round1_data_prep_v1/{d}_sorted_original_rows.npy',mmap_mode='r')
  assert len(order)==n and np.array_equal(np.sort(order),np.arange(n))
  inverse=np.empty(n,dtype=np.int64);inverse[order]=np.arange(n)
  cache=np.load(R/f'outputs/prediction_cache_v2/{d}/rf_All49.npy',mmap_mode='r')['raw_probability']
  labels=np.load(R/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
  store=np.load(R/f'outputs/feature_store_v2/{d}_X.npy',mmap_mode='r');assert store.shape==(n,49) and store.dtype==np.float32
  ends=np.empty(n,dtype=np.int64);rawlabels=np.empty(n,dtype=np.int8)
  hcache=np.zeros((2742 if d=='ToN' else 24,16),dtype=np.int64);hfresh=hcache.copy()
  scope_names=['calibration','evaluation'] if d=='ToN' else ['source_test']
  stats={s:dict(domain=d,scope=s,compared_rows=0,feature_elements=0,feature_mismatch_rows=0,feature_mismatch_elements=0,
    feature_bit_mismatch_elements=0,imputed_input_mismatch_elements=0,imputed_nonfinite_elements=0,
    probability_compared_rows=0,probability_bit_equal_rows=0,probability_unequal_rows=0,max_abs_probability_difference=0.,prediction_set_changed_rows=0,raw_prediction_changed_rows=0,
    probability_status='PASS' if model is not None else 'NOT_VERIFIED') for s in scope_names}
  labeldiff=0;familydiff=0;invalidlabel=0;cursor=0;batchno=0;started=time.monotonic();lastlog=started
  types={f:pa.float64() for f in features};types.update(Label=pa.int8(),Attack=pa.string(),FLOW_END_MILLISECONDS=pa.int64())
  reader=pcsv.open_csv(R/'datasets'/filename,read_options=pcsv.ReadOptions(block_size=16*1024*1024),
   convert_options=pcsv.ConvertOptions(include_columns=features+['Label','Attack','FLOW_END_MILLISECONDS'],column_types=types,strings_can_be_null=False,null_values=['']))
  for batch in reader:
   size=batch.num_rows;ranks=inverse[cursor:cursor+size]
   y=batch.column('Label').to_numpy(zero_copy_only=False); rawlabels[cursor:cursor+size]=y
   t=batch.column('FLOW_END_MILLISECONDS').to_numpy(zero_copy_only=False);ends[cursor:cursor+size]=t
   invalidlabel+=int(np.count_nonzero((y!=0)&(y!=1)))
   benign=np.asarray(pc.equal(pc.utf8_lower(pc.utf8_trim_whitespace(batch.column('Attack'))),'benign'))
   familydiff+=int(np.count_nonzero((y==0)!=benign))
   bad=(y!=labels[ranks]);labeldiff+=int(bad.sum())
   for ii in np.flatnonzero(bad)[:max(0,100-len(first))]:first.append(dict(domain=d,kind='label',original_row=int(cursor+ii),rank=int(ranks[ii]),feature='Label',expected=str(y[ii]),actual=str(labels[ranks[ii]]),batch=batchno))
   ev=((ranks>=100000)&(ranks<27520000)) if d=='ToN' else ((ranks>=2130000)&(ranks<2365424))
   isc=np.isin(ranks,calr) if d=='ToN' else np.zeros(size,dtype=bool)
   selected=np.flatnonzero(ev|isc)
   if len(selected):
    # Rebuild from raw CSV, no feature-store input is used for inference.
    x=np.column_stack([pc.fill_null(batch.column(f),np.nan).to_numpy(zero_copy_only=False)[selected] for f in features])
    x[~np.isfinite(x)]=np.nan;x=x.astype(np.float32)
    sr=ranks[selected]; yy=y[selected]; xx=store[sr]
    unequal=~((x==xx)|(np.isnan(x)&np.isnan(xx)))
    bitunequal=x.view(np.uint32)!=xx.view(np.uint32)
    transformed=np.where(np.isfinite(x.astype(np.float64)),x.astype(np.float64),fill[None,:]).astype(np.float32)
    transformed_store=np.where(np.isfinite(xx.astype(np.float64)),xx.astype(np.float64),fill[None,:]).astype(np.float32)
    inputbad=transformed.view(np.uint32)!=transformed_store.view(np.uint32)
    cached=np.asarray(cache[sr]);fresh=np.full(len(sr),np.nan)
    if model is not None:
     for a in range(0,len(sr),131072):
      gx=cp.asarray(np.ascontiguousarray(transformed[a:a+131072]),dtype=cp.float32)
      proba=model.predict_proba(gx);assert isinstance(proba,cp.ndarray) and proba.shape==(len(gx),2)
      fresh[a:a+131072]=cp.asnumpy(proba[:,1].astype(cp.float64))
     assert np.isfinite(fresh).all() and np.all((fresh>=0)&(fresh<=1))
    for s in scope_names:
     mask=isc[selected] if s=='calibration' else ev[selected]
     st=stats[s];st['compared_rows']+=int(mask.sum());st['feature_elements']+=int(mask.sum())*49
     st['feature_mismatch_rows']+=int(np.any(unequal[mask],axis=1).sum());st['feature_mismatch_elements']+=int(unequal[mask].sum())
     st['feature_bit_mismatch_elements']+=int(bitunequal[mask].sum());st['imputed_input_mismatch_elements']+=int(inputbad[mask].sum())
     st['imputed_nonfinite_elements']+=int((~np.isfinite(transformed[mask])).sum())
     if model is not None and mask.any():
      pb=cached[mask];pf=fresh[mask];same=pb.view(np.uint64)==pf.view(np.uint64)
      st['probability_compared_rows']+=len(pb);st['probability_bit_equal_rows']+=int(same.sum());st['probability_unequal_rows']+=int((~same).sum())
      st['max_abs_probability_difference']=max(st['max_abs_probability_difference'],float(np.abs(pb-pf).max()))
      st['prediction_set_changed_rows']+=int(np.count_nonzero(state(pb,qb,qa)!=state(pf,qb,qa)))
      st['raw_prediction_changed_rows']+=int(np.count_nonzero((pb>=.5)!=(pf>=.5)))
    for i,j in np.argwhere(unequal)[:max(0,100-len(first))]:
     first.append(dict(domain=d,kind='feature',original_row=int(cursor+selected[i]),rank=int(sr[i]),feature=features[j],expected=str(x[i,j]),actual=str(xx[i,j]),batch=batchno))
    if model is not None:
     for i in np.flatnonzero(cached.view(np.uint64)!=fresh.view(np.uint64))[:max(0,100-len(first))]:
      first.append(dict(domain=d,kind='probability',original_row=int(cursor+selected[i]),rank=int(sr[i]),feature='P(Attack)',expected=repr(float(fresh[i])),actual=repr(float(cached[i])),batch=batchno))
    mask=ev[selected];hcache+=histogram(sr[mask],cached[mask],yy[mask],d,qb,qa)
    if model is not None:hfresh+=histogram(sr[mask],fresh[mask],yy[mask],d,qb,qa)
    for i in np.flatnonzero(isc[selected]):
     idx=cal_index[int(sr[i])];p=float(cached[i]);pr=float(fresh[i]) if model is not None else None
     calref[idx]=dict(rank=int(sr[i]),original_row=int(cursor+selected[i]),true_label=int(yy[i]),original_raw_probability=p,
       reinferred_probability=pr if pr is not None else 'NA',nonconformity_score=p if yy[i]==0 else 1-p,
       reinferred_score=(pr if yy[i]==0 else 1-pr) if pr is not None else 'NA',window_id=f'ToN:heldout:10000:T_heldout:{sr[i]//10000}')
    batches.append(dict(domain=d,batch=batchno,original_start=cursor,original_end_exclusive=cursor+size,selected_rows=len(selected),
      feature_mismatch_elements=int(unequal.sum()),probability_unequal_rows=int(np.count_nonzero(cached.view(np.uint64)!=fresh.view(np.uint64))) if model is not None else 'NA',
      max_abs_probability_difference=float(np.max(np.abs(cached-fresh))) if model is not None else 'NA'))
   cursor+=size;batchno+=1
   if time.monotonic()-lastlog>20:log(f'{d}: CSV rows {cursor:,}/{n:,}; selected rows {sum(v["compared_rows"] for v in stats.values()):,}');lastlog=time.monotonic()
  assert cursor==n
  # Independent two-key order, using raw CSV timestamps, not prepared timestamps.
  reference_order=np.lexsort((np.arange(n,dtype=np.int64),ends))
  orderbad=np.flatnonzero(reference_order!=order)
  for rank in orderbad[:max(0,100-len(first))]:first.append(dict(domain=d,kind='sort',original_row=int(order[rank]),rank=int(rank),feature='original_row',expected=str(reference_order[rank]),actual=str(order[rank]),batch='NA'))
  saved_ends=np.load(R/f'outputs/round1_data_prep_v1/{d}_sorted_end_ms.npy',mmap_mode='r')
  timebad=int(np.count_nonzero(ends[order]!=saved_ends))
  ws=[w for w in csvrows(R/'outputs/round1_data_prep_v1/window_manifest.csv') if w['domain']==d and w['window_size']=='10000' and w['scheme']==('heldout' if d=='ToN' else 'source')]
  for w in ws:
   a,b=int(w['rank_start']),int(w['rank_end_exclusive']); expected=[int(reference_order[a]),int(reference_order[b-1]),int(ends[reference_order[a]]),int(ends[reference_order[b-1]])]
   actual=[int(w[k]) for k in ['first_original_row','last_original_row','first_available_ms','last_available_ms']]
   boundary.append(dict(domain=d,window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,expected_first_original=expected[0],expected_last_original=expected[1],mismatches=sum(x!=y for x,y in zip(expected,actual))))
  alignment.append(dict(domain=d,csv_rows=cursor,rank_rows=n,duplicate_original_rows=0,missing_original_rows=0,
    labels_compared=n,label_mismatches=labeldiff,label_family_mismatches=familydiff,invalid_labels=invalidlabel,
    independently_sorted_rank_mismatches=len(orderbad),raw_timestamp_mismatches=timebad,
    window_boundary_mismatches=sum(v['mismatches'] for v in boundary if v['domain']==d),
    comparison_scope='all rows for ordering/labels; inference and features restricted to requested calibration/evaluation'))
  if d=='ToN':
   startup_y=rawlabels[reference_order[:100000]]
   chosen=np.sort(np.concatenate([np.flatnonzero(startup_y==y)[:200] for y in (0,1)]))
   stop=int(chosen.max());scan=stop+1
   expected_sample=[dict(rank=int(r),original_row=int(reference_order[r]),label=int(startup_y[r]),window_id=f'ToN:heldout:10000:T_heldout:{r//10000}',window_index=int(r//10000)) for r in chosen]
   diffs=[]
   for i,(expected,actual) in enumerate(zip(expected_sample,cal)):
    for k,v in expected.items():
     if str(v)!=actual[k]:diffs.append(dict(index=i,field=k,expected=v,actual=actual[k]))
   assert len(chosen)==len(cal)==400
   outcsv('sample_first_differences.csv',diffs[:100])
   outj('sample_alignment.json',dict(status='PASS' if not diffs else 'FAIL',differences=len(diffs),scan_rows=scan,stop_rank=stop,
    prefix_benign=int((startup_y[:scan]==0).sum()),prefix_attack=int((startup_y[:scan]==1).sum()),
    counts=[int((startup_y[chosen]==y).sum()) for y in (0,1)],unique_original_rows=len(set(int(order[r]) for r in calr)),
    evaluation_overlap=int(np.count_nonzero(calr>=100000)),support_100k=[int((startup_y==y).sum()) for y in (0,1)],
    ordering='lexsort(raw FLOW_END_MILLISECONDS, zero-based original CSV data row); header excluded',
    selection_evidence='independent first200 of each class derived from CSV Label, not cache labels'))
  npz[d+'_cache']=hcache;npz[d+'_reinferred']=hfresh
  comparisons.extend(stats.values());log(d+' complete: '+json.dumps(list(stats.values())))
  del ends,rawlabels,reference_order,inverse,store,cache,labels
 outcsv('calibration_reference.csv',calref);outcsv('data_alignment_comparison.csv',alignment)
 outcsv('feature_probability_comparison.csv',comparisons);outcsv('first_upstream_differences.csv',first)
 outcsv('window_boundary_comparison.csv',boundary);outcsv('inference_batches.csv',batches)
 with (O/'reference_histograms.npz').open('xb') as f:np.savez(f,**npz)
 outj('upstream_complete.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),model_verified=model is not None,
   feature_probability_rows=comparisons,alignment=alignment,probabilities_not_stored=True,
   old_files_unchanged_by_size_mtime=snapshot()==loadj(O/'old_inventory_before.json')))
 log('Upstream full scan and inference finished.')
if __name__=='__main__':main()
