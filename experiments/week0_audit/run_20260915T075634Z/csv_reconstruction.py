from audit_common import *
import pyarrow as pa,pyarrow.csv as pacsv,pyarrow.compute as pc
P=read(PILOT/'protocol.json');sample=rows(OUT/'frozen_sample_manifest.csv');assert sha(OUT/'frozen_sample_manifest.csv')==read(OUT/'sample_freeze.json')['manifest_sha256']
features=read(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json')['feature_names']
assert len(features)==49 and not set(features)&{'Label','Attack','FLOW_START_MILLISECONDS','FLOW_END_MILLISECONDS','IPV4_SRC_ADDR','IPV4_DST_ADDR','original_row','rank','record_id'}
fill=np.load(ROOT/'outputs/source_v2/preprocessing/preprocessing_stats.npz')['fill'];feature_reports=[];label_reports=[];failures=[]
for d in ['UNSW','ToN']:
 inp=P['inputs'][d];order=np.load(inp['original_rows'],mmap_mode='r');end=np.load(inp['availability_times'],mmap_mode='r');N=len(order)
 assert np.array_equal(np.sort(order),np.arange(N))
 assert (end[1:]>=end[:-1]).all() and (order[1:][end[1:]==end[:-1]]>order[:-1][end[1:]==end[:-1]]).all()
 inv=np.empty(N,dtype=np.int32);inv[order]=np.arange(N,dtype=np.int32)
 ycache=np.load(inp['labels'],mmap_mode='r');acache=np.load(inp['attack_codes'],mmap_mode='r');Xcache=np.load(ROOT/f'outputs/feature_store_v2/{d}_X.npy',mmap_mode='r')
 ss=[r for r in sample if r['domain']==d];ranks=np.array([int(r['rank']) for r in ss]);original=np.array([int(r['original_row']) for r in ss]);assert np.array_equal(order[ranks],original)
 sortedpos=np.argsort(original);sortedoriginal=original[sortedpos];rawfeatures=np.empty((len(ss),49),dtype=np.float64);seen=np.zeros(len(ss),bool)
 ranges=[('S_prob_cal',1420000,1780000),('S_source_test',2130000,2365424)] if d=='UNSW' else [('T_heldout',0,27520000)]
 result={name:dict(domain=d,partition=name,rows_checked=0,label_mismatches=0,attack_category_mismatches=0,first_mismatch=None) for name,_,_ in ranges};timebad=0
 reverse={v:int(k) for k,v in inp['attack_codebook'].items()};cursor=0
 path=ROOT/('datasets/NF-UNSW-NB15-v3.csv' if d=='UNSW' else 'datasets/NF-ToN-IoT-v3.csv')
 types={f:pa.float64() for f in features};types.update(Label=pa.int8(),Attack=pa.string(),FLOW_END_MILLISECONDS=pa.int64())
 reader=pacsv.open_csv(path,read_options=pacsv.ReadOptions(block_size=16*1024*1024),convert_options=pacsv.ConvertOptions(include_columns=features+['Label','Attack','FLOW_END_MILLISECONDS'],column_types=types,strings_can_be_null=False,null_values=['']))
 for batch in reader:
  stop=cursor+batch.num_rows;rr=inv[cursor:stop];y=batch.column('Label').to_numpy();enc=pc.dictionary_encode(batch.column('Attack'));names=enc.dictionary.to_pylist()
  mapped=np.array([reverse.get(n,-1) for n in names],dtype=np.int16)[enc.indices.to_numpy()]
  timebad+=int(np.count_nonzero(batch.column('FLOW_END_MILLISECONDS').to_numpy()!=end[rr]))
  for name,lo,hi in ranges:
   select=(rr>=lo)&(rr<hi);loc=np.flatnonzero(select);yc=ycache[rr[select]];ac=acache[rr[select]]
   dy=y[select]!=yc;da=mapped[select]!=ac;res=result[name];res['rows_checked']+=len(loc);res['label_mismatches']+=int(dy.sum());res['attack_category_mismatches']+=int(da.sum())
   if (dy|da).any() and res['first_mismatch'] is None:
    j=int(loc[np.flatnonzero(dy|da)[0]]);res['first_mismatch']=dict(domain=d,rank=int(rr[j]),original_row=cursor+j,csv_label=int(y[j]),cache_label=int(ycache[rr[j]]),csv_attack=batch.column('Attack')[j].as_py(),cache_attack_code=int(acache[rr[j]]))
  a0,b0=np.searchsorted(sortedoriginal,[cursor,stop]);positions=sortedpos[a0:b0];local=sortedoriginal[a0:b0]-cursor
  if len(local):
   for j,name in enumerate(features):rawfeatures[positions,j]=pc.fill_null(batch.column(name),np.nan).to_numpy()[local]
   seen[positions]=True
  cursor=stop
  if cursor//5000000!=(cursor-batch.num_rows)//5000000:print(d,'CSV rows',cursor,flush=True)
 assert cursor==N and seen.all()
 # Reconstruct exact Phase1 staging: CSV f64 -> nonfinite NaN -> f32 feature store;
 # then store f32 -> f64 fill -> f32 RF input. No scaling/units changed.
 raw_normal=np.where(np.isfinite(rawfeatures),rawfeatures,np.nan)
 with np.errstate(over='ignore'):stage=raw_normal.astype(np.float32)
 cached=np.asarray(Xcache[ranks]);reconstructed=np.where(np.isfinite(stage.astype(np.float64)),stage.astype(np.float64),fill).astype(np.float32)
 cached_input=np.where(np.isfinite(cached.astype(np.float64)),cached.astype(np.float64),fill).astype(np.float32)
 for j,name in enumerate(features):
  same=(stage[:,j]==cached[:,j])|(np.isnan(stage[:,j])&np.isnan(cached[:,j]));bit=stage[:,j].view('u4')==cached[:,j].view('u4');fin=np.isfinite(stage[:,j])&np.isfinite(cached[:,j]);err=np.abs(stage[fin,j].astype('f8')-cached[fin,j].astype('f8'))
  bad=np.flatnonzero(~same);feature_reports.append(dict(domain=d,feature_index=j,feature=name,sample_rows=len(ss),value_or_missing_mismatches=int((~same).sum()),bit_mismatches=int((~bit).sum()),reconstructed_nonfinite=int((~np.isfinite(stage[:,j])).sum()),cached_nonfinite=int((~np.isfinite(cached[:,j])).sum()),max_abs_finite_error=float(err.max()) if len(err) else None,filled_input_bit_mismatches=int((reconstructed[:,j].view('u4')!=cached_input[:,j].view('u4')).sum()),first_mismatch_rank=int(ranks[bad[0]]) if len(bad) else None))
  if len(bad):
   i=int(bad[0]);failures.append(dict(domain=d,rank=int(ranks[i]),original_row=int(original[i]),feature=name,csv_f64_hex=float(rawfeatures[i,j]).hex(),reconstructed_f32_hex=float(stage[i,j]).hex(),feature_store_hex=float(cached[i,j]).hex()))
 np.savez_compressed(OUT/f'{d}_sample_features.npz',ranks=ranks,original_rows=original,csv_raw_float64=rawfeatures,reconstructed_store_float32=stage,rf_input_float32=reconstructed)
 for res in result.values():label_reports.append(res)
 write(f'{d}_csv_identity_validation.json',dict(status='PASS' if not timebad and all(not r['label_mismatches'] and not r['attack_category_mismatches'] for r in result.values()) else 'FAIL',raw_csv_rows=cursor,end_time_mapping_rows=N,end_time_mismatches=timebad,full_identity_permutation=True,stable_time_sort=True,sample_rows=len(ss),sample_features_reconstructed=True,CSV_passes=1,columns_parsed=features+['Label','Attack','FLOW_END_MILLISECONDS']))
 print(d,'CSV reconstruction complete',len(ss),flush=True)
table('feature_comparison.csv',feature_reports);write('label_identity_comparison.json',label_reports);write('feature_first_differences.json',failures)
write('csv_reconstruction_validation.json',dict(status='PASS' if not failures and all(r['label_mismatches']==r['attack_category_mismatches']==0 for r in label_reports) and all(r['bit_mismatches']==0 and r['filled_input_bit_mismatches']==0 for r in feature_reports) else 'FAIL',feature_elements_checked=sum(r['sample_rows'] for r in feature_reports),feature_rows=len(sample),label_rows_checked=sum(r['rows_checked'] for r in label_reports),inference_not_yet_run=True))
