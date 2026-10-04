from audit_common import *
import pickle,traceback,importlib.metadata
assert read(OUT/'csv_reconstruction_validation.json')['status']=='PASS'
assert sha(OUT/'frozen_sample_manifest.csv')==read(OUT/'sample_freeze.json')['manifest_sha256']
P=read(PILOT/'protocol.json');q=float.fromhex(read(PILOT/'q_freeze.json')['q_serialized']['value']);before=now()
try:
 import cupy as cp
 modelpath=ROOT/'outputs/source_v2/models/rf_All49/model.pkl'
 assert sha(modelpath)==read(PILOT/'environment.json')['model_sha256']
 with modelpath.open('rb') as f:model=pickle.load(f)
 classes=model.classes_;classes=classes.get() if hasattr(classes,'get') else np.asarray(classes)
 assert classes.tolist()==[0,1] and int(model.n_features_in_)==49
 env=dict(status='PASS',backend=type(model).__module__+'.'+type(model).__name__,classes=classes.tolist(),feature_count=int(model.n_features_in_),cuml=importlib.metadata.version('cuml-cu12'),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),model_sha256=sha(modelpath))
 write('model_reload.json',env)
 outrows=[];summaries=[]
 for d in ['UNSW','ToN']:
  z=np.load(OUT/f'{d}_sample_features.npz');ranks=z['ranks'];original=z['original_rows'];X=z['rf_input_float32'];assert X.dtype==np.float32 and np.isfinite(X).all()
  cache=np.load(P['inputs'][d]['probability_cache'],mmap_mode='r');ys=np.load(P['inputs'][d]['labels'],mmap_mode='r')
  chunks=[]
  for a in range(0,len(X),131072):
   # Exact original RF API, float32 contiguous CUDA input; no original helper called.
   gx=cp.ascontiguousarray(cp.asarray(X[a:a+131072],dtype=cp.float32));result=model.predict_proba(gx);cp.cuda.Stream.null.synchronize();chunks.append(cp.asnumpy(cp.asarray(result)))
  prob=np.concatenate(chunks);old=np.asarray(cache['raw_probability'][ranks]);new=prob[:,1].astype(np.float64);p0=prob[:,0].astype(np.float64)
  assert prob.shape==(len(ranks),2) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
  bits=new.view('u8')==old.view('u8');absdiff=np.abs(new-old);codes=lambda p:(p<=q).astype(int)+2*(1.0-p<=q).astype(int)
  oldc=codes(old);newc=codes(new);comp=p0-(1.0-new);sums=p0+new-1.0
  for i,r in enumerate(ranks):outrows.append(dict(domain=d,rank=int(r),original_row=int(original[i]),y=int(ys[r]),cached_attack=float(old[i]),cached_attack_hex=float(old[i]).hex(),recomputed_benign=float(p0[i]),recomputed_benign_hex=float(p0[i]).hex(),recomputed_attack=float(new[i]),recomputed_attack_hex=float(new[i]).hex(),bitwise_equal=bool(bits[i]),absolute_error=float(absdiff[i]),two_column_sum_error=float(sums[i]),benign_minus_one_minus_attack=float(comp[i]),old_set=int(oldc[i]),new_set=int(newc[i]),set_changed=bool(oldc[i]!=newc[i])))
  summaries.append(dict(domain=d,sampled_rows=len(ranks),output_dtype=str(prob.dtype),batch_rows=131072,bit_mismatch_count=int((~bits).sum()),max_abs_error=float(absdiff.max()),abs_error_quantiles=dict(zip(['min','p1','p5','p10','p50','p90','p95','p99','max'],np.quantile(absdiff,[0,.01,.05,.1,.5,.9,.95,.99,1]).tolist())),q_set_change_count=int((oldc!=newc).sum()),max_two_column_sum_error=float(np.abs(sums).max()),two_column_sum_nonzero_count=int((sums!=0).sum()),max_benign_complement_error=float(np.abs(comp).max()),benign_complement_nonzero_count=int((comp!=0).sum()),tolerance_used=None,status='PASS' if bits.all() else 'FAIL_REQUIRES_BATCH_LOCALIZATION'))
 table('inference_comparison.csv',outrows)
 write('inference_validation.json',dict(status='PASS' if all(x['bit_mismatch_count']==0 for x in summaries) else 'FAIL',coverage='SAMPLED_INFERENCE_ONLY',full_cache_inference_verified=False,sampling_frozen_at_utc=read(OUT/'sample_freeze.json')['frozen_at_utc'],inference_started_at_utc=before,completed_at_utc=now(),summary=summaries,algorithm_unchanged='Continue using 1-p; native benign column differences diagnostic only',inference_api='same frozen RF predict_proba on original CSV reconstructed float32 samples',no_softmax=True,no_retraining=True))
 print(json.dumps(summaries,indent=2))
except Exception:
 write('inference_execution_failure.json',dict(status='BLOCKED_OR_FAILURE',started_at_utc=before,exception=traceback.format_exc()));raise
