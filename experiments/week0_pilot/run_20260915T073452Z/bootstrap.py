from common import *
assert read(OUT/'evaluation_complete.json')['status']=='PASS'
assert not (OUT/'bootstrap_manifest.json').exists(),'bootstrap already complete'
p=read(OUT/'protocol.json'); assert p['statistics']['seed']==20260915
w=np.load(OUT/'ToN_window_counts.npz')['counts'].sum(axis=1)
assert w.shape==(2752,2,2,4)
blocks=np.stack([w[a:min(a+10,len(w))].sum(0) for a in range(0,len(w),10)])
B=len(blocks);assert B==276 and int(blocks[-1].sum())==20000
blockrows=[dict(block_id=i,first_window_index=i*10,last_window_index_exclusive=min((i+1)*10,2752),windows=min(10,2752-i*10),rank_start=i*100000,rank_end_exclusive=min((i+1)*100000,27520000),rows=int(t.sum())) for i,t in enumerate(blocks)]
csvwrite('bootstrap_blocks.csv',blockrows)
rng=np.random.Generator(np.random.PCG64(20260915));indices=rng.integers(0,B,size=(2000,B),dtype=np.int64)
multiplicity=np.stack([np.bincount(row,minlength=B) for row in indices]).astype(np.int64)
assert (multiplicity.sum(1)==276).all()
np.save(OUT/'bootstrap_indices.npy',indices);np.save(OUT/'bootstrap_multiplicity.npy',multiplicity)
np.savez_compressed(OUT/'bootstrap_block_counts.npz',counts=blocks)
repcounts=(multiplicity@blocks.reshape(B,-1)).reshape(2000,2,2,4)
reps=[summarize(t) for t in repcounts];point=summarize(w.sum(0));rows=[];rep_rows=[]
for i,r in enumerate(reps):
 rr={'replicate':i}
 for k in BOOT_METRICS:
  rr[k]=r[k]
  for suf in ['_numerator','_denominator']:
   if k+suf in r:rr[k+suf]=r[k+suf]
 rep_rows.append(rr)
for k in BOOT_METRICS:
 vals=np.array([np.nan if r[k] is None else r[k] for r in reps]);valid=vals[np.isfinite(vals)]
 ci=np.quantile(valid,[.025,.975],method='linear') if len(valid) else [None,None]
 rows.append(dict(domain='ToN',metric=k,estimate=point[k],lower_95=None if ci[0] is None else float(ci[0]),upper_95=None if ci[1] is None else float(ci[1]),valid_replicates=len(valid),undefined_replicates=2000-len(valid),replicates=2000,interval='percentile_linear',scope='overall',conditional_on='frozen model, source q, observed target sequence'))
csvwrite('bootstrap_intervals.csv',rows);csvwrite('bootstrap_replicate_metrics.csv',rep_rows)
write('bootstrap_manifest.json',dict(status='PASS',created_at_utc=now(),replicates=2000,blocks=276,windows=2752,final_block_windows=2,seed=20260915,rng='PCG64',draws_per_replicate=276,shared_for_all_metrics=True,method='percentile 95%, numpy.quantile(method=linear); undefined omitted per metric with effective count',files={n:sha(OUT/n) for n in ['bootstrap_blocks.csv','bootstrap_indices.npy','bootstrap_multiplicity.npy','bootstrap_block_counts.npz','bootstrap_replicate_metrics.csv','bootstrap_intervals.csv']},q_sha256=sha(OUT/'q_freeze.json'),protocol_sha256=sha(OUT/'protocol.json'),code_hashes={n:sha(OUT/n) for n in ['bootstrap.py','common.py']},source_CI=False,temporal_CI=False,category_CI=False,source_target_paired_CI=False))
print('PASS: 2000 shared block replicates, direct within-replicate gaps and percentile intervals')
