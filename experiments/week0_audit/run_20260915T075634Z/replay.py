from audit_common import *
from fractions import Fraction
P=read(PILOT/'protocol.json');F=read(PILOT/'q_freeze.json');oldq=float.fromhex(F['q_serialized']['value']);diffs=[];compared=0
# Independent order statistic: np.partition rather than old full sort/helper.
c=np.load(P['inputs']['UNSW']['probability_cache'],mmap_mode='r');yy=np.load(P['inputs']['UNSW']['labels'],mmap_mode='r')
p=np.asarray(c['raw_probability'][1420000:1780000]);y=yy[1420000:1780000]
a=p.copy();a[y==1]=np.float64(1)-p[y==1]
k=((len(a)+1)*9+9)//10;q=float(np.partition(a,k-1)[k-1]);assert k==324001
cal=dict(q_hex=q.hex(),k_one_based=k,k_zero_based=k-1,score_sha256=hashlib.sha256(a.tobytes()).hexdigest(),below=int((a<q).sum()),equal=int((a==q).sum()),above=int((a>q).sum()),classes={str(v):dict(n=int((y==v).sum()),covered=int(((y==v)&(a<=q)).sum())) for v in [0,1]})
assert q.hex()==F['q_serialized']['value'] and cal['score_sha256']==F['score_sha256']
assert [cal[z] for z in ['below','equal','above']]==[F[z] for z in ['scores_less_than_q','scores_equal_q','scores_greater_than_q']]
write('independent_q_verification.json',dict(status='PASS',verification_only_not_new_selection=True,**cal))
windows=rows(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv');samples={};uniform=[];newtensor={};allmetrics={};targeted=[];rng=np.random.Generator(np.random.PCG64(20260915))
def add(d,r,reason):samples.setdefault((d,int(r)),set()).add(reason)
def compare_row(old,new,artifact,key):
 global compared
 for field,val in new.items():
  if field not in old:continue
  ov=old[field]; compared+=1
  match=ov=='NA' if val is None else float(ov)==val
  if not match:diffs.append(dict(artifact=artifact,key=str(key),field=field,old=ov,new=val))
for domain,lo,hi,scheme,part in [('UNSW',2130000,2365424,'source','S_source_test'),('ToN',0,27520000,'heldout','T_heldout')]:
 inp=P['inputs'][domain];cache=np.load(inp['probability_cache'],mmap_mode='r');labels=np.load(inp['labels'],mmap_mode='r');categories=np.load(inp['attack_codes'],mmap_mode='r');orig=np.load(inp['original_rows'],mmap_mode='r')
 ww=[w for w in windows if w['domain']==domain and w['scheme']==scheme and w['partition']==part and w['window_size']=='10000']
 intervals=[(int(w['rank_start']),int(w['rank_end_exclusive']),w['window_id']) for w in ww]
 if domain=='UNSW':intervals.append((2360000,2365424,None))
 K=len(inp['attack_codebook']);new=np.zeros((len(intervals),K,2,2,4),dtype=np.int64);old=np.load(PILOT/f'{domain}_window_counts.npz')['counts'];row_dif=0
 oldcsv=[r for r in rows(PILOT/'window_metrics.csv') if r['domain']==domain]
 for wi,(start,end,wid) in enumerate(intervals):
  p=cache['raw_probability'][start:end];assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all() and np.array_equal(p,p.astype('f4').astype('f8'))
  left=np.less_equal(p,q);right=np.less_equal(np.subtract(np.float64(1),p),q)
  code=left.astype('u1')+right.astype('u1')*2
  # Exact old expression re-expressed locally, not importing/calling old helpers. Old per-flow sets were never saved.
  legacy=np.column_stack((p<=oldq,1.0-p<=oldq));legacycode=legacy[:,0].astype(int)+2*legacy[:,1].astype(int)
  bad=code!=legacycode;row_dif+=int(bad.sum())
  if bad.any():
   i=int(np.flatnonzero(bad)[0]);write('first_set_difference.json',dict(domain=domain,rank=start+i,original_row=int(orig[start+i]),p_hex=float(p[i]).hex(),y=int(labels[start+i]),old_set=int(legacycode[i]),new_set=int(code[i]),reason='independent vs locally replayed old expression'))
  y=labels[start:end];cat=categories[start:end];raw=(p>=.5).astype(int)
  for z in range(4):
   take=code==z;packed=cat[take].astype(int)*4+y[take].astype(int)*2+raw[take]
   new[wi,:,:,:,z]=np.bincount(packed,minlength=K*4).reshape(K,2,2)
  if not np.array_equal(new[wi],old[wi]):
   loc=np.argwhere(new[wi]!=old[wi])[0].tolist();diffs.append(dict(artifact=f'{domain}_window_counts.npz',key=str(wi),field=str(loc),old=int(old[wi][tuple(loc)]),new=int(new[wi][tuple(loc)])))
  compare_row(oldcsv[wi],reference_metrics(new[wi].sum(axis=0)),'window_metrics.csv',(domain,wi))
  for r in np.linspace(start,end-1,17,dtype=np.int64):add(domain,r,'window17_equal_rank' if wid else 'retained_tail17')
  if domain=='ToN':
   for r in np.flatnonzero((code==1)|(code==2))+start:add(domain,r,'all_ToN_accepted_directed')
  if wi%700==0:print(domain,wi,'windows replayed',flush=True)
 assert row_dif==0
 newtensor[domain]=new;allmetrics[domain]=reference_metrics(new.sum(axis=(0,1)))
 np.savez_compressed(OUT/f'{domain}_independent_window_counts.npz',counts=new)
 for filename in ['overall_metrics.csv','prediction_set_metrics.csv','selective_metrics.csv','rejection_diagnostics.csv']:
  oldrow=next(r for r in rows(PILOT/filename) if r['domain']==domain);compare_row(oldrow,allmetrics[domain],filename,domain)
 for oldrow in rows(PILOT/'class_metrics.csv'):
  if oldrow['domain']==domain:
   t=new.sum(axis=(0,1));t[1 if oldrow['true_class']=='Benign' else 0]=0;compare_row(oldrow,reference_metrics(t),'class_metrics.csv',(domain,oldrow['true_class']))
 for oldrow in rows(PILOT/'attack_category_metrics.csv'):
  if oldrow['domain']==domain:compare_row(oldrow,reference_metrics(new[:,int(oldrow['category_code'])].sum(0)),'attack_category_metrics.csv',(domain,oldrow['category_code']))
 for oldrow in rows(PILOT/'set_type_error_counts.csv'):
  if oldrow['domain']==domain:
   z=int(oldrow['set_code']);t=new.sum(axis=(0,1))[:,:,z];n=int(t.sum());err=int(t[0,1]+t[1,0]);compare_row(oldrow,dict(records=n,raw_correct=n-err,raw_errors=err,raw_error_rate=err/n if n else None,raw_error_rate_numerator=err,raw_error_rate_denominator=n),'set_type_error_counts.csv',(domain,z))
 if domain=='ToN':
  for oldrow in rows(PILOT/'temporal_metrics.csv'):
   seg=int(oldrow['segment'])-1;compare_row(oldrow,reference_metrics(new[seg*688:(seg+1)*688].sum(axis=(0,1))),'temporal_metrics.csv',seg+1)
  for oldrow in rows(PILOT/'temporal_attack_category_metrics.csv'):
   seg=int(oldrow['segment'])-1;cat=int(oldrow['category_code']);compare_row(oldrow,reference_metrics(new[seg*688:(seg+1)*688,cat].sum(0)),'temporal_attack_category_metrics.csv',(seg+1,cat))
# Sampling is deterministic and frozen before any new model inference.
for r in np.linspace(1420000,1779999,3601,dtype=np.int64):add('UNSW',r,'calibration_uniform3601')
for domain,lo,hi,name in [('UNSW',1420000,1780000,'cal'),('UNSW',2130000,2365424,'test'),('ToN',0,27520000,'target')]:
 cache=np.load(P['inputs'][domain]['probability_cache'],mmap_mode='r');lab=np.load(P['inputs'][domain]['labels'],mmap_mode='r')
 for boundary,bname in [(q,'benign'),(1.0-q,'attack')]:
  for side in ['below','above']:
   candidates=np.array([],dtype=np.int64);distances=np.array([],dtype=np.float64)
   for a0 in range(lo,hi,262144):
    p=cache['raw_probability'][a0:min(a0+262144,hi)];ix=np.flatnonzero(p<=boundary if side=='below' else p>=boundary)
    if len(ix):
     dd=np.abs(p[ix]-boundary);at=np.argpartition(dd,min(63,len(dd)-1))[:64];ii=ix[at]+a0
     candidates=np.r_[candidates,ii];distances=np.r_[distances,dd[at]]
     keep=np.lexsort((candidates,distances))[:64];candidates=candidates[keep];distances=distances[keep]
   for r in candidates:add(domain,r,f'directed_{name}_{bname}_{side}_nearest64')
 for r in range(((lo+131071)//131072)*131072,hi,131072):
  for rr in [r-1,r,r+1]:
   if lo<=rr<hi:add(domain,rr,'producer_batch_boundary')
 if domain=='UNSW':
  p=cache['raw_probability'][lo:hi];y=lab[lo:hi];a=np.where(y==0,p,1.0-p)
  for accepted in [False,True]:
   group=np.flatnonzero((y==1)&((a<=q)==accepted));sel=rng.choice(group,size=min(256,len(group)),replace=False)
   for r in sel+lo:add(domain,r,f'directed_{name}_Attack_'+('covered' if accepted else 'rejected'))
rs=[]
for (d,r),reason in sorted(samples.items()):
 order=np.load(P['inputs'][d]['original_rows'],mmap_mode='r');rs.append(dict(domain=d,rank=r,original_row=int(order[r]),sample_roles=';'.join(sorted(reason)),sampling_seed=20260915,independent_performance_sample=False))
table('frozen_sample_manifest.csv',rs)
write('sample_freeze.json',dict(frozen_at_utc=now(),manifest_sha256=sha(OUT/'frozen_sample_manifest.csv'),samples=len(rs),by_domain={d:sum(r['domain']==d for r in rs) for d in ['UNSW','ToN']},ToN_accepted_included=sum('all_ToN_accepted_directed' in r['sample_roles'] for r in rs),baseline='17 linspace rank points per original complete evaluation window; source tail17; calibration3601 uniform',directed='near frozen singleton boundaries, all target accepts, source Attack covered/rejected, original production batch edges; not independent performance evaluation',seed=20260915,inference_not_started=True))
table('replay_differences.csv',diffs,['artifact','key','field','old','new'])
write('cache_replay_validation.json',dict(status='PASS' if not diffs else 'FAIL',source_calibration_rows=360000,source_test_rows=235424,ToN_rows=27520000,rowwise_legacy_expression_comparisons=27755424,stored_old_row_sets_available=False,old_per_row_limitation='Legacy expression independently re-expressed for every row; no old per-flow set trace exists. Saved tensors compared exactly cell-by-cell per original window/category/label/raw/set.',tensor_cells_compared=sum(x.size for x in newtensor.values()),csv_fields_compared=compared,differences=len(diffs),forbidden_old_helpers_called=False,cache_correctness_not_producer_proof=True))
assert not diffs,diffs[:1]
print('PASS independent replay; samples frozen',len(rs),flush=True)
