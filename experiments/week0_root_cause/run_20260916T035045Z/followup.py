from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,csv
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run');F=R/'outputs/feature_store_v2';D=R/'outputs/round1_data_prep_v1';P=R/'outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z'
def table(n,x):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
features=json.loads((F/'feature_store_manifest.json').read_text())['feature_order'];q=float.fromhex(json.loads((P/'q_freeze.json').read_text())['q_serialized']['value']);focus=['MIN_TTL','MAX_TTL','MIN_IP_PKT_LEN','SHORTEST_FLOW_PKT'];checks=[]
for domain,filename,n_eval in [('UNSW','NF-UNSW-NB15-v3.csv',2365424),('ToN','NF-ToN-IoT-v3.csv',27520000)]:
 X=np.load(F/f'{domain}_X.npy',mmap_mode='r');orig=np.load(D/f'{domain}_sorted_original_rows.npy',mmap_mode='r');inv=np.empty(len(orig),np.int64);inv[orig]=np.arange(len(orig));offset=0;total=Counter();bad=Counter();first={};rawzeros=defaultdict(Counter)
 for frame in pd.read_csv(R/'datasets'/filename,usecols=focus+['Label','PROTOCOL'],chunksize=131072):
  ranks=inv[offset:offset+len(frame)];offset+=len(frame);keep=ranks<n_eval;yy=frame.Label.to_numpy()
  for name in focus:
   raw=frame[name].to_numpy(dtype=np.float64);raw[~np.isfinite(raw)]=np.nan;expected=raw.astype(np.float32);actual=X[ranks,features.index(name)];eq=(expected==actual)|(np.isnan(expected)&np.isnan(actual));bad[name]+=int((~eq&keep).sum());total[name]+=int(keep.sum())
   if name not in first and (~eq&keep).any():i=np.flatnonzero(~eq&keep)[0];first[name]={'rank':int(ranks[i]),'raw':float(raw[i]),'stored':float(actual[i])}
   for y in [0,1]:rawzeros[(name,y)]['rows']+=int((keep&(yy==y)).sum());rawzeros[(name,y)]['zeros']+=int((keep&(yy==y)&(raw==0)).sum())
  if offset%1048576<131072:print('RAW CHECK',domain,offset,flush=True)
 for name in focus:
  for y in [0,1]:checks.append(dict(domain=domain,feature=name,y=y,checked_rows=total[name],mismatches=bad[name],class_rows=rawzeros[(name,y)]['rows'],raw_zero_count=rawzeros[(name,y)]['zeros'],first_mismatch=json.dumps(first.get(name)),scope='entire evaluated domain; original CSV -> original_row -> frozen feature store; float64-to-float32 path'))
 assert not sum(bad.values())
table('full_raw_four_feature_verification.csv',checks)
# Exact raw line fingerprint including time, IP, 49 features, labels: identity-independent repeated content.
orig=np.load(D/'UNSW_sorted_original_rows.npy',mmap_mode='r');inv=np.empty(len(orig),np.int64);inv[orig]=np.arange(len(orig));train=set();later=defaultdict(list)
with (R/'datasets/NF-UNSW-NB15-v3.csv').open('rb') as f:
 next(f)
 for i,line in enumerate(f):
  h=hashlib.sha256(line.rstrip(b'\r\n')).digest();rank=inv[i]
  if rank<1420000:train.add(h)
  elif rank<1780000:later['S_prob_cal'].append(h)
  elif rank<2130000:later['S_competence'].append(h)
  else:later['S_source_test'].append(h)
table('source_full_csv_record_reuse.csv',[dict(partition=k,n=len(hh),raw_line_sha256_matches_train=sum(h in train for h in hh),within_partition_repeated_rows=len(hh)-len(set(hh)),train_unique_full_records=len(train),train_rows=1420000,scope='SHA256 of complete original CSV line excluding newline; includes timestamps IP labels; no match means exact byte duplicates absent; not semantic duplicate proof') for k,hh in later.items()])
# Are repeated feature vectors the sole reason for tight score distribution? Fixed-q subgroup descriptions only.
X=np.load(F/'UNSW_X.npy',mmap_mode='r');Y=np.load(F/'UNSW_label.npy',mmap_mode='r');cache=np.load(R/'outputs/prediction_cache_v2/UNSW/rf_All49.npy',mmap_mode='r');tr=X[:1420000];h=pd.util.hash_pandas_object(pd.DataFrame(tr),index=False).to_numpy();order=np.argsort(h,kind='stable');hs=h[order];out=[]
for part,a,b in [('S_prob_cal',1420000,1780000),('S_source_test',2130000,2365424)]:
 x=X[a:b];hh=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy();idx=np.searchsorted(hs,hh);ok=idx<len(hs);ix=np.minimum(idx,len(hs)-1);ok &=hs[ix]==hh;trrows=tr[order[ix]];match=ok&((x==trrows)|(np.isnan(x)&np.isnan(trrows))).all(axis=1);p=np.asarray(cache['raw_probability'][a:b]);yy=Y[a:b];scores=np.where(yy==0,p,1-p)
 for y in [0,1]:
  for same in [False,True]:
   mask=(yy==y)&(match==same);z=scores[mask]
   out.append(dict(partition=part,y=y,exact_feature_match_train=same,n=int(mask.sum()),fixed_q_covered=int((z<=q).sum()),fixed_q_ties=int((z==q).sum()),score_median=float(np.median(z)),raw_correct=int((((p>=.5)==yy)&mask).sum()),no_new_q=True))
table('feature_reuse_fixed_q_subgroups.csv',out)
print('COMPLETE',flush=True)
