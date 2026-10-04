from pathlib import Path
import json,csv,hashlib,sys,platform
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run');A=R/'outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z';P=R/'outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z';F=R/'outputs/feature_store_v2'
def j(p):return json.loads(p.read_text())
def dump(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,x):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
features=j(F/'feature_store_manifest.json')['feature_order'];q=float.fromhex(j(P/'q_freeze.json')['q_serialized']['value']);groups=[('UNSW','S_train',0,1420000),('UNSW','S_prob_cal',1420000,1780000),('UNSW','S_competence',1780000,2130000),('UNSW','S_source_test',2130000,2365424),('ToN','T_heldout',0,27520000)]
provenance={};used=[]
def load(p):
 used.append(p);s=p.stat();provenance[str(p)]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns};return np.load(p,mmap_mode='r')
X={d:load(F/f'{d}_X.npy') for d in ['UNSW','ToN']};Y={d:load(F/f'{d}_label.npy') for d in X};Prob={d:load(R/f'outputs/prediction_cache_v2/{d}/rf_All49.npy') for d in X}
# Count exact class-score components at frozen threshold, no threshold selection.
cal=[]
for y in [0,1]:
 p=np.asarray(Prob['UNSW']['raw_probability'][1420000:1780000]);yy=Y['UNSW'][1420000:1780000];p=p[yy==y];a=p if y==0 else 1-p
 cal.append(dict(y=y,n=len(p),score_below_q=int((a<q).sum()),score_equal_q=int((a==q).sum()),score_above_q=int((a>q).sum()),covered=int((a<=q).sum()),raw_correct=int(((p>=.5)==y).sum())))
table('calibration_rank_decomposition.csv',cal)
# Full-domain feature marginal counts; same-label contrasts rather than pooled shift only.
focus=['MIN_TTL','MAX_TTL','SHORTEST_FLOW_PKT','MIN_IP_PKT_LEN','LONGEST_FLOW_PKT','L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO','FLOW_DURATION_MILLISECONDS'];histrows=[];ttlrows=[]
for d,part,start,end in groups:
 counts=defaultdict(Counter);ttl=defaultdict(lambda:np.zeros(7,dtype=np.float64))
 for a in range(start,end,131072):
  b=min(end,a+131072);xx=X[d][a:b];yy=Y[d][a:b];pp=np.asarray(Prob[d]['raw_probability'][a:b]);score=np.where(yy==0,pp,1-pp)
  for y in [0,1]:
   mask=yy==y
   for name in focus:
    z=xx[mask,features.index(name)];uv,ct=np.unique(z,return_counts=True);counts[(y,name)].update(dict(zip(uv.tolist(),ct.tolist())))
   # TTL pair full aggregation; no per-flow records written.
   z=xx[mask][:,[features.index('MIN_TTL'),features.index('MAX_TTL')]];ps=pp[mask];sc=score[mask]
   vals,inv=np.unique(z,axis=0,return_inverse=True)
   ag=np.column_stack([np.bincount(inv),np.bincount(inv,weights=ps),np.bincount(inv,weights=(ps>=.5)),np.bincount(inv,weights=(ps<=q)),np.bincount(inv,weights=(1-ps<=q)),np.bincount(inv,weights=(sc<=q)),np.bincount(inv,weights=((ps>=.5)==y))]) if len(vals) else np.empty((0,7))
   for pair,v in zip(vals,ag):ttl[(y,*pair.tolist())]+=v
 for (y,name),ct in counts.items():
  total=sum(ct.values());top=ct.most_common(12)
  for val,n in top:histrows.append(dict(domain=d,partition=part,y=y,feature=name,value=str(val),count=n,class_rows=total,fraction=n/total,unique_values=len(ct)))
 for (y,tmin,tmax),v in ttl.items():
  ttlrows.append(dict(domain=d,partition=part,y=y,min_ttl=tmin,max_ttl=tmax,n=int(v[0]),mean_p=v[1]/v[0],raw_attack=int(v[2]),benign_singleton=int(v[3]),attack_singleton=int(v[4]),label_covered=int(v[5]),raw_correct=int(v[6])))
 print('FULL FEATURE GROUP',d,part,flush=True)
table('feature_top_values_by_class.csv',histrows);table('ttl_pair_probability_counts.csv',ttlrows)
# Exact feature-vector reuse across source partitions: hashes select candidates, all candidate matches checked elementwise.
train=X['UNSW'][:1420000];h=pd.util.hash_pandas_object(pd.DataFrame(train),index=False).to_numpy();order=np.argsort(h,kind='stable');hs=h[order]
reuse=[]
for d,part,start,end in groups:
 if d!='UNSW':continue
 stats=defaultdict(Counter)
 for a in range(start,end,65536):
  b=min(end,a+65536);x=X[d][a:b];hh=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy();idx=np.searchsorted(hs,hh);ok=idx<len(hs);clipped=np.minimum(idx,len(hs)-1);ok &=hs[clipped]==hh
  orig=order[clipped];same=((x==train[orig])|(np.isnan(x)&np.isnan(train[orig]))).all(axis=1)&ok
  for y in [0,1]:
   mask=Y[d][a:b]==y;st=stats[y];st['n']+=int(mask.sum());st['hash_candidates']+=int((ok&mask).sum());st['exact_feature_match_to_train']+=int((same&mask).sum());st['candidate_not_exact']+=int((ok&~same&mask).sum());st['match_same_label']+=int((same&mask&(Y[d][orig]==y)).sum());st['match_other_label']+=int((same&mask&(Y[d][orig]!=y)).sum())
 for y,st in stats.items():reuse.append(dict(domain=d,partition=part,y=y,**st,comparison='all49 float32 stage features; NaNs equal; first hash candidate exact checked; not full raw row duplication'))
 print('REUSE',part,flush=True)
table('source_exact_feature_reuse.csv',reuse)
# IP/entity reuse across source partitions from raw CSV; not feature-input leakage by itself.
orig=load(R/'outputs/round1_data_prep_v1/UNSW_sorted_original_rows.npy');inverse=np.empty(len(orig),dtype=np.int64);inverse[orig]=np.arange(len(orig));entities=defaultdict(Counter);offset=0
for frame in pd.read_csv(R/'datasets/NF-UNSW-NB15-v3.csv',usecols=['IPV4_SRC_ADDR','IPV4_DST_ADDR','Label'],chunksize=131072):
 rr=inverse[offset:offset+len(frame)];offset+=len(frame)
 for _,part,start,end in groups[:4]:
  z=frame[(rr>=start)&(rr<end)]
  for col in ['IPV4_SRC_ADDR','IPV4_DST_ADDR']:
   entities[(part,col)].update(z[col].value_counts().to_dict())
entity=[]
for (part,col),ct in entities.items():
 trainset=set(entities[('S_train',col)]);shared=set(ct)&trainset
 entity.append(dict(partition=part,entity_role=col,unique_entities=len(ct),entities_shared_with_train=len(shared),rows_with_entity_seen_in_train=sum(ct[x] for x in shared),rows=sum(ct.values()),interpretation='entity overlap under row/time split; not proof forbidden leakage; raw IP excluded from model'))
table('source_entity_overlap.csv',entity)
dump('input_stat_protection.json',{'before':provenance,'unchanged':all(p.stat().st_size==provenance[str(p)]['bytes'] and p.stat().st_mtime_ns==provenance[str(p)]['mtime_ns'] for p in used)})
dump('environment.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'completed_at':datetime.now(timezone.utc).isoformat()})
print('COMPLETE',flush=True)
