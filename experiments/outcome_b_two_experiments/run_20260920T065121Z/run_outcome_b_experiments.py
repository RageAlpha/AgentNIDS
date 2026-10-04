"""Staged Outcome B supplement. Read-only historical inputs; no file digests."""
import csv,json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run'); OUT=Path(__file__).resolve().parent
RF=ROOT/'outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z'
XGB=ROOT/'outputs/post_p7_diagnostics/xgboost_variant_b_ton_pilot/run_20260918T070417Z'
C1=ROOT/'outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/protocol.json'
MODELS=['rf_All49','xgb_All49','mlp_All49']
ENDS={'ToN':27520000,'BoT':16930000,'UNSW':2360000}; NS={'ToN':27520260,'BoT':16933808,'UNSW':2365424}
SEGS={'ToN':[100000,6960000,13820000,20670000,27520000],'BoT':[100000,4310000,8520000,12730000,16930000]}
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvread(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def save(p,rows):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def txt(p,t):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(t)
def arr(p):return np.load(p,mmap_mode='r',allow_pickle=False)
def gate(q):return 'Depreciation stops' if q>=.90 else ('Allows for full evaluation' if q<.80 else 'The Professor’s Judgment')
def windows(d):return sorted([r for r in csvread(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv') if r['domain']==d and r['window_size']=='10000' and r['scheme']==('source' if d=='UNSW' else 'heldout')],key=lambda r:int(r['rank_start']))
def validate_domain(d,models):
 n=NS[d]; base=ROOT/'outputs'; fm=read(base/'feature_store_v2/feature_store_manifest.json');cm=read(base/f'prediction_cache_v2/{d}/manifest.json')
 assert fm['row_order']=='availability_sorted_rank' and cm['row_order']=='availability_rank'
 op=base/f'round1_data_prep_v1/{d}_sorted_original_rows.npy';yp=base/f'feature_store_v2/{d}_label.npy'
 assert Path(cm['original_row_index'])==op and Path(fm['rank_source'])==op.parent
 y,o=arr(yp),arr(op);assert y.shape==o.shape==(n,) and cm['rows']==read(base/f'feature_store_v2/{d}_manifest.json')['rows']==n
 seen=np.zeros(n,dtype=bool)
 for a in range(0,n,1000000):
  z=o[a:a+1000000];assert ((z>=0)&(z<n)).all() and len(np.unique(z))==len(z) and not seen[z].any();seen[z]=True
 assert seen.all()
 ws=windows(d); assert len(ws)==ENDS[d]//10000
 for i,w in enumerate(ws):
  a,b=int(w['rank_start']),int(w['rank_end_exclusive']);assert a==i*10000 and b==a+10000
  assert (int(o[a]),int(o[b-1]))==(int(w['first_original_row']),int(w['last_original_row']))
  assert int(w['first_available_ms'])<=int(w['last_available_ms'])
  if i:assert int(ws[i-1]['last_available_ms'])<=int(w['first_available_ms'])
 if d in SEGS:
  c=read(C1);e=next(v for v in c['evaluation'] if v['domain']==d)
  assert e['evaluation_rank_start']==100000 and e['evaluation_rank_end_exclusive']==ENDS[d]
  assert c['temporal_segments'][d]==[(b-a)//10000 for a,b in zip(SEGS[d][:-1],SEGS[d][1:])]
 caches={}; meta=[]
 for m in models:
  mm=read(base/f'prediction_cache_v2/{d}/{m}_manifest.json');tm=read(base/f'source_v2/models/{m}/training_manifest.json')
  mp=base/f'source_v2/models/{m}'/tm['artifact'];cp=ROOT/mm['file']
  assert mp.is_file() and mp.stat().st_size==tm['artifact_size_bytes']
  assert tm['training_domain']=='UNSW' and tm['training_partition']=='S_train' and tm['model_id']==mm['model_id']==m
  assert tm['feature_indices']==mm['feature_indices']==list(range(49)) and tm['feature_names']==mm['feature_names']==fm['feature_order']
  assert mm['domain']==d and mm['rows']==n
  caches[m]=arr(cp);assert caches[m].shape==(n,) and caches[m].dtype['raw_probability']==np.dtype('float64')
  meta.append(dict(model=m,model_path=str(mp),cache_path=str(cp),training_manifest=str(base/f'source_v2/models/{m}/training_manifest.json'),cache_manifest=str(base/f'prediction_cache_v2/{d}/{m}_manifest.json'),preprocessing=tm['preprocessing']))
 return y,o,ws,caches,dict(domain=d,rows=n,labels=str(yp),original_rows=str(op),models=meta,original_rows_permutation=True,window_boundaries_match=True,already_rank_sorted=True,no_second_sort=True,complete_windows=len(ws),excluded_tail=n-ENDS[d])
def check_sample(rows,y,o,ws):
 assert len(rows)==400 and len({int(r['rank']) for r in rows})==len({int(r['original_row']) for r in rows})==400
 for r in rows:
  k=int(r['rank']);assert 0<=k<100000 and int(r['label'])==int(y[k]) and int(r['original_row'])==int(o[k])
  assert int(r['window_index'])==k//10000 and r['window_id']==ws[k//10000]['window_id']
 assert [sum(int(r['label'])==v for r in rows) for v in (0,1)]==[200,200]
def calibrate(folder,m,d,rows,cache):
 ranks=np.array([int(r['rank']) for r in rows]);y=np.array([int(r['label']) for r in rows]);p=np.asarray(cache['raw_probability'][ranks]);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 s=np.where(y==0,p,1.-p);q=[float(np.sort(s[y==v])[180]) for v in (0,1)]
 assert all(q[v]==float(np.partition(s[y==v],180)[180]) for v in (0,1))
 save(folder/'calibration_scores.csv',[dict(**r,raw_probability=float(pp),true_class_score=float(ss)) for r,pp,ss in zip(rows,p,s)])
 f=dict(model=m,domain=d,frozen_at_utc=now(),q_Benign=q[0],q_Attack=q[1],threshold_sum=q[0]+q[1],min_p_Attack=1.-q[1],min_p_Benign=1.-q[0],gate=gate(q[1]),epsilon=.1,n_Benign=200,n_Attack=200,order_statistic_one_based=181,order_statistic_zero_based=180,interpolation=False,dtype='float64',evaluation_started=False)
 write(folder/'threshold_freeze.json',f);return f
def table(fs):
 keys=[('q_Benign','q_Benign'),('q_Attack','q_Attack'),('q_Benign + q_Attack','threshold_sum'),('Attack inclusion lower bound: 1-q_Attack','min_p_Attack'),('Benign inclusion lower bound: 1-q_Benign','min_p_Benign')]
 return '\n'.join(['| Quantity | '+' | '.join(f["model"]+' / '+f['domain'] for f in fs)+' |','|---|'+'---|'*len(fs)]+['| '+label+' | '+' | '.join(repr(f[k]) for f in fs)+' |' for label,k in keys])
def ton():
 fm=read(ROOT/'outputs/feature_store_v2/feature_store_manifest.json');pm=read(ROOT/'outputs/source_v2/preprocessing/preprocessing_manifest.json')
 assert pm['fit_domain']=='UNSW' and pm['fit_partition']=='S_train' and pm['feature_order']==fm['feature_order']
 stats=ROOT/'outputs/source_v2/preprocessing'/pm['statistics_file'];assert stats.is_file()
 with np.load(stats) as z:
  for k in ['fill','mean','scale']:assert z[k].shape==(49,) and np.isfinite(z[k]).all()
  assert (z['scale']>0).all()
 y,o,ws,caches,validation=validate_domain('ToN',['mlp_All49'])
 rows=csvread(RF/'calibration_sample_manifest.csv');assert rows==csvread(XGB/'calibration_sample_manifest.csv')
 check_sample(rows,y,o,ws);assert max(int(r['rank']) for r in rows)==4019
 for v in (0,1):assert [int(r['rank']) for r in rows if int(r['label'])==v]==np.flatnonzero(y[:4020]==v)[:200].tolist()
 protocol=dict(created_at_utc=now(),authority=[str(ROOT/'docs/OutcomeB_2Expirements'/p) for p in ['Reply_to_Siyuan_OutcomeB_BeginWriting.md','OutcomeB_Two_Experiments_Plan_ZH.md']],historical_rf=str(RF),historical_xgb=str(XGB),historical_C1=str(C1),epsilon=.1,per_class=200,probability='raw_probability=P(Attack); 1-p=P(Benign); unchanged float64',labels={'0':'Benign','1':'Attack'},preprocessing_manifest=str(ROOT/'outputs/source_v2/preprocessing/preprocessing_manifest.json'),preprocessing_stats=str(stats),threshold='ascending true-class scores, index 180, no interpolation',gate={'stop':'q_Attack>=0.90','evaluate':'q_Attack<0.80','pending':'0.80<=q_Attack<0.90'},startup=[0,100000],segments=SEGS,source_if_ton_permitted=[2130000,2365424],bootstrap_if_ton_permitted=dict(block_windows=10,blocks=275,last_block_windows=2,replicates=2000,generator='PCG64',seed=20260915,percentiles=[2.5,97.5]),reporting_order=['MLP/ToN thresholds','BoT all thresholds','permitted MLP/ToN evaluation','permitted BoT RF, XGBoost, MLP evaluations'],no_digests=True,no_retraining=True,no_probability_modification=True,code_evidence=['src/nids/data/feature_store.py: Label copied then gathered once by original row order','src/nids/replay/predictions.py: RF class 1, XGB binary:logistic, MLP sigmoid; raw saved before Platt','src/nids/p2.py: frozen source model and preprocessing manifests'])
 write('protocol.json',protocol);write('ton_mlp/input_validation.json',validation)
 save('ton_mlp/calibration_sample_manifest.csv',rows)
 f=calibrate(Path('ton_mlp'),'mlp_All49','ToN',rows,caches['mlp_All49'])
 report='# MLP / ToN threshold report\n\n'+table([f])+'\n\nRF is in phase with XGBoost manifest; Reuse the original 400 records directly: 200 Benign + 200 Attack, unique ranks and original row indices, maximum rank=4019; It’s not a new sample. The start-up period remains at [0,100000).\n\nDoors are controlled:'+f['gate']+'. Full evaluation and source testing have not yet been carried out.\n'
 txt('ton_mlp/threshold_report.md',report);print(report)
def bot():
 assert (OUT/'ton_threshold_report_displayed.json').exists()
 y,o,ws,caches,validation=validate_domain('BoT',MODELS)
 support=[int(np.count_nonzero(y[:100000]==v)) for v in (0,1)];assert support==[9071,90929]
 rows=[];ns=[0,0];positions={}
 for k in range(100000):
  v=int(y[k]);assert v in (0,1)
  if ns[v]<200:
   rows.append(dict(original_row=int(o[k]),rank=k,label=v,window_id=ws[k//10000]['window_id'],window_index=k//10000));ns[v]+=1
   if ns[v]==200:positions[str(v)]=k
  if ns==[200,200]:break
 assert ns==[200,200] and positions['0']==14863
 check_sample(rows,y,o,ws)
 summary=dict(collected_per_class=ns,total_collected=len(rows),class_200th_rank=positions,stop_rank_inclusive=k,actual_labels_scanned_for_collection=k+1,startup_support_check_labels=100000,startup_support=support,startup=[0,100000],shared_models=MODELS,selection='earliest 200 per true class; stop when both full',completed_at_utc=now())
 save('bot/calibration_sample_manifest.csv',rows);write('bot/collection_summary.json',summary);write('bot/input_validation.json',validation)
 fs=[calibrate(Path('bot')/m,m,'BoT',rows,caches[m]) for m in MODELS]
 report='# BoT Three model threshold report\n\n'+table(fs)+'\n\n'+json.dumps(summary,ensure_ascii=False)+'\n\n'+'\n'.join(f['model']+'：'+f['gate']+'; The full assessment was not carried out.' for f in fs)+'\n'
 txt('bot/threshold_report.md',report);print(report)
if __name__=='__main__':
 stage=sys.argv[1]
 with (OUT/'commands.sh').open('a') as f:f.write('python '+str(Path(__file__).resolve())+' '+stage+'\n')
 print(now(),stage,flush=True)
 {'ton':ton,'bot':bot}[stage]()
