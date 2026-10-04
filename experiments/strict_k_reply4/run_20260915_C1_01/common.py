"""Strict-K C1 audit utilities. No fitting or evaluation on import."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import hashlib
import sys
import numpy as np

ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT=Path(__file__).resolve().parent
PRE=ROOT/'outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight'
PLAN=ROOT/'docs/StrictK_Reply4_Experiment_Plan.md'
PLAN_SHA='0df7315ea8120376489311bc064cb1da3657291bfaf9d015a3da5afa3f3992ff'
MODELS=['rf_Traffic','rf_All49','xgb_Protocol','xgb_All49','mlp_Behavior','mlp_All49']
FIELDS={'Raw':'raw_probability','Platt':'probability'}
DOMAINS={'ToN':('F1',27520000,2742,343,57),'CIC':('F3',20110000,2001,400,0),'BoT':('F2',16930000,1683,41,359)}
SEGMENTS={'ToN':[686,686,685,685],'CIC':[501,500,500,500],'BoT':[421,421,421,420]}
COUNTS=['TN','FP','FN','TP']
RATES=['FPR','Recall','BA','Accuracy']
METRICS=COUNTS+RATES
UNSUPPORTED='TARGET_CLASS_SUPPORT_INSUFFICIENT'
DEGENERATE='NO_NONTRIVIAL_CLASSIFICATION_SOLUTION'
def now(): return datetime.now(timezone.utc).isoformat()
def log(s): print(now(),s,flush=True)
def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(p,rs):
    assert rs
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        w.writerows({k:'NA' if r.get(k) is None else r[k] for k in keys} for r in rs)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
    return h.hexdigest()
def inventory():
    result={}
    for folder in ['configs','docs','src','datasets','outputs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts:
                s=p.stat(); r=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
                if s.st_size<=2000000 and p.suffix in ['.json','.md','.py','.csv','.sh']: r['sha256']=sha(p)
                result[str(p)]=r
    return result
def div(a,b): return a/b if b else None
def metrics(c):
    if c is None: return dict.fromkeys(METRICS)
    tn,fp,fn,tp=map(int,c); n=tn+fp+fn+tp
    fpr=div(fp,tn+fp); recall=div(tp,fn+tp)
    return dict(TN=tn,FP=fp,FN=fn,TP=tp,FPR=fpr,Recall=recall,
        BA=(1-fpr+recall)/2 if fpr is not None and recall is not None else None,Accuracy=div(tn+tp,n))
def confusion(y,p,t): return np.bincount(y.astype(np.int64)*2+(p>=np.float64(t)),minlength=4)
def rate_arrays(counts):
    tn,fp,fn,tp=np.asarray(counts,dtype=np.float64).T
    def ratio(a,b): return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    fpr=ratio(fp,tn+fp); recall=ratio(tp,fn+tp)
    return dict(FPR=fpr,Recall=recall,BA=(1-fpr+recall)/2,Accuracy=ratio(tn+tp,tn+fp+fn+tp))
def key(d,m,o): return dict(domain=d,fold=DOMAINS[d][0],model=m,output=o)
def cache(d,m):
    path=ROOT/f'outputs/prediction_cache_v2/{d}/{m}.npy'
    a=np.load(path,mmap_mode='r'); meta=read(path.with_name(m+'_manifest.json'))
    train=read(ROOT/f'outputs/source_v2/models/{m}/training_manifest.json')
    assert meta['domain']==d and meta['model_id']==m and meta['feature_indices']==train['feature_indices']
    assert meta['feature_names']==train['feature_names']
    assert len(a)==meta['rows'] and path.stat().st_size==meta['bytes']==a.offset+a.nbytes
    assert all(a.dtype[f]==np.dtype('float64') for f in FIELDS.values())
    return a
def check_frozen():
    for name,record in [('protocol_freeze.json','files'),('threshold_freeze.json','files')]:
        frozen=read(OUT/name)
        assert all(sha(Path(p))==h for p,h in frozen[record].items()), name
    assert all(sha(PRE/n)==h for n,h in read(PRE/'rank_commitment.json')['sha256'].items())
    for p,h in read(OUT/'protocol.json')['preflight_file_sha256'].items(): assert sha(p)==h
