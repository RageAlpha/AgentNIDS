import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

COUNT_KEYS = ['N','N_B','N_A','TN','FN','FP','TP','abstain_B','abstain_A','pred_B','pred_A','accepted_pred_B','accepted_pred_A','windows_B','windows_A']

def now(): return datetime.now(timezone.utc).isoformat()

def readj(p): return json.loads(Path(p).read_text())

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def writej(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def csv(p,rows): pd.DataFrame(rows).to_csv(p,index=False,na_rep='NA')

def require(ok,msg):
    if not ok: raise ValueError(msg)

def ratio(a,b): return a/b if b else None

class Assets:
    def __init__(self,root,domain,protocol):
        self.root=Path(root);self.domain=domain;self.protocol=protocol
        store=self.root/'outputs/feature_store_v2';pre=self.root/'outputs/source_v2/preprocessing'
        fm=readj(store/'feature_store_manifest.json');dm=readj(store/f'{domain}_manifest.json')
        pm=readj(pre/'preprocessing_manifest.json')
        cm=readj(self.root/f'outputs/prediction_cache_v2/{domain}/rf_All49_manifest.json')
        tm=readj(self.root/'outputs/source_v2/models/rf_All49/training_manifest.json')
        self.names=fm['feature_order'];require(len(self.names)==49,'feature count')
        require(self.names==pm['feature_order']==cm['feature_names']==tm['feature_names'],'feature order conflict')
        require(fm['row_order']=='availability_sorted_rank','feature rank semantics')
        require(cm['model_id']==tm['model_id']=='rf_All49' and cm['domain']==domain,'model/domain identity')
        require(tm['preprocessing']=='S_train_imputed_original_numeric_representation','RF preprocessing semantics')
        require(pm['fit_domain']=='UNSW' and pm['fit_partition']=='S_train','source statistics identity')
        stats=np.load(pre/pm['statistics_file']);self.fill=stats['fill'];self.scale=stats['scale']
        require(self.fill.shape==self.scale.shape==(49,) and np.isfinite(self.fill).all() and np.isfinite(self.scale).all() and (self.scale>0).all(),'fill/scale invalid')
        self.paths={'features':str(store/dm['files']['features']), 'labels':str(store/dm['files']['binary_label']), 'probability':str(self.root/cm['file']), 'original_rows':str(self.root/f'outputs/round1_data_prep_v1/{domain}_sorted_original_rows.npy'),'statistics':str(pre/pm['statistics_file'])}
        self.x=np.load(self.paths['features'],mmap_mode='r');self.p=np.load(self.paths['probability'],mmap_mode='r');self.orig=np.load(self.paths['original_rows'],mmap_mode='r')
        require(self.x.shape==(dm['rows'],49) and self.p.shape==self.orig.shape==(dm['rows'],) and cm['rows']==dm['rows'],'array row shape conflict')
        require('raw_probability' in self.p.dtype.names,'missing raw probability')
        self.cat=[i for i,n in enumerate(self.names) if n in protocol['evidence']['categorical_features']]
        self.num=[i for i,n in enumerate(self.names) if i not in self.cat and n!='DNS_QUERY_ID']
        require(len(self.cat)==11 and len(self.num)==37,'distance dimensions')
        end=protocol['domains'][domain]['evaluation'][1];require(end<=len(self.x),'evaluation range')
        self.info=dict(self.paths,rows=len(self.x),feature_order=self.names,categorical_indices=self.cat,numeric_indices=self.num,feature_representation='rank-aligned raw float32 promoted to float64, source fill once, no standardization',raw_semantics='RF predict_proba[:,1]; recompute raw_pred; cached predicted_class ignored',loader_evidence=['src/nids/data/feature_store.py:104-116','src/nids/replay/predictions.py:33-37,70-113'])
    def features(self,r):
        x=np.asarray(self.x[r],dtype=np.float64);x=np.where(np.isfinite(x),x,self.fill)
        require(np.isfinite(x).all(),'nonfinite imputed features');return x
    def probs(self,r):
        p=np.asarray(self.p['raw_probability'][r],dtype=np.float64)
        require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'probability range');return p
    def labels(self,r,calibration=False,frozen=None):
        require(calibration and np.all((r>=0)&(r<100000)) or frozen is not None and Path(frozen).is_file(),'evaluation labels accessed before freeze')
        a=np.load(self.paths['labels'],mmap_mode='r');require(a.shape==self.p.shape,'label shape')
        y=np.array(a[r]);require(np.isin(y,[0,1]).all(),'binary label semantics');return y

def distances(x,r,cat,num,scale):
    # Deliberate feature-order float64 accumulation; no approximate search.
    d=np.zeros((len(x),len(r)),dtype=np.float64)
    for j in sorted(cat+num):
        if j in cat: d += x[:,j,None]!=r[None,:,j]
        else: d += np.minimum(np.abs(x[:,j,None]-r[None,:,j])/scale[j],10.)/10.
    return d/48.

def evidence(x,r,ry,pred,cat,num,scale):
    d=distances(x,r,cat,num,scale)
    # R is ascending by availability_rank; stable sort implements (distance, rank).
    nn=np.argsort(d,axis=1,kind='stable')[:,:15]
    support=(ry[nn]==pred[:,None]).mean(axis=1)
    return support,np.take_along_axis(d,nn,axis=1).mean(axis=1),nn

def window_counts(y,pred,a,nw):
    z=lambda m: m.reshape(nw,100).sum(axis=1,dtype=np.int64)
    return np.column_stack([np.full(nw,100),z(y==0),z(y==1),z((a==0)&(y==0)),z((a==0)&(y==1)),z((a==1)&(y==0)),z((a==1)&(y==1)),z((a==-1)&(y==0)),z((a==-1)&(y==1)),z(pred==0),z(pred==1),z((a!=-1)&(pred==0)),z((a!=-1)&(pred==1)),z(a==0)>0,z(a==1)>0]).astype(np.int64)

def metrics(count):
    d=dict(zip(COUNT_KEYS,map(int,count)));b=d['TN']+d['FN'];a=d['TP']+d['FP'];n=d['N'];nb=d['N_B'];na=d['N_A']
    d.update(accepted_B=b,accepted_A=a,R_B=ratio(d['FN'],b),R_A=ratio(d['FP'],a),U_B=ratio(d['TN'],nb),U_A=ratio(d['TP'],na),acceptance=ratio(b+a,n),acceptance_pred_B=ratio(d['accepted_pred_B'],d['pred_B']),acceptance_pred_A=ratio(d['accepted_pred_A'],d['pred_A']),abstention_true_B=ratio(d['abstain_B'],nb),abstention_true_A=ratio(d['abstain_A'],na))
    d['U_bal']=(d['U_B']+d['U_A'])/2 if nb and na else None
    limited= b<100 or a<100 or d['windows_B']<10 or d['windows_A']<10 or nb==0 or na==0
    fail=[]
    for key,op,val in [('R_B','le',.05),('R_A','le',.05),('U_B','ge',.1),('U_A','ge',.1)]:
        if d[key] is None or (d[key]>val if op=='le' else d[key]<val): fail.append(key)
    if limited: fail.append('action_count_or_window_support')
    d['target_status']='EVAL_SUPPORT_LIMITED' if limited else ('JOINT_TARGET_MET' if not fail else 'JOINT_TARGET_NOT_MET')
    d['failure_metrics']=';'.join(fail);return d
