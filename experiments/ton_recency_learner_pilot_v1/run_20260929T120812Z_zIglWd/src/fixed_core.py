"""Fixed protocol v1: exact mixed distance, exhaustive gate selection and counts."""
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

METHODS = ['RAW','G0','G1H','G1F','G2','Q400','CONST_A','CONST_B','ABSTAIN']
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

def check_alignment(actual,expected): require(np.array_equal(actual,expected),'rank/original_row alignment conflict')

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

def grid(p):
    pred=p>=.5;s=np.maximum(p,1-p);values=[.5,1.]
    for c in [0,1]:
        if np.any(pred==c): values.extend(np.quantile(s[pred==c],[0,.25,.5,.75,1],method='linear'))
    return sorted(set(map(float,values)))

def mask(rule,s,l):
    if rule['direction']=='OFF': return np.zeros(len(s),dtype=bool)
    return ((s>=rule['threshold']) if rule['direction']=='HIGH' else (s<=rule['threshold'])) & (l>=rule['u'])

def branch_candidates(p,y,l,t,cls,method):
    pred=p>=.5;s=np.maximum(p,1-p);out=[]
    defs=[dict(id='OFF_T999_U000',direction='OFF',threshold=None,threshold_index=None,u=0.,u_index=0)]
    for direction in (['HIGH'] if method in ['G0','G1H'] else ['HIGH','LOW']):
        for ti,th in enumerate(t):
            for ui,u in enumerate([0.,.5,.75,1.] if method=='G2' else [0.]):
                defs.append(dict(id=f'{direction}_T{ti:03d}_U{ui:03d}',direction=direction,threshold=th,threshold_index=ti,u=u,u_index=ui))
    for rule in defs:
        m=(pred==cls)&mask(rule,s,l);n=int(m.sum());err=int((m&(y!=cls)).sum());correct=n-err
        off=rule['direction']=='OFF'
        out.append(dict(rule,accepted=n,errors=err,correct=correct,risk=ratio(err,n),utility=ratio(correct,int((y==cls).sum())),feasible=bool(off or ((pred==cls).sum()>=10 and n>=10 and err/n<=.05))))
    return out

def select(p,y,l,method,domain):
    t=grid(p);bs=branch_candidates(p,y,l,t,0,method);aa=branch_candidates(p,y,l,t,1,method)
    nb=int((y==0).sum());na=int((y==1).sum());require(nb>0 and na>0,'missing true class in selection')
    allrows=[];pairs=[]
    for cls,ls in [('B',bs),('A',aa)]:
        allrows.extend(dict(domain=domain,method=method,record_type='branch',action=cls,**r) for r in ls)
    for b in bs:
        for a in aa:
            if method=='G0' and b['direction']!='OFF' and a['direction']!='OFF' and b['threshold_index']!=a['threshold_index']: continue
            ub=b['correct']/nb;ua=a['correct']/na;ident='B_'+b['id']+'__A_'+a['id'];feasible=b['feasible'] and a['feasible']
            allrows.append(dict(domain=domain,method=method,record_type='pair',id=ident,B_id=b['id'],A_id=a['id'],TN=b['correct'],FN=b['errors'],TP=a['correct'],FP=a['errors'],N_B=nb,N_A=na,U_B=ub,U_A=ua,U_bal=(ub+ua)/2,feasible=feasible))
            if feasible:
                # Integer numerators avoid rounding changes to mathematically equal objectives.
                direction={'HIGH':0,'LOW':1,'OFF':2}
                key=(-(b['correct']*na+a['correct']*nb),-min(b['correct']*na,a['correct']*nb),b['errors']+a['errors'],b['u'],a['u'],direction[b['direction']],direction[a['direction']],ident)
                pairs.append((key,b,a,ident))
    _,b,a,ident=min(pairs,key=lambda v:v[0]);on=sum(r['direction']!='OFF' for r in [b,a])
    return dict(B=b,A=a,id=ident,grid=t,status=['NO_FEASIBLE_POLICY','PARTIAL_ACTION_POLICY','TWO_ACTION_POLICY'][on],selection_rows=len(y),N_B=nb,N_A=na),allrows

def actions(p,l,rules):
    pred=(p>=.5).astype(np.int8);s=np.maximum(p,1-p);out={}
    for method in METHODS:
        if method=='RAW': a=pred.copy()
        elif method=='CONST_A': a=np.ones(len(p),dtype=np.int8)
        elif method=='CONST_B': a=np.zeros(len(p),dtype=np.int8)
        elif method=='ABSTAIN': a=np.full(len(p),-1,dtype=np.int8)
        else:
            a=np.full(len(p),-1,dtype=np.int8)
            for cls,key in [(0,'B'),(1,'A')]: a[(pred==cls)&mask(rules[method][key],s,l)]=cls
            require(np.all((a==-1)|(a==pred)),'raw label changed')
        out[method]=a
    return out

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

def selftest():
    p=np.array([.5,.5,.2,.2]);s=np.maximum(p,1-p);l=np.array([0,.5,.75,1.])
    for direction in ['HIGH','LOW']:
        r=dict(direction=direction,threshold=.5,u=0.)
        assert np.array_equal(mask(r,s,l),mask(r,s,np.ones(4)))
        assert mask(r,s,l)[:2].all()
    assert not mask(dict(direction='OFF'),s,l).any()
    v=np.array([100,50,50,20,5,10,30,20,15,25,75,25,40,1,1]);m=metrics(v)
    assert m['R_B']==5/25 and m['R_A']==10/40 and m['U_B']==20/50 and m['U_A']==30/50
    z=metrics(np.array([100,50,50,0,0,0,0,50,50,50,50,0,0,0,0]));assert z['R_B'] is None and z['R_A'] is None
    try: check_alignment(np.array([2,1]),np.array([1,2]))
    except ValueError: pass
    else: raise AssertionError('alignment failure not detected')
    # Distance ties: first 15 reference ranks, not label-dependent.
    x=np.zeros((2,49));r=np.zeros((20,49));ry=np.arange(20)%2
    su,di,nn=evidence(x,r,ry,np.array([0,1]),list(range(11)),list(range(11,48)),np.ones(49))
    assert np.array_equal(nn,np.tile(np.arange(15),(2,1))) and (di==0).all()
    return dict(threshold_equality=True,off=True,action_risk_denominators=True,zero_denominator_null=True,g2_u0_equals_g1f=True,misalignment_detected=True,stable_reference_ties=True)
