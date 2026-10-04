import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fixed_core import metrics
GRID=[i/100 for i in range(50,101,5)]

def attack_probability(model,x):
    p=model.predict_proba(x);indices=np.flatnonzero(model.classes_==1)
    return p[:,int(indices[0])] if len(indices) else np.zeros(len(x),dtype=np.float64)

def gate(score,rule):
    h=(score>=.5).astype(np.int8);support=np.where(h==1,score,1-score);out=np.full(len(score),-1,dtype=np.int8)
    for cls,key in [(0,'B'),(1,'A')]:
        if rule[key]['threshold'] is not None:out[(h==cls)&(support>=rule[key]['threshold'])]=cls
    return out

def choose(score,y,method):
    h=score>=.5;support=np.where(h,score,1-score);nb=int((y==0).sum());na=int((y==1).sum());branches={};rows=[];best=None
    for cls,key in [(0,'B'),(1,'A')]:
        branches[key]=[]
        for i,t in enumerate(GRID+[None]):
            accepted=(h==cls)&(support>=t) if t is not None else np.zeros(len(y),bool)
            n=int(accepted.sum());err=int((accepted&(y!=cls)).sum());valid=t is None or n>=10 and 20*err<=n
            r=dict(id=f'T{i:03d}',threshold=t,accepted=n,errors=err,correct=n-err,risk=err/n if n else None,feasible=bool(valid))
            branches[key].append(r);rows.append(dict(method=method,record_type='branch',action=key,**r))
    missing=nb==0 or na==0
    for bi,b in enumerate(branches['B']):
        for ai,a in enumerate(branches['A']):
            feasible=b['feasible'] and a['feasible'] and (not missing or bi==11 and ai==11)
            ub=b['correct']/nb if nb else None;ua=a['correct']/na if na else None
            row=dict(method=method,record_type='pair',B_id=b['id'],A_id=a['id'],N_B=nb,N_A=na,TN=b['correct'],FN=b['errors'],FP=a['errors'],TP=a['correct'],U_B=ub,U_A=ua,U_bal=(ub+ua)/2 if not missing else None,feasible=bool(feasible));rows.append(row)
            if feasible:
                key=(-(b['correct']*na+a['correct']*nb),-min(b['correct']*na,a['correct']*nb),b['errors']+a['errors'],bi,ai)
                if best is None or key<best[0]:best=(key,b,a)
    _,b,a=best;on=sum(r['threshold'] is not None for r in [b,a])
    return dict(B=b,A=a,N_B=nb,N_A=na,status='CALIBRATION_CLASS_SUPPORT_MISSING' if missing else ['NO_FEASIBLE_POLICY','PARTIAL_ACTION_POLICY','TWO_ACTION_POLICY'][on],feasible_enabled_candidates={c:sum(v['threshold'] is not None and v['feasible'] for v in ls) for c,ls in branches.items()}),rows

def tests():
    r={'B':{'threshold':.6},'A':{'threshold':.5}}
    assert np.array_equal(gate(np.array([.4,.5,.49,1.]),r),[0,1,-1,1])
    off={'B':{'threshold':None},'A':{'threshold':None}};assert (gate(np.array([0.,1.]),off)==-1).all()
    y=np.r_[np.zeros(19,int),1,np.ones(20,int)];score=np.r_[np.full(20,.1),np.full(20,.9)];rule,_=choose(score,y,'test');assert rule['B']['accepted']==20 and rule['B']['errors']==1
    missing,_=choose(np.full(20,.1),np.zeros(20,int),'test');assert missing['status']=='CALIBRATION_CLASS_SUPPORT_MISSING'
    x=np.arange(20).reshape(-1,1)
    for cls in [0,1]:
        m=RandomForestClassifier(n_estimators=1,random_state=1).fit(x,np.full(20,cls));assert np.all(attack_probability(m,x)==cls)
    v=metrics(np.array([100,50,50,20,5,10,30,20,15,25,75,25,40,1,1]));assert v['R_B']==5/25 and v['R_A']==10/40
    ranks=np.array([13709999,13710000,13809999,13810000]);assert np.array_equal(ranks[(ranks>=13710000)&(ranks<13810000)],[13710000,13809999])
    return dict(half_open_interval=True,score_half_Attack=True,gate_inclusive_no_epsilon=True,OFF=True,integer_risk_boundary=True,single_class_RF_mapping=True,missing_C_class_OFF=True,action_risk_denominators=True)

def boot_metrics(c):
    with np.errstate(divide='ignore',invalid='ignore'):
        ub=c[:,3]/c[:,1];ua=c[:,6]/c[:,2]
        return dict(U_bal=(ub+ua)/2,U_B=ub,U_A=ua,R_B=c[:,4]/(c[:,3]+c[:,4]),R_A=c[:,5]/(c[:,5]+c[:,6]))

def ci(x):
    v=x[np.isfinite(x)];rate=len(v)/1000
    lo,hi=np.percentile(v,[2.5,97.5]) if rate>=.9 else (None,None)
    return dict(ci_low=lo,ci_high=hi,valid_fraction=rate,valid_replicates=len(v))
