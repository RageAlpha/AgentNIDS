"""Prespecified probability tails, scalar kNN and local-only gates."""
import numpy as np
from calibration_gates import select as select_mixed_gate,accept,counts


def q1d(p,reference_p,reference_y):
    # Reference inputs must already be ascending in availability rank.
    dist=np.abs(np.asarray(p,dtype=np.float64)[:,None]-reference_p[None,:])
    nn=np.argsort(dist,axis=1,kind='stable')[:,:15]
    return reference_y[nn].mean(axis=1),nn


def gate_action(h,s,q,rule):
    out=np.full(len(h),-1,dtype=np.int8)
    for c,key in [(0,'B'),(1,'A')]:
        out[(h==c)&accept(rule[key],s,q if c else 1-q)]=c
    return out


def tail_action(p,rule):
    r=p if rule['direction']=='NORMAL' else 1-p
    out=np.full(len(p),-1,dtype=np.int8)
    out[r<=rule['b']]=0;out[r>=rule['a']]=1
    return out


def select_tail(p,y,method):
    nb=int((y==0).sum());na=int((y==1).sum());rows=[];best=None;summary={};grids={}
    for di,direction in enumerate(['NORMAL','REVERSE']):
        r=p if di==0 else 1-p
        grid=np.concatenate(([-1.],np.unique(r),[2.]));grids[direction]=grid.tolist()
        order=np.argsort(r,kind='stable');sr=r[order];sy=y[order]
        prefixB=np.r_[0,np.cumsum(sy==0)];prefixA=np.r_[0,np.cumsum(sy==1)]
        lower=np.searchsorted(sr,grid,side='right');upper=np.searchsorted(sr,grid,side='left')
        valid=0;both=0;total=0
        for bi,b in enumerate(grid):
            bn=int(lower[bi]);tn=int(prefixB[bn]);fn=int(prefixA[bn]);bf=bn==0 or bn>=10 and fn/bn<=.05
            for ai in range(bi+1,len(grid)):
                a=float(grid[ai]);idx=int(upper[ai]);an=len(y)-idx;tp=na-int(prefixA[idx]);fp=nb-int(prefixB[idx]);af=an==0 or an>=10 and fp/an<=.05
                feasible=bf and af;ident=f'{direction}_B{bi:03d}_A{ai:03d}';total+=1
                row=dict(method=method,record_type='pair',id=ident,direction=direction,b_index=bi,a_index=ai,b=float(b),a=a,accepted_B=bn,accepted_A=an,TN=tn,FN=fn,FP=fp,TP=tp,N_B=nb,N_A=na,R_B=fn/bn if bn else None,R_A=fp/an if an else None,U_B=tn/nb,U_A=tp/na,U_bal=.5*(tn/nb+tp/na),feasible_B=bf,feasible_A=af,feasible=feasible)
                rows.append(row)
                if feasible:
                    valid+=1;both+=int(bn>0 and an>0)
                    key=(-(tn*na+tp*nb),-min(tn*na,tp*nb),fn+fp,di,bi,ai)
                    if best is None or key<best[0]:best=(key,row)
        summary[direction]=dict(grid_size=len(grid),b_lt_a_candidates=total,feasible_pairs=valid,feasible_two_action_pairs=both)
    chosen=dict(best[1]);on=int(chosen['accepted_B']>0)+int(chosen['accepted_A']>0)
    chosen.update(status=['NO_FEASIBLE_POLICY','PARTIAL_ACTION_POLICY','TWO_ACTION_POLICY'][on],selection_rows=len(y),candidate_summary=summary,grids=grids,metrics=counts(y,tail_action(p,chosen)))
    return chosen,rows


def select_local(h,y,q):
    # Same constrained G2 family with s>=0.5 always true, retaining only local support.
    # Two directions are deliberately NOT searched; IDs give only u and OFF.
    nb=int((y==0).sum());na=int((y==1).sum());branches={};rows=[];best=None
    for cls,label in [(0,'B'),(1,'A')]:
        branches[label]=[]
        for ui,u in enumerate([0.,.5,.75,1.,None]):
            mask=(h==cls)&((q if cls else 1-q)>=u) if u is not None else np.zeros(len(h),bool)
            n=int(mask.sum());err=int((mask&(y!=cls)).sum());feasible=u is None or n>=10 and err/n<=.05
            r=dict(id=f'LOCAL_U{ui:03d}' if u is not None else 'OFF_U000',direction='HIGH' if u is not None else 'OFF',threshold=.5 if u is not None else None,u=u if u is not None else 0.,accepted=n,errors=err,correct=n-err,risk=err/n if n else None,feasible=feasible)
            branches[label].append(r);rows.append(dict(method='LOCAL_GATE',record_type='branch',action=label,**r))
    for b in branches['B']:
        for a in branches['A']:
            ident='B_'+b['id']+'__A_'+a['id'];feasible=b['feasible'] and a['feasible'];tn=b['correct'];tp=a['correct']
            rows.append(dict(method='LOCAL_GATE',record_type='pair',id=ident,B_id=b['id'],A_id=a['id'],TN=tn,FN=b['errors'],FP=a['errors'],TP=tp,N_B=nb,N_A=na,U_bal=.5*(tn/nb+tp/na),feasible=feasible))
            if feasible:
                key=(-(tn*na+tp*nb),-min(tn*na,tp*nb),b['errors']+a['errors'],b['u'],a['u'],ident)
                if best is None or key<best[0]:best=(key,b,a,ident)
    _,b,a,ident=best;on=sum(z['direction']!='OFF' for z in [b,a])
    rule=dict(B=b,A=a,id=ident,status=['NO_FEASIBLE_POLICY','PARTIAL_ACTION_POLICY','TWO_ACTION_POLICY'][on],selection_rows=len(y),feasible_enabled_candidates={c:sum(z['feasible'] and z['direction']!='OFF' for z in ls) for c,ls in branches.items()})
    rule['metrics']=counts(y,gate_action(h,np.full(len(h),.5),q,rule))
    return rule,rows


def selftest():
    p=np.array([.1,.2,.2,.5,.8,.8,.9]);rule=dict(direction='NORMAL',b=.2,a=.8)
    assert np.array_equal(tail_action(p,rule),[0,0,0,-1,1,1,1])
    assert np.array_equal(tail_action(p,dict(direction='REVERSE',b=.25,a=.75)),[1,1,1,-1,0,0,0])
    assert np.all(tail_action(p,dict(direction='NORMAL',b=-1.,a=2.))==-1)
    y=np.r_[np.ones(10,dtype=int),np.zeros(10,dtype=int)];px=np.r_[np.full(10,.1),np.full(10,.9)]
    selected,_=select_tail(px,y,'synthetic');assert selected['direction']=='REVERSE' and selected['metrics']['TP']==10 and selected['metrics']['TN']==10
    selected,_=select_tail(np.full(20,.5),np.r_[np.zeros(10,int),np.ones(10,int)],'synthetic');assert selected['status']=='NO_FEASIBLE_POLICY'
    q,nn=q1d(np.array([.5]),np.full(20,.5),np.r_[np.ones(15,int),np.zeros(5,int)]);assert q[0]==1 and np.array_equal(nn[0],np.arange(15))
    m=counts(np.array([0,1,0,1,1]),np.array([0,0,1,1,-1]));assert m['R_B']==.5 and m['R_A']==.5 and m['U_A']==1/3
    assert counts(np.array([0,1]),np.array([-1,-1]))['R_A'] is None
    return dict(inclusive_boundaries=True,tied_scores_grouped=True,OFF=True,reverse_direction=True,tail_selection_synthetic=True,action_denominators=True,zero_denominator_null=True,scalar_knn_stable_ties=True)
