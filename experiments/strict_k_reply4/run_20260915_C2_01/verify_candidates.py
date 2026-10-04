"""Independent candidate audit: unique/inverse + group histogram, versus selector sort/prefix."""
from common import *
import cupy as cp
from fractions import Fraction
engine=score_engine()

def main():
    check_frozen(); sample=rows(PRE/'calibration_sample_manifest.csv'); devscope=rows(PRE/'development_R_select_scope.csv')
    candidate=rows(OUT/'C2_acceptance_candidates.csv'); ts=rows(OUT/'C2_acceptance_thresholds.csv'); checks=[]
    for d,(fold,end,nw,nb,na) in DOMAINS.items():
        sr=[r for r in sample if r['domain']==d]; rank=np.array([int(r['availability_rank']) for r in sr]); y=np.array([int(r['binary_label']) for r in sr])
        for m in MODELS:
            for o,field in FIELDS.items():
                fit=read(B2/f'models/{fold}__{m}__{o}.json')
                p=cp.asarray(cache(d,m)[field][rank]); u=engine.entropy(p)
                target={'one_minus_u':1-u,'fu':engine.predict(u,fit)}; bins=y*2+(cp.asnumpy(p)>=.5)
                data=[]
                for r in [r for r in devscope if r['fold']==fold]:
                    a,b=int(r['rank_start']),int(r['rank_end_exclusive']); dd=r['domain']
                    pp=cp.asarray(cache(dd,m)[field][a:b]); yy=cp.asarray(np.load(ROOT/f'outputs/feature_store_v2/{dd}_label.npy',mmap_mode='r')[a:b],dtype=cp.int64)
                    uu=engine.entropy(pp)
                    data.append((b-a,yy*2+(pp>=.5),{'one_minus_u':1-uu,'fu':engine.predict(uu,fit)}))
                for s in SCORES:
                    k=dict(**key(d,m,o),score=s); stem=f'{fold}__{m}__{o}__{s}'
                    t=next(r for r in ts if all(r[v]==k[v] for v in k)); sc=cp.asnumpy(target[s])
                    cr=[r for r in candidate if all(r[v]==k[v] for v in k)]
                    if d=='CIC': assert not cr and t['threshold']=='NA' and t['status']==UNSUPPORTED
                    else:
                        assert np.array_equal([float(r['threshold']) for r in cr],np.unique(np.r_[sc,-1.,2.]))
                        for r in cr:
                            a=float(r['threshold']); c=np.bincount(bins[sc>=a],minlength=4)
                            assert list(c)==[int(r[v]) for v in COUNTS]
                            assert (r['feasible']=='True')==(c.sum()>=40 and 20*(c[1]+c[2])<=c.sum())
                        feasible=[r for r in cr if r['feasible']=='True']
                        winner=min(feasible,key=lambda r:(-int(r['accepted_rows']),Fraction(int(r['accepted_errors']),int(r['accepted_rows'])),-float(r['threshold']))) if feasible else next(r for r in cr if float(r['threshold'])==2)
                        assert winner['selected']=='True' and float(winner['threshold'])==float(t['threshold'])
                    allscore=cp.concatenate([v[2][s] for v in data]); allcode=cp.concatenate([v[1]+4*i for i,v in enumerate(data)])
                    levels,inverse=cp.unique(allscore,return_inverse=True)
                    group=cp.bincount(inverse*8+allcode,minlength=len(levels)*8).reshape(-1,8)
                    cumulative=cp.asnumpy(cp.cumsum(group[::-1],axis=0,dtype=cp.int64)); lev=cp.asnumpy(levels[::-1])
                    chunks=rows(OUT/f'C0_candidate_audit/{stem}_chunks.csv')
                    for r in chunks:
                        a,b=int(r['group_start']),int(r['group_end_exclusive'])
                        assert hashlib.sha256(lev[a:b].tobytes()+cumulative[a:b].tobytes()).hexdigest()==r['sha256_float64_levels_then_int64_cumulative_counts']
                    n1,n2=[v[0] for v in data]; A1=cumulative[:,:4].sum(axis=1); A2=cumulative[:,4:].sum(axis=1)
                    E1=cumulative[:,1]+cumulative[:,2]; E2=cumulative[:,5]+cumulative[:,6]
                    # Divide out common factors: independent exact inequality implementation.
                    from math import gcd
                    g=gcd(n1,n2); w1=n2//g; w2=n1//g; denom=2*n1*w1
                    ca=A1*w1+A2*w2; ce=E1*w1+E2*w2
                    feasible=(10*ca>=denom)&(20*ce<=ca)
                    ix=np.flatnonzero(feasible); audit=read(OUT/f'C0_candidate_audit/{stem}.json')
                    assert len(ix)==audit['feasible_group_count'] and len(lev)+2==audit['candidate_count']
                    chosen=int(ix[-1]) if len(ix) else None
                    assert chosen==audit['selected_group']
                    if chosen is not None:
                        a1,a2,e1,e2=map(int,[A1[chosen],A2[chosen],E1[chosen],E2[chosen]])
                        coverage=(Fraction(a1,n1)+Fraction(a2,n2))/2
                        risk=(Fraction(e1,n1)+Fraction(e2,n2))/(Fraction(a1,n1)+Fraction(a2,n2))
                        assert coverage>=Fraction(1,10) and risk<=Fraction(1,20)
                        assert float(lev[chosen])==audit['threshold']
                    else: assert audit['threshold']==2.0
                    checks.append(dict(**k,C2_direct_counts_and_exact_fraction_ties=True,C0_all_groups_independent_histogram_digest_match=True,
                        C0_reduced_integer_constraints_match=True,exact_fraction_selected_point_verified=True if chosen is not None else None,
                        candidate_count=len(lev)+2,classification_threshold=.5))
                    del allscore,allcode,levels,inverse,group,cumulative,lev
                del data; cp.get_default_memory_pool().free_all_blocks()
                log(f'Independent full candidate audit {d}/{m}/{o} passed.')
    check_frozen(); write(OUT/'independent_candidate_validation.json',dict(status='PASS',checks=checks))

if __name__=='__main__': main()
