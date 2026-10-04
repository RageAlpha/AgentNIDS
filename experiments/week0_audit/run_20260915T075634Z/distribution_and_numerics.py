from audit_common import *
from fractions import Fraction
P=read(PILOT/'protocol.json');q=float.fromhex(read(PILOT/'q_freeze.json')['q_serialized']['value']);rationalq=Fraction.from_float(q);out=[];bound=[];rational_tests=0;differences=[];classrows=[]
points=[0,.01,.05,.1,.5,.9,.95,.99,1];names=['min','p1','p5','p10','p50','p90','p95','p99','max']
for d,lo,hi,part in [('UNSW',1420000,1780000,'S_prob_cal'),('UNSW',2130000,2365424,'S_source_test'),('ToN',0,27520000,'T_heldout')]:
 cache=np.load(P['inputs'][d]['probability_cache'],mmap_mode='r');lab=np.load(P['inputs'][d]['labels'],mmap_mode='r');pall=np.array(cache['raw_probability'][lo:hi]);yall=np.asarray(lab[lo:hi]);assert np.array_equal(pall,pall.astype('f4').astype('f8'))
 for yy in [0,1]:
  p=pall[yall==yy];a=p if yy==0 else 1.0-p
  for field,v in [('attack_probability',p),('true_class_nonconformity',a)]:
   quant=np.quantile(v,points,method='linear');out.append(dict(domain=d,partition=part,true_class=yy,field=field,rows=len(v),unique_values=int(len(np.unique(v))),zero_count=int((v==0).sum()),one_count=int((v==1).sum()),**{n:float(value) for n,value in zip(names,quant)},quantile_rule='descriptive linear fixed quantiles; never changes conformal q'))
  covered=a<=q;right=(1.0-p)<=q;left=p<=q;rawok=(p>=.5)==yy
  classrows.append(dict(domain=d,partition=part,true_class=yy,rows=len(p),covered=int(covered.sum()),coverage=float(covered.mean()),raw_correct=int(rawok.sum()),raw_correct_rejected=int((rawok&~left&~right).sum()),raw_attack_prediction_count=int((p>=.5).sum()),accepted_benign=int(left.sum()),accepted_attack=int(right.sum())))
  for bname,b in [('benign_probability_boundary',q),('attack_probability_boundary',1.0-q)]:
   unique=np.unique(p);lower=unique[unique<b];upper=unique[unique>b];near=([float(lower[-1])] if len(lower) else [])+([float(upper[0])] if len(upper) else [])
   bound.append(dict(domain=d,partition=part,true_class=yy,boundary=bname,boundary_decimal=repr(b),boundary_hex=b.hex(),p_below=int((p<b).sum()),p_equal=int((p==b).sum()),p_above=int((p>b).sum()),nearest_below_hex=float(lower[-1]).hex() if len(lower) else None,nearest_below_count=int((p==lower[-1]).sum()) if len(lower) else 0,nearest_above_hex=float(upper[0]).hex() if len(upper) else None,nearest_above_count=int((p==upper[0]).sum()) if len(upper) else 0))
   ix=np.linspace(0,len(p)-1,min(1001,len(p)),dtype=int)
   for value in list(p[ix])+near+[b]:
    v=float(value);exact=Fraction.from_float(v);ref=(exact<=rationalq,1-exact<=rationalq);machine=(v<=q,1.0-v<=q);rational_tests+=1
    if ref!=machine:differences.append(dict(domain=d,partition=part,p_hex=v.hex(),exact=str(ref),float64=str(machine)))
 print(d,part,'distribution done',flush=True)
table('probability_score_diagnostics.csv',out);table('boundary_counts.csv',bound);table('class_coverage_rejection.csv',classrows);table('rational_comparison_differences.csv',differences,['domain','partition','p_hex','exact','float64'])
# Explicit neighboring float32 values at the two singleton boundaries; do not alter observed inputs.
edge=[]
for b in [q,1.0-q,.5]:
 f=np.float32(b)
 for v in [np.nextafter(f,np.float32(-np.inf)),f,np.nextafter(f,np.float32(np.inf))]:
  p=float(v);e=Fraction.from_float(p);ref=(e<=rationalq,1-e<=rationalq);got=(p<=q,1-p<=q);assert ref==got;edge.append(dict(p_hex=p.hex(),benign=bool(got[0]),attack=bool(got[1]),exact_rational_matches=True))
# Verification-only reconstruction of S_train fill statistics, never writes frozen preprocessor.
manifest=read(ROOT/'outputs/source_v2/preprocessing/preprocessing_manifest.json');features=manifest['feature_order'];config=read(ROOT/'configs/execution_v2.json')
x=np.load(ROOT/'outputs/feature_store_v2/UNSW_X.npy',mmap_mode='r');aux=np.load(ROOT/'outputs/feature_store_v2/UNSW_categorical.npy',mmap_mode='r');saved=np.load(ROOT/'outputs/source_v2/preprocessing/preprocessing_stats.npz')['fill'];catnames=config['shift']['categorical_features'];fr=[]
for j,name in enumerate(features):
 vals=np.asarray(aux[:1420000,catnames.index(name)] if name in catnames else x[:1420000,j],dtype='f8');vals=vals[np.isfinite(vals)]
 if not len(vals):v=0.
 elif name in catnames:
  u,c=np.unique(vals,return_counts=True);v=float(u[c==c.max()].min())
 else:v=float(np.median(vals))
 fr.append(dict(feature=name,rank_start=0,rank_end_exclusive=1420000,statistic='mode_smallest_tie' if name in catnames else 'finite_median',expected_hex=float(saved[j]).hex(),recomputed_hex=v.hex(),match=v==saved[j]))
table('preprocessing_fill_verification.csv',fr)
write('numerical_validation.json',dict(status='PASS' if not differences and all(r['match'] for r in fr) else 'FAIL',exact_rational_checks=rational_tests,exact_rational_differences=len(differences),float32_neighbors=edge,probabilities_not_modified=True,q_not_tuned=True,fill_statistics_recomputed_from_frozen_S_train_only=True,fill_matches=all(r['match'] for r in fr),unchanged_protocol_complement='1.0-p',descriptive_quantile_method='linear; independent of conformal order statistic'))
