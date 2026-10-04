from pathlib import Path
import csv,json,os,platform
from datetime import datetime,timezone
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/nids_diagnostic_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/karri/projects/NIDSProject_Professor_Run');O=Path(__file__).resolve().parent;B=R/'outputs/post_p7_diagnostics'
T=B/'temporal_stratified_400_ton/run_20260918T131742Z';RF=B/'variant_b_ton_pilot/run_20260916T085242Z';XC=B/'xgboost_variant_b_ton_pilot/run_20260918T070417Z';XE=B/'xgboost_variant_b_ton_exploratory_eval/run_20260918T071352Z'
MODELS=['rf_All49','xgb_All49'];SCHEMES=['Early-400','Temporal-Stratified-400'];EDGES=np.array([.5,.6,.7,.8,.9,.95,.99,.999,1.]);TARGETS=[.01,.05,.1,.2,.4,.6,.8,1.]
PERIODS={'overall':(100000,27520000),'E1':(100000,6960000),'E2':(6960000,13820000),'E3':(13820000,20670000),'E4':(20670000,27520000)}
FIELDS=['n','errors','confidence_sum','true_benign','true_attack','fp','fn'];CHECK=[]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def js(n,x):
 with (O/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(n,data):
 with (O/n).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def ck(name,ok,detail=''):
 CHECK.append(dict(check=name,passed=bool(ok),detail=detail));assert ok,(name,detail)
def ratio(a,b):return float(a/b) if b else 'NA'
def binids(c):return np.minimum(np.searchsorted(EDGES,c,side='right')-1,7)
def binlabel(i):return f'[{EDGES[i]:g},{EDGES[i+1]:g}'+(']' if i==7 else ')')
def export(fig,name):
 fig.savefig(O/'figures'/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(O/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def group_p(p,y):
 vals,inv=np.unique(p,return_inverse=True);n=np.bincount(inv);attack=np.bincount(inv,weights=y).astype(np.int64)
 return vals,n-attack,attack
def merge(groups):
 vals=np.concatenate([g[0] for g in groups]);u,ix=np.unique(vals,return_inverse=True)
 return u,np.bincount(ix,weights=np.concatenate([g[1] for g in groups])).astype(np.int64),np.bincount(ix,weights=np.concatenate([g[2] for g in groups])).astype(np.int64)
def score_group(g,k):
 x=g[0] if k==0 else 1.-g[0];w=g[k+1];mask=w>0;x=x[mask];w=w[mask];u,inv=np.unique(x,return_inverse=True)
 return u,np.bincount(inv,weights=w).astype(np.int64)
def weighted_quantile(x,w,qs):
 c=np.cumsum(w);h=(int(c[-1])-1)*np.asarray(qs);lo=np.floor(h).astype(np.int64);hi=np.ceil(h).astype(np.int64)
 a=x[np.searchsorted(c,lo,side='right')];b=x[np.searchsorted(c,hi,side='right')]
 return a+(b-a)*(h-lo)
def counts(g):
 p,b,a=g;pred=p>=.5
 return dict(n=int((b+a).sum()),true_benign=int(b.sum()),true_attack=int(a.sum()),tn=int(b[~pred].sum()),fp=int(b[pred].sum()),tp=int(a[pred].sum()),fn=int(a[~pred].sum()))
def raw_metrics(model,period,c):
 denb=c['true_benign'];dena=c['true_attack'];n=c['n'];err=c['fp']+c['fn']
 vals={'record_count':(n,1),'true_benign_fraction':(denb,n),'true_attack_fraction':(dena,n),'tp':(c['tp'],1),'tn':(c['tn'],1),'fp':(c['fp'],1),'fn':(c['fn'],1),'raw_error_rate':(err,n),'accuracy':(n-err,n),'benign_recall':(c['tn'],denb),'attack_recall':(c['tp'],dena),'balanced_accuracy':(c['tn']*dena+c['tp']*denb,2*dena*denb)}
 return [dict(model=model,period=period,rank_start=PERIODS[period][0],rank_end_exclusive=PERIODS[period][1],metric=m,numerator=a,denominator=b,value=ratio(a,b),unit='count' if m in ['record_count','tp','tn','fp','fn'] else 'fraction') for m,(a,b) in vals.items()]

def main():
 protocol=dict(registered_at_utc=now(),purpose='Post-hoc descriptive error-confidence and true-class score distribution diagnosis; original Outcome B unchanged',models=MODELS,domain='ToN',historical_runs=dict(rf_early=str(RF),xgb_early=str(XC),xgb_exploratory=str(XE),temporal_comparison=str(T)),evaluation_periods=PERIODS,excluded_startup=[0,100000],excluded_tail=260,probability='raw_probability=P(Attack); read float64 unchanged',labels={'0':'Benign','1':'Attack'},definitions=dict(prediction='p>=0.5',confidence='max(p,1-p); label-free',error='prediction != y',alpha_true='p if y=0 else 1-p; requires true label'),confidence_bins=[binlabel(i) for i in range(8)],bin_subsets=['all','predicted_Benign','predicted_Attack'],empty_bin='retain; proportion, rate and mean NA',bin_proportion_denominator='same period and prediction subset',descriptive_quantiles=dict(probabilities=[.1,.25,.5,.75,.9,.95,.99],method='linear: h=(n-1)*q; interpolate adjacent order statistics; computed exactly from weighted unique-value groups; never refit threshold'),frozen_threshold_rule='historical class-specific order statistic 181/200, unchanged',bootstrap=dict(matrix=str(T/'bootstrap_indices.npy'),blocks=275,block_windows=10,last_block_windows=2,replicates=2000,seed=20260915,generator='PCG64',percentiles=[2.5,97.5],method='linear',invalid='exclude zero-denominator replicate per metric; report valid count',scope='overall per confidence bin and predicted subset; time segments descriptive; conditional on fixed frozen model'),raw_risk_coverage=dict(order='descending confidence, entire exact float64 ties included; Raw labels unchanged',targets=TARGETS,selection='first whole tie-group boundary reaching/exceeding target',recall_denominators='all true Benign/Attack in same period (end-to-end correct acceptance); additional recall within accepted true class reported',purpose='post-hoc diagnostic, not deployment thresholds; no best point, AURC or significance test'),score_plots=dict(full=[0,1],attack_zoom=[[.9,1],[.99,1]],normalization='empirical CDF per group',rendering='ECDF on fixed 2001-point x grid per displayed range plus observed calibration values and frozen thresholds; exact counts at each displayed x, steps are display compression; raw risk curves compressed to <=3000 whole-group boundaries plus requested target points, all numeric target results exact'),processing='cache mmap; 10000-row window aggregation; bounded one-period probability workspace <=6,860,000 float64 entries filled in 100000-row chunks; retain unique-value grouped counts, never copy full cache; no per-flow output',restrictions='no sampling, fitting, threshold changes, probability transforms, new rules, other domains/models/methods, hashes, messages or subsequent experiments',environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__))
 js('protocol.json',protocol)
 ck('raw_half_boundary',np.array_equal(np.array([np.nextafter(.5,0),.5,np.nextafter(.5,1)])>=.5,[False,True,True]))
 ck('bin_boundaries',np.array_equal(binids(EDGES),[0,1,2,3,4,5,6,7,7]))
 ck('bin_just_below_boundaries',np.array_equal(binids(np.nextafter(EDGES[1:],0)),np.arange(8)))
 ck('empty_NA',ratio(0,0)=='NA')
 ck('quantile_independent_expanded_reference',np.allclose(weighted_quantile(np.array([.1,.3,.9]),np.array([2,3,1]),[.1,.25,.5,.75,.9,.95,.99]),np.quantile(np.array([.1,.1,.3,.3,.3,.9]),[.1,.25,.5,.75,.9,.95,.99],method='linear')))
 tp=read(T/'protocol.json');rp=read(RF/'protocol.json');xp=read(XC/'protocol.json');ep=read(XE/'protocol.json')
 ck('protocol_linkage',tp['early_manifest']==str(RF/'calibration_sample_manifest.csv') and xp['inputs']['original_calibration_manifest']==str(RF/'calibration_sample_manifest.csv') and ep['threshold_source']==str(XC/'threshold_freeze.json') and tp['evaluation']==rp['ton_evaluation']==ep['ton_evaluation']==list(PERIODS['overall']))
 ck('evaluation_period_identity',tp['evaluation_periods']==[list(PERIODS[x]) for x in ['E1','E2','E3','E4']])
 early=rows(RF/'calibration_sample_manifest.csv');strat=rows(T/'calibration_sample_manifest.csv');ck('early_sample_linkage',early==rows(XC/'calibration_sample_manifest.csv')==rows(T/'early_sample_manifest.csv'))
 y=np.load(R/'outputs/feature_store_v2/ToN_label.npy',mmap_mode='r');original=np.load(R/'outputs/round1_data_prep_v1/ToN_sorted_original_rows.npy',mmap_mode='r');ck('original_identity_permutation',np.array_equal(np.sort(original),np.arange(len(original))) and len(y)==len(original)==27520260)
 wm=[r for r in rows(R/'outputs/round1_data_prep_v1/window_manifest.csv') if r['domain']=='ToN' and r['window_size']=='10000' and r['partition']=='T_heldout'];wm.sort(key=lambda r:int(r['rank_start']));ck('window_count',len(wm)==2752)
 for i,w in enumerate(wm):assert int(w['rank_start'])==10000*i and int(w['rank_end_exclusive'])==(i+1)*10000 and int(w['first_original_row'])==original[i*10000] and int(w['last_original_row'])==original[(i+1)*10000-1]
 ck('window_rank_and_identity_endpoints',True)
 for name,sample in zip(SCHEMES,[early,strat]):
  ck(name+'_identity',len(sample)==400 and len({r['rank'] for r in sample})==400 and all(int(r['rank'])<100000 and int(r['original_row'])==original[int(r['rank'])] and int(r['label'])==y[int(r['rank'])] and r['window_id']==wm[int(r['rank'])//10000]['window_id'] for r in sample))
  ck(name+'_class_support',sum(int(r['label'])==0 for r in sample)==200 and sum(int(r['label'])==1 for r in sample)==200)
 idx=np.load(T/'bootstrap_indices.npy',allow_pickle=False);ck('shared_historical_matrix',idx.shape==(2000,275) and np.array_equal(idx,np.load(RF/'bootstrap_indices.npy')) and np.array_equal(idx,np.load(XE/'bootstrap_indices.npy')))
 with (O/'bootstrap_indices.npy').open('xb') as f:np.save(f,idx,allow_pickle=False)
 histories={m:rows((RF if m=='rf_All49' else XE)/'window_counts.csv') for m in MODELS};historical=rows(T/'metrics.csv');histlookup={(r['model'],r['scheme'],r['period'],r['metric']):r for r in historical}
 save('existing_conformal_summary.csv',historical)
 rawout=[];binout=[];intervalout=[];distout=[];cdfout=[];riskout=[];riskplotout=[];highout=[];inputrecords={}
 wf=(O/'window_confidence_bin_counts.csv').open('x',newline='');writer=None
 for model in MODELS:
  cachepath=R/f'outputs/prediction_cache_v2/ToN/{model}.npy';cache=np.load(cachepath,mmap_mode='r');cm=read(R/f'outputs/prediction_cache_v2/ToN/{model}_manifest.json');tm=read(R/f'outputs/source_v2/models/{model}/training_manifest.json');fs=read(R/'outputs/feature_store_v2/feature_store_manifest.json')
  ck(model+'_cache_identity',cache.shape==y.shape and cache.dtype['raw_probability']==np.dtype('float64') and cm['model_id']==tm['model_id']==model and cm['feature_indices']==tm['feature_indices']==list(range(49)) and cm['feature_names']==tm['feature_names']==fs['feature_order'] and cm['bytes']==cachepath.stat().st_size and tm==tp['inputs'][model]['model_identity'])
  ck(model+'_rank_lineage',fs['row_order']=='availability_sorted_rank' and fs['rank_source']==str(R/'outputs/round1_data_prep_v1'))
  frozen=read(T/f'{model}_threshold_freeze.json');old=read((RF if model=='rf_All49' else XC)/'threshold_freeze.json');ck(model+'_early_frozen_identity',frozen[SCHEMES[0]]==old)
  inputrecords[model]=dict(cache=str(cachepath),cache_manifest=cm,model_record=tm,frozen_thresholds=frozen,raw_semantics_evidence='src/nids/replay/predictions.py RF predict_proba[:,1]; XGB binary:logistic inplace_predict; rank slices preserved; feature_store.py sorted_y=raw_y[order]')
  calgroups={}
  for scheme,sample in zip(SCHEMES,[early,strat]):
   rr=np.array([int(r['rank']) for r in sample]);yy=np.array([int(r['label']) for r in sample]);p=cache['raw_probability'][rr].astype(np.float64)
   if scheme==SCHEMES[1]:saved=rows(T/f'{model}_calibration_scores.csv');field='true_class_score'
   elif model=='xgb_All49':saved=rows(XC/'calibration_scores.csv');field='true_class_nonconformity_score'
   else:saved=None;field=None
   alpha=np.where(yy==0,p,1.-p)
   if saved is not None:
    byrank={int(r['rank']):r for r in saved};ss=np.array([float(byrank[r][field]) for r in rr]);ck(model+scheme+'_saved_calibration_scores',np.array_equal(ss,alpha) and all(float(byrank[r]['raw_probability'])==v for r,v in zip(rr,p)));alpha=ss
   save(f'{model}_{scheme}_score_inputs.csv',[dict(rank=int(r),original_row=int(original[r]),label=int(k),raw_probability=float(v),alpha_true=float(s),source='saved historical scores verified against cache' if saved is not None else 'historical RF scores absent; exact saved list cache lookup only') for r,k,v,s in zip(rr,yy,p,alpha)])
   calgroups[scheme]=[np.unique(alpha[yy==k],return_counts=True) for k in range(2)]
  win=np.zeros((2742,3,8,7),dtype=np.float64);groups={}
  for per,(lo,hi) in list(PERIODS.items())[1:]:
   # Bounded period workspace; the full structured cache is never copied.
   workspace=np.empty(hi-lo,dtype=np.float64)
   for a in range(lo,hi,100000):
    b=min(a+100000,hi);workspace[a-lo:b-lo]=cache['raw_probability'][a:b]
   ck(model+per+'_finite_probability_and_binary_labels',np.isfinite(workspace).all() and np.all((workspace>=0)&(workspace<=1)) and np.all((y[lo:hi]==0)|(y[lo:hi]==1)))
   for a in range(lo,hi,10000):
    wi=(a-100000)//10000;p=workspace[a-lo:a-lo+10000];yy=y[a:a+10000];pred=p>=.5;conf=np.maximum(p,1-p);err=pred!=yy;bi=binids(conf)
    assert ((bi>=0)&(bi<8)).all()
    for subset,mask in enumerate([np.ones(len(p),dtype=bool),~pred,pred]):
     for j,weights in enumerate([np.ones(len(p)),err,conf,yy==0,yy==1,pred&(yy==0),(~pred)&(yy==1)]):win[wi,subset,:,j]=np.bincount(bi[mask],weights=weights[mask],minlength=8)
    assert np.array_equal(win[wi,0,:,:2],win[wi,1,:,:2]+win[wi,2,:,:2])
    assert int(win[wi,0,:,0].sum())==10000 and int(win[wi,0,:,1].sum())==int(histories[model][wi]['raw_errors'])
    # Independent direct boolean coverage against each historical window.
    for scheme in SCHEMES:
     q=frozen[scheme];bn=int(((yy==0)&(p<=q['q_Benign'])).sum());an=int(((yy==1)&((1-p)<=q['q_Attack'])).sum())
     if scheme==SCHEMES[0]:assert bn==int(histories[model][wi]['benign_covered']) and an==int(histories[model][wi]['attack_covered'])
   groups[per]=group_p(workspace,y[lo:hi]);del workspace
   print('AGGREGATED',model,per,'unique probability values',len(groups[per][0]),flush=True)
  groups['overall']=merge(list(groups.values()))
  ck(model+'_window_counts_all_partition_checks',np.all(win[:,0,:,0].sum(axis=1)==10000) and np.allclose(win[:,0],win[:,1]+win[:,2],rtol=0,atol=1e-10))
  ck(model+'_historical_raw_errors_every_window',True,'independent direct comparison of 2742 window error counts to historical saved counts')
  for wi in range(2742):
   a=100000+wi*10000;per=next(k for k,(lo,hi) in list(PERIODS.items())[1:] if lo<=a<hi)
   for s,subset in enumerate(['all','predicted_Benign','predicted_Attack']):
    for bi in range(8):
     vals=win[wi,s,bi];r=dict(model=model,window_id=wm[wi+10]['window_id'],rank_start=a,rank_end_exclusive=a+10000,period=per,prediction_subset=subset,bin=binlabel(bi),**{k:(float(vals[j]) if k=='confidence_sum' else int(vals[j])) for j,k in enumerate(FIELDS)})
     if writer is None:writer=csv.DictWriter(wf,fieldnames=list(r));writer.writeheader()
     writer.writerow(r)
  for per,(lo,hi) in PERIODS.items():
   c=counts(groups[per]);rawout+=raw_metrics(model,per,c)
   hist=histlookup[model,SCHEMES[0],per,'raw_error_rate'];ck(model+per+'_raw_counts_history',c['fp']+c['fn']==int(hist['numerator']) and c['n']==int(hist['denominator']))
   total=win[(lo-100000)//10000:(hi-100000)//10000].sum(axis=0)
   for s,subset in enumerate(['all','predicted_Benign','predicted_Attack']):
    subsetn=int(total[s,:,0].sum())
    for bi in range(8):
     vals=total[s,bi];n=int(vals[0]);errors=int(vals[1]);cs=float(vals[2]);r=dict(model=model,period=per,prediction_subset=subset,bin=binlabel(bi),n=n,scope_subset_n=subsetn,bin_fraction=ratio(n,subsetn) if n else 'NA',errors=errors,error_rate=ratio(errors,n),confidence_sum=cs,mean_confidence=ratio(cs,n),correct=n-errors,accuracy=ratio(n-errors,n),confidence_minus_accuracy=ratio(cs-(n-errors),n),true_benign=int(vals[3]),true_attack=int(vals[4]),fp=int(vals[5]),fn=int(vals[6]));binout.append(r)
   # Fixed high confidence definition for explanatory summary: union of predefined bins >=.99.
   for s,subset in enumerate(['all','predicted_Benign','predicted_Attack']):
    v=total[s,6:].sum(axis=0);highout.append(dict(model=model,period=per,prediction_subset=subset,confidence_range='[0.99,1.0]',n=int(v[0]),errors=int(v[1]),error_rate=ratio(v[1],v[0]),scope_subset_n=int(total[s,:,0].sum()),fraction_of_subset=ratio(v[0],total[s,:,0].sum()),all_raw_errors=c['fp']+c['fn'],fraction_of_all_raw_errors=ratio(v[1],c['fp']+c['fn'])))
   make_risk(model,per,groups[per],c,riskout,riskplotout)
  ck(model+'_period_counts_sum',all(sum(counts(groups[p])[k] for p in ['E1','E2','E3','E4'])==counts(groups['overall'])[k] for k in counts(groups['overall'])))
  blocks=np.add.reduceat(win,np.arange(0,2742,10),axis=0)
  for s,subset in enumerate(['all','predicted_Benign','predicted_Attack']):
   for bi in range(8):
    sums=blocks[:,s,bi,:2][idx].sum(axis=1);values=np.divide(sums[:,1],sums[:,0],out=np.full(2000,np.nan),where=sums[:,0]>0);valid=values[np.isfinite(values)];ci=np.percentile(valid,[2.5,97.5],method='linear') if len(valid) else ['NA','NA'];n=int(win[:,s,bi,0].sum());e=int(win[:,s,bi,1].sum())
    intervalout.append(dict(model=model,period='overall',prediction_subset=subset,bin=binlabel(bi),n=n,errors=e,error_rate=ratio(e,n),low_95=ci[0],high_95=ci[1],valid_replicates=len(valid),total_replicates=2000))
  for k in range(2):
   sg={s:calgroups[s][k] for s in SCHEMES};sg.update({per:score_group(g,k) for per,g in groups.items()})
   for name,(x,w) in sg.items():
    qs=weighted_quantile(x,w,[.1,.25,.5,.75,.9,.95,.99]);distout.append(dict(model=model,true_class=['Benign','Attack'][k],group=name,n=int(w.sum()),minimum=float(x[0]),maximum=float(x[-1]),**{f'q{int(q*100):02d}':float(val) for q,val in zip([.1,.25,.5,.75,.9,.95,.99],qs)},quantile_method='linear'))
    for scheme in SCHEMES:
     q=frozen[scheme]['q_'+['Benign','Attack'][k]];lt=int(w[x<q].sum());eq=int(w[x==q].sum());gt=int(w[x>q].sum());n=int(w.sum());cdfout.append(dict(model=model,true_class=['Benign','Attack'][k],group=name,threshold_scheme=scheme,q=q,n=n,less_count=lt,equal_count=eq,greater_count=gt,less_equal_count=lt+eq,less_fraction=ratio(lt,n),equal_fraction=ratio(eq,n),greater_fraction=ratio(gt,n),less_equal_fraction=ratio(lt+eq,n)))
     if name in PERIODS:
      h=histlookup[model,scheme,name,['benign_coverage','attack_coverage'][k]];ck(model+name+scheme+str(k)+'_cdf_matches_historical_coverage',lt+eq==int(h['numerator']) and n==int(h['denominator']) and abs((lt+eq)/n-float(h['value']))<1e-14)
   make_ecdf(model,k,sg,frozen)
  make_binplot(model,binout,intervalout)
  print('FINISHED MODEL',model,flush=True)
 wf.close()
 save('raw_metrics.csv',rawout);save('confidence_bin_metrics.csv',binout);save('confidence_bin_intervals.csv',intervalout);save('score_distribution_summary.csv',distout);save('score_threshold_cdf.csv',cdfout);save('raw_confidence_risk_coverage_points.csv',riskout);save('raw_confidence_risk_coverage_display.csv',riskplotout);save('high_confidence_summary.csv',highout);js('input_records.json',inputrecords)
 ck('all_empty_bins_retained',len(binout)==2*5*3*8)
 ck('all_requested_curve_points',len(riskout)==2*5*8)
 js('validation.json',dict(status='PASS',checks=CHECK,completed_at_utc=now(),independent_checks='direct per-window boolean counts vs preexisting historical counts; exact weighted quantile vs expanded toy data; exact score CDF counts vs saved class coverage; saved calibration scores vs original cache; independent delivery audit follows',unverified=['No model inference rerun: cache/model lineage checked by manifests, source code and historical counts; no file hashes computed.','No original CSV reparse: labels/rank lineage checked through feature-store source and metadata, original-row permutation, window boundaries and saved sample identities.','No label-return timestamps verified; calibration has only 200 per class.','No source-domain performance measured; no causal attribution or future-data guarantee.']))
 print('COMPLETE',O,flush=True)


def make_risk(model,per,g,c,out,display):
 p,b,a=g;conf=np.maximum(p,1-p);u,inv=np.unique(conf,return_inverse=True);pred=p>=.5
 def agg(v):return np.bincount(inv,weights=v).astype(np.int64)[::-1]
 u=u[::-1];n=agg(b+a);fp=agg(b*pred);fn=agg(a*(~pred));tp=agg(a*pred);tn=agg(b*(~pred));N=np.cumsum(n);FP=np.cumsum(fp);FN=np.cumsum(fn);TP=np.cumsum(tp);TN=np.cumsum(tn);risk=(FP+FN)/N;coverage=N/c['n']
 ck(model+per+'_curve_full_acceptance',int(N[-1])==c['n'] and int(FP[-1])==c['fp'] and int(FN[-1])==c['fn'])
 ck(model+per+'_unique_ties_descending',np.all(np.diff(u)<0) and np.all(n>0))
 points=[]
 for target in TARGETS:
  j=int(np.searchsorted(N,int(np.ceil(target*c['n'])),side='left'));points.append(j)
  assert N[j]>=target*c['n'] and (j==0 or N[j-1]<target*c['n'])
  out.append(dict(model=model,period=per,target_coverage=target,diagnostic_confidence_boundary=float(u[j]),tie_group_n=int(n[j]),previous_accepted=int(N[j]-n[j]),accepted=int(N[j]),total=c['n'],actual_coverage=float(coverage[j]),errors=int(FP[j]+FN[j]),risk=float(risk[j]),accepted_tn=int(TN[j]),accepted_tp=int(TP[j]),accepted_fp=int(FP[j]),accepted_fn=int(FN[j]),total_true_benign=c['true_benign'],total_true_attack=c['true_attack'],benign_recall_all_true=ratio(TN[j],c['true_benign']),attack_recall_all_true=ratio(TP[j],c['true_attack']),tp_fraction_all_attacks=ratio(TP[j],c['true_attack']),benign_recall_within_accepted_true=ratio(TN[j],TN[j]+FP[j]),attack_recall_within_accepted_true=ratio(TP[j],TP[j]+FN[j])))
 ii=np.unique(np.r_[np.linspace(0,len(N)-1,min(len(N),3000)).astype(int),points]);
 for j in ii:display.append(dict(model=model,period=per,actual_coverage=float(coverage[j]),risk=float(risk[j]),accepted=int(N[j]),errors=int(FP[j]+FN[j]),confidence=float(u[j])))
 fig,ax=plt.subplots(figsize=(8,5));ax.plot(coverage[ii]*100,risk[ii]*100,label='Raw confidence ordering (whole ties)');ax.scatter(coverage[points]*100,risk[points]*100,s=24,color='black',label='Fixed target points');ax.axhline((c['fp']+c['fn'])/c['n']*100,color='gray',ls='--',label='Raw full acceptance');ax.set(xlabel='Actual acceptance (%)',ylabel='Raw error risk (%)',xlim=(0,100),ylim=(0,100),title=f'{model} | all true / predicted classes | {per}\nrank [{PERIODS[per][0]}, {PERIODS[per][1]}) | N={c["n"]:,}');ax.legend(fontsize=8);ax.grid(alpha=.25);export(fig,f'{model}_{per}_raw_risk_coverage')

def make_ecdf(model,k,sg,frozen):
 cls=['Benign','Attack'][k]
 for lo,hi in ([(0,1)] if k==0 else [(0,1),(.9,1),(.99,1)]):
  fig,ax=plt.subplots(figsize=(10,6));grid=np.unique(np.r_[np.linspace(lo,hi,2001),*[g[0][(g[0]>=lo)&(g[0]<=hi)] for name,g in sg.items() if name in SCHEMES],*[frozen[s]['q_'+cls] for s in SCHEMES]])
  grid=grid[(grid>=lo)&(grid<=hi)]
  for name in SCHEMES+['overall','E1','E2','E3','E4']:
   x,w=sg[name];cs=np.r_[0,np.cumsum(w)];cdf=cs[np.searchsorted(x,grid,side='right')]/w.sum();ax.step(grid,cdf,where='post',label=f'{name} (n={w.sum():,})',lw=2 if name in SCHEMES else 1.2,ls='--' if name in SCHEMES else '-')
  for s,col in zip(SCHEMES,['black','magenta']):ax.axvline(frozen[s]['q_'+cls],color=col,ls=':',label=f'{s} frozen q={frozen[s]["q_"+cls]:.10g}')
  ax.set(xlim=(lo,hi),ylim=(0,1),xlabel=f'True-class nonconformity score (unitless) | zoom [{lo}, {hi}]',ylabel='Normalized empirical CDF (fraction)',title=f'{model} | true {cls} | calibration vs ToN evaluation\nstartup [0,100000); evaluation [100000,27520000); E1-E4 as protocol');ax.legend(fontsize=8,loc='best');ax.grid(alpha=.25);export(fig,f'{model}_{cls}_ecdf_{lo:g}_{hi:g}')

def make_binplot(model,allrows,intervals):
 fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True)
 for ax,sub in zip(axes,['all','predicted_Benign','predicted_Attack']):
  rr=[r for r in allrows if r['model']==model and r['period']=='overall' and r['prediction_subset']==sub];ci=[r for r in intervals if r['model']==model and r['prediction_subset']==sub];valid=[i for i,r in enumerate(rr) if r['n']];yv=np.array([rr[i]['error_rate']*100 for i in valid]);ax.plot(valid,yv,'o-',label='Observed error (%)')
  ax.vlines(valid,[ci[i]['low_95']*100 for i in valid],[ci[i]['high_95']*100 for i in valid],color='C0',alpha=.7,label='Block bootstrap 95% CI')
  ax.plot(valid,[(1-rr[i]['mean_confidence'])*100 for i in valid],'s--',label='1 - mean confidence (%)')
  ax.set_xticks(range(8),[f'{binlabel(i)}\nn={rr[i]["n"]:,}' for i in range(8)],rotation=65,ha='right',fontsize=7);ax.set(title=f'{sub}; N={sum(r["n"] for r in rr):,}',xlabel='Confidence bin (unitless)',ylim=(0,100));ax.grid(alpha=.2)
 axes[0].set_ylabel('Error rate (%)');axes[0].legend(fontsize=8);fig.suptitle(f'{model} | all true classes | overall rank [100000,27520000)');fig.tight_layout();export(fig,f'{model}_confidence_bins')

if __name__=='__main__':
 try:main()
 except Exception as e:
  js('failure.json',dict(error=repr(e),time=now(),checks=CHECK));raise
