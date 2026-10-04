import csv,json,hashlib,math,sys,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run');OUT=Path(__file__).resolve().parent
PILOT=ROOT/'outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z';PREF=ROOT/'outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(n,o): (OUT/n).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rs,fields=None):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows({k:'NA' if v is None else v for k,v in r.items()} for r in rs)
def rows(p):return list(csv.DictReader(Path(p).open()))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory():return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for base in ['outputs','docs','configs','src','datasets','scripts','logs'] for p in (ROOT/base).rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
def reference_metrics(t):
 # Independent masks over 16 integer cells, no pilot helpers imported.
 y,r,c=np.indices((2,2,4));T=np.asarray(t,dtype=np.int64);N=int(T.sum())
 def count(mask):return int(T[mask].sum())
 B=count(y==0);A=count(y==1);acc=(c==1)|(c==2);rej=~acc;err=y!=r
 tn=count((y==0)&(c==1));fp=count((y==0)&(c==2));fn=count((y==1)&(c==1));tp=count((y==1)&(c==2))
 ac=count(acc);re=count(rej);E=count(err);ER=count(err&rej);EC=count(~err&rej)
 cb=count((y==0)&((c==1)|(c==3)));ca=count((y==1)&((c==2)|(c==3)))
 d=dict(rows=N,benign=B,attack=A,empty_count=count(c==0),benign_singleton_count=count(c==1),attack_singleton_count=count(c==2),multi_label_count=count(c==3),accepted=ac,rejected=re,prediction_set_covered=cb+ca,benign_covered=cb,attack_covered=ca,TN=tn,FP=fp,FN=fn,TP=tp,raw_TN=count((y==0)&(r==0)),raw_FP=count((y==0)&(r==1)),raw_FN=count((y==1)&(r==0)),raw_TP=count((y==1)&(r==1)),raw_errors=E,raw_correct=N-E,rejected_raw_errors=ER,accepted_raw_errors=E-ER,rejected_raw_correct=EC)
 for name,num,den in [('singleton_rate',ac,N),('acceptance_rate',ac,N),('multi_label_rate',d['multi_label_count'],N),('empty_set_rate',d['empty_count'],N),('mean_set_size',ac+2*d['multi_label_count'],N),('prediction_set_coverage',cb+ca,N),('benign_prediction_set_coverage',cb,B),('attack_prediction_set_coverage',ca,A),('rejection_rate',re,N),('raw_error_rate',E,N),('raw_error_given_rejected',ER,re),('raw_error_given_accepted',E-ER,ac),('rejected_given_raw_error',ER,E),('rejected_given_raw_correct',EC,N-E),('selective_risk',fp+fn,ac),('benign_acceptance_rate',tn+fp,B),('attack_acceptance_rate',fn+tp,A),('accepted_benign_FPR',fp,tn+fp),('accepted_attack_miss_rate',fn,fn+tp),('effective_attack_detection',tp,A),('wrong_accepted_attack_fraction',fn,A),('attack_abstention_rate',A-tp-fn,A),('raw_attack_recall',d['raw_TP'],A),('raw_benign_FPR',d['raw_FP'],B)]:
  d[name]=num/den if den else None;d[name+'_numerator']=num;d[name+'_denominator']=den
 d['raw_error_rate_gap']=None if not re or not ac else ER/re-(E-ER)/ac
 d['rejection_rate_error_correct_gap']=None if not E or not N-E else ER/E-EC/(N-E)
 return d
