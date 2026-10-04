import csv,json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT=Path(__file__).resolve().parent
PREF=ROOT/'outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z'
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvread(p): return list(csv.DictReader(Path(p).open()))
def csvwrite(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  w.writerows({k:('NA' if v is None else v) for k,v in r.items()} for r in rows)
def inventory():
 return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for b in ['outputs','configs','docs','src','datasets','logs','scripts'] for p in (ROOT/b).rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
def file_record(p,role='inherited',version='SHA256 snapshot'):
 p=Path(p); return dict(path=str(p),version=version,sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,role=role)

def summarize(t):
 """Sufficient counts axes: true label(0/1), raw prediction(0/1), set code(0..3)."""
 t=np.asarray(t,dtype=np.int64);assert t.shape==(2,2,4) and (t>=0).all()
 N=int(t.sum()); B=int(t[0].sum()); A=int(t[1].sum())
 sets=t.sum(axis=(0,1)); accepted=int(sets[1]+sets[2]); rejected=N-accepted
 TN=int(t[0,:,1].sum()); FP=int(t[0,:,2].sum()); FN=int(t[1,:,1].sum()); TP=int(t[1,:,2].sum())
 rawTN=int(t[0,0].sum());rawFP=int(t[0,1].sum());rawFN=int(t[1,0].sum());rawTP=int(t[1,1].sum())
 errors=rawFP+rawFN; Er=int(t[0,1,[0,3]].sum()+t[1,0,[0,3]].sum()); Ea=errors-Er
 cb=int(t[0,:,[1,3]].sum());ca=int(t[1,:,[2,3]].sum())
 r=dict(rows=N,benign=B,attack=A,empty_count=int(sets[0]),benign_singleton_count=int(sets[1]),attack_singleton_count=int(sets[2]),multi_label_count=int(sets[3]),accepted=accepted,rejected=rejected,prediction_set_covered=cb+ca,benign_covered=cb,attack_covered=ca,TN=TN,FP=FP,FN=FN,TP=TP,raw_TN=rawTN,raw_FP=rawFP,raw_FN=rawFN,raw_TP=rawTP,raw_errors=errors,raw_correct=N-errors,rejected_raw_errors=Er,accepted_raw_errors=Ea,rejected_raw_correct=rejected-Er)
 def rate(k,num,den):
  r[k]=num/den if den else None;r[k+'_numerator']=int(num);r[k+'_denominator']=int(den)
 for k,num,den in [('singleton_rate',accepted,N),('acceptance_rate',accepted,N),('multi_label_rate',sets[3],N),('empty_set_rate',sets[0],N),('mean_set_size',accepted+2*sets[3],N),('prediction_set_coverage',cb+ca,N),('benign_prediction_set_coverage',cb,B),('attack_prediction_set_coverage',ca,A),('rejection_rate',rejected,N),('raw_error_rate',errors,N),('raw_error_given_rejected',Er,rejected),('raw_error_given_accepted',Ea,accepted),('rejected_given_raw_error',Er,errors),('rejected_given_raw_correct',rejected-Er,N-errors),('selective_risk',FP+FN,accepted),('benign_acceptance_rate',TN+FP,B),('attack_acceptance_rate',FN+TP,A),('accepted_benign_FPR',FP,TN+FP),('accepted_attack_miss_rate',FN,FN+TP),('effective_attack_detection',TP,A),('wrong_accepted_attack_fraction',FN,A),('attack_abstention_rate',A-TP-FN,A),('raw_attack_recall',rawTP,A),('raw_benign_FPR',rawFP,B)]:rate(k,int(num),int(den))
 r['raw_error_rate_gap']=None if rejected==0 or accepted==0 else Er/rejected-Ea/accepted
 r['rejection_rate_error_correct_gap']=None if errors==0 or N-errors==0 else Er/errors-(rejected-Er)/(N-errors)
 assert TN+FP+FN+TP==accepted and Ea==FP+FN
 assert int(t[0,1,1]+t[1,1,1]+t[0,0,2]+t[1,0,2])==0,'singleton differs from raw prediction'
 if N:
  assert math.isclose(r['singleton_rate']+r['multi_label_rate']+r['empty_set_rate'],1,abs_tol=1e-15)
  assert math.isclose(r['mean_set_size'],r['singleton_rate']+2*r['multi_label_rate'],abs_tol=1e-15)
 return r

def table_fields(r,keys):
 result={k:r[k] for k in ['domain','rows','benign','attack'] if k in r}
 for k in keys:
  result[k]=r[k]
  for suffix in ['_numerator','_denominator']:
   if k+suffix in r:result[k+suffix]=r[k+suffix]
 return result

BOOT_METRICS=['prediction_set_coverage','rejection_rate','acceptance_rate','singleton_rate','multi_label_rate','empty_set_rate','mean_set_size','benign_prediction_set_coverage','attack_prediction_set_coverage','selective_risk','raw_error_given_rejected','raw_error_given_accepted','rejected_given_raw_error','rejected_given_raw_correct','effective_attack_detection','wrong_accepted_attack_fraction','attack_abstention_rate','benign_acceptance_rate','attack_acceptance_rate','accepted_benign_FPR','accepted_attack_miss_rate','raw_error_rate_gap','rejection_rate_error_correct_gap']
