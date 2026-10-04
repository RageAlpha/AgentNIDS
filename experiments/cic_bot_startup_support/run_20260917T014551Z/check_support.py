"""Read-only startup support check. No probability/model or evaluation-label access."""
import csv,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run');OUT=Path(__file__).resolve().parent
PRE=ROOT/'outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight'
ROWDEF='zero-based original CSV data row, header excluded; rank is existing availability order (FLOW_END_MILLISECONDS then original row)'
def j(p):return json.loads(Path(p).read_text())
def rr(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def wj(name,data):
 with (OUT/name).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
def wc(name,data):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def main():
 reply=ROOT/'docs/Ton_variant/Reply_to_Siyuan_Week0_Analysis_and_Next_Step.md'
 assert '## 5. CIC and BoT status before full Phase 2' in reply.read_text()
 pre=j(PRE/'protocol_draft.json');assert pre['startup']['rank_interval']==[0,100000] and pre['startup']['first_complete_windows']==10 and pre['startup']['window_size']==10000
 scopes={r['domain']:r for r in rr(PRE/'evaluation_scope.csv')}
 manifest=ROOT/'outputs/round1_data_prep_v1/window_manifest.csv';allwindows=rr(manifest)
 fm=j(ROOT/'outputs/feature_store_v2/feature_store_manifest.json')
 assert fm['row_order']=='availability_sorted_rank' and Path(fm['rank_source'])==ROOT/'outputs/round1_data_prep_v1'
 config=j(ROOT/'outputs/round1_data_prep_v1/round1_config.json')
 assert config['replay']['sort_keys']==['FLOW_END_MILLISECONDS','original_zero_based_row_index']
 code=(ROOT/'src/nids/data/feature_store.py').read_text();assert 'sorted_y[start:stop] = raw_y[selected]' in code
 inputs={};paths=[reply,PRE/'protocol_draft.json',PRE/'evaluation_scope.csv',manifest,ROOT/'outputs/feature_store_v2/feature_store_manifest.json',ROOT/'src/nids/data/feature_store.py',ROOT/'outputs/round1_data_prep_v1/round1_config.json']
 for d,file in [('CIC','NF-CICIDS2018-v3.csv'),('BoT','NF-BoT-IoT-v3.csv')]:
  inputs[d]=dict(labels=str(ROOT/f'outputs/feature_store_v2/{d}_label.npy'),mapping=str(ROOT/f'outputs/round1_data_prep_v1/{d}_sorted_original_rows.npy'),
   times=str(ROOT/f'outputs/round1_data_prep_v1/{d}_sorted_end_ms.npy'),raw_csv=str(ROOT/'datasets'/file),domain_manifest=str(ROOT/f'outputs/feature_store_v2/{d}_manifest.json'))
  paths += [Path(v) for v in inputs[d].values() if Path(v).exists()]
 before={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
 wj('protocol.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),authority=str(reply),authority_section=5,
  preflight_scope=str(PRE/'evaluation_scope.csv'),preflight_protocol=str(PRE/'protocol_draft.json'),window_manifest=str(manifest),inputs=inputs,
  rank_range=[0,100000],windows=10,window_size=10000,label_mapping={'0':'Benign','1':'Attack'},row_definition=ROWDEF,
  cache_order='availability rank already applied; labels sliced directly, never labels[original_rows]',
  raw_read_policy='stream bytes past unrelated CSV rows without parsing labels; parse only startup original-row records; no evaluation labels accessed',
  purpose='class support and 200th positions only; not a frozen calibration sample',ton_conclusion='No-go unchanged',no_Week1=True))
 summaries=[];windows=[];positions=[];checks=[];firstdiff=[]
 for d in ['CIC','BoT']:
  ip=inputs[d];a=np.load(ip['labels'],mmap_mode='r');order=np.load(ip['mapping'],mmap_mode='r');times=np.load(ip['times'],mmap_mode='r')
  dm=j(ip['domain_manifest']);assert a.ndim==order.ndim==times.ndim==1 and len(a)==len(order)==len(times)==dm['rows']
  assert len(a)==int(scopes[d]['tail_rank_end_exclusive'])
  # Only these label elements are accessed, never the downstream evaluation slice.
  y=np.array(a[:100000]);o=np.array(order[:100000]);t=np.array(times[:100000])
  assert len(y)==100000 and np.all((y==0)|(y==1)) and len(np.unique(o))==100000 and np.all((o>=0)&(o<len(a)))
  assert np.all(t[1:]>=t[:-1]);tie=t[1:]==t[:-1];assert np.all(o[1:][tie]>o[:-1][tie])
  ws=[r for r in allwindows if r['domain']==d and r['scheme']=='heldout' and r['partition']=='T_heldout' and r['window_size']=='10000' and int(r['rank_start'])<100000]
  # Preserve manifest order; no sorting or redefinition of the stream.
  assert len(ws)==10
  sc=scopes[d];assert [int(sc[k]) for k in ['startup_rank_start','startup_rank_end_exclusive','startup_rows','startup_complete_windows']]==[0,100000,100000,10]
  assert ws[0]['window_id']==sc['first_startup_window_id'] and ws[-1]['window_id']==sc['last_startup_window_id']
  assert int(sc['first_available_ms'])==t[0] and int(sc['startup_last_available_ms'])==t[-1]
  common=dict(label_path=ip['labels'],mapping_path=ip['mapping'],raw_csv_path=ip['raw_csv'],row_definition=ROWDEF)
  for i,w in enumerate(ws):
   lo,hi=int(w['rank_start']),int(w['rank_end_exclusive']);assert (lo,hi)==(i*10000,(i+1)*10000)
   assert [int(w[k]) for k in ['first_original_row','last_original_row','first_available_ms','last_available_ms']]==[int(o[lo]),int(o[hi-1]),int(t[lo]),int(t[hi-1])]
   assert w['predecessor_window_id']==('' if i==0 else ws[i-1]['window_id'])
   nb=int((y[lo:hi]==0).sum());na=int((y[lo:hi]==1).sum());assert nb+na==10000
   windows.append(dict(domain=d,window_id=w['window_id'],rank_start=lo,rank_end_exclusive=hi,total=hi-lo,benign=nb,attack=na,**common))
  nb=int((y==0).sum());na=int((y==1).sum());assert nb+na==100000
  assert sum(w['benign'] for w in windows if w['domain']==d)==nb and sum(w['attack'] for w in windows if w['domain']==d)==na
  ranks200=[]
  for label,name in [(0,'Benign'),(1,'Attack')]:
   found=np.flatnonzero(y==label);rank=int(found[199]) if len(found)>=200 else None
   if rank is not None:assert int((y[:rank]==label).sum())==199 and int((y[:rank+1]==label).sum())==200
   ranks200.append(rank)
   positions.append(dict(domain=d,class_label=label,class_name=name,startup_class_count=len(found),rank_200th=rank if rank is not None else 'NA',
    original_row_200th=int(o[rank]) if rank is not None else 'NA',window_id=ws[rank//10000]['window_id'] if rank is not None else 'NA',
    scanned_rows_to_200th=rank+1 if rank is not None else 'NA',rank_start=0,rank_end_exclusive=100000,**common))
  rawpath=Path(ip['raw_csv']);rawstatus='NOT_VERIFIED';labeldiff='NA';timediff='NA';mappingdiff='NA';parsed=0;traversed=0
  rawnb=rawna='NA'
  if rawpath.exists():
   wanted={int(original):rank for rank,original in enumerate(o)}
   ry=np.full(100000,-1,dtype=np.int8);rt=np.zeros(100000,dtype=np.int64);familydiff=0
   # These NetFlow CSVs are one physical line per data record. All selected records
   # must have the expected column count and exact mapped end timestamp.
   with rawpath.open('rb') as f:
    header=next(csv.reader([f.readline().decode('utf-8-sig').rstrip('\r\n')]))
    iy,ia,it=[header.index(k) for k in ['Label','Attack','FLOW_END_MILLISECONDS']]
    last=int(o.max());lastlog=time.monotonic()
    for original,line in enumerate(f):
     if original in wanted:
      rank=wanted[original];fields=next(csv.reader([line.decode('utf-8').rstrip('\r\n')]))
      assert len(fields)==len(header),(d,original,'CSV columns')
      yy=int(fields[iy].strip());assert yy in (0,1)
      ry[rank]=yy;rt[rank]=int(fields[it].strip());parsed+=1
      familydiff+=int((yy==0)!=(fields[ia].strip().lower()=='benign'))
     traversed=original+1
     if original==last:break
     if time.monotonic()-lastlog>20:log(f'{d}: traversed raw bytes for {traversed:,} physical records; decoded startup rows {parsed:,}');lastlog=time.monotonic()
   assert parsed==100000 and np.all(ry>=0)
   labeldiff=int((ry!=y).sum());timediff=int((rt!=t).sum());mappingdiff=timediff
   for rank in np.flatnonzero((ry!=y)|(rt!=t))[:20]:firstdiff.append(dict(domain=d,rank=int(rank),original_row=int(o[rank]),cache_label=int(y[rank]),raw_label=int(ry[rank]),cache_end_ms=int(t[rank]),raw_end_ms=int(rt[rank])))
   rawnb=int((ry==0).sum());rawna=int((ry==1).sum())
   assert labeldiff==timediff==familydiff==0,(d,'raw mismatch',labeldiff,timediff,familydiff)
   assert (rawnb,rawna)==(nb,na)
   rawstatus='PASS'
  both=nb>=200 and na>=200
  eligibility='meet the quantitative conditions of the 200+200 category; Quantitative only, not authorized to operate Mondrian or Week1' if both else 'The current 10 window does not meet the Mondrian quantitative requirements; Variant A only in reply if the field continues.'
  summaries.append(dict(domain=d,rank_start=0,rank_end_exclusive=100000,total=100000,benign=nb,benign_denominator=100000,benign_fraction=nb/100000,
   attack=na,attack_denominator=100000,attack_fraction=na/100000,benign_at_least_200=nb>=200,attack_at_least_200=na>=200,both_200=both,
   scan_rows_to_collect_200_each=max(ranks200)+1 if both else 'NA',raw_label_verification=rawstatus,reply_eligibility=eligibility,**common))
  checks.append(dict(domain=d,status='PASS',array_rows=len(a),array_lengths_match=True,label_elements_read=100000,evaluation_label_elements_read=0,
   complete_windows=10,invalid_labels=0,unique_original_rows=100000,availability_monotonic=True,original_row_tie_order=True,
   historical_scope_and_window_boundaries_match=True,window_counts_sum_to_overall=True,positions_200th_validated=True,
   labels_already_availability_sorted=True,raw_verification=rawstatus,raw_records_decoded=parsed,raw_physical_records_traversed=traversed,
   nonstartup_raw_labels_parsed=0,raw_label_mismatches=labeldiff,raw_time_identity_mismatches=timediff,raw_benign=rawnb,raw_attack=rawna))
  log(f'{d}: total=100000 Benign={nb} Attack={na} both200={both}; raw check {rawstatus}')
 wc('startup_class_support.csv',summaries);wc('startup_window_class_counts.csv',windows);wc('class_200th_positions.csv',positions)
 wc('raw_label_comparison.csv',[dict(domain=c['domain'],status=c['raw_verification'],compared_rows=c['raw_records_decoded'],label_differences=c['raw_label_mismatches'],timestamp_identity_differences=c['raw_time_identity_mismatches'],raw_benign=c['raw_benign'],raw_attack=c['raw_attack']) for c in checks])
 if firstdiff:wc('first_differences.csv',firstdiff)
 after={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths};assert before==after
 wj('validation.json',dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),domains=checks,inputs_unchanged_by_size_mtime=True,
  raw_verification_complete=all(c['raw_verification']=='PASS' for c in checks),file_digest_operations=False,new_calibration_sample=False,probabilities_read=False,thresholds_computed=False,Week1_started=False))
 def table(h,rr):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
 text='# CIC / BoT Start-up time category support quantity checks\n\nAccording to Professor reply 5 section. Only the existing rank [0,100000) was estimated, not using the old 400 sample or the full range category ratio. ToN Variant B No-go remains unchanged, this time not representing permission to continue the full Phase2.\n\n'
 text+=table(['domain','Total start-up period','Benign','Benign ratio','Attack','Attack ratio','200+200','Subsequent qualification by reply'],[[r['domain'],r['total'],r['benign'],f"{100*r['benign_fraction']:.3f}%",r['attack'],f"{100*r['attack_fraction']:.3f}%",r['both_200'],r['reply_eligibility']] for r in summaries])
 text+='\n\n## Location of 200\n\nBoth rank and original_row start with 0; The scanning bar number starts at 1. The original number does not contain the CSV headings. NA indicates a lack of 200 during the fixed start-up period.\n\n'+table(['domain','Categories','Actual support','Article 200 rank','Original CSV line','The original window_id','Number of scans'],[[r[k] for k in ['domain','class_name','startup_class_count','rank_200th','original_row_200th','window_id','scanned_rows_to_200th']] for r in positions])
 text+='\n\n'+table(['domain','The first 200+200 scan required to align'],[[r['domain'],r['scan_rows_to_collect_200_each']] for r in summaries])
 text+='\n\nThis time only records the number with the 200 position, and no new calibration samples have been registered. Even before reaching the 200, complete statistics of all 100,000 items are still available.\n\n## Original 10 window count\n\n'+table(['domain','window_id','rank_start','rank_end_exclusive','Total number','Benign','Attack'],[[r[k] for k in ['domain','window_id','rank_start','rank_end_exclusive','total','benign','attack']] for r in windows])
 text+='\n\n## sequence, identity and original labels verification\n\nStrict-K pre-examination is used only to confirm the range and window of the start-up period; Both the old 400 random sample and the old 20/ class thresholds are not used in this inspection. The 10k window with the original heldout/T_heldout is used, with the previous 10 window continuously covering the [0,100000). label.npy has been ranked according to availability rank and read [:100000] directly, without repeating the original mapping line. availability is FLOW_END_MILLISECONDS with the same time line sequence along the original CSV data line; Windows are not rearranged or expanded.\n\n'
 for c in checks:
  text+=f"- {c['domain']} Check the 100,000, 10 and complete windows; Two-way counting, window-by-window and overall, window endpoints are consistent with historical pre-checks, labels has no illegal / missing value, and identity is the only start-up period. Original CSV independent labels = test {c['raw_verification']} Just analyze.{c['raw_records_decoded']:,} the original records corresponding to the start-up period; labels Difference ={c['raw_label_mismatches']} The original end time is rank identity mapping difference ={c['raw_time_identity_mismatches']}。\n"
 text+='\nThe original CSV uses a sequence byte scan, skips the non-target line and does not parse its labels; Therefore, BoT passes through a larger CSV threshold, but no labels fields are statistically recorded or read for subsequent evaluation. Select the number of rows and finish time mapping simultaneously. labels is read directly from the original CSV Label, 0=Benign, 1=Attack, and checks whether the Benign is crossed with the Attack field. Prior to Article 200, both Articles 199 and 200 containing them have been validated; 200 notation NA does not exist.\n\n## input and boundary\n\n'
 for d,ip in inputs.items():text+='\n'+d+'：\n'+''.join(f'- {k}: `{v}`\n' for k,v in ip.items())
 text+=f'\nThe window manifest: `{manifest}`. Pre-examination range: `{PRE}/evaluation_scope.csv`; Protocol: `{PRE}/protocol_draft.json`. Sorting specification: `outputs/round1_data_prep_v1/round1_config.json`; labels generated: `src/nids/data/feature_store.py` is read as raw_y and sorted_y is logically written as order. All CSV results record absolute input paths, rank range and original line definition.\n '
 text+='\nthe number of categories is not sufficient to be calibration with time representation, is not equal to Mondrian effective and is not authorized to start Week1; When insufficient Variant A only is a conditional subsequent branch given by the professor, this time it does not run Variant A. No model or probability reading, no fit threshold, no modification of historical documents, no hash or SHA256 computational verification. When it is finished, stop and do not send messages to the professor.\n'
 (OUT/'startup_support_report.md').write_text(text)
 log('All required startup checks complete; stopped.')
if __name__=='__main__':
 try:main()
 except Exception as exc:
  wj('failure.json',dict(status='STOPPED',error=repr(exc),no_definition_repair=True));raise
