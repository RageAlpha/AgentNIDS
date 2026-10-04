"""Independent startup-only review; no original analysis code imported/executed."""
import csv,json,itertools
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path('/home/karri/projects/NIDSProject_Professor_Run');O=Path(__file__).resolve().parent
OLD=R/'outputs/post_p7_diagnostics/cic_bot_startup_support/run_20260917T014551Z'
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def write(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def save(n,x):
 with (O/n).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def inventory():return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in OLD.rglob('*') if p.is_file()}
def log(t):print(datetime.now(timezone.utc).isoformat(),t,flush=True)
def main():
 before=inventory();protocol=read(OLD/'protocol.json')
 # Read original implementation and all delivered results without running its code.
 original_code=(OLD/'check_support.py').read_text();oldreport=(OLD/'startup_support_report.md').read_text()
 oldcsv={p.name:rows(p) for p in OLD.glob('*.csv')};validation=read(OLD/'validation.json')
 assert protocol['rank_range']==[0,100000] and protocol['windows']==10 and protocol['window_size']==10000
 pre=read(protocol['preflight_protocol']);assert pre['startup']['rank_interval']==[0,100000]
 scopes={r['domain']:r for r in rows(protocol['preflight_scope'])}
 manifest=rows(protocol['window_manifest'])
 fm=read(R/'outputs/feature_store_v2/feature_store_manifest.json');assert fm['row_order']=='availability_sorted_rank'
 featurecode=(R/'src/nids/data/feature_store.py').read_text();sortcode=(R/'outputs/round1_data_prep_v1/prepare_data.py').read_text()
 assert 'sorted_y[start:stop] = raw_y[selected]' in featurecode and 'np.argsort(ends, kind="stable")' in sortcode
 differences=[];compared=0
 def check(context,expected,actual):
  nonlocal compared
  for k,v in expected.items():
   compared+=1
   if str(actual.get(k))!=str(v):differences.append(dict(context=context,field=k,reference=str(v),original=str(actual.get(k))))
 summaries=[];positions=[];windows=[];rawchecks=[]
 for d in ['CIC','BoT']:
  paths=protocol['inputs'][d]
  # Read array metadata, but only startup elements, never downstream labels.
  ymap=np.load(paths['labels'],mmap_mode='r');omap=np.load(paths['mapping'],mmap_mode='r');tmap=np.load(paths['times'],mmap_mode='r')
  assert ymap.shape==omap.shape==tmap.shape==(read(paths['domain_manifest'])['rows'],)
  assert len(ymap)==int(scopes[d]['tail_rank_end_exclusive'])
  yy=ymap[:100000].copy();orig=omap[:100000].copy();ends=tmap[:100000].copy()
  assert len(set(orig.tolist()))==100000 and min(orig)>=0 and max(orig)<len(omap)
  assert np.all(ends[1:]>=ends[:-1]) and np.all(orig[1:][ends[1:]==ends[:-1]]>orig[:-1][ends[1:]==ends[:-1]])
  ws=[w for w in manifest if w['domain']==d and w['scheme']=='heldout' and w['partition']=='T_heldout' and w['window_size']=='10000' and int(w['rank_start'])<100000]
  assert len(ws)==10 and ws[0]['window_id']==scopes[d]['first_startup_window_id'] and ws[-1]['window_id']==scopes[d]['last_startup_window_id']
  assert [int(scopes[d][k]) for k in ['startup_rank_start','startup_rank_end_exclusive','startup_rows','startup_complete_windows']]==[0,100000,100000,10]
  # Parse only selected original rows. Validate quote pairing on skipped physical
  # lines too, so original row indexing cannot silently cross a multiline record.
  wanted={int(v):i for i,v in enumerate(orig)};raw_y=[None]*100000;raw_t=[None]*100000;family_errors=0;decoded=0
  with Path(paths['raw_csv']).open('rb') as f:
   header=next(csv.reader([f.readline().decode('utf-8-sig')]));li=header.index('Label');ti=header.index('FLOW_END_MILLISECONDS');ai=header.index('Attack')
   for rowno,line in enumerate(itertools.islice(f,int(max(orig))+1)):
    assert line.count(b'"')%2==0,(d,rowno,'possible multiline CSV: stop')
    if rowno not in wanted:continue
    fields=next(csv.reader([line.decode('utf-8')]));assert len(fields)==len(header)
    rank=wanted[rowno];label=int(fields[li].strip());assert label in (0,1)
    raw_y[rank]=label;raw_t[rank]=int(fields[ti]);decoded+=1
    family_errors+=int((fields[ai].strip().lower()=='benign')!=(label==0))
  assert decoded==100000
  labeldiff=sum(int(a)!=b for a,b in zip(yy,raw_y));timediff=sum(int(a)!=b for a,b in zip(ends,raw_t))
  assert labeldiff==timediff==family_errors==0
  # Independent Python chronological cumulative counting, based on raw CSV labels.
  count=[0,0];first200={};window_counts=[[0,0] for _ in range(10)]
  for rank,label in enumerate(raw_y):
   count[label]+=1;window_counts[rank//10000][label]+=1
   if count[label]==200:first200[label]=rank
  for i,w in enumerate(ws):
   lo,hi=i*10000,(i+1)*10000
   assert [int(w[k]) for k in ['rank_start','rank_end_exclusive','first_original_row','last_original_row','first_available_ms','last_available_ms']]==[lo,hi,int(orig[lo]),int(orig[hi-1]),raw_t[lo],raw_t[hi-1]]
   assert w['predecessor_window_id']==('' if i==0 else ws[i-1]['window_id'])
   nb,na=window_counts[i];assert nb+na==10000
   ref=dict(domain=d,window_id=w['window_id'],rank_start=lo,rank_end_exclusive=hi,total=10000,benign=nb,attack=na)
   old=next(x for x in oldcsv['startup_window_class_counts.csv'] if x['window_id']==w['window_id']);check('window/'+w['window_id'],ref,old);windows.append(ref)
  assert sum(count)==100000 and [sum(v[i] for v in window_counts) for i in [0,1]]==count
  for label in [0,1]:
   rank=first200.get(label)
   if rank is not None:assert raw_y[:rank].count(label)==199 and raw_y[:rank+1].count(label)==200
   ref=dict(domain=d,class_label=label,class_name=['Benign','Attack'][label],startup_class_count=count[label],rank_200th=rank if rank is not None else 'NA',
    original_row_200th=int(orig[rank]) if rank is not None else 'NA',window_id=ws[rank//10000]['window_id'] if rank is not None else 'NA',
    scanned_rows_to_200th=rank+1 if rank is not None else 'NA',rank_start=0,rank_end_exclusive=100000)
   old=next(x for x in oldcsv['class_200th_positions.csv'] if x['domain']==d and int(x['class_label'])==label);check('position/'+d+'/'+str(label),ref,old);positions.append(ref)
  nb,na=count;both=min(count)>=200
  ref=dict(domain=d,rank_start=0,rank_end_exclusive=100000,total=100000,benign=nb,benign_denominator=100000,benign_fraction=nb/100000,
   attack=na,attack_denominator=100000,attack_fraction=na/100000,benign_at_least_200=nb>=200,attack_at_least_200=na>=200,both_200=both,
   scan_rows_to_collect_200_each=max(first200.values())+1 if both else 'NA',raw_label_verification='PASS')
  old=next(x for x in oldcsv['startup_class_support.csv'] if x['domain']==d);check('summary/'+d,ref,old);summaries.append(ref)
  for oldrow in [x for group in oldcsv.values() for x in group if x.get('domain')==d]:
   for field,key in [('label_path','labels'),('mapping_path','mapping'),('raw_csv_path','raw_csv')]:
    if field in oldrow:check('input_path/'+d,{field:paths[key]},oldrow)
  expected_line='| '+' | '.join(map(str,[d,100000,nb,f'{nb/1000:.3f}%',na,f'{na/1000:.3f}%',both,old['reply_eligibility']]))+' |'
  assert expected_line in oldreport
  rawchecks.append(dict(domain=d,raw_labels_compared=decoded,label_differences=labeldiff,timestamp_identity_differences=timediff,label_family_differences=family_errors,
   evaluation_labels_parsed=0,selected_original_rows_unique=True,labels_already_rank_sorted=True,positions_cumulative_count_verified=True))
  log(f'{d}: independent raw-label counts {count}; original result differences so far {len(differences)}')
 assert len(oldcsv['startup_class_support.csv'])==2 and len(oldcsv['startup_window_class_counts.csv'])==20 and len(oldcsv['class_200th_positions.csv'])==4
 save('reference_support.csv',summaries);save('reference_positions.csv',positions);save('reference_window_counts.csv',windows);save('raw_comparison.csv',rawchecks)
 save('differences.csv',differences or [dict(status='NO_DIFFERENCES',count=0)])
 unchanged=before==inventory();assert unchanged and not differences
 result=dict(status='PASS',reviewed_run=str(OLD),completed_at_utc=datetime.now(timezone.utc).isoformat(),comparison_fields=compared,differences=len(differences),
  startup_labels_independently_verified=200000,domains=rawchecks,original_files_size_mtime_unchanged=unchanged,
  method='independent raw CSV cumulative counts; original code read only, not imported or executed',
  no_evaluation_labels=True,no_models_probabilities_thresholds=True,no_hash_operations=True,ToN_No_go_unchanged=True)
 write('validation.json',result)
 write('protocol.json',dict(review_scope='latest CIC/BoT startup support task',original_run=str(OLD),inputs=protocol['inputs'],rank_range=[0,100000],window_manifest=protocol['window_manifest'],preflight_scope=protocol['preflight_scope'],
  row_definition=protocol['row_definition'],label_order='existing availability rank; no reordering',read_scope='decode labels only at startup original rows; skip unrelated physical CSV records'))
 text='# CIC / BoT Start-up Period Support quantity independent Review\n\n**PASS: No error was found and the original conclusions remained unchanged.**\n\nUsing independent Python as the cumulative number of rank, read labels directly from the original CSV correspondence; No call or re-enactment of the original check_support.py. The original document is read-only and no extracts are counted.\n\n| range | total | Benign | Attack | 200+200 | alignment required scanning rules |\n| --- | --- | --- | --- | --- | --- |\n'
 for r in summaries:text+=f"| {r['domain']} | 100000 | {r['benign']} | {r['attack']} | {r['both_200']} | {r['scan_rows_to_collect_200_each']} |\n"
 text+='\nCIC has 0 clause Attack and, according to reply section 5, if continued in this domain, Variant A only should be adopted. Both classes of BoT reach 200 and meet only quantitative conditions; This does not establish temporal representativeness, Mondrian validity, or authorization to start Week1. No-go of ToN is unchanged.\n\n| range | category | 200 article rank | original CSV line number | window_id | scanning number |\n| --- | --- | --- | --- | --- | --- |\n'
 for r in positions:text+='| '+' | '.join(str(r[k]) for k in ['domain','class_name','rank_200th','original_row_200th','window_id','scanned_rows_to_200th'])+' |\n'
 text+=f'\nindependent has verified the original labels of the 100,000 clause in each domain, the total number of 20 original windows, the 200 position of the 4 clause, the overall proportion and conditional qualification, total.{compared} The difference between the two fields is 0. Prior to Article 200, Article 199, containing that Article 200; Inadequacy is NA. The original labels corresponds to the caching difference 0, the original end time corresponds to the mapping difference 0, labels and Benign corresponding to the letter mapping.\n\nlabels caches have been sorted according to availability rank and have not been replicated using original_row mapping. The endpoints of the original window, the front and back, the Strict-K range, the length of the array, the timestamp and the sequence of the same timestamp are consistent. Read only the previous 100,000 cache of labels; CSV sequence scans by skipping unrelated lines, not analyzing their labels, and checking the physical line arithmetic pairing to exclude line errors caused by cross-line recording. No rearrangement or checking of labels ratings.\n\nAll figures and the original report are consistent. This time only completed identity, labels, counting and location verification within the existing start-up period, without extending the re-construction to the full range of sequencing correctness. No new calibration samples were generated, no predicted probabilities or thresholds, no models or other experiments were run, no messages were sent.\n\nThe original input absolute path, rank range, line definition is maintained at protocol.json; independent windows and location reference tables, differences records, scripts, commands and logs are kept in this directory. Stop when it’s done.\n '
 (O/'review_report.md').write_text(text);log('Review PASS; stopped.')
if __name__=='__main__':main()
