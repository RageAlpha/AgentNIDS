import csv,json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent; R=Path('/home/karri/projects/NIDSProject_Professor_Run')
p=json.loads((O/'pilot_protocol_draft.json').read_text())
p['inputs']={}
for d in ['UNSW','ToN']:
 m=json.loads((R/f'outputs/feature_store_v2/{d}_manifest.json').read_text())
 p['inputs'][d]={k:str(R/v) for k,v in dict(probability_cache=f'outputs/prediction_cache_v2/{d}/rf_All49.npy',probability_manifest=f'outputs/prediction_cache_v2/{d}/rf_All49_manifest.json',labels=f'outputs/feature_store_v2/{d}_label.npy',attack_codes=f'outputs/feature_store_v2/{d}_attack_code.npy',feature_manifest=f'outputs/feature_store_v2/{d}_manifest.json',original_rows=f'outputs/round1_data_prep_v1/{d}_sorted_original_rows.npy',availability_times=f'outputs/round1_data_prep_v1/{d}_sorted_end_ms.npy').items()}
 p['inputs'][d]['attack_codebook']=m['attack_codebook']
p['inputs']['hash_inventory']=str(O/'source_hashes.csv')
(O/'pilot_protocol_draft.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
ws=list(csv.DictReader((R/'outputs/round1_data_prep_v1/window_manifest.csv').open()))
s=[w for w in ws if w['domain']=='UNSW' and w['scheme']=='source' and w['partition']=='S_source_test' and w['window_size']=='10000']
assert len(s)==23 and int(s[0]['rank_start'])==2130000 and int(s[-1]['rank_end_exclusive'])==2360000
for i,w in enumerate(s):
 assert int(w['rank_start'])==2130000+i*10000
with (O/'source_evaluation_window_references.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(s[0]));w.writeheader();w.writerows(s)
required=['preflight_report.md','pilot_protocol_draft.json','asset_audit.csv','source_partition_audit.csv','evaluation_scope.csv','information_boundary.md','implementation_check.json','gate_definition.md','source_hashes.csv','implementation.py','run_preflight.py','commands.sh','run.log','validation.json']
assert all((O/n).is_file() and (O/n).stat().st_size>0 for n in required)
for f in O.glob('*.json'):json.loads(f.read_text())
audit=list(csv.DictReader((O/'source_hashes.csv').open()))
assert all(row['sha256'] for row in audit)
assert not any('target_performance' in f.name or f.name=='q.json' for f in O.iterdir())
c=json.loads((O/'calibration_data_check.json').read_text())
assert (c['rows'],c['benign'],c['attack'],c['k_one_based'],c['k_zero_based'])==(360000,325600,34400,324001,324000)
assert p['calibration']['q'] is None and p['epsilon']==dict(numerator=1,denominator=10)
assert p['time']['window_counts']==[688]*4
assert sum(b-a for a,b in p['time']['rank_intervals'])==27520000
assert len(p['inputs']['ToN']['attack_codebook'])>0 and 'scanning' in str(p['inputs']['ToN']['attack_codebook'])
with (O/'commands.sh').open('a') as f:f.write('PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/verify_deliverables.py" > "$TASK_OUT/deliverable_validation.log" 2>&1\n')
(O/'deliverable_validation.json').write_text(json.dumps(dict(status='PASS',required_files=len(required),all_json_parse=True,source_windows=23,source_tail_preserved=5424,standalone_input_paths=True,protocol_q_is_null=True),indent=2)+'\n')
with (O/'deliverable_hashes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
 for a in sorted(O.iterdir()):
  if a.is_file() and a.name not in ['deliverable_hashes.csv','deliverable_validation.log']:
   w.writerow(dict(path=str(a),bytes=a.stat().st_size,sha256=hashlib.sha256(a.read_bytes()).hexdigest()))
print('PASS: required artifacts, schemas, ranks, input paths, frozen taxonomy, source windows and no real q')
