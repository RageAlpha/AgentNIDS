from pathlib import Path
import json,csv,hashlib,ast
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run')
base=json.loads((R/'outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/upstream_inventory_after.json').read_text());changes=[]
for name,s in base.items():
 p=Path(name)
 if not p.exists() or p.stat().st_size!=s['bytes'] or p.stat().st_mtime_ns!=s['mtime_ns']:changes.append(name)
assert not changes,changes
for p in O.glob('*.json'):json.loads(p.read_text())
for p in O.glob('*.py'):ast.parse(p.read_text())
v=json.loads((O/'validation.json').read_text());assert v['full_four_feature_elements_verified']==119541696 and v['full_four_feature_mismatches']==0 and not v['q_reselected'] and not v['model_retrained_or_modified']
report=(O/'root_cause_report.md').read_text();assert f'{28401/235424*100:.4f}%' in report
res={'status':'PASS','completed_at':datetime.now(timezone.utc).isoformat(),'old_registered_files_metadata_checked':len(base),'old_registered_file_changes':changes,'json_and_python_syntax':'PASS','limits':'full hashes of used inputs in source_hashes.csv; full all49 backend reinference not claimed'}
(O/'deliverable_validation.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
with (O/'deliverable_hashes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
 for p in sorted(O.iterdir()):
  if p.is_file() and p.name not in ['deliverable_hashes.csv','delivery.log']:w.writerow({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps(res,ensure_ascii=False))
