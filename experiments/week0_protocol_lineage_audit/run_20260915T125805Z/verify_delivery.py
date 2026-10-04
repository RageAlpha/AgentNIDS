from pathlib import Path
import csv,json,hashlib,ast
from datetime import datetime,timezone
O=Path(__file__).resolve().parent
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
required=['audit_report.md','roadmap_compliance_matrix.csv','asset_lineage.csv','historical_stage_impact.csv','unresolved_items.csv','audit_validation.json','audit.py','write_report.py','finalize.py','commands.sh','audit.log','source_hashes.csv','artifact_differences.csv','environment.json','manifest_join_verification.json','sample_coverage_verification.json']
assert all((O/n).is_file() and (O/n).stat().st_size>0 for n in required)
for p in O.glob('*.json'):json.loads(p.read_text())
for p in O.glob('*.py'):ast.parse(p.read_text())
mat=rows(O/'roadmap_compliance_matrix.csv')
for r in mat:
 path,line=r['original_location'].rsplit(':',1);source=Path(path).read_text().splitlines();excerpt='\n'.join(source[int(line)-1:int(line)-1+len(r['original_excerpt'].splitlines())]);assert excerpt==r['original_excerpt']
assert all(r['classification'] in {'EXPLICIT_MATCH','IMPLEMENTATION_DETAIL','DEVIATION','AMBIGUOUS','NOT_VERIFIED'} for r in mat)
v=json.loads((O/'audit_validation.json').read_text());assert v['overall_conclusion'].startswith('B.') and not v['Week1_started'] and not v['new_inference_or_experiments']
stat=json.loads((O/'upstream_inventory_before.json').read_text());changed=[]
for name,s in stat.items():
 p=Path(name)
 if not p.exists() or p.stat().st_size!=s['bytes'] or p.stat().st_mtime_ns!=s['mtime_ns']:changed.append(name)
assert not changed,changed
out={'status':'PASS','completed_at_utc':datetime.now(timezone.utc).isoformat(),'required_files':len(required),'roadmap_exact_excerpt_checks':len(mat),'all_json_parse':True,'all_python_parse':True,'old_artifacts_stat_checked':len(stat),'old_artifacts_changed':changed,'note':'No new inference, threshold computation or historical experiment executed; source byte hashes in source_hashes.csv and artifact_differences.csv'}
(O/'deliverable_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
with (O/'deliverable_hashes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
 for p in sorted(O.rglob('*')):
  if p.is_file() and p.name not in {'deliverable_hashes.csv','delivery.log'}:
   w.writerow({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps(out,ensure_ascii=False))
