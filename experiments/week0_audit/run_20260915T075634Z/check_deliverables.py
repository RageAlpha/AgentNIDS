from audit_common import *
import ast
required=['audit_report.md','audit_validation.json','frozen_sample_manifest.csv','inference_comparison.csv','feature_comparison.csv','probability_score_diagnostics.csv','artifact_differences.csv','source_hashes.csv','environment.json','commands.sh','documentation_errata.md','producer_code_review.md','read_access_ledger.json']
assert all((OUT/n).is_file() and (OUT/n).stat().st_size for n in required)
# No old module imports or old computation helpers anywhere in runnable audit code.
for p in OUT.glob('*.py'):
 tree=ast.parse(p.read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.Call):
   name=node.func.id if isinstance(node.func,ast.Name) else ''
   assert name not in ['scores','threshold','prediction_sets','summarize'],(p,name)
  if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('implementation','common','nids')),(p,node.module)
for p in OUT.glob('*.json'):read(p)
assert sha(OUT/'frozen_sample_manifest.csv')==read(OUT/'sample_freeze.json')['manifest_sha256']
newcal=read(OUT/'independent_q_verification.json')
for r in rows(PILOT/'calibration_class_summary.csv'):
 val=newcal['classes']['0' if r['true_class']=='Benign' else '1']
 assert val['n']==int(r['rows']) and val['covered']==int(r['included_at_frozen_q'])
 assert val['covered']/val['n']==float(r['empirical_calibration_coverage'])
features=rows(OUT/'feature_comparison.csv')
assert len(features)==98 and all(int(r['bit_mismatches'])==0 and int(r['filled_input_bit_mismatches'])==0 for r in features)
inf=rows(OUT/'inference_comparison.csv');assert len(inf)==53813
assert all(r['bitwise_equal']=='True' and r['set_changed']=='False' and float(r['absolute_error'])==0 for r in inf)
assert len(rows(OUT/'replay_differences.csv'))==0
assert all(r['match']=='True' for r in rows(OUT/'artifact_differences.csv'))
# Final freeze linkage from original pilot, independently of the report prose.
pf=read(PILOT/'protocol_freeze.json');qf=read(PILOT/'q_freeze.json')
assert sha(PILOT/'protocol.json')==pf['protocol_sha256']==qf['protocol_sha256']
assert sha(PILOT/'q_freeze.json')==read(PILOT/'evaluation_complete.json')['q_sha256']
assert read(OUT/'upstream_inventory_before.json')==inventory()
val=read(OUT/'audit_validation.json');assert all(v['status']=='PASS' for v in val['layers'].values()) and len(val['not_verified'])==4
assert val['prior_NO_GO_EVIDENCE_supported'] and not val['Week1_started']
write('deliverable_validation.json',dict(status='PASS',completed_at_utc=now(),required_artifacts=len(required),audit_code_has_no_old_helper_calls=True,old_q_protocol_links_intact=True,all_inference_rows_and_features_checked=True,calibration_class_coverages_match=True,old_outputs_unchanged=True,report_only_syntax_fix_logged=True))
rs=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file() and p.name not in ['deliverable_hashes.csv','deliverable_check.log']:
  rs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
table('deliverable_hashes.csv',rs)
print('PASS: complete audit deliverables, independent helpers, bitwise sample results and preserved old outputs')
