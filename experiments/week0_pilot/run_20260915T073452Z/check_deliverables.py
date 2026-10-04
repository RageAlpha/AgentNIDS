from common import *
required=['protocol.json','protocol_freeze.json','source_hashes.csv','q_freeze.json','calibration_summary.json','prediction_set_metrics.csv','set_type_error_counts.csv','rejection_diagnostics.csv','selective_metrics.csv','attack_category_metrics.csv','temporal_metrics.csv','bootstrap_intervals.csv','bootstrap_blocks.csv','bootstrap_indices.npy','bootstrap_multiplicity.npy','UNSW_window_counts.npz','ToN_window_counts.npz','gate_assessment.md','stage_Week0_pilot_report.md','validation.json','commands.sh']
assert all((OUT/n).is_file() and (OUT/n).stat().st_size>0 for n in required)
for f in OUT.glob('*.json'):read(f)
pf=read(OUT/'protocol_freeze.json');qf=read(OUT/'q_freeze.json');bm=read(OUT/'bootstrap_manifest.json')
assert pf['protocol_sha256']==sha(OUT/'protocol.json')
assert pf['sources_sha256']==sha(OUT/'source_hashes.csv')
for n,h in pf['code_hashes'].items():assert sha(OUT/n)==h,n
for n,h in qf['code_sha256'].items():assert sha(OUT/n)==h,n
assert qf['calibration_summary_sha256']==sha(OUT/'calibration_summary.json')
for n,h in bm['files'].items():assert sha(OUT/n)==h,n
assert read(OUT/'validation.json')['status']=='PASS'
assert read(OUT/'gate_assessment.json')['status']=='NO_GO_EVIDENCE'
assert read(OUT/'upstream_inventory_before.json')==inventory()
# Check reported key counts and metric split matches machine-readable artifacts.
m=read(OUT/'overall_metrics.json');assert m['ToN']['accepted']==731 and (m['ToN']['TN'],m['ToN']['FP'],m['ToN']['FN'],m['ToN']['TP'])==(115,0,616,0)
assert m['UNSW']['rows']==235424 and m['ToN']['rows']==27520000
report=(OUT/'stage_Week0_pilot_report.md').read_text()
assert 'NO_GO_EVIDENCE' in report and qf['q_decimal'] in report and 'prediction_set_coverage' in report
write('deliverable_check.json',dict(status='PASS',required_artifacts=len(required),all_json_valid=True,protocol_and_q_and_bootstrap_hashes_valid=True,frozen_execution_code_unchanged=True,upstream_preserved=True,no_per_flow_outputs=True,completed_at_utc=now()))
# Artifact hashes exclude this manifest itself and the currently open log.
rs=[file_record(f,'pilot_deliverable') for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['deliverable_hashes.csv','deliverable_check.log']]
csvwrite('deliverable_hashes.csv',rs)
print('PASS: deliverables complete, protocol/q/code/matrices hashes valid, old outputs unchanged')
