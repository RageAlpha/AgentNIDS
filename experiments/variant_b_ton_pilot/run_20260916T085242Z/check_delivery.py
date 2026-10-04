"""Read back saved deliverables; no evaluation, threshold fitting, or digest operations."""
import csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
P=Path(__file__).resolve().parent
required=['protocol.json','calibration_sample_manifest.csv','collection_summary.json','threshold_freeze.json','evaluation_metrics.csv','temporal_metrics.csv','source_class_coverage.csv','bootstrap_intervals.csv','bootstrap_indices.npy','window_counts.csv','bootstrap_block_counts.csv','bootstrap_replicate_metrics.csv','source_window_counts.csv','aggregate_counts.csv','validation.json','stage_report.md','gate_assessment.md','pilot.py','commands.sh','prepare.log','evaluate.log']
for n in required: assert (P/n).is_file() and (P/n).stat().st_size>0,n
def rows(n):
 with (P/n).open() as f:return list(csv.DictReader(f))
freeze=json.loads((P/'threshold_freeze.json').read_text())
start=json.loads((P/'evaluation_start.json').read_text())
assert freeze['registered_at_utc']<start['started_at_utc']
assert freeze['q_Benign']==start['q_Benign'] and freeze['q_Attack']==start['q_Attack']
assert len(rows('calibration_sample_manifest.csv'))==400
w=rows('window_counts.csv'); assert len(w)==2742
agg=rows('aggregate_counts.csv'); total=agg[0]
for k in total:
 if k=='scope':continue
 assert sum(int(r[k]) for r in w)==int(total[k])==sum(int(r[k]) for r in agg[1:]),k
for n in ['evaluation_metrics.csv','temporal_metrics.csv','source_class_coverage.csv']:
 for r in rows(n):
  a,b=int(r['numerator']),int(r['denominator'])
  assert (r['value']=='NA') if b==0 else np.isclose(float(r['value']),a/b,rtol=1e-14,atol=0),r
indices=np.load(P/'bootstrap_indices.npy',allow_pickle=False)
assert indices.shape==(2000,275) and indices.min()>=0 and indices.max()<275
assert all(int(r['valid_replicates'])==2000 for r in rows('bootstrap_intervals.csv'))
assert json.loads((P/'validation.json').read_text())['status']=='PASS'
assert 'No-go' in (P/'gate_assessment.md').read_text()
result=dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),required_files=required,
 persisted_ratios_counts_and_matrix_verified=True,
 editorial_note='After execution, only the report generator prose and two Markdown reports were expanded to interpret segment contributions. Calibration, registered thresholds, protocol, evaluation, bootstrap and all numerical outputs were unchanged.',
 output_directory=str(P),stopped_after_mini_pilot=True)
with (P/'delivery_check.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result,ensure_ascii=False))
