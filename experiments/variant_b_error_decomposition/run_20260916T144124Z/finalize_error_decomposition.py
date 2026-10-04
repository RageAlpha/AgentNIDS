"""Add probability-interval interpretation and verify saved deliverables."""
from pathlib import Path
import csv,json
O=Path(__file__).resolve().parent
def rows(n):
 with (O/n).open() as f:return list(csv.DictReader(f))
p=json.loads((O/'analysis_protocol.json').read_text());v=json.loads((O/'validation.json').read_text())
t=next(r for r in rows('error_transition_counts.csv') if r['scope']=='overall' and r['true_class']=='all' and r['singleton_prediction']=='all')
assert int(t['new_false_negative'])==int(t['corrected_false_positive'])==0
assert 1-p['q_Attack']<p['q_Benign']<.5
text=f'''\n## probability range and complementary explanation of research priorities\n\nUnder this threshold, the receiving record for reevaluation is Raw Benign → Variant B Attack with p at (.{p['q_Benign']},0.5). This is why we have a new error report on the real Benign.{int(t['new_false_positive']):,} The real Attack corrects the leak.{int(t['corrected_false_negative']):,} There is no reversal. All refused to record as binary elements, and p below 0.5 and Raw all predicted Benign; So the Raw error rate for rejected real Benign is 0% and the Raw error rate for rejected real Attack is 100%. This is a direct result of fixed probability and threshold conditions, not causal attribution.\n\nAs for the arithmetic increase in the current sample, the labels rule of thumb can be studied first: its net error-added contribution is greater than the screening contribution, and the direction is clearly focused on the Benign → Attack misstatement of the / misstatement weighting. But the screening itself also raises overall Raw to risk, so processing alone cannot explain or eliminate all of it.{v['reference_total_pp']:.6f} This is an increase in risk by one percentage point. The rules of screening are still a independent research issue. Subsequent research priorities are diagnostic recommendations and cannot be equated with some untested improvement in effective.\n '''
with (O/'decomposition_report.md').open('a') as f:f.write(text)
required=['error_transition_counts.csv','risk_decomposition.csv','temporal_decomposition.csv','class_decomposition.csv','bootstrap_intervals.csv','window_decomposition_counts.csv','validation.json','decomposition_report.md','analyze.py','commands.sh','analysis.log']
assert all((O/n).is_file() and (O/n).stat().st_size for n in required)
for file in ['risk_decomposition.csv','temporal_decomposition.csv','class_decomposition.csv']:
 for r in rows(file):
  n,d=int(r['numerator']),int(r['denominator'])
  assert r['value']=='NA' if d==0 else float(r['value'])==n/d
print('Saved deliverables and all numerator/denominator ratios verified. PASS.')
