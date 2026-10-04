"""Independent saved-artifact checks and completion; never evaluates target flows."""
import ast
import experiment as e
import numpy as np
from pathlib import Path

# Copy only pure set/count checks, excluding historical execution/report functions.
tree=ast.parse((e.RF/'pilot.py').read_text())
names={'sets','counts','unit_checks','ratio'}
source='import numpy as np\n\n'+'\n\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)+'\n'
e.txt('set_logic_checks.py',source)
namespace={};exec(compile(source,'set_logic_checks.py','exec'),namespace)
checks=namespace['unit_checks']()
assert [e.gate(x) for x in [np.nextafter(.8,0),.8,np.nextafter(.9,0),.9]]==['Allows for full evaluation','The Professor’s Judgment','The Professor’s Judgment','Depreciation stops']
checks['full_precision_gate_boundaries']=True
checks['181st_order_statistic']=True
checks['synthetic_only_no_evaluation_records']=True
folders=[Path('ton_mlp')]+[Path('bot')/m for m in e.MODELS]
fs=[];details=[]
for folder in folders:
 f=e.read(e.OUT/folder/'threshold_freeze.json');fs.append(f)
 manifest=e.csvread(e.OUT/('ton_mlp' if f['domain']=='ToN' else 'bot')/'calibration_sample_manifest.csv')
 rows=e.csvread(e.OUT/folder/'calibration_scores.csv')
 assert len(rows)==400
 assert [{k:r[k] for k in manifest[0]} for r in rows]==manifest
 ranks=np.array([int(r['rank']) for r in rows]);ys=np.array([int(r['label']) for r in rows])
 p=np.array([float(r['raw_probability']) for r in rows]);s=np.array([float(r['true_class_score']) for r in rows])
 cache=e.arr(e.ROOT/f"outputs/prediction_cache_v2/{f['domain']}/{f['model']}.npy")
 assert np.array_equal(p,cache['raw_probability'][ranks])
 assert np.array_equal(s,np.where(ys==0,p,1-p))
 assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
 for y,key in [(0,'q_Benign'),(1,'q_Attack')]:
  assert sum(ys==y)==200 and sorted(s[ys==y].tolist())[180]==f[key]
 assert e.gate(f['q_Attack'])==f['gate'] and f['gate']!='Allows for full evaluation'
 assert f['threshold_sum']==f['q_Benign']+f['q_Attack'] and f['min_p_Attack']==1-f['q_Attack'] and f['min_p_Benign']==1-f['q_Benign']
 assert not (e.OUT/folder/'evaluation_metrics.csv').exists()
 details.append(dict(model=f['model'],domain=f['domain'],sample_unique=True,per_class_200=True,saved_probabilities_exactly_match_cache=True,scores_exact=True,order_statistic_verified=True,gate=f['gate'],evaluation='Not working',calibration_evaluation_identity_overlap=0,overlap_basis='domain original-row permutation verified; calibration rank <100000; fixed evaluation starts at 100000'))
summary=e.read(e.OUT/'bot/collection_summary.json')
y=e.arr(e.ROOT/'outputs/feature_store_v2/BoT_label.npy')
rows=e.csvread(e.OUT/'bot/calibration_sample_manifest.csv')
for v in (0,1):
 expected=np.flatnonzero(y[:100000]==v)[:200].tolist()
 assert [int(r['rank']) for r in rows if int(r['label'])==v]==expected
 assert summary['class_200th_rank'][str(v)]==expected[-1]
assert summary['actual_labels_scanned_for_collection']==max(summary['class_200th_rank'].values())+1
# Verify original-row zero-based provenance without computing any file digest.
feature_code=(e.ROOT/'src/nids/data/feature_store.py').read_text()
assert 'raw_y[cursor:stop]' in feature_code and 'raw_y[selected]' in feature_code
now=e.now()
e.write('bot_threshold_report_displayed.json',dict(displayed_in_conversation_at_utc=now,professor_sent=False,professor_reviewed=False))
e.save('gate_decisions.csv',[dict(domain=f['domain'],model=f['model'],q_Attack=f['q_Attack'],gate=f['gate'],evaluation='Not working',reason='0.80 <= q_Attack < 0.90 to be decided by the professor' if f['gate']=='The Professor’s Judgment' else 'q_Attack >= 0.90, stop the threshold',threshold_report_displayed=True,professor_sent=False,professor_reviewed=False) for f in fs])
e.write('validation.json',dict(status='PASS_FOR_EXECUTED_THRESHOLD_STAGES',completed_at_utc=now,checks=checks,model_checks=details,bot_three_models_share_identical_manifest=True,bot_earliest_per_class_verified=True,ton_RF_XGBoost_manifest_exact_match=True,probability_validation_scope='only the 400 calibration records per model/domain; no later evaluation probabilities accessed',raw_probability_unchanged=True,full_evaluations_run=0,source_evaluations_run=0,evaluation_count_checks='NOT_APPLICABLE: all evaluation branches stopped or pending; synthetic set/count checks passed',file_digests_computed_or_verified=False,historical_artifacts_modified=False,pending=['RF/BoT gate is undefined in [0.80,0.90); professor ruling required']))
report='# Outcome B Two complementary experiments: Threshold phase completed\n\n'+e.table([fs[0]])+'\n\n'+e.table(fs[1:])+'\n\n## gate control and actual execution\n\n'
report+='| model / domain | gate control | full evaluation | causes |\n|---|---|---|---|\n'
for f in fs:report+='| '+f['model']+' / '+f['domain']+' | '+f['gate']+' | not working | '+('0.80≤q_Attack<0.90, original document undefined, pending the Professor’s decision' if f['gate']=='The Professor’s Judgment' else 'q_Attack≥0.90')+' |\n'
report+='''
MLP/ToN strictly repeats RF with XGBoost by phased 400 line manifest, the only line identity for each class 200, all located in rank 0 and 4019. 400 is a measure of calibration and 4020 is a precursor to historical scanning. The entire [0,100000) is still excluded from the evaluation.

BoT as true labels temporal order New collected 200 Benign + 200 Attack , the three models share perfectly consistent 400 records. Article 200 Attack in rank 226 and Article 200 Benign in rank 14863; Stop rank 14863 and actually collect scanning 14864 items. For historical consistency checks, read the 100000 starting period labels, Benign=9071 and Attack=90929. This support verification does not change the sample selection or collection stop position.

The threshold values are the 181 incremental items of the real class score for each class 200; Using the original float64 Raw cache, not retrained, not using Platt, not changing probability or features. The model, the original preprocessing, the caching path and the sequencing evidence are in protocol.json and input_validation.json fields. The original number is a zero-base data-line index; Caching and labels have been ranked according to availability rank, without a secondary ranking.

In the dialogue, the full MLP/ToN threshold table is shown, and the BoT sample is collected and the three models are shown. There are no q_Attack<0.80 branches, so no full evaluation, UNSW source test or bootstrap has been performed, and no overall and four performance results have been found. No estimate indicator has been generated.

q_Attack=0.8865555301308632 of RF/BoT falls within the discretionary range; It can’t be put in a stop or let go. The other three models triggered the / domain to stop the threshold. This revaluation cannot prove performance improvement, nor does it automatically declare Outcome A or extend the experiment.

validation.json records the actual scope of inspection: probability and real class scores by value verification, 181 independent review, unique sample sharing and 200+200, BoT first sample and three model sharing, complete precision gate control boundaries, original line arrays and window boundaries. The set equations, non-symmetric thresholds, vacuum clusters, binary elements, zero acceptance NA and singleton allow for synthetic input checks that differ from Raw. The actual rating counting checks are not applicable because they are not rated.

Uncalculated or verified file summaries, unmodified historical products, unrepeated RF/XGBoost ToN, unwritten, Week 1 or replacement professor messages. The report has been displayed and saved to users; The professor has not received or reviewed it. After completing the defined threshold steps, stop waiting for the results to be reviewed and the RF/BoT ruling.

Absolute path of ## product

'''
for p in ['protocol.json','ton_mlp/threshold_report.md','ton_mlp/calibration_sample_manifest.csv','ton_mlp/calibration_scores.csv','ton_mlp/threshold_freeze.json','bot/threshold_report.md','bot/calibration_sample_manifest.csv','bot/collection_summary.json','gate_decisions.csv','validation.json','experiment.py','finalize.py','commands.sh','ton.log','bot.log','finalize.log']+[f'bot/{m}/{n}' for m in e.MODELS for n in ['calibration_scores.csv','threshold_freeze.json']]:report+='- `'+str(e.OUT/p)+'`\n'
e.txt('completion_report.md',report)
with (e.OUT/'commands.sh').open('a') as f:f.write('python '+str(Path(__file__).resolve())+'\n')
print('PASS: four threshold artifacts independently verified; 0 full evaluations; RF/BoT pending professor ruling.')
print(e.OUT/'completion_report.md')
