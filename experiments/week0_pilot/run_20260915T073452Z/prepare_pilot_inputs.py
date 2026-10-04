import sys,platform,importlib.metadata,shutil
import numpy as np
from common import *
from implementation import synthetic_tests
assert not (OUT/'protocol_freeze.json').exists(),'refuse to overwrite protocol freeze'
write('upstream_inventory_before.json',inventory())
# Read every reviewed file and original roadmap; preserve originals.
required=['preflight_report.md','pilot_protocol_draft.json','gate_definition.md','numerical_contract.md','implementation.py','implementation_check.json','source_partition_audit.csv','evaluation_scope.csv','source_hashes.csv','validation.json']
for n in required:(PREF/n).read_text()
(ROOT/'docs/Research_Roadmap_Phase2_Conformal_Coordination.md').read_text()
refs=csvread(PREF/'source_hashes.csv'); conflicts=[]
for r in refs:
 if not Path(r['path']).exists() or sha(r['path'])!=r['sha256']:conflicts.append(r['path'])
write('input_hash_check.json',dict(status='PASS' if not conflicts else 'CONFLICT_STOP',checked=len(refs),conflicts=conflicts))
assert not conflicts,conflicts
refs += [file_record(p,'reviewed_preflight') for p in sorted(PREF.iterdir()) if p.is_file()]
stageA=ROOT/'outputs/post_p7_diagnostics/stage_A/run_20260914_A1/detector_raw_platt_metrics.csv'
refs.append(file_record(stageA,'historical_raw_count_comparator'))
# Requested Windows summary is background only.
win=Path('/mnt/c/Users/ysy/Desktop/literature/StrictK_A_B1_B2_C1_C2_Integrated_Summary_20260915.md')
(OUT/'provenance').mkdir()
background=dict(original_windows_path=r'C:\Users\ysy\Desktop\literature\StrictK_A_B1_B2_C1_C2_Integrated_Summary_20260915.md',wsl_path=str(win),used_for_computation=False)
if win.exists():
 dest=OUT/'provenance'/win.name;shutil.copyfile(win,dest);dest.read_text();background.update(status='COPIED',copy=str(dest),sha256=sha(win));assert sha(dest)==background['sha256']
else:background.update(status='NOT_SYNCED',technical_blocker=False,fallback='Existing A/B1/B2/C1/C2 stage reports and P7 summary; no summary reconstruction')
write('provenance/background_summary.json',background)
env=dict(python=sys.executable,python_version=platform.python_version(),versions={})
for n in ['numpy','cuml-cu12','cupy-cuda12x','scikit-learn','torch','pyarrow']:env['versions'][n]=importlib.metadata.version(n)
old=read(PREF/'model_load_result.json')
unchanged=all(env[k]==old[k] for k in ['python','python_version','versions'])
env.update(unchanged_from_preflight=unchanged,model_sha256=sha(ROOT/'outputs/source_v2/models/rf_All49/model.pkl'),model_loading='Reuse successful preflight original environment load: identical executable and package versions, model hash verified; no inference needed',preflight_model_load_sha256=sha(PREF/'model_load_result.json'))
write('environment.json',env)
assert unchanged,'environment changed: original RF reload required before proceeding'
# Identity partitions and exact declared scopes; full identity files hashes match preflight full scan.
parts=csvread(ROOT/'outputs/round1_data_prep_v1/split_manifest.csv')
source=[r for r in parts if r['domain']=='UNSW' and r['scheme']=='source']
assert [(r['partition'],int(r['rank_start']),int(r['rank_end_exclusive'])) for r in source]==[('S_train',0,1420000),('S_prob_cal',1420000,1780000),('S_competence',1780000,2130000),('S_source_test',2130000,2365424)]
order=np.load(ROOT/'outputs/round1_data_prep_v1/UNSW_sorted_original_rows.npy',mmap_mode='r')
assert np.array_equal(np.sort(order),np.arange(len(order)))
assert len(np.intersect1d(order[:1420000],order[1420000:1780000]))==0
assert len(np.intersect1d(order[2130000:],order[1420000:1780000]))==0
windows=csvread(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv')
w=[r for r in windows if r['domain']=='ToN' and r['scheme']=='heldout' and r['partition']=='T_heldout' and r['window_size']=='10000']
assert len(w)==2752 and [(int(r['rank_start']),int(r['rank_end_exclusive'])) for r in w]==[(i*10000,(i+1)*10000) for i in range(2752)]
assert w==csvread(PREF/'evaluation_window_references.csv')
csvwrite('evaluation_window_references.csv',w)
csvwrite('source_partition_audit.csv',csvread(PREF/'source_partition_audit.csv'))
csvwrite('evaluation_scope.csv',csvread(PREF/'evaluation_scope.csv'))
write('implementation_check.json',synthetic_tests())
assert sha(OUT/'implementation.py')==sha(PREF/'implementation.py')
protocol=read(PREF/'pilot_protocol_draft.json')
protocol.update(version='week0-pilot-v1.0',run_id=OUT.name,created_at_utc=now(),status='AUTHORIZED_FROZEN_BEFORE_CALIBRATION',open_protocol_choices=[],gate='gate_definition.md')
protocol.pop('pilot_sequence_after_approval',None)
protocol['calibration'].update(q_status='TO_BE_COMPUTED_ONCE_AND_SAVED_IN_Q_FREEZE',q_freeze_path=str(OUT/'q_freeze.json'))
protocol['evaluation']['ToN']['window_reference']=str(OUT/'evaluation_window_references.csv')
protocol['inputs']['hash_inventory']=str(OUT/'source_hashes.csv')
protocol['provenance']={'roadmap_explicit':protocol['provenance']['roadmap_explicit'],'inherited_frozen_protocol':protocol['provenance']['existing_project'],'approved_execution_refinements':['RF-only float32 origin checked without modifying p; float64 direct complement comparisons','Complete original source test with tail and complete P7 ToN including startup','One source-calibrated q frozen before all evaluation','ToN overall shared 2000-replicate PCG64(20260915) 276-block bootstrap; 10 windows/block, final 2','Source, temporal and taxonomy analyses descriptive; no source-target paired CI','Human joint evidence assessment, no new numerical gate cutoffs','Previously unblinded exploratory pilot; no Week1 or professor confirmation inferred']}
protocol['statistics']=dict(scope='ToN overall only; source/time/category descriptive',replicates=2000,block_windows=10,blocks=276,final_block_windows=2,rng='numpy.random.Generator(PCG64(20260915))',seed=20260915,sampling='rng.integers(0,276,size=(2000,276),dtype=np.int64); independent draws of whole blocks with replacement',shared_matrix=True,interval='95% percentile numpy.quantile(method=linear), bootstrap estimates only',undefined='NA; exclude nonfinite estimates separately per metric, report valid and invalid replicate counts',difference_intervals='compute two gaps within each same replicate, then percentile; never subtract separate CI endpoints',source='only 3 blocks at frozen length, below minimum20; no CI, no changed blocks or removed tail',source_target_difference='descriptive only',conditioned_on='Frozen RF, fixed source calibration q, observed target sequence; excludes source-calibration resampling and gives no future-domain coverage/risk guarantee',metrics=BOOT_METRICS)
protocol['approved_choices']=dict(authorized_by='Current user instruction: preflight reviewed, execute formal Week0 only',background_summary=background['status'],gate='Predefined four states with joint evidence human review; no numeric effect threshold',source_selection_boundary='Existing saved code/logs show no S_prob_cal-driven current Raw RF selection; undocumented human attempts not ruled out')
protocol['execution_sequence']=['prepare.py','calibrate.py (exclusive q_freeze creation)','evaluate.py (loads q; no calibration call)','bootstrap.py','verify.py','report.py']
write('protocol.json',protocol)
shutil.copyfile(PREF/'gate_definition.md',OUT/'gate_definition.md')
shutil.copyfile(PREF/'numerical_contract.md',OUT/'numerical_contract.md')
csvwrite('source_hashes.csv',refs)
write('protocol_freeze.json',dict(version='week0-protocol-freeze-v1.0',frozen_at_utc=now(),protocol_sha256=sha(OUT/'protocol.json'),sources_sha256=sha(OUT/'source_hashes.csv'),implementation_sha256=sha(OUT/'implementation.py'),code_hashes={f.name:sha(f) for f in OUT.glob('*.py')},authorization='User reviewed preflight and authorized this Week0 run; no Week1',calibration_started=False,evaluation_started=False))
print('PASS: inputs match preflight; protocol frozen, synthetic tests passed; ready for calibration',flush=True)
