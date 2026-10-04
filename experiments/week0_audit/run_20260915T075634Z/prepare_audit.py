from audit_common import *
import importlib.metadata
write('upstream_inventory_before.json',inventory())
refs=rows(PILOT/'source_hashes.csv');diff=[]
for r in refs:
 new=sha(r['path']);diff.append(dict(path=r['path'],expected_sha256=r['sha256'],actual_sha256=new,match=new==r['sha256'],comparison='pilot frozen input'))
for base in [PILOT,PREF]:
 for p in sorted(base.rglob('*')):
  if p.is_file():diff.append(dict(path=str(p),expected_sha256=sha(p),actual_sha256=sha(p),match=True,comparison='audit-start snapshot; old output'))
extra=[ROOT/'datasets/NF-ToN-IoT-v3.csv',ROOT/'outputs/feature_store_v2/ToN_X.npy',ROOT/'outputs/round1_data_prep_v1/prepare_data.py',ROOT/'outputs/round1_data_prep_v1/round1_config.json']
for p in extra:diff.append(dict(path=str(p),expected_sha256=sha(p),actual_sha256=sha(p),match=True,comparison='audit-start snapshot; no earlier full-content hash claimed'))
table('artifact_differences.csv',diff)
assert all(r['match'] for r in diff),'Frozen input conflict; stop'
for base in [PILOT,PREF]:
 for p in base.glob('*.py'):p.read_text()
for name in ['docs/Research_Roadmap_Phase2_Conformal_Coordination.md','src/nids/data/feature_store.py','src/nids/data/preprocessing.py','src/nids/data/partitions.py','src/nids/replay/predictions.py','src/nids/models/base_models.py','src/nids/p2.py','configs/execution_v2.json','outputs/round1_data_prep_v1/prepare_data.py']:(ROOT/name).read_text()
write('environment.json',dict(python=sys.executable,python_version=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ['numpy','pyarrow','cuml-cu12','cupy-cuda12x','torch','scikit-learn']}))
write('audit_protocol.json',dict(version='week0-independent-audit-v1.0',created_at_utc=now(),pilot=str(PILOT),preflight=str(PREF),model='rf_All49',q_hex=read(PILOT/'q_freeze.json')['q_serialized']['value'],epsilon='1/10',source_cal=[1420000,1780000],source_test=[2130000,2365424],ToN=[0,27520000],gate_frozen='NO_GO_EVIDENCE',seed=20260915,uniform_calibration_samples=3601,per_complete_window_samples=17,source_tail_samples=17,targeted_sampling='all 731 accepted ToN; nearest 64 records on each side of both singleton boundaries in each partition; 256 source true Attack in covered/rejected groups; original inference batch boundaries',numerical_input='float32 values losslessly stored float64; compare only, no rounding/clipping/softmax',independence='No calls/imports to old scores/threshold/prediction_sets/summarize; reference order statistic and masked integer counting',upstream='Full raw CSV Label/Attack and end-time mapping; sampled exact features/inference',inference_claim='Sampled only unless explicitly expanded; bit equality first, no invented tolerance',no_week1=True))
print('PASS frozen inputs; new audit protocol saved',flush=True)
