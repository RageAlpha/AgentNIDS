"""Read saved preflight outputs and array headers only; never redraw or evaluate."""
from pathlib import Path
from collections import Counter
import csv
import json
import hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
def read(name): return json.loads((OUT/name).read_text())
def rows(name):
    with (OUT/name).open(newline='') as f: return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

manifest=rows('calibration_sample_manifest.csv')
assert len(manifest)==1200
assert len({r['record_id'] for r in manifest})==1200
commit=read('rank_commitment.json')
assert all(sha(OUT/n)==h for n,h in commit['sha256'].items())
for support in rows('calibration_support.csv'):
    d=support['domain']; records=[r for r in manifest if r['domain']==d]
    ranks=np.array([int(r['availability_rank']) for r in records])
    saved=np.load(OUT/f'{d}_sorted_ranks.npy')
    assert np.array_equal(ranks,saved) and np.array_equal(np.sort(np.load(OUT/f'{d}_draw_order_ranks.npy')),saved)
    assert len(set(ranks))==400 and 0<=ranks.min()<=ranks.max()<100000
    assert sum(int(r['binary_label']) for r in records)==int(support['attack'])
    assert 400-int(support['attack'])==int(support['benign'])
    counts=Counter(r['original_attack_category'] for r in records)
    assert counts==json.loads(support['original_attack_category_counts_json'])
    assert Counter(r['window_id'] for r in records)==json.loads(support['window_sample_counts_json'])
    assert all(r['FLOW_END_MILLISECONDS']==r['availability_ms'] for r in records)
    assert all(int(r['FLOW_START_MILLISECONDS'])<=int(r['FLOW_END_MILLISECONDS']) for r in records)
    order=np.load(ROOT/f'outputs/round1_data_prep_v1/{d}_sorted_original_rows.npy',mmap_mode='r')
    # Header-only check explicitly covers future R_select labels and native category arrays.
    for name in ['label','attack_code']:
        arr=np.load(ROOT/f'outputs/feature_store_v2/{d}_{name}.npy',mmap_mode='r')
        assert arr.shape==order.shape and arr.dtype.kind in 'iu'
        assert (ROOT/f'outputs/feature_store_v2/{d}_{name}.npy').stat().st_size==arr.offset+arr.nbytes
scope=rows('evaluation_scope.csv')
assert sum(int(r['evaluation_rows']) for r in scope)==64260000
assert sum(int(r['evaluation_complete_windows']) for r in scope)==6426
for r in scope:
    assert int(r['evaluation_rows'])==int(r['evaluation_complete_windows'])*10000
    assert int(r['startup_wait_ms'])==int(r['startup_last_available_ms'])-int(r['first_available_ms'])
    assert int(r['evaluation_first_available_ms'])>=int(r['startup_last_available_ms'])
assets=rows('asset_availability.csv')
assert Counter(r['kind'] for r in assets)==dict(detector=6,Platt=6,probability_cache=36,frozen_fu=18,R_select_label_cache=6)
for r in assets:
    p=Path(r['path']); assert p.stat().st_size==int(r['bytes'])
    if r['sha256']!='NA': assert sha(p)==r['sha256']
assert read('upstream_inventory_before.json')==read('upstream_inventory_after.json')
protocol=read('protocol_draft.json')
assert protocol['C1']['status']==protocol['C2']['status']==protocol['C0_select']['status']=='NOT_RUN'
result=dict(status='PASS',calibration_records=1200,rank_commitment_unchanged=True,
    sample_support_and_category_and_window_tables_consistent=True,source_times_valid=True,
    target_label_and_category_cache_headers_cover_full_R_select=True,
    asset_counts=dict(Counter(r['kind'] for r in assets)),model_and_fu_hashes_verified=True,
    no_random_sampling_performed=True,no_label_or_probability_arrays_values_read=True)
(OUT/'deliverable_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
