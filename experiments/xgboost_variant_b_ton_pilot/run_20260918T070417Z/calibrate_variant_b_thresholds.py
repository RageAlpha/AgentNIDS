"""Reuse the saved 400 identities; compute thresholds only. No model inference."""
import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT = Path(__file__).resolve().parent
RF = ROOT / 'outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z'

def read(path):
    return json.loads(path.read_text())

def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    cfg_path = ROOT / 'configs/execution_v2.json'
    cfg = read(cfg_path)
    assert Path(cfg['project_root']) == ROOT
    model_id = 'xgb_All49'
    folder = ROOT / cfg['outputs']['source'] / 'models' / model_id
    training_path = folder / 'training_manifest.json'
    training = read(training_path)
    model = folder / training['artifact']
    assert training['model_id'] == model_id and training['status'] == 'COMPLETE'
    assert training['family'] == 'xgb' and training['view'] == 'All49'
    assert training['parameters'] == cfg['models']['xgb']
    assert training['parameters']['objective'] == 'binary:logistic'
    assert model.stat().st_size == training['artifact_size_bytes'] > 0
    cache_dir = ROOT / cfg['outputs']['base_predictions'] / 'ToN'
    cm = read(cache_dir / 'manifest.json')
    mm = read(cache_dir / (model_id + '_manifest.json'))
    cache_path = ROOT / mm['file']
    store = ROOT / cfg['outputs']['feature_store']
    fm = read(store / 'feature_store_manifest.json')
    lm = read(store / 'ToN_manifest.json')
    prepared = Path(cfg['prepared_run'])
    order_path = prepared / 'ToN_sorted_original_rows.npy'
    labels_path = store / lm['files']['binary_label']
    assert mm['model_id'] == model_id and mm['domain'] == cm['domain'] == 'ToN'
    assert mm in cm['models']
    assert mm['feature_indices'] == training['feature_indices'] == list(range(49))
    assert mm['feature_names'] == training['feature_names'] == fm['feature_order']
    assert cm['row_order'] == 'availability_rank'
    assert Path(cm['original_row_index']) == order_path
    assert fm['row_order'] == 'availability_sorted_rank' and Path(fm['rank_source']) == prepared
    assert cache_path.stat().st_size == mm['bytes']
    source_manifest = RF / 'calibration_sample_manifest.csv'
    original_bytes = source_manifest.read_bytes()
    records = list(csv.DictReader(original_bytes.decode().splitlines()))
    assert len(records) == 400
    ranks = np.array([int(r['rank']) for r in records], dtype=np.int64)
    original_rows = np.array([int(r['original_row']) for r in records], dtype=np.int64)
    y = np.array([int(r['label']) for r in records], dtype=np.uint8)
    assert len(np.unique(ranks)) == len(np.unique(original_rows)) == 400
    assert np.all((ranks >= 0) & (ranks < 100000))
    assert np.all((y == 0) | (y == 1)) and int((y == 0).sum()) == int((y == 1).sum()) == 200
    collection = read(RF / 'collection_summary.json')
    old_protocol = read(RF / 'protocol.json')
    assert old_protocol['labels'] == {'0': 'Benign', '1': 'Attack'}
    assert collection['scanned_rows'] == 4020 and collection['stop_rank_inclusive'] == 4019
    assert collection['startup_limit'] == 100000 and ranks.max() == 4019
    assert collection['collected_benign'] == collection['collected_attack'] == 200
    assert collection['total_collected'] == 400
    cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
    labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
    order = np.load(order_path, mmap_mode='r', allow_pickle=False)
    n = cm['rows']
    assert n == mm['rows'] == lm['rows']
    assert cache.shape == labels.shape == order.shape == (n,)
    assert np.array_equal(order[ranks], original_rows) and np.array_equal(labels[ranks], y)
    assert np.all((original_rows >= 0) & (original_rows < n))
    last = records[int(np.argmax(ranks))]
    assert int(last['original_row']) == collection['stop_original_row']
    assert last['window_id'] == collection['stop_window_id']
    with (prepared / 'window_manifest.csv').open() as stream:
        windows = {r['window_id']: r for r in csv.DictReader(stream)
                   if r['domain'] == 'ToN' and r['scheme'] == 'heldout'
                   and r['partition'] == 'T_heldout' and int(r['window_size']) == 10000
                   and int(r['rank_start']) < 100000}
    assert len(windows) == 10
    for r in records:
        w = windows[r['window_id']]
        rank = int(r['rank'])
        assert int(w['rank_start']) <= rank < int(w['rank_end_exclusive'])
        assert int(r['window_index']) == rank // 10000
        assert int(w['first_original_row']) == int(order[int(w['rank_start'])])
        assert int(w['last_original_row']) == int(order[int(w['rank_end_exclusive']) - 1])
    # The only probability reads are these 400 manifest ranks, in saved order.
    p = np.asarray(cache['raw_probability'][ranks], dtype=np.float64)
    assert p.shape == (400,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    scores = np.where(y == 0, p, 1.0 - p)
    k = math.ceil((200 + 1) * (1 - 0.10))
    assert k == 181
    qb, qa = [float(np.sort(scores[y == c])[k - 1]) for c in (0, 1)]
    values = dict(q_Benign=qb, q_Attack=qa, q_Benign_plus_q_Attack=qb + qa,
                  Attack_inclusion_lower_bound=1.0 - qa, Benign_inclusion_lower_bound=1.0 - qb)
    if qa >= 0.90:
        branch, status = 'A / Outcome B', 'Full evaluation has not been carried out due to the suspension of the threshold; Outcome B by mail, without adding conformal experiments.'
    elif qa < 0.80:
        branch, status = 'B / eligible, awaiting professor report', 'fulfilling the conditions for the initiation of the full assessment set out in the correspondence and requiring the user to report the devaluation to the professor first; Outcome A has not yet been announced.'
    else:
        branch, status = 'C / unspecified interval', 'The original letter did not specify how to handle the 0.80 <= q_Attack < 0.90 area, and the professor made it clear. Outcome A or Outcome B is not determined.'
    inputs = dict(config=str(cfg_path), model_record=str(training_path), model_artifact=str(model),
                  cache=str(cache_path), cache_manifest=str(cache_dir / (model_id + '_manifest.json')),
                  domain_cache_manifest=str(cache_dir / 'manifest.json'), labels=str(labels_path),
                  original_rows=str(order_path), window_manifest=str(prepared / 'window_manifest.csv'),
                  original_calibration_manifest=str(source_manifest), original_rf_run=str(RF))
    protocol = dict(authority=str(ROOT / 'docs/XGBOOST_TON/Reply_to_Siyuan_VariantB_Analysis_and_Next_Step.md'),
                    authority_sections='3 Steps 1–3 plus current user constraints', inputs=inputs,
                    model_identity={k: training[k] for k in ['version','model_id','family','view','feature_count','completed_at_utc','training_domain','training_partition','preprocessing']},
                    probability_field='raw_probability', probability_meaning='P(Attack)', labels=old_protocol['labels'],
                    reuse_rule='Exact saved 400 identities in original manifest order; no recollection or replacement',
                    epsilon=0.10, per_class=200, order_statistic_one_based=181, order_statistic_zero_based=180,
                    precision='float64 after cache read; no clipping, rounding, interpolation or probability transform',
                    scores={'Benign':'p','Attack':'1-p'}, startup_rank_range=[0,100000],
                    historical_collection={'scanned_rows':4020,'stop_rank_inclusive':4019},
                    scope='Calibration and threshold report only; zero evaluation windows; no Phase 2; no professor message',
                    evidence=['predictions.py gpu_raw_probability: xgb binary logistic inplace_predict; raw stored separately from transformed probability',
                              'predictions.py predict_domain: model from configured source; cache slices retain feature-store rank',
                              'feature_store.py: sorted_y = raw_y[order]; feature-store metadata identifies prepared rank source'],
                    environment={'python':platform.python_version(),'numpy':np.__version__})
    save('protocol.json', protocol)
    with (OUT / 'calibration_sample_manifest.csv').open('xb') as stream:
        stream.write(original_bytes)
    assert (OUT / 'calibration_sample_manifest.csv').read_bytes() == original_bytes
    with (OUT / 'calibration_scores.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) + ['raw_probability','true_class_nonconformity_score'])
        writer.writeheader()
        for r, probability, score in zip(records, p, scores):
            writer.writerow(dict(r, raw_probability=repr(float(probability)), true_class_nonconformity_score=repr(float(score))))
    freeze = dict(values, frozen_at_utc=datetime.now(timezone.utc).isoformat(), precision='float64',
                  serialization='Python float shortest round-trip decimal; no rounding', epsilon=0.10,
                  sample_count=400, per_class={'Benign':200,'Attack':200}, order_statistic_one_based=181,
                  order_statistic_zero_based=180, gate_branch=branch, gate_status=status)
    save('threshold_freeze.json', freeze)
    loaded = read(OUT / 'threshold_freeze.json')
    assert all(float(loaded[key]).hex() == value.hex() for key, value in values.items())
    with (OUT / 'calibration_scores.csv').open() as stream:
        saved = list(csv.DictReader(stream))
    rp = np.array([float(r['raw_probability']) for r in saved])
    rs = np.array([float(r['true_class_nonconformity_score']) for r in saved])
    assert np.array_equal(rp, p) and np.array_equal(rs, scores)
    for c, q in [(0,qb),(1,qa)]:
        independent = sorted(float(r['raw_probability']) if c == 0 else 1.0-float(r['raw_probability'])
                             for r in saved if int(r['label']) == c)
        assert independent[180] == q
        assert sum(s < q for s in independent) <= 180 and sum(s <= q for s in independent) >= 181
    checks = dict(status='PASS', missing_items=[], exact_manifest_bytes_preserved=True,
                  sample_count=400, benign_count=200, attack_count=200, unique_original_rows=True,
                  rank_original_row_label_window_alignment=True, startup_membership=True,
                  historical_stop_rank=4019, historical_scanned_rows=4020, no_recollection=True,
                  model_config_and_cache_identity=True, row_counts_match=n,
                  finite_unit_interval_probabilities=True, probability_records_read=400,
                  scores_recomputed_from_saved_probabilities=True, order_statistic_verified=True,
                  threshold_float64_roundtrip_exact=True, scores_float64_roundtrip_exact=True,
                  evaluation_windows_run=0, phase_2_started=False, model_loaded_or_run=False)
    save('validation.json', checks)
    table = '\n'.join(f'| {key} | {repr(value)} |' for key,value in values.items())
    report = f'''# XGBoost-All49 / ToN Variant B Threshold report

{status}

The original 400 clause manifest is copied by byte, identity and the original sequence are perfectly consistent: Benign=200 , Attack=200 . rank, original, row, labels, window numbers and startup attributes have all been verified. The original startup is rank [0,100000); 4020 points, rank 4019 is only historically aligned and has not been collected again.

fixedε =0.10; Raw P(Attack) is converted to float64 , Benign point =p , Attack point =1-p , various 181 elevations , zero base 180), does not insert, does not change, does not yield.

| Threshold | Complete accuracy ((backward recovery of float64)) |
|---|---|
{table}

The door control branch:{branch}. Using an uncompromising q_Attack judgment. Benign in the subgroup 1-q_Benign is the subgroup P(Benign) (equivalent to P(Attack)<=q_Benign).

It is necessary to check all passes, missing items: No. The scope of validation is for input metadata and 400 corresponding records, not for full audit history. No accepted error rate or accuracy results were generated.

The actual input:
''' + '\n'.join(f'- {key}: `{value}`' for key,value in inputs.items()) + f'''

I’m going to tell you something.

Professor Omar, I reused exactly the same 400 ToN calibration flows (200 Benign and 200 Attack) with frozen XGBoost-All49 Raw probabilities. At epsilon=0.10, the class-specific 181st-order thresholds are q_Benign={qb!r}, q_Attack={qa!r}; their sum is {qb+qa!r}. The Attack and Benign inclusion lower bounds are {1-qa!r} and {1-qb!r}, respectively. Gate status: {branch}. No evaluation windows have been run. This is a local threshold report for your review.

Local reporting is not the same as sending any messages to professors who have already reported to them.

This time only completed the calibration with the threshold report, without running any evaluation windows, and without entering Phase 2.
'''
    with (OUT / 'threshold_report.md').open('x') as stream:
        stream.write(report)
    print(json.dumps(dict(values, gate_branch=branch, gate_status=status, validation='PASS', output=str(OUT)), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
