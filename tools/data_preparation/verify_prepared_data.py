"""Reopen saved manifests; optionally load one source window's feature views.

No fitting, calibration, shift computation, or target model evaluation.
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--smoke-source-window', action='store_true')
    args = p.parse_args()
    root = args.run_dir.resolve()
    config = json.loads((root / 'run_config.json').read_text())
    assert config['status'] == 'DATA_PREPARATION_COMPLETE_NOT_EXPERIMENT_FROZEN'
    splits = read_csv(root / 'split_manifest.csv')
    windows = read_csv(root / 'window_manifest.csv')
    groups = read_csv(root / 'group_manifest.csv')
    tails = read_csv(root / 'tail_manifest.csv')
    blocks = read_csv(root / 'bootstrap_manifest.csv')
    lookup = {w['window_id']: w for w in windows}
    assert len(lookup) == len(windows)
    indices = {s['domain']: np.load(root / s['sorted_row_index'], mmap_mode='r') for s in splits}
    ends = {d: np.load(root / f'{d}_sorted_end_ms.npy', mmap_mode='r') for d in indices}
    part_key = lambda r: (r['domain'], r['scheme'], r['partition'])
    window_key = lambda r: (*part_key(r), r['window_size'])
    split_lookup = {part_key(s): s for s in splits}
    grouped_windows = defaultdict(list)
    for w in windows:
        s = split_lookup[part_key(w)]
        a, b = int(w['rank_start']), int(w['rank_end_exclusive'])
        assert int(s['rank_start']) <= a < b <= int(s['rank_end_exclusive'])
        assert b - a == int(w['window_size'])
        assert int(w['first_original_row']) == indices[w['domain']][a]
        assert int(w['last_original_row']) == indices[w['domain']][b-1]
        assert int(w['first_available_ms']) == ends[w['domain']][a]
        assert int(w['last_available_ms']) == ends[w['domain']][b-1]
        if w['predecessor_window_id']:
            prev = lookup[w['predecessor_window_id']]
            assert (prev['domain'], prev['scheme'], prev['window_size'], prev['stream_id']) == (w['domain'], w['scheme'], w['window_size'], w['stream_id'])
            assert int(prev['rank_end_exclusive']) <= a
            assert int(prev['last_available_ms']) <= int(w['first_available_ms'])
            assert w['predecessor_last_available_ms'] == prev['last_available_ms']
            assert w['initialization'] == 'False'
        else:
            assert w['initialization'] == 'True' and float(w['shift_initial_value']) == 0.0
        grouped_windows[window_key(w)].append(w)
    for t in tails:
        s = split_lookup[part_key(t)]
        ws = grouped_windows[window_key(t)]
        cursor = int(s['rank_start'])
        for w in ws:
            assert int(w['rank_start']) == cursor
            cursor = int(w['rank_end_exclusive'])
        assert cursor == int(t['rank_start'])
        assert int(t['rank_end_exclusive']) == int(s['rank_end_exclusive'])
        assert int(t['excluded_rows']) == int(t['rank_end_exclusive']) - cursor
        assert len(ws) == int(t['complete_windows'])
    block_coverage = Counter()
    for b in blocks:
        ws = grouped_windows[window_key(b)]
        a, z = int(b['rank_start']), int(b['rank_end_exclusive'])
        selected = [w for w in ws if a <= int(w['rank_start']) < z]
        assert len(selected) == int(b['window_count']) <= 10
        assert selected[0]['window_id'] == b['first_window_id']
        assert selected[-1]['window_id'] == b['last_window_id']
        assert selected[-1]['rank_end_exclusive'] == b['rank_end_exclusive']
        block_coverage.update(w['window_id'] for w in selected)
    assert set(block_coverage) == set(lookup) and all(v == 1 for v in block_coverage.values())
    seen_groups = set()
    for g in groups:
        key = (g['domain'], g['scheme'], g['group_id'])
        assert key not in seen_groups
        seen_groups.add(key)
        s = split_lookup[part_key(g)]
        assert int(s['rank_start']) <= int(g['rank_start']) < int(g['rank_end_exclusive']) <= int(s['rank_end_exclusive'])
    for key, s in split_lookup.items():
        cursor = int(s['rank_start'])
        for g in (g for g in groups if part_key(g) == key):
            assert int(g['rank_start']) == cursor
            assert int(g['rank_end_exclusive']) - cursor <= 10000
            cursor = int(g['rank_end_exclusive'])
        assert cursor == int(s['rank_end_exclusive'])
    folds = json.loads((root / 'fold_manifest.json').read_text())
    for f in folds.values():
        assert len(set(f['development'] + [f['heldout'], f['source_domain']])) == 4
    result = dict(status='PASS', verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  serialized_windows_checked=len(windows), serialized_blocks_checked=len(blocks),
                  serialized_groups_checked=len(groups), source_window_smoke='NOT_REQUESTED')
    if args.smoke_source_window:
        w = next(w for w in windows if w['domain'] == 'UNSW' and w['partition'] == 'S_train' and w['window_size'] == '10000')
        ids = indices['UNSW'][int(w['rank_start']):int(w['rank_end_exclusive'])]
        position = {int(row): i for i, row in enumerate(ids)}
        views = read_csv(root / 'feature_view_manifest_v1.csv')
        names = [v['feature'] for v in views]
        assert len(set(names)) == 49 and not set(names).intersection({'Label', 'Attack', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'})
        matrix = np.empty((len(ids), len(names)), dtype=np.float64)
        labels = np.empty(len(ids), dtype=np.int8)
        found = 0
        path = Path(config['data_root']) / config['experiment_config']['datasets']['UNSW']
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            cols = [header.index(n) for n in names]
            label_col = header.index('Label')
            for row_id, row in enumerate(reader):
                if row_id in position:
                    i = position[row_id]
                    # Preserve empty numeric fields as missing; do not fit/impute here.
                    matrix[i] = [float(row[j]) if row[j].strip() else np.nan for j in cols]
                    labels[i] = int(row[label_col])
                    found += 1
                if found == len(ids):
                    break
        assert found == len(ids)
        result['source_window_smoke'] = dict(status='PASS', window_id=w['window_id'], rows=found,
                    view_shapes={v: [found, sum(r['view'] == v for r in views)] for v in ('Traffic', 'Protocol', 'Behavior')},
                    benign=int((labels == 0).sum()), attack=int((labels == 1).sum()),
                    nonfinite_feature_values=int((~np.isfinite(matrix)).sum()),
                    missing_by_feature={name: int(np.isnan(matrix[:, j]).sum()) for j, name in enumerate(names) if np.isnan(matrix[:, j]).any()},
                    infinity_by_feature={name: int(np.isinf(matrix[:, j]).sum()) for j, name in enumerate(names) if np.isinf(matrix[:, j]).any()},
                    missing_policy='preserved_as_nan_no_imputation_fitted', fitted_artifacts=0)
    (root / 'artifact_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    source = Path(__file__).resolve()
    if source != root / source.name:
        shutil.copyfile(source, root / source.name)
    marker = '\n## drop-disk review and small-scale read-only input\n'
    report = (root / 'data_preparation_report.md').read_text(encoding='utf-8').split(marker)[0]
    extra = [marker, f" Reviewing time:{result['verified_at_utc']}。",
             f" Reopened and verified {len(windows):,} A window.{len(blocks):,} It’s a resampling block.{len(groups):,} The group allocation: all approved.", '',
             'partition UTC Available time range (including the partition tail record):', '',
             '| data set | use | partition | early UTC | late UTC |', '|---|---|---|---|---|']
    def utc(ms):
        return datetime.fromtimestamp(int(ms)/1000, timezone.utc).isoformat() if ms else 'Empty partition'
    for s in splits:
        extra.append(f"| {s['domain']} | {s['scheme']} | {s['partition']} | {utc(s['first_available_ms'])} | {utc(s['last_available_ms'])} |")
    if args.smoke_source_window:
        smoke = result['source_window_smoke']
        extra += ['', f" Only small-scale exams have been passed:{smoke['window_id']} 10,000 records source domain, and the three group arrays are 10000×17, 10000×17, and 10000×15.",
                  f" The window Benign={smoke['benign']}、Attack={smoke['attack']} Non-limited features value of ={smoke['nonfinite_feature_values']}. This only checks the features readings of a single window, and does not represent the full features quality scan of the table.",
                  f" Count the empty fields:{json.dumps(smoke['missing_by_feature'], ensure_ascii=False)}. After the first direct numerical conversion of the discovered empty field, the reader has been changed to retain NaN; Not deleted records or filled in fit values. Follow-up by agreement S_train fit preprocessing , re-apply to other divisions."]
        extra += [f" Count the infinite values:{json.dumps(smoke['infinity_by_feature'], ensure_ascii=False)}. At this stage, non-finite values processing protocols must be implemented prior to training; PASS readings here do not mean that features can be sent directly to the model."]
    import shlex
    extra += ['', 'Small-scale data inputs that can be run directly (non-trained):', '', '```bash',
              shlex.join([sys.executable, str(source), '--run-dir', str(root), '--smoke-source-window']), '```', '',
              'A new --output directory must be specified when the prepare_data.py is fully restarted; Script refuses to cover existing directories.', '']
    (root / 'data_preparation_report.md').write_text(report + '\n'.join(extra), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
