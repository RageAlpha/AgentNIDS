"""Read the existing heldout role without exposing labels to prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PRIMARY = ['rf_Traffic', 'xgb_Protocol', 'mlp_Behavior']
CONTROL = ['rf_All49', 'xgb_All49', 'mlp_All49']
METHODS = ['prior', 'raw_h', 'u', 'hu', 'hus', 'one_minus_u']


def checked_windows(prepared, domain, rows):
    part = prepared.partition(domain, 'heldout', 'T_heldout')
    original = {r['window_id']: r for r in prepared.windows
                if r['domain'] == domain and r['scheme'] == 'heldout'
                and r['partition'] == 'T_heldout' and int(r['window_size']) == 10000}
    windows = sorted([r for r in rows if r['scheme'] == 'heldout'
                      and r['partition'] == 'T_heldout' and r['window_size'] == 10000],
                     key=lambda r: r['rank_start'])
    if not windows or len(windows) != len(original):
        raise ValueError('Heldout window inventory is empty or inconsistent')
    order, ends = prepared.sorted_rows(domain), prepared.sorted_end_ms(domain)
    cursor = part.rank_start
    previous = None
    for w in windows:
        r = original[w['window_id']]
        a, b = w['rank_start'], w['rank_end_exclusive']
        if a != cursor or b-a != 10000 or b > part.rank_end_exclusive:
            raise ValueError('Heldout windows must be a contiguous complete-window prefix')
        for key in ['rank_start', 'rank_end_exclusive', 'first_original_row',
                    'last_original_row', 'first_available_ms', 'last_available_ms']:
            if int(r[key]) != w[key]:
                raise ValueError(f'Window manifest mismatch: {key}')
        if (int(order[a]), int(order[b-1]), int(ends[a]), int(ends[b-1])) != (
                w['first_original_row'], w['last_original_row'],
                w['first_available_ms'], w['last_available_ms']):
            raise ValueError('Heldout identity mismatch')
        expected_id = previous['window_id'] if previous else ''
        if (w['predecessor_window_id'] or '') != expected_id or r['predecessor_window_id'] != expected_id:
            raise ValueError('Heldout predecessor must belong to its own completed stream')
        if (w['previous_window_id'] or '') != expected_id or w['initialization'] != (previous is None):
            raise ValueError('Invalid heldout initialization')
        expected_shift = previous['jsd'] if previous else 0.0
        if not np.isfinite(w['shift']) or w['shift'] != expected_shift:
            raise ValueError('Shift does not equal the completed predecessor JSD')
        if previous and (w['stream_id'] != previous['stream_id']
                         or previous['last_available_ms'] > w['first_available_ms']):
            raise ValueError('Noncausal predecessor')
        cursor, previous = b, w
    if part.rank_end_exclusive-cursor >= 10000:
        raise ValueError('Complete heldout windows are missing')
    return part, windows


class Heldout:
    """Prediction-only object: deliberately has no label path or labels method."""

    def __init__(self, config, prepared, fold, score_folder):
        self.domain = config['folds'][fold]['heldout']
        if self.domain in config['folds'][fold]['development']:
            raise ValueError('Heldout domain appears in development domains')
        root = Path(config['project_root'])
        rows = pq.read_table(root/config['outputs']['shift']/'10k'/f'{self.domain}_windows.parquet').to_pylist()
        part, self.windows = checked_windows(prepared, self.domain, rows)
        self.a, self.b = part.rank_start, self.windows[-1]['rank_end_exclusive']
        self.n, self.fold, self.partition = self.b-self.a, fold, 'T_heldout'
        self.shift = np.repeat([r['shift'] for r in self.windows], 10000)
        self.cache = [np.load(root/config['outputs']['base_predictions']/self.domain/f'{m}.npy', mmap_mode='r') for m in PRIMARY]
        self.pre = [json.loads((root/config['outputs']['round1']/'P4'/fold/'models'/f'{m}.json').read_text()) for m in PRIMARY]
        self.score_folder = Path(score_folder)
        self.audit = dict(domain=self.domain, partition=self.partition, rows=self.n,
                          windows=len(self.windows), rank_start=self.a, rank_end_exclusive=self.b,
                          excluded_tail=part.rank_end_exclusive-self.b,
                          initialization_rows=sum(w['initialization'] for w in self.windows)*10000)

    def open_scores(self):
        self.pre_cache = [np.load(self.score_folder/f'{m}.npy', mmap_mode='r') for m in PRIMARY]

    def accessor(self, start, stop):
        from nids.coordination.evidence import VisibleEvidence
        return VisibleEvidence(self, start, stop)

    def write_identities(self, prepared, path):
        order, ends = prepared.sorted_rows(self.domain), prepared.sorted_end_ms(self.domain)
        with pq.ParquetWriter(path, pa.schema([
                ('record_id', pa.string()), ('rank', pa.int64()), ('original_row', pa.int64()),
                ('availability_ms', pa.int64()), ('window_id', pa.string()),
                ('previous_window_id', pa.string()), ('initialization', pa.bool_()),
                ('complete_window', pa.bool_()), ('shift', pa.float64())]), compression='zstd') as writer:
            for w in self.windows:
                a, b = w['rank_start'], w['rank_end_exclusive']
                writer.write_table(pa.table(dict(
                    record_id=[f'{self.domain}:{int(i)}' for i in order[a:b]],
                    rank=np.arange(a, b), original_row=np.asarray(order[a:b], dtype=np.int64),
                    availability_ms=np.asarray(ends[a:b], dtype=np.int64),
                    window_id=[w['window_id']]*(b-a), previous_window_id=[w['previous_window_id']]*(b-a),
                    initialization=np.full(b-a, w['initialization']), complete_window=np.ones(b-a, dtype=bool),
                    shift=np.full(b-a, w['shift'])), schema=writer.schema), row_group_size=10000)


def evaluation_arrays(config, data):
    """Called only after all fold predictions have been committed."""
    import cupy as cp
    root = Path(config['project_root'])
    labels = cp.asarray(np.load(root/config['outputs']['feature_store']/f'{data.domain}_label.npy', mmap_mode='r')[data.a:data.b])
    caches = {m: np.load(root/config['outputs']['base_predictions']/data.domain/f'{m}.npy', mmap_mode='r') for m in PRIMARY+CONTROL}
    probability = {m: cp.asarray(v['probability'][data.a:data.b]) for m, v in caches.items()}
    for m, v in caches.items():
        if not bool(cp.all((probability[m] >= .5) == cp.asarray(v['predicted_class'][data.a:data.b]))):
            raise ValueError('Base prediction threshold mismatch')
    audit = dict(data.audit, benign=int((labels == 0).sum()), attack=int((labels == 1).sum()))
    return dict(audit=audit, labels=labels, probability=probability,
                errors=cp.column_stack([(probability[m] >= .5) != labels for m in PRIMARY]),
                control_errors=cp.column_stack([(probability[m] >= .5) != labels for m in CONTROL]))
