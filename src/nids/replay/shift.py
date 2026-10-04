from __future__ import annotations

import json
from pathlib import Path

import cupy as cp
import numpy as np


def fit_jsd_reference(
    output: Path,
    features: list[str],
    numeric_indices: list[int],
    categorical_indices: list[int],
    imputed_source: cp.ndarray,
    source_categorical: cp.ndarray,
    categorical_fill: cp.ndarray,
    quantiles: int,
) -> dict:
    numeric_arrays: dict[str, np.ndarray] = {}
    categorical_arrays: dict[str, np.ndarray] = {}
    feature_records = []
    probabilities = cp.linspace(0, 1, quantiles + 1, dtype=cp.float64)[1:-1]
    for index in numeric_indices:
        name = features[index]
        internal = cp.unique(cp.quantile(imputed_source[:, index], probabilities))
        bins = cp.searchsorted(internal, imputed_source[:, index], side="right")
        counts = cp.bincount(bins, minlength=internal.size + 1).astype(cp.float64)
        reference = counts / counts.sum()
        numeric_arrays[f"{name}__internal_edges"] = cp.asnumpy(internal)
        numeric_arrays[f"{name}__probabilities"] = cp.asnumpy(reference)
        feature_records.append({
            "feature": name, "kind": "numeric", "effective_bins": int(internal.size + 1),
            "duplicate_edges_collapsed": int((quantiles - 1) - internal.size),
            "outer_edges": ["-Infinity", "Infinity"],
        })
    filled_categorical = cp.where(cp.isnan(source_categorical), categorical_fill, source_categorical)
    for auxiliary_index, feature_index in enumerate(categorical_indices):
        name = features[feature_index]
        values, counts = cp.unique(filled_categorical[:, auxiliary_index], return_counts=True)
        reference = counts.astype(cp.float64) / counts.sum()
        categorical_arrays[f"{name}__values"] = cp.asnumpy(values)
        categorical_arrays[f"{name}__probabilities"] = cp.asnumpy(reference)
        feature_records.append({
            "feature": name, "kind": "categorical", "source_vocabulary_size": int(values.size),
            "unseen_category": "OTHER",
        })
    np.savez(output / "jsd_numeric_reference.npz", **numeric_arrays)
    np.savez(output / "jsd_categorical_reference.npz", **categorical_arrays)
    manifest = {
        "status": "COMPLETE", "fit_partition": "S_train", "fit_rows": int(imputed_source.shape[0]),
        "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode()),
        "numeric_rule": f"{quantiles}_source_equal_frequency_quantiles_duplicate_edges_collapsed",
        "categorical_rule": "source_vocabulary_plus_OTHER", "logarithm_base": 2,
        "aggregate": "mean_over_49_marginals", "features": feature_records,
        "numeric_reference_file": "jsd_numeric_reference.npz",
        "categorical_reference_file": "jsd_categorical_reference.npz",
    }
    (output / "jsd_source_reference.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest



class FrozenJSD:
    """GPU marginal histograms against the unchanged P1 source references."""

    def __init__(self, config: dict):
        from cupyx.scipy.special import rel_entr
        self.rel_entr = rel_entr
        root = Path(config['project_root']) / config['outputs']['source'] / 'preprocessing'
        self.features = json.loads((root / 'preprocessing_manifest.json').read_text())['feature_order']
        with np.load(root / 'preprocessing_stats.npz') as stats:
            self.fill = cp.asarray(stats['fill'])
        self.categorical = config['shift']['categorical_features']
        self.references = []
        with np.load(root / 'jsd_numeric_reference.npz') as numeric, np.load(root / 'jsd_categorical_reference.npz') as categorical:
            for index, name in enumerate(self.features):
                if name in self.categorical:
                    vocab = cp.asarray(categorical[f'{name}__values'])
                    p = cp.concatenate((cp.asarray(categorical[f'{name}__probabilities']), cp.zeros(1)))
                    self.references.append((index, self.categorical.index(name), vocab, p))
                else:
                    self.references.append((index, None, cp.asarray(numeric[f'{name}__internal_edges']),
                                            cp.asarray(numeric[f'{name}__probabilities'])))

    def batch(self, values: np.ndarray, categorical: np.ndarray) -> cp.ndarray:
        # Batch dimensions are [complete windows, window_size rows, columns].
        x = cp.asarray(values, dtype=cp.float64)
        x = cp.where(cp.isfinite(x), x, self.fill)
        c = cp.asarray(categorical, dtype=cp.float64)
        cat_indices = [self.features.index(name) for name in self.categorical]
        c = cp.where(cp.isfinite(c), c, self.fill[cat_indices])
        result = cp.empty((len(x), len(self.features)), dtype=cp.float64)
        for index, auxiliary, boundaries, p in self.references:
            k = len(p)
            if auxiliary is None:
                bins = cp.searchsorted(boundaries, x[:,:,index], side='right')
            else:
                v = c[:,:,auxiliary]
                position = cp.searchsorted(boundaries, v)
                matches = (position < len(boundaries)) & (boundaries[cp.minimum(position, len(boundaries)-1)] == v)
                bins = cp.where(matches, position, len(boundaries))
            flat = bins + cp.arange(len(x), dtype=cp.int64)[:,None] * k
            q = cp.bincount(flat.ravel(), minlength=len(x)*k).reshape(len(x),k) / x.shape[1]
            mean = (p[None,:] + q) / 2
            # Both-zero cells have their mathematical limiting contribution zero.
            safe_mean = cp.where(mean > 0, mean, 1.)
            jsd = (self.rel_entr(p[None,:], safe_mean) + self.rel_entr(q, safe_mean)).sum(axis=1) / (2 * cp.log(2.))
            result[:,index] = jsd
        assert bool(cp.isfinite(result).all())
        assert float(result.min()) >= -1e-14 and float(result.max()) <= 1+1e-14
        return result


def compute_domain_shift(config: dict, prepared, domain: str, window_size=10000, algorithm='jsd', output_root=None, reference=None) -> dict:
    import time
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nids.models.base_models import GpuMemoryMonitor
    from .predictions import write_json

    started = time.perf_counter()
    project = Path(config['project_root'])
    output = Path(output_root) if output_root is not None else project / config['outputs']['shift'] / '10k'
    output.mkdir(parents=True, exist_ok=True)
    store = project / config['outputs']['feature_store']
    x = np.load(store / f'{domain}_X.npy', mmap_mode='r')
    c = np.load(store / f'{domain}_categorical.npy', mmap_mode='r')
    windows = [r for r in prepared.windows if r['domain']==domain and int(r['window_size'])==window_size and (output_root is None or r['scheme']=='development')]
    intervals = sorted({(int(r['rank_start']),int(r['rank_end_exclusive'])) for r in windows})
    monitor = GpuMemoryMonitor(0)
    monitor.start()
    reference = FrozenJSD(config) if reference is None else reference
    marginal = {}
    batch_windows = min(config['batches']['jsd_windows'],max(1,320000//window_size)) if algorithm=='jsd' else 1
    for start in range(0,len(intervals),batch_windows):
        batch = intervals[start:start+batch_windows]
        features = np.stack([x[a:b] for a,b in batch])
        categories = np.stack([c[a:b] for a,b in batch])
        distances = cp.asnumpy(reference.batch(features,categories))
        if start % 100 == 0:
            print(f'{domain}/{algorithm}/{window_size}: {start}/{len(intervals)} windows',flush=True)
        for pair, distance in zip(batch, distances):
            marginal[pair] = distance
    lookup = {r['window_id']:r for r in windows}
    ends = prepared.sorted_end_ms(domain)
    order = prepared.sorted_rows(domain)
    rows = []
    for r in windows:
        a,b = int(r['rank_start']),int(r['rank_end_exclusive'])
        assert b-a == window_size
        assert int(order[a]) == int(r['first_original_row']) and int(order[b-1]) == int(r['last_original_row'])
        assert int(ends[a]) == int(r['first_available_ms']) and int(ends[b-1]) == int(r['last_available_ms'])
        previous_id = r['predecessor_window_id']
        initial = r['initialization']=='True'
        assert initial == (not previous_id)
        if previous_id:
            previous = lookup[previous_id]
            assert previous['stream_id']==r['stream_id'] and previous['scheme']==r['scheme']
            pa_,pb = int(previous['rank_start']),int(previous['rank_end_exclusive'])
            assert pb <= a and int(previous['last_available_ms']) <= int(r['first_available_ms'])
            assert int(previous['last_available_ms']) == int(r['predecessor_last_available_ms'])
            shift = float(marginal[(pa_,pb)].mean())
        else:
            shift = 0.
        row = dict(r)
        for key in ['rank_start','rank_end_exclusive','window_size','first_original_row','last_original_row','first_available_ms','last_available_ms']:
            row[key] = int(row[key])
        row['predecessor_last_available_ms'] = int(r['predecessor_last_available_ms']) if previous_id else None
        row['previous_window_id'] = previous_id or None
        row['initialization'] = initial
        row['cross_partition_context'] = r['cross_partition_context']=='True'
        row[algorithm] = float(marginal[(a,b)].mean())
        row['shift'] = shift
        if algorithm=='jsd':
            row['marginal_jsd'] = marginal[(a,b)].tolist()
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), output / f'{domain}_windows.parquet', compression='zstd')
    report = dict(domain=domain, manifest_windows=len(rows), unique_data_windows=len(intervals),
                  initialization_windows=sum(r['initialization'] for r in rows),
                  cross_partition_windows=sum(r['cross_partition_context'] for r in rows),
                  seconds=time.perf_counter()-started, gpu_memory=monitor.stop(), device='cuda:0',
                  algorithm=algorithm,window_size=window_size,distance_min=min(r[algorithm] for r in rows), distance_max=max(r[algorithm] for r in rows),
                  shift_min=min(r['shift'] for r in rows), shift_max=max(r['shift'] for r in rows),
                  checks='window_identity_and_predecessor_completion_PASS')
    if algorithm == 'jsd':
        report.update(jsd_min=report['distance_min'],jsd_max=report['distance_max'])
    write_json(output / f'{domain}_manifest.json',report)
    print(f"{domain}/{algorithm}/{window_size}: {len(rows):,} window identities, {len(intervals):,} computed intervals, {report['seconds']:.2f}s",flush=True)
    return report
