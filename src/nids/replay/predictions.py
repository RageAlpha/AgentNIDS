from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from cupyx.scipy.special import xlogy

from nids.data.preprocessing import SourcePreprocessor
from nids.models.base_models import MODEL_SPECS, GpuMemoryMonitor, load_model

PREDICTION_DTYPE = np.dtype([
    ('raw_probability', '<f8'), ('probability', '<f8'),
    ('predicted_class', 'u1'), ('u', '<f8'), ('h', '<f8'),
])


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_preprocessor(config: dict) -> SourcePreprocessor:
    path = Path(config['project_root']) / config['outputs']['source'] / 'preprocessing'
    with np.load(path / 'preprocessing_stats.npz') as stats:
        return SourcePreprocessor(stats['fill'], stats['mean'], stats['scale'])


def gpu_raw_probability(model, spec, values: cp.ndarray) -> cp.ndarray:
    if spec.family == 'rf':
        result = model.predict_proba(values)
        assert isinstance(result, cp.ndarray) and result.device.id == 0
        return result[:, 1].astype(cp.float64)
    if spec.family == 'xgb':
        # In-place booster prediction preserves CUDA residency, including on reload.
        result = model.get_booster().inplace_predict(values)
        assert isinstance(result, cp.ndarray) and result.device.id == 0
        return result.astype(cp.float64)
    with torch.inference_mode():
        logits = model(torch.from_dlpack(values))
        assert logits.device.type == 'cuda'
        return cp.from_dlpack(torch.sigmoid(logits)).astype(cp.float64)


def gpu_evidence(raw: cp.ndarray, calibrator: dict, h: cp.ndarray, threshold: float):
    epsilon = calibrator['input_probability_epsilon']
    clipped = cp.clip(raw, epsilon, 1 - epsilon)
    logit = cp.log(clipped) - cp.log1p(-clipped)
    # Same float64 PyTorch sigmoid as the frozen P2 calibrators.
    with torch.inference_mode():
        probability = cp.from_dlpack(torch.sigmoid(
            torch.from_dlpack(logit) * calibrator['coefficient'] + calibrator['intercept']
        ))
    predicted = (probability >= threshold).astype(cp.uint8)
    uncertainty = -(xlogy(probability, probability) + xlogy(1-probability, 1-probability)) / cp.log(2.)
    return probability, predicted, uncertainty, h[predicted]


def predict_domain(config: dict, domain: str) -> dict:
    started = time.perf_counter()
    project = Path(config['project_root'])
    source = project / config['outputs']['source']
    store = project / config['outputs']['feature_store']
    output = project / config['outputs']['base_predictions'] / domain
    output.mkdir(parents=True, exist_ok=True)
    values = np.load(store / f'{domain}_X.npy', mmap_mode='r')
    feature_order = json.loads((store / 'feature_store_manifest.json').read_text())['feature_order']
    preprocessor = load_preprocessor(config)
    rows_per_batch = config['batches']['inference_rows']
    reports = []
    for model_id in config['models']['schedule']:
        model_started = time.perf_counter()
        spec = MODEL_SPECS[model_id]
        folder = source / 'models' / model_id
        manifest = json.loads((folder / 'training_manifest.json').read_text())
        indices = manifest['feature_indices']
        assert manifest['feature_names'] == [feature_order[i] for i in indices]
        assert manifest['parameters'] == config['models'][spec.family]
        model = load_model(spec, folder / manifest['artifact'], 'cuda:0')
        if spec.family == 'xgb':
            assert json.loads(model.get_booster().save_config())['learner']['generic_param']['device'] == 'cuda:0'
        calibrator = json.loads((source / 'calibrators' / f'{model_id}.json').read_text())
        table = json.loads((source / 'competence' / f'{model_id}.json').read_text())['table']
        h = cp.asarray([r['h'] for r in sorted(table, key=lambda r: r['predicted_class'])])
        cache = np.lib.format.open_memmap(output / f'{model_id}.npy', mode='w+', dtype=PREDICTION_DTYPE, shape=(len(values),))
        monitor = GpuMemoryMonitor(0)
        monitor.start()
        minimum = np.ones(5)
        maximum = np.zeros(5)
        events_ms = 0.
        for start in range(0, len(values), rows_per_batch):
            end = min(start + rows_per_batch, len(values))
            x = cp.ascontiguousarray(preprocessor.transform(values[start:end], spec.standardized)[:, indices])
            assert bool(cp.isfinite(x).all())
            before, after = cp.cuda.Event(), cp.cuda.Event()
            before.record()
            raw = gpu_raw_probability(model, spec, x)
            after.record()
            after.synchronize()
            events_ms += cp.cuda.get_elapsed_time(before, after)
            columns = (raw, *gpu_evidence(raw, calibrator, h, config['competence']['attack_threshold']))
            for index, (name, column) in enumerate(zip(PREDICTION_DTYPE.names, columns)):
                assert bool(cp.isfinite(column).all())
                low, high = float(column.min()), float(column.max())
                assert 0 <= low <= high <= 1
                minimum[index] = min(minimum[index], low)
                maximum[index] = max(maximum[index], high)
                cache[name][start:end] = cp.asnumpy(column)
        cache.flush()
        memory = monitor.stop()
        report = dict(model_id=model_id, domain=domain, rows=len(values), feature_indices=indices,
                      feature_names=manifest['feature_names'], device='cuda:0',
                      gpu_input_and_output_verified=True, prediction_cuda_event_ms=events_ms,
                      range={name: [float(a), float(b)] for name,a,b in zip(PREDICTION_DTYPE.names,minimum,maximum)},
                      seconds=time.perf_counter()-model_started, gpu_memory=memory,
                      file=str((output / f'{model_id}.npy').relative_to(project)),
                      bytes=(output / f'{model_id}.npy').stat().st_size)
        reports.append(report)
        write_json(output / f'{model_id}_manifest.json', report)
        print(f"{domain}/{model_id}: {len(values):,} rows, {report['seconds']:.2f}s, CUDA verified", flush=True)
        del model, cache, x, columns, raw
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    result = dict(domain=domain, rows=len(values), models=reports, seconds=time.perf_counter()-started,
                  row_order='availability_rank', record_id='domain:original_row',
                  original_row_index=str(Path(config['prepared_run']) / f'{domain}_sorted_original_rows.npy'))
    write_json(output / 'manifest.json', result)
    return result
