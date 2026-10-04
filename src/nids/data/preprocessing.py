from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

from .partitions import PreparedData
from nids.replay.shift import fit_jsd_reference


class SourcePreprocessor:
    def __init__(self, fill: np.ndarray, mean: np.ndarray, scale: np.ndarray):
        self.fill = cp.asarray(fill, dtype=cp.float64)
        self.mean = cp.asarray(mean, dtype=cp.float64)
        self.scale = cp.asarray(scale, dtype=cp.float64)

    def transform(self, values: np.ndarray, standardized: bool) -> cp.ndarray:
        result = cp.asarray(values, dtype=cp.float64)
        result = cp.where(cp.isfinite(result), result, cp.nan)
        result = cp.where(cp.isnan(result), self.fill, result)
        if standardized:
            result = (result - self.mean) / self.scale
        return result.astype(cp.float32)


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fit_source_preprocessing(config: dict, prepared: PreparedData) -> dict:
    started = time.perf_counter()
    project = Path(config["project_root"])
    feature_root = project / config["outputs"]["feature_store"]
    output = project / config["outputs"]["source"] / "preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    feature_manifest = json.loads((feature_root / "feature_store_manifest.json").read_text(encoding="utf-8"))
    features = feature_manifest["feature_order"]
    categorical = config["shift"]["categorical_features"]
    categorical_indices = [features.index(name) for name in categorical]
    numeric_indices = [index for index in range(49) if index not in categorical_indices]
    source = prepared.partition("UNSW", "source", "S_train")
    source_x = np.load(feature_root / "UNSW_X.npy", mmap_mode="r")[source.rank_start:source.rank_end_exclusive]
    source_categorical = np.load(feature_root / "UNSW_categorical.npy", mmap_mode="r")[source.rank_start:source.rank_end_exclusive]
    gpu_x = cp.asarray(source_x, dtype=cp.float64)
    print(f"UNSW S_train: fitting preprocessing on CUDA:0 with {source.rows:,} rows", flush=True)
    gpu_x = cp.where(cp.isfinite(gpu_x), gpu_x, cp.nan)
    fill = cp.zeros(49, dtype=cp.float64)
    all_missing = []
    for index in numeric_indices:
        valid = gpu_x[:, index][~cp.isnan(gpu_x[:, index])]
        if valid.size == 0:
            fill[index] = config["preprocessing"]["all_source_missing_fill"]
            all_missing.append(features[index])
        else:
            fill[index] = cp.median(valid)
    gpu_categorical = cp.asarray(source_categorical, dtype=cp.float64)
    gpu_categorical = cp.where(cp.isfinite(gpu_categorical), gpu_categorical, cp.nan)
    for auxiliary_index, feature_index in enumerate(categorical_indices):
        valid = gpu_categorical[:, auxiliary_index][~cp.isnan(gpu_categorical[:, auxiliary_index])]
        if valid.size == 0:
            fill[feature_index] = config["preprocessing"]["all_source_missing_fill"]
            all_missing.append(features[feature_index])
        else:
            values, counts = cp.unique(valid, return_counts=True)
            largest = counts.max()
            fill[feature_index] = values[counts == largest].min()
    imputed = cp.where(cp.isnan(gpu_x), fill, gpu_x)
    mean = imputed.mean(axis=0, dtype=cp.float64)
    scale = imputed.std(axis=0, dtype=cp.float64)
    constant = cp.asnumpy(scale == 0)
    scale[scale == 0] = config["preprocessing"]["constant_scale"]
    fill_cpu, mean_cpu, scale_cpu = cp.asnumpy(fill), cp.asnumpy(mean), cp.asnumpy(scale)
    np.savez(output / "preprocessing_stats.npz", fill=fill_cpu, mean=mean_cpu, scale=scale_cpu)
    jsd_manifest = fit_jsd_reference(
        output=output,
        features=features,
        numeric_indices=numeric_indices,
        categorical_indices=categorical_indices,
        imputed_source=imputed,
        source_categorical=gpu_categorical,
        categorical_fill=fill[categorical_indices],
        quantiles=int(config["shift"]["numeric_quantiles"]),
    )
    preprocessor = SourcePreprocessor(fill_cpu, mean_cpu, scale_cpu)
    transformed_checks = []
    validation_rows = int(config["batches"]["inference_rows"])
    for domain in config["datasets"]:
        print(f"{domain}: validating finite tree and MLP transforms", flush=True)
        values = np.load(feature_root / f"{domain}_X.npy", mmap_mode="r")
        finite_tree = 0
        finite_mlp = 0
        for start in range(0, len(values), validation_rows):
            stop = min(start + validation_rows, len(values))
            tree = preprocessor.transform(values[start:stop], standardized=False)
            mlp = preprocessor.transform(values[start:stop], standardized=True)
            finite_tree += int(cp.count_nonzero(cp.isfinite(tree)).get())
            finite_mlp += int(cp.count_nonzero(cp.isfinite(mlp)).get())
        expected = len(values) * 49
        assert finite_tree == expected
        assert finite_mlp == expected
        transformed_checks.append({
            "domain": domain, "rows": len(values), "tree_finite_values": finite_tree,
            "mlp_finite_values": finite_mlp, "expected_finite_values": expected,
        })
    pool_total = cp.get_default_memory_pool().total_bytes()
    manifest = {
        "version": "source-preprocessing-v2.0", "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_domain": "UNSW", "fit_partition": "S_train", "fit_rank_start": source.rank_start,
        "fit_rank_end_exclusive": source.rank_end_exclusive, "fit_rows": source.rows,
        "numeric_fill": "finite_median", "categorical_fill": "mode_smallest_value_tiebreak",
        "all_source_missing_fill": config["preprocessing"]["all_source_missing_fill"],
        "all_source_missing_features": all_missing,
        "standardization": "S_train_imputed_mean_population_standard_deviation",
        "constant_scale_features": [feature for feature, flag in zip(features, constant) if flag],
        "statistics_file": "preprocessing_stats.npz", "feature_order": features,
        "artifact_sizes_bytes": {
            name: (output / name).stat().st_size for name in [
                "preprocessing_stats.npz", "jsd_numeric_reference.npz",
                "jsd_categorical_reference.npz", "jsd_source_reference.json",
            ]
        },
        "fit_device": str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode()),
        "fit_device_index": 0, "gpu_memory_pool_peak_proxy_bytes": pool_total,
        "transformed_full_store_finite_checks": transformed_checks,
        "jsd_reference": jsd_manifest, "elapsed_seconds": time.perf_counter() - started,
    }
    _save_json(output / "preprocessing_manifest.json", manifest)
    return manifest
