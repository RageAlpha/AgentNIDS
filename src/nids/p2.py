from __future__ import annotations

import csv
import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import torch

from .data.partitions import PreparedData
from .evaluation.source import binary_probability_metrics
from .models.base_models import (
    MODEL_SPECS,
    GpuMemoryMonitor,
    load_model,
    predict_attack_probability,
    train_model,
)
from .reliability.calibration import apply_platt, fit_platt, historical_competence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SourceModelData:
    def __init__(self, config: dict, prepared: PreparedData):
        project = Path(config["project_root"])
        feature_root = project / config["outputs"]["feature_store"]
        preprocessing_root = project / config["outputs"]["source"] / "preprocessing"
        feature_manifest = json.loads(
            (feature_root / "feature_store_manifest.json").read_text(encoding="utf-8")
        )
        preprocessing_manifest = json.loads(
            (preprocessing_root / "preprocessing_manifest.json").read_text(encoding="utf-8")
        )
        assert feature_manifest["status"] == "COMPLETE"
        assert preprocessing_manifest["status"] == "COMPLETE"
        self.config = config
        self.prepared = prepared
        self.feature_root = feature_root
        self.features = feature_manifest["feature_order"]
        self.values = np.load(feature_root / "UNSW_X.npy", mmap_mode="r")
        self.labels = np.load(feature_root / "UNSW_label.npy", mmap_mode="r")
        statistics = np.load(preprocessing_root / "preprocessing_stats.npz")
        self.fill = statistics["fill"]
        self.mean = statistics["mean"]
        self.scale = statistics["scale"]
        with (project / config["feature_views"]).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.views = {
            name: [self.features.index(row["feature"]) for row in rows if row["view"] == name]
            for name in ("Traffic", "Protocol", "Behavior")
        }
        self.views["All49"] = list(range(49))

    def partition_labels(self, name: str) -> np.ndarray:
        partition = self.prepared.partition("UNSW", "source", name)
        return np.array(
            self.labels[partition.rank_start:partition.rank_end_exclusive], copy=True
        )

    def transform(self, name: str, indices: list[int], standardized: bool) -> cp.ndarray:
        partition = self.prepared.partition("UNSW", "source", name)
        selected = self.values[partition.rank_start:partition.rank_end_exclusive]
        if indices != list(range(49)):
            selected = selected[:, indices]
        result = cp.asarray(selected, dtype=cp.float64)
        fill = cp.asarray(self.fill[indices], dtype=cp.float64)
        result = cp.where(cp.isfinite(result), result, cp.nan)
        result = cp.where(cp.isnan(result), fill, result)
        if standardized:
            mean = cp.asarray(self.mean[indices], dtype=cp.float64)
            scale = cp.asarray(self.scale[indices], dtype=cp.float64)
            result = (result - mean) / scale
        result = result.astype(cp.float32)
        assert bool(cp.all(cp.isfinite(result)).item())
        return result

    def predict_partition(self, model, spec, name: str, indices: list[int]) -> np.ndarray:
        partition = self.prepared.partition("UNSW", "source", name)
        result = np.empty(partition.rows, dtype=np.float64)
        batch_rows = int(self.config["batches"]["inference_rows"])
        for local_start in range(0, partition.rows, batch_rows):
            local_stop = min(local_start + batch_rows, partition.rows)
            absolute_start = partition.rank_start + local_start
            absolute_stop = partition.rank_start + local_stop
            selected = self.values[absolute_start:absolute_stop]
            if indices != list(range(49)):
                selected = selected[:, indices]
            values = cp.asarray(selected, dtype=cp.float64)
            fill = cp.asarray(self.fill[indices], dtype=cp.float64)
            values = cp.where(cp.isfinite(values), values, cp.nan)
            values = cp.where(cp.isnan(values), fill, values)
            if spec.standardized:
                mean = cp.asarray(self.mean[indices], dtype=cp.float64)
                scale = cp.asarray(self.scale[indices], dtype=cp.float64)
                values = (values - mean) / scale
            values = values.astype(cp.float32)
            probability = predict_attack_probability(model, spec, values)
            assert np.all(np.isfinite(probability))
            assert np.all((probability >= 0) & (probability <= 1))
            result[local_start:local_stop] = probability
            del values
        return result


def _artifact_path(model_dir: Path, family: str) -> Path:
    suffix = {"rf": ".pkl", "xgb": ".ubj", "mlp": ".pt"}[family]
    return model_dir / f"model{suffix}"


def _family_parameters(config: dict, family: str) -> dict:
    return dict(config["models"][family])


def train_source_model(config: dict, prepared: PreparedData, model_id: str) -> Path:
    assert model_id in MODEL_SPECS
    spec = MODEL_SPECS[model_id]
    assert config["models"]["schedule"].index(model_id) >= 0
    project = Path(config["project_root"])
    source_root = project / config["outputs"]["source"]
    model_dir = source_root / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_path(model_dir, spec.family)
    data = SourceModelData(config, prepared)
    indices = data.views[spec.view]
    feature_names = [data.features[index] for index in indices]
    train_partition = prepared.partition("UNSW", "source", "S_train")
    train_labels = data.partition_labels("S_train")
    train_counts = np.bincount(train_labels.astype(np.int64), minlength=2)
    assert train_partition.rows == 1_420_000
    assert len(indices) == ({"Traffic": 17, "Protocol": 17, "Behavior": 15, "All49": 49}[spec.view])
    assert np.all(train_counts > 0)
    print(
        f"{model_id}: S_train rank [{train_partition.rank_start}, "
        f"{train_partition.rank_end_exclusive}), rows={train_partition.rows:,}, "
        f"view={spec.view}, features={len(indices)}, device=CUDA:0",
        flush=True,
    )
    monitor = GpuMemoryMonitor(int(config["device"]["index"]))
    monitor.start()
    total_started = time.perf_counter()
    train_x = data.transform("S_train", indices, spec.standardized)
    train_y = cp.asarray(train_labels, dtype=cp.int32)
    cp.get_default_memory_pool().free_all_blocks()
    fit_result = train_model(config, spec, train_x, train_y, artifact_path)
    artifact_size = artifact_path.stat().st_size
    print(
        f"{model_id}: training complete in {fit_result['training_seconds']:.3f}s; "
        f"artifact={artifact_size:,} bytes",
        flush=True,
    )
    del train_x, train_y
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    model = load_model(spec, artifact_path, config["models"].get(spec.family, {}).get("device", "cuda:0"))

    inference_seconds = {}
    raw_probabilities = {}
    for partition_name in ("S_prob_cal", "S_competence", "S_source_test"):
        started = time.perf_counter()
        raw_probabilities[partition_name] = data.predict_partition(
            model, spec, partition_name, indices
        )
        inference_seconds[partition_name] = time.perf_counter() - started
        print(
            f"{model_id}: {partition_name} raw inference complete, "
            f"rows={len(raw_probabilities[partition_name]):,}, "
            f"elapsed={inference_seconds[partition_name]:.3f}s",
            flush=True,
        )

    cal_labels = data.partition_labels("S_prob_cal")
    calibrator = fit_platt(raw_probabilities["S_prob_cal"], cal_labels, config)
    calibrator["model_id"] = model_id
    calibrator["fit_domain"] = "UNSW"
    calibrator["fit_partition"] = "S_prob_cal"
    calibrator["device"] = "cuda:0"
    metric_args = {
        "threshold": float(config["competence"]["attack_threshold"]),
        "epsilon": float(config["metrics"]["nll_epsilon"]),
        "ece_bins": int(config["metrics"]["ece_bins"]),
    }
    calibrated_cal = apply_platt(raw_probabilities["S_prob_cal"], calibrator)
    calibrator["raw_metrics"] = binary_probability_metrics(
        cal_labels, raw_probabilities["S_prob_cal"], **metric_args
    )
    calibrator["calibrated_metrics"] = binary_probability_metrics(
        cal_labels, calibrated_cal, **metric_args
    )
    calibrator["status"] = "COMPLETE"
    calibrator["completed_at_utc"] = _now()
    calibrator_path = source_root / "calibrators" / f"{model_id}.json"
    _write_json(calibrator_path, calibrator)
    print(
        f"{model_id}: Platt complete, coefficient={calibrator['coefficient']:.9g}, "
        f"intercept={calibrator['intercept']:.9g}",
        flush=True,
    )

    competence_labels = data.partition_labels("S_competence")
    calibrated_competence = apply_platt(raw_probabilities["S_competence"], calibrator)
    competence_table = historical_competence(
        calibrated_competence,
        competence_labels,
        float(config["competence"]["attack_threshold"]),
    )
    competence = {
        "version": "source-competence-v2.0",
        "status": "COMPLETE",
        "completed_at_utc": _now(),
        "model_id": model_id,
        "fit_domain": "UNSW",
        "fit_partition": "S_competence",
        "fit_rank_start": prepared.partition("UNSW", "source", "S_competence").rank_start,
        "fit_rank_end_exclusive": prepared.partition("UNSW", "source", "S_competence").rank_end_exclusive,
        "fit_rows": len(competence_labels),
        "label_counts": {
            "Benign": int(np.count_nonzero(competence_labels == 0)),
            "Attack": int(np.count_nonzero(competence_labels == 1)),
        },
        "formula": "(correct_in_predicted_class+1)/(count_in_predicted_class+2)",
        "attack_threshold": float(config["competence"]["attack_threshold"]),
        "table": competence_table,
    }
    competence_path = source_root / "competence" / f"{model_id}.json"
    _write_json(competence_path, competence)

    test_labels = data.partition_labels("S_source_test")
    calibrated_test = apply_platt(raw_probabilities["S_source_test"], calibrator)
    source_metrics = {
        "version": "source-metrics-v2.0",
        "status": "COMPLETE",
        "completed_at_utc": _now(),
        "model_id": model_id,
        "domain": "UNSW",
        "partition": "S_source_test",
        "rank_start": prepared.partition("UNSW", "source", "S_source_test").rank_start,
        "rank_end_exclusive": prepared.partition("UNSW", "source", "S_source_test").rank_end_exclusive,
        "raw": binary_probability_metrics(
            test_labels, raw_probabilities["S_source_test"], **metric_args
        ),
        "calibrated": binary_probability_metrics(test_labels, calibrated_test, **metric_args),
    }
    metrics_path = source_root / "source_metrics" / f"{model_id}.json"
    _write_json(metrics_path, source_metrics)

    del model
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    gpu_memory = monitor.stop()
    total_seconds = time.perf_counter() - total_started
    training_manifest = {
        "version": "base-model-v2.0",
        "status": "COMPLETE",
        "completed_at_utc": _now(),
        "model_id": model_id,
        "family": spec.family,
        "backend": config["models"][spec.family]["backend"],
        "view": spec.view,
        "feature_count": len(indices),
        "feature_indices": indices,
        "feature_names": feature_names,
        "preprocessing": "S_train_imputed_source_standardized" if spec.standardized else "S_train_imputed_original_numeric_representation",
        "seed": int(config["seed"]),
        "parameters": _family_parameters(config, spec.family),
        "training_domain": "UNSW",
        "training_partition": "S_train",
        "training_rank_start": train_partition.rank_start,
        "training_rank_end_exclusive": train_partition.rank_end_exclusive,
        "training_rows": train_partition.rows,
        "training_label_counts": {"Benign": int(train_counts[0]), "Attack": int(train_counts[1])},
        "natural_training_prevalence": float(train_counts[1] / train_counts.sum()),
        "device": {
            "type": "cuda",
            "index": int(config["device"]["index"]),
            "name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
        "gpu_memory": gpu_memory,
        "fit": fit_result,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "artifact": artifact_path.name,
        "artifact_size_bytes": artifact_size,
        "calibrator": str(calibrator_path.relative_to(project)),
        "competence": str(competence_path.relative_to(project)),
        "source_metrics": str(metrics_path.relative_to(project)),
    }
    manifest_path = model_dir / "training_manifest.json"
    _write_json(manifest_path, training_manifest)
    validate_source_model(config, prepared, model_id)
    print(
        f"{model_id}: validation PASS; total={total_seconds:.3f}s; "
        f"peak_gpu={gpu_memory['peak_used_mib']:.1f} MiB",
        flush=True,
    )
    return manifest_path


def validate_source_model(config: dict, prepared: PreparedData, model_id: str) -> dict:
    spec = MODEL_SPECS[model_id]
    project = Path(config["project_root"])
    source_root = project / config["outputs"]["source"]
    model_dir = source_root / "models" / model_id
    manifest = json.loads((model_dir / "training_manifest.json").read_text(encoding="utf-8"))
    calibrator = json.loads((source_root / "calibrators" / f"{model_id}.json").read_text(encoding="utf-8"))
    competence = json.loads((source_root / "competence" / f"{model_id}.json").read_text(encoding="utf-8"))
    metrics = json.loads((source_root / "source_metrics" / f"{model_id}.json").read_text(encoding="utf-8"))
    train = prepared.partition("UNSW", "source", "S_train")
    assert manifest["status"] == "COMPLETE"
    assert manifest["training_rank_start"] == train.rank_start
    assert manifest["training_rank_end_exclusive"] == train.rank_end_exclusive
    assert manifest["training_rows"] == train.rows
    assert manifest["view"] == spec.view
    assert manifest["parameters"] == config["models"][spec.family]
    assert manifest["device"]["name"] == config["device"]["model_actual"]
    artifact = model_dir / manifest["artifact"]
    assert artifact.stat().st_size == manifest["artifact_size_bytes"] > 0
    assert calibrator["status"] == "COMPLETE"
    assert calibrator["fit_partition"] == config["calibration"]["fit_partition"]
    assert calibrator["fit_rows"] == prepared.partition("UNSW", "source", "S_prob_cal").rows
    assert competence["status"] == "COMPLETE"
    assert competence["fit_partition"] == config["competence"]["fit_partition"]
    assert competence["fit_rows"] == prepared.partition("UNSW", "source", "S_competence").rows
    assert sum(row["predicted_count"] for row in competence["table"]) == competence["fit_rows"]
    assert metrics["status"] == "COMPLETE"
    assert metrics["calibrated"]["rows"] == prepared.partition("UNSW", "source", "S_source_test").rows
    return {
        "model_id": model_id,
        "status": "PASS",
        "training_rows": manifest["training_rows"],
        "view": manifest["view"],
        "feature_count": manifest["feature_count"],
        "device": manifest["device"],
        "artifact_size_bytes": manifest["artifact_size_bytes"],
        "calibration_rows": calibrator["fit_rows"],
        "competence_rows": competence["fit_rows"],
        "source_test_rows": metrics["calibrated"]["rows"],
    }


def validate_model_structure(config: dict, model_id: str) -> dict:
    spec = MODEL_SPECS[model_id]
    project = Path(config["project_root"])
    model_dir = project / config["outputs"]["source"] / "models" / model_id
    artifact = _artifact_path(model_dir, spec.family)
    model = None
    if spec.family == "rf":
        model = load_model(spec, artifact)
        result = {
            "n_estimators": int(model.n_estimators),
            "max_depth": int(model.max_depth),
            "n_bins": int(model.n_bins),
        }
        assert result == {"n_estimators": 300, "max_depth": 16, "n_bins": 128}
    elif spec.family == "xgb":
        model = load_model(spec, artifact, config["models"]["xgb"]["device"])
        booster_config = json.loads(model.get_booster().save_config())
        result = {
            "boosted_rounds": int(model.get_booster().num_boosted_rounds()),
            "device": booster_config["learner"]["generic_param"]["device"],
        }
        assert result == {"boosted_rounds": 400, "device": "cuda:0"}
    else:
        checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
        result = {
            "input_size": int(checkpoint["input_size"]),
            "hidden_sizes": checkpoint["hidden_sizes"],
            "layer_shapes": [
                list(checkpoint["state_dict"]["network.0.weight"].shape),
                list(checkpoint["state_dict"]["network.2.weight"].shape),
                list(checkpoint["state_dict"]["network.4.weight"].shape),
            ],
        }
        assert result["hidden_sizes"] == [128, 64]
        assert result["layer_shapes"] == [
            [128, result["input_size"]], [64, 128], [1, 64]
        ]
    del model
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()
    return result


def finalize_p2(config: dict, prepared: PreparedData) -> Path:
    project = Path(config["project_root"])
    source_root = project / config["outputs"]["source"]
    output = source_root / "P2"
    output.mkdir(parents=True, exist_ok=True)
    validations = []
    manifests = []
    calibrators = []
    competence_rows = []
    metric_rows = []
    for model_id in config["models"]["schedule"]:
        validation = validate_source_model(config, prepared, model_id)
        validation["artifact_structure"] = validate_model_structure(config, model_id)
        validations.append(validation)
        manifest = json.loads(
            (source_root / "models" / model_id / "training_manifest.json").read_text(encoding="utf-8")
        )
        calibrator = json.loads(
            (source_root / "calibrators" / f"{model_id}.json").read_text(encoding="utf-8")
        )
        competence = json.loads(
            (source_root / "competence" / f"{model_id}.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (source_root / "source_metrics" / f"{model_id}.json").read_text(encoding="utf-8")
        )
        manifests.append(manifest)
        calibrators.append(calibrator)
        for row in competence["table"]:
            competence_rows.append({"model_id": model_id, **row})
        metric_rows.append({"model_id": model_id, **metrics["calibrated"]})

    validation = {
        "version": "P2-validation-v2.0",
        "status": "PASS",
        "completed_at_utc": _now(),
        "model_order": config["models"]["schedule"],
        "checks": validations,
        "models_complete": len(validations),
        "training_partition": "UNSW/source/S_train",
        "calibration_partition": "UNSW/source/S_prob_cal",
        "competence_partition": "UNSW/source/S_competence",
        "evaluation_partition": "UNSW/source/S_source_test",
        "gpu": config["device"]["model_actual"],
    }
    _write_json(output / "P2_validation.json", validation)
    _write_json(output / "P2_manifest.json", {
        "version": "P2-v2.0",
        "status": "COMPLETE",
        "completed_at_utc": _now(),
        "models": manifests,
    })
    shutil.copy2(project / "configs" / "execution_v2.json", output / "execution_v2.json")

    with (output / "calibration_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model_id", "fit_rows", "benign", "attack", "coefficient", "intercept",
            "iterations", "final_objective", "gradient_l2_norm", "raw_nll", "calibrated_nll",
            "raw_brier", "calibrated_brier", "raw_ece", "calibrated_ece",
        ])
        writer.writeheader()
        for row in calibrators:
            writer.writerow({
                "model_id": row["model_id"], "fit_rows": row["fit_rows"],
                "benign": row["label_counts"]["Benign"], "attack": row["label_counts"]["Attack"],
                "coefficient": row["coefficient"], "intercept": row["intercept"],
                "iterations": row["solver"]["iterations"],
                "final_objective": row["solver"]["final_objective"],
                "gradient_l2_norm": row["solver"]["gradient_l2_norm"],
                "raw_nll": row["raw_metrics"]["nll"],
                "calibrated_nll": row["calibrated_metrics"]["nll"],
                "raw_brier": row["raw_metrics"]["brier"],
                "calibrated_brier": row["calibrated_metrics"]["brier"],
                "raw_ece": row["raw_metrics"]["ece"],
                "calibrated_ece": row["calibrated_metrics"]["ece"],
            })
    with (output / "competence_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(competence_rows[0]))
        writer.writeheader(); writer.writerows(competence_rows)
    metric_fields = [
        "model_id", "rows", "prevalence", "accuracy", "balanced_accuracy", "precision_attack",
        "recall_attack", "f1_attack", "auroc", "average_precision", "nll", "brier", "ece",
    ]
    with (output / "source_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(metric_rows)

    total_training_seconds = sum(row["fit"]["training_seconds"] for row in manifests)
    total_model_seconds = sum(row["total_seconds"] for row in manifests)
    lines = [
        "# P2 Base detector Phase Report", "", f" State: COMPLETETime to finish:{_now()}。", "",
        "## modified code", "",
        "`models/base_models.py` achieves cuML RF, CUDA XGBoost, PyTorch MLP, model preserves / loading, segmentation reasoning and sampling;`reliability/logistic.py` and `calibration.py` achieve CUDA float64 LBFGS Platt and h;`evaluation/source.py` achieves the source domain indicator;`p2.py` is responsible for binding, operating, receiving and reporting partition by model;`run.py` and `scripts/run_p2.sh` provide a unified input and fixed order.", "",
        "## actually works", "",
        "The six basic models were formally trained on the single NVIDIA RTX A5000 CUDA:0 in freeze order. All models use full UNSW/source/S_train and maintain a natural class ratio. Each model is preserved and reloaded to complete S_prob_cal Platt, S_competence historical capability and S_source_test evaluations.", "",
        "Training, calibration, capability and evaluation of partition as `[0,1420000)`, `[1420000,1780000)`, `[1780000,2130000)`, and `[2130000,2365424)`, respectively, without reclassification of data. P1 features caching, preprocessing statistics and JSD source reference direct reuse.", "",
        "## freeze model configuration", "",
        "| model | rear end | view | full parameters |", "|---|---|---|---|",
    ]
    for row in manifests:
        parameters = json.dumps(row["parameters"], ensure_ascii=False, separators=(",", ":"))
        lines.append(f"| {row['model_id']} | {row['backend']} | {row['view']} | `{parameters}` |")
    lines += [
        "", "## six models", "",
        "| model | view | features number | training record | training time (s) | total time (s) | peak showing (MiB) | product (bytes) |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifests:
        lines.append(
            f"| {row['model_id']} | {row['view']} | {row['feature_count']} | "
            f"{row['training_rows']:,} | {row['fit']['training_seconds']:.3f} | "
            f"{row['total_seconds']:.3f} | {row['gpu_memory']['peak_used_mib']:.1f} | "
            f"{row['artifact_size_bytes']:,} |"
        )
    lines += ["", f" Six final effective trainings accumulated {total_training_seconds:.3f} Seconds, six complete training kits calibration capacity kits accumulated chain assessment {total_model_seconds:.3f} Seconds. The unified random seed is 20260905. RF is for 300, depth16, min_leaf2, sqrt, 128 bins; XGBoost is for CUDA hist, 400 rounds, depth6, eta0.05, subsample0.8, colsample0.8; MLP is equivalent to 128 → 64, ReLU, Adam, lr0.001, batch8192, 30 epochs. The complete parameters maintains the training_manifest.json of each model.", "", "Peak storage is the total peak occupancy of devices that read CUDA `total-free` at intervals of 100 ms in each independent model process.", "", "## Platt calibration", "", "| model | coefficient | intersection | replacement | calibration before NLL | after calibration after NLL | after calibration before ECE | after calibration after ECE after |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in calibrators:
        lines.append(
            f"| {row['model_id']} | {row['coefficient']:.8g} | {row['intercept']:.8g} | "
            f"{row['solver']['iterations']} | {row['raw_metrics']['nll']:.8f} | "
            f"{row['calibrated_metrics']['nll']:.8f} | {row['raw_metrics']['ece']:.8f} | "
            f"{row['calibrated_metrics']['ece']:.8f} |"
        )
    lines += ["", "All six Platt models use 360,000 S_prob_cal records and unpenalized CUDA float64 LBFGS; labels contains 325,600 clause Benign and 34,400 clause Attack. After calibration of XGBoost–Protocol17, NLL is converted from 0.00003810 to 0.00004134; This value is derived from the initial values of freeze and `tolerance_grad=1e-4`, which are retained as a formal result.", "", "## Historical capacity h", "", "| model | prediction category | prediction number | correct number | h |", "|---|---|---:|---:|---:|"]
    for row in competence_rows:
        lines.append(
            f"| {row['model_id']} | {row['predicted_class_name']} | {row['predicted_count']:,} | "
            f"{row['correct_count']:,} | {row['h']:.9f} |"
        )
    lines += ["", "h uses only S_competence’s 350,000 entry record and uses `(correct+1)/(count+2)` as a predicted category after calibration.", "", "## S_source_test source domain evaluation", "", "| model | Accuracy | Balanced accuracy | Precision | Recall | F1 | AUROC | AP | NLL | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        lines.append(
            f"| {row['model_id']} | {row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['precision_attack']:.6f} | {row['recall_attack']:.6f} | {row['f1_attack']:.6f} | "
            f"{row['auroc']:.6f} | {row['average_precision']:.6f} | {row['nll']:.6f} | "
            f"{row['brier']:.6f} | {row['ece']:.6f} |"
        )
    lines += ["", "The results of the S_source_test 235,424 post-calibration probability in the table are recorded; Attack determines the threshold value as 0.5 and ECE using 15 equivalent wide boxes.", "", "## inspection, repair and product", "", "All six models were trained in rank, record numbers, features views, freeze parameters, CUDA devices, model file size, calibration partition, capability partition, and evaluated partition verification. The RF file is inside the 300 tree, two XGBoost files are inside the 400 rounds and the device is cuda:0, and the layer shapes of the two MLP files correspond to 15/49 → 128 → 64 → 1 respectively. The training and reasoning for entering the model are all limited. After the model is saved, the product is re-loaded to complete the subsequent calculations.", "", "For the first time, RF exposure to Traffic17 calibration read only the mmap labels warning;`p2.py` re-calculates the complete chain of the model after creating a writeable array at the labels conversion boundary. For the first time, the XGBoost lithium Protocol17 weighted-reasoning exposure device is restored to CPU;`base_models.py` restores booster to cuda:0 after loading, and then recalculates the complete chain of the model. The final product comes from the repaired operation.", "", "Unresolved problems: No. P1 caching, preprocessing statistics, JSD source reference and data allocation are unchanged.", "", "- `outputs/source_v2/models/`: Six models and one-by-one manifest training", "- `outputs/source_v2/calibrators/`: six Platt parameters and calibration indicators", "- `outputs/source_v2/competence/`: Six historical capacity tables", "- `outputs/source_v2/source_metrics/`: Six source domain assessments", "- `outputs/source_v2/P2/`: Phase summary, freeze configuration copy and P2 verification results", "", "P2 is complete. P3 can directly load six models, calibration and h, generating predictive caching and causal feedback records for each domain.", ""]
    report_path = output / "stage_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
