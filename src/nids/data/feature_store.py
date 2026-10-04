from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .partitions import PreparedData


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _encode_attack(column: pa.Array, codebook: dict[str, int]) -> np.ndarray:
    encoded = pc.dictionary_encode(column)
    local_labels = encoded.dictionary.to_pylist()
    local_to_global = np.empty(len(local_labels), dtype=np.int16)
    for index, label in enumerate(local_labels):
        if label not in codebook:
            codebook[label] = len(codebook)
        local_to_global[index] = codebook[label]
    return local_to_global[encoded.indices.to_numpy(zero_copy_only=False)]


def build_feature_store(config: dict, prepared: PreparedData) -> dict:
    started = time.perf_counter()
    project = Path(config["project_root"])
    data_root = Path(config["data_root"])
    output = project / config["outputs"]["feature_store"]
    output.mkdir(parents=True, exist_ok=True)
    temp = output / "temporary_original_order"
    temp.mkdir(exist_ok=True)
    with (project / config["feature_views"]).open(encoding="utf-8-sig") as handle:
        import csv
        features = [row["feature"] for row in csv.DictReader(handle)]
    categorical = config["shift"]["categorical_features"]
    categorical_indices = [features.index(name) for name in categorical]
    audit = json.loads((Path(config["prepared_run"]) / "data_audit.json").read_text(encoding="utf-8"))
    row_counts = {row["domain"]: int(row["rows"]) for row in audit}
    domain_results = []
    for domain, filename in config["datasets"].items():
        print(f"{domain}: converting CSV to original-order feature cache", flush=True)
        domain_start = time.perf_counter()
        rows = row_counts[domain]
        raw_x_path = temp / f"{domain}_X_original.npy"
        raw_y_path = temp / f"{domain}_label_original.npy"
        raw_a_path = temp / f"{domain}_attack_original.npy"
        raw_c_path = temp / f"{domain}_categorical_original.npy"
        raw_x = np.lib.format.open_memmap(raw_x_path, mode="w+", dtype=np.float32, shape=(rows, 49))
        raw_y = np.lib.format.open_memmap(raw_y_path, mode="w+", dtype=np.uint8, shape=(rows,))
        raw_a = np.lib.format.open_memmap(raw_a_path, mode="w+", dtype=np.int16, shape=(rows,))
        raw_c = np.lib.format.open_memmap(raw_c_path, mode="w+", dtype=np.float64, shape=(rows, len(categorical)))
        null_counts = np.zeros(49, dtype=np.int64)
        nan_value_counts = np.zeros(49, dtype=np.int64)
        positive_infinity = np.zeros(49, dtype=np.int64)
        negative_infinity = np.zeros(49, dtype=np.int64)
        codebook: dict[str, int] = {}
        types = {name: pa.float64() for name in features}
        types.update({"Label": pa.int8(), "Attack": pa.string()})
        reader = pacsv.open_csv(
            data_root / filename,
            read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
            convert_options=pacsv.ConvertOptions(
                include_columns=features + ["Label", "Attack"],
                column_types=types,
                strings_can_be_null=False,
                null_values=[""],
            ),
        )
        cursor = 0
        batch_rows = int(config["batches"]["csv_rows"])
        for record_batch in reader:
            for offset in range(0, record_batch.num_rows, batch_rows):
                batch = record_batch.slice(offset, min(batch_rows, record_batch.num_rows - offset))
                count = batch.num_rows
                matrix = np.empty((count, 49), dtype=np.float64)
                for column_index, feature in enumerate(features):
                    column = batch.column(feature)
                    null_counts[column_index] += column.null_count
                    values = np.array(
                        pc.fill_null(column, np.nan).to_numpy(zero_copy_only=False), copy=True
                    )
                    nan_value_counts[column_index] += np.count_nonzero(np.isnan(values)) - column.null_count
                    positive_infinity[column_index] += np.count_nonzero(np.isposinf(values))
                    negative_infinity[column_index] += np.count_nonzero(np.isneginf(values))
                    values[~np.isfinite(values)] = np.nan
                    matrix[:, column_index] = values
                stop = cursor + count
                raw_x[cursor:stop] = matrix
                raw_c[cursor:stop] = matrix[:, categorical_indices]
                raw_y[cursor:stop] = batch.column("Label").to_numpy(zero_copy_only=False)
                raw_a[cursor:stop] = _encode_attack(batch.column("Attack"), codebook)
                cursor = stop
        assert cursor == rows
        print(f"{domain}: decoded {rows:,} rows; applying prepared rank order", flush=True)
        raw_x.flush(); raw_y.flush(); raw_a.flush(); raw_c.flush()
        order = prepared.sorted_rows(domain)
        sorted_x = np.lib.format.open_memmap(output / f"{domain}_X.npy", mode="w+", dtype=np.float32, shape=(rows, 49))
        sorted_y = np.lib.format.open_memmap(output / f"{domain}_label.npy", mode="w+", dtype=np.uint8, shape=(rows,))
        sorted_a = np.lib.format.open_memmap(output / f"{domain}_attack_code.npy", mode="w+", dtype=np.int16, shape=(rows,))
        sorted_c = np.lib.format.open_memmap(output / f"{domain}_categorical.npy", mode="w+", dtype=np.float64, shape=(rows, len(categorical)))
        gather_rows = int(config["batches"]["csv_rows"])
        for start in range(0, rows, gather_rows):
            stop = min(start + gather_rows, rows)
            selected = order[start:stop]
            sorted_x[start:stop] = raw_x[selected]
            sorted_y[start:stop] = raw_y[selected]
            sorted_a[start:stop] = raw_a[selected]
            sorted_c[start:stop] = raw_c[selected]
        sorted_x.flush(); sorted_y.flush(); sorted_a.flush(); sorted_c.flush()
        assert np.array_equal(sorted_y, raw_y[order])
        for split in (row for row in prepared.splits if row["domain"] == domain):
            start, stop = int(split["rank_start"]), int(split["rank_end_exclusive"])
            assert int(sorted_y[start:stop].sum()) == int(split["attack"])
        attack_labels = [None] * len(codebook)
        for label, code in codebook.items():
            attack_labels[code] = label
        result = {
            "domain": domain,
            "source_csv": str(data_root / filename),
            "rows": rows,
            "features": 49,
            "categorical_auxiliary_columns": categorical,
            "empty_counts": dict(zip(features, null_counts.tolist())),
            "nan_value_counts": dict(zip(features, nan_value_counts.tolist())),
            "positive_infinity_counts": dict(zip(features, positive_infinity.tolist())),
            "negative_infinity_counts": dict(zip(features, negative_infinity.tolist())),
            "missing_after_normalization": dict(zip(features, (null_counts + nan_value_counts + positive_infinity + negative_infinity).tolist())),
            "attack_codebook": {str(index): label for index, label in enumerate(attack_labels)},
            "elapsed_seconds": time.perf_counter() - domain_start,
            "files": {
                "features": f"{domain}_X.npy", "binary_label": f"{domain}_label.npy",
                "attack_code": f"{domain}_attack_code.npy", "categorical": f"{domain}_categorical.npy",
            },
        }
        result["file_sizes_bytes"] = {
            role: (output / name).stat().st_size for role, name in result["files"].items()
        }
        _save_json(output / f"{domain}_manifest.json", result)
        domain_results.append(result)
        print(f"{domain}: feature cache complete in {result['elapsed_seconds']:.3f} seconds", flush=True)
        del raw_x, raw_y, raw_a, raw_c, sorted_x, sorted_y, sorted_a, sorted_c
        raw_x_path.unlink(); raw_y_path.unlink(); raw_a_path.unlink(); raw_c_path.unlink()
    temp.rmdir()
    manifest = {
        "version": "feature-store-v2.0", "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "row_order": "availability_sorted_rank",
        "rank_source": str(prepared.root), "dtype": "float32", "nonfinite_policy": "positive_and_negative_infinity_to_NaN",
        "records_removed": 0, "feature_order": features, "domains": domain_results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _save_json(output / "feature_store_manifest.json", manifest)
    return manifest
