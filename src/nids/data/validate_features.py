from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def validate_feature_stage(config: dict) -> dict:
    project = Path(config["project_root"])
    prepared = Path(config["prepared_run"])
    feature_root = project / config["outputs"]["feature_store"]
    source_root = project / config["outputs"]["source"] / "preprocessing"
    store = json.loads((feature_root / "feature_store_manifest.json").read_text(encoding="utf-8"))
    domain_checks = []
    for domain in store["domains"]:
        name = domain["domain"]
        values = np.load(feature_root / f"{name}_X.npy", mmap_mode="r")
        nan_count = 0
        infinity_count = 0
        for start in range(0, len(values), 262144):
            batch = values[start:start + 262144]
            nan_count += int(np.count_nonzero(np.isnan(batch)))
            infinity_count += int(np.count_nonzero(np.isinf(batch)))
        expected_nan = sum(domain["missing_after_normalization"].values())
        assert nan_count == expected_nan and infinity_count == 0
        source_audit = json.loads((prepared / f"{Path(domain['source_csv']).stem}_scan.json").read_text(encoding="utf-8"))
        source_stat = Path(domain["source_csv"]).stat()
        assert source_stat.st_size == source_audit["bytes"]
        assert source_stat.st_mtime_ns == source_audit["mtime_ns"]
        domain_checks.append({
            "domain": name, "rows": len(values), "cache_nan_values": nan_count,
            "cache_infinity_values": infinity_count, "source_bytes": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns, "source_unchanged_from_data_preparation": True,
        })
    stats = np.load(source_root / "preprocessing_stats.npz")
    assert all(np.isfinite(stats[name]).all() for name in stats.files)
    reference_counts = {}
    for filename, expected in [("jsd_numeric_reference.npz", 37), ("jsd_categorical_reference.npz", 12)]:
        reference = np.load(source_root / filename)
        probability_names = [name for name in reference.files if name.endswith("__probabilities")]
        assert len(probability_names) == expected
        assert all(np.isclose(reference[name].sum(), 1.0) for name in probability_names)
        assert all(np.all(reference[name] >= 0) for name in probability_names)
        reference_counts[filename] = expected
    preprocessing = json.loads((source_root / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    assert all(
        row["tree_finite_values"] == row["expected_finite_values"]
        and row["mlp_finite_values"] == row["expected_finite_values"]
        for row in preprocessing["transformed_full_store_finite_checks"]
    )
    result = {
        "version": "P1-validation-v2.0", "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "domains": domain_checks, "preprocessing_arrays_finite": True,
        "jsd_reference_probability_checks": reference_counts,
        "full_transforms_finite": True, "records_removed": 0,
    }
    (feature_root / "P1_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = feature_root / "stage_report.md"
    marker = "\n## official product verification\n"
    report = report_path.read_text(encoding="utf-8").split(marker)[0]
    report += marker + "\n"
    report += f" Check time:{result['validated_at_utc']}。\n\n"
    report += "The four-dimensional cache is checked by batch: Inf is 0 and NaN is completely consistent with the conversion of manifest."
    report += "The original CSV file size and time of modification are consistent with the data preparation stage."
    report += "preprocessing statistics are all finite, with both the 37 numerical value and the 12 classification JSD probability vector being non-negative and equal to 1.\n"
    report_path.write_text(report, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = validate_feature_stage(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

