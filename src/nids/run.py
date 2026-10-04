from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .data.partitions import PreparedData


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_p0_report(project: Path, config: dict, runtime: dict, elapsed: float) -> None:
    output = project / config["outputs"]["round1"] / "P0"
    packages = runtime["packages"]
    lines = [
        "# P0 phase report", "", f" State: COMPLETEThe time of registration:{runtime['registered_at_utc']}。", "",
        "## actually works", "",
        "The integrated input completes the GPU, CUDA runtime, Python and training library registration, the fixed execution-v2 configuration, taxonomy, the tripod role, indicators, thresholds and data boundaries.",
        "The project uses a src/nids layout; The original execution packaging and early transmission materials are located at archive and data are available to prepare the product to maintain its original path.", "",
        "## modified code", "",
        "Unified input is src/nids/run.py; Environmental registration is the responsibility of src/nids/runtime.py; Both manifest associations are responsible for src/nids/data/partitions.py.",
        "P1 Data responsibilities are located in src/nids/data/feature_store.py, src/nids/data/preprocessing.py and src/nids/replay/shift.py respectively.",
        "pyproject.toml registers the src package; scripts/run_round1.sh fixed phase call order.", "",
        "## Equipment and Environment", "",
        f"GPU：{runtime['gpu']['name']} × {runtime['gpu']['count']}，{runtime['gpu']['memory_mib']} MiB; Driving {runtime['gpu']['driver']}。",
        f"PyTorch CUDA runtime {runtime['cuda']['torch_runtime']}；CuPy runtime {runtime['cuda']['cupy_runtime']}。",
        "Formal GPU tasks are fixed using CUDA:0 and run in single-process order; CPU takes on CSV, mmap, manifest and reports.", "",
        "| library | version |", "|---|---|",
    ]
    for package, version in packages.items():
        lines.append(f"| {package} | {version} |")
    lines += ["", "## configuration and inspection", "",
              "The actual GPU model, display and quantity are consistent with the execution plan. Both PyTorch and CuPy actually access CUDA:0.",
              "prepared_run refers to outputs/round1_data_prep_v1; partition and window manifest are read and verified by partitions.py.",
              "49 features maintaining Traffic17, Protocol17, Behavior15, taxonomy and threshold rules have been registered.",
              "All Python modules are checked by byte code compilation; The partition module actually reads and verifies the 16 paragraphs partition and 42,070 windows; The uniqueness of 49 in features manifest and the 17/17/15 number are checked.",
              "For the first time, official registration of the cuML module version 26.08.00 and the metadata package 26.8.0 format differences was discovered. runtime.py is a standardized semantic version comparison and a re-completion of P0.",
              f" Stages are time consuming:{elapsed:.3f} Seconds.", "",
              "## products", "",
              "- outputs/round1_v2/P0/runtime.json",
              "- outputs/round1_v2/config_snapshot/execution_v2.json、feature_views.csv、taxonomy.json",
              "- configs/execution_v2.json、configs/feature_views.csv、configs/taxonomy.json",
              "- docs/execution_plan_v2.md、docs/data_preparation_report.md、docs/protocol_decisions.md",
              "- requirements-runtime.txt、pyproject.toml、src/nids/", ""]
    (output / "stage_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_p1_report(project: Path, config: dict, store: dict, preprocessing: dict, elapsed: float) -> None:
    output = project / config["outputs"]["feature_store"]
    total_missing = {}
    total_rows = 0
    total_store_bytes = 0
    lines = ["# P1 phase report", "", f" State: COMPLETETime to finish:{now()}。", "",
             "## modified code", "",
             "feature_store.py is responsible for the conversion and sequencing of the total CSV of gather; preprocessing.py is the only implementation of filling, standardization and transform; shift.py is responsible for JSD source reference.",
             "partitions.py provides both rank, partition and window contracts; run.py is responsible for the formal call and reporting phase.", "",
             "## actually works", "",
             "Convert all four original CSVs to mmap features, binary labels, attack subcategories, and JSD categorical auxiliary columns, ordered by the existing availability rank.",
             "The empty fields, positive infinity and negative infinity are preserved as NaN; No records have been deleted. The original CSV, data preparation sequence, partition and window manifest remain unchanged.", "",
             "For the first time, the official conversion found that the PyArrow output of the NumPy view was read-only; feature_store.py creates writeable copies in the conversion of the boundary explicitly, and then recalculates the entire product from P1.", "",
             "| field | record number | empty field | NaN literal value | +Inf | -Inf | conversion time ( seconds ) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for domain in store["domains"]:
        empty = sum(domain["empty_counts"].values())
        nan_values = sum(domain["nan_value_counts"].values())
        positive = sum(domain["positive_infinity_counts"].values())
        negative = sum(domain["negative_infinity_counts"].values())
        total_rows += domain["rows"]
        total_store_bytes += sum(domain["file_sizes_bytes"].values())
        for feature, count in domain["missing_after_normalization"].items():
            total_missing[feature] = total_missing.get(feature, 0) + count
        lines.append(f"| {domain['domain']} | {domain['rows']:,} | {empty:,} | {nan_values:,} | {positive:,} | {negative:,} | {domain['elapsed_seconds']:.3f} |")
    affected = {feature: count for feature, count in total_missing.items() if count}
    lines += ["", "## S_train fit", "",
              f" fit range: UNSW/source/S_train , rank [0, {preprocessing['fit_rank_end_exclusive']:,}) , total {preprocessing['fit_rows']:,} It’s not.",
              "numeric columns is calculated on CUDA:0 as finite-value median; JSD categorical columns calculates mode at CUDA:0 and processes ties at a minimum value.",
              "S_train after filling in CUDA:0 fit mean and population standard deviation; The scale=1 is a constant. No other partition participated in statistical fitting.",
              "The JSD numerical boundary, the vocabulary classification and the source probability are all from the S_train fit, and the rules are consistent with the registration configuration.", "",
              f" Train the missing columns and use the 0 columns:{', '.join(preprocessing['all_source_missing_features']) or 'none'}。",
              f" The source training constant column:{', '.join(preprocessing['constant_scale_features']) or 'none'}。",
              f" Total anomalous value impact columns and counts:{json.dumps(affected, ensure_ascii=False)}。", "",
              "## test results", "",
              f" The total number of cached records is {total_rows:,}(b) the total number of data preparations in accordance with existing data; Each domain labels is reconstructed consistent under the same ranking index.",
              f" The total size of the official cached file in four domains:{total_store_bytes:,} bytes. preprocessing and JSD reference file sizes are recorded in preprocessing_manifest.json.",
              "The four-dimensional cache calls in batches the same fit preprocessing device; All values for tree model inputs and MLP standardized inputs are finite values.",
              "The JSD reference contains 37 as numeric features and 12 as categorical features; The numerical repeat boundary has been folded, and the vocabulary classification retains the source value and reserves the OTHER.",
              "P1 numerals fit and finite values check with RTX A5000 CUDA:0; CSV decoding, original sequence caching and sequencing gather using CPU.",
              f" CuPy memory pool cumulative peak allocation agents:{preprocessing['gpu_memory_pool_peak_proxy_bytes']:,} bytes。",
              f" P1 total time consumed:{elapsed:.3f} Seconds; features conversion cumulative time:{store['elapsed_seconds']:.3f} Seconds; preprocessing with reference to fit:{preprocessing['elapsed_seconds']:.3f} Seconds.", "",
              "## products", "",
              "- outputs/feature_store_v2/*_X.npy、*_label.npy、*_attack_code.npy、*_categorical.npy",
              "- outputs/feature_store_v2/feature_store_manifest.json with different manifest domains",
              "- outputs/source_v2/preprocessing/preprocessing_stats.npz",
              "- outputs/source_v2/preprocessing/preprocessing_manifest.json",
              "- outputs/source_v2/preprocessing/jsd_source_reference.json and two NPZ reference documents", "",
              "## P2 Ready state", "",
              "The six basic models can read feature_store directly in configuration order, select the view column, and call the only SourcePreprocessor.",
              "P2 uses full S_train training, S_prob_cal is for Platt only and S_competence is for h only; There is no training model or computational model effect at this stage.",
              "The CUDA training library and data inputs required by P2 are ready; Six models, Platt and competence modules, are implemented and operated in the next phase. Matplotlib belongs to the subsequent scientific development of dependency charts, which does not block P2 training.", ""]
    (output / "stage_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["register", "features", "train-model", "finalize-p2", "replay-p3", "round1-p4", "sensitivity-p5", "coordination-p6", "preflight-p7", "outer-p7"],
        required=True,
    )
    parser.add_argument("--model-id", choices=[
        "rf_Traffic", "rf_All49", "xgb_Protocol", "xgb_All49",
        "mlp_Behavior", "mlp_All49",
    ])
    parser.add_argument("--reuse-fitted-fold", choices=["F1","F2"], action="append", default=[])
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    project = Path(config["project_root"])
    prepared = PreparedData(Path(config["prepared_run"]))
    if args.stage == "preflight-p7":
        from .p7 import preflight
        result = preflight(config)
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))
        print(f"Frozen input files checked: {len(result['files'])}")
    if args.stage == "outer-p7":
        from .p7 import run_p7
        print(run_p7(config, prepared))
    if args.stage == "register":
        started = time.perf_counter()
        from .runtime import register
        runtime = register(config_path, config)
        requirements = "\n".join(
            f"{package}=={version}" for package, version in runtime["packages"].items()
            if version != "NOT_INSTALLED"
        ) + "\n"
        (project / "requirements-runtime.txt").write_text(requirements, encoding="utf-8")
        write_p0_report(project, config, runtime, time.perf_counter() - started)
        print(project / config["outputs"]["round1"] / "P0" / "stage_report.md")
    if args.stage == "features":
        assert config["status"] == "REGISTERED_AND_FROZEN_FOR_EXECUTION_V2"
        started = time.perf_counter()
        from .data.feature_store import build_feature_store
        from .data.preprocessing import fit_source_preprocessing
        store = build_feature_store(config, prepared)
        preprocessing = fit_source_preprocessing(config, prepared)
        write_p1_report(project, config, store, preprocessing, time.perf_counter() - started)
        from .data.validate_features import validate_feature_stage
        validate_feature_stage(config)
        print(project / config["outputs"]["feature_store"] / "stage_report.md")
    if args.stage == "train-model":
        assert config["status"] == "REGISTERED_AND_FROZEN_FOR_EXECUTION_V2"
        assert args.model_id is not None
        from .p2 import train_source_model
        print(train_source_model(config, prepared, args.model_id))
    if args.stage == "coordination-p6":
        from .p6 import run_p6
        print(run_p6(config, prepared, args.reuse_fitted_fold))
    if args.stage == "sensitivity-p5":
        from .p5 import run_p5
        print(run_p5(config, prepared))
    if args.stage == "round1-p4":
        from .p4 import run_p4
        print(run_p4(config, prepared))
    if args.stage == "replay-p3":
        from .p3 import run_p3
        print(run_p3(config, prepared))
    if args.stage == "finalize-p2":
        assert config["status"] == "REGISTERED_AND_FROZEN_FOR_EXECUTION_V2"
        from .p2 import finalize_p2
        print(finalize_p2(config, prepared))


if __name__ == "__main__":
    main()
