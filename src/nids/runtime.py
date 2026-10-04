from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import Version


TRACKED_PACKAGES = [
    "numpy", "pyarrow", "cupy-cuda12x", "torch", "xgboost", "cuml-cu12",
    "cudf-cu12", "scipy", "scikit-learn", "pandas", "matplotlib",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register(config_path: Path, config: dict) -> dict:
    import cupy as cp
    import cuml
    import pyarrow
    import torch
    import xgboost

    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    ).stdout.strip().splitlines()
    smi_text = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True).stdout
    driver_cuda = re.search(r"CUDA Version:\s*([0-9.]+)", smi_text).group(1)
    assert len(query) == 1
    index, name, memory, driver = [value.strip() for value in query[0].split(",")]
    properties = torch.cuda.get_device_properties(0)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert xgboost.build_info()["USE_CUDA"] is True
    assert Version(cuml.__version__) == Version(importlib.metadata.version("cuml-cu12"))
    assert name == "NVIDIA RTX A5000" and int(memory) == 24564
    cupy_properties = cp.cuda.runtime.getDeviceProperties(0)
    assert cupy_properties["name"].decode() == name
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    runtime = {
        "version": "runtime-v2.0", "registered_at_utc": utc_now(),
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "gpu": {"index": int(index), "name": name, "memory_mib": int(memory), "count": len(query),
                "driver": driver, "torch_total_memory_bytes": properties.total_memory},
        "cuda": {"driver_capability": driver_cuda, "torch_runtime": torch.version.cuda,
                 "cupy_runtime": cp.cuda.runtime.runtimeGetVersion(),
                 "cupy_driver": cp.cuda.runtime.driverGetVersion()},
        "torch_cuda_available": torch.cuda.is_available(),
        "xgboost_version": xgboost.__version__, "pyarrow_version": pyarrow.__version__,
        "xgboost_build": xgboost.build_info(), "cuml_version": cuml.__version__,
        "gpu_execution": "single_GPU_sequential", "automatic_cpu_fallback": False,
    }
    output = Path(config["project_root"]) / config["outputs"]["round1"] / "P0"
    snapshot = Path(config["project_root"]) / config["outputs"]["round1"] / "config_snapshot"
    output.mkdir(parents=True, exist_ok=True); snapshot.mkdir(parents=True, exist_ok=True)
    (output / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config["status"] = "REGISTERED_AND_FROZEN_FOR_EXECUTION_V2"
    config["frozen_at_utc"] = runtime["registered_at_utc"]
    config["device"]["runtime_versions"] = runtime["cuda"]
    config["device"]["model_actual"] = name
    config["device"]["memory_mib_actual"] = int(memory)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (snapshot / "execution_v2.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(Path(config["project_root"]) / config["feature_views"], snapshot / "feature_views.csv")
    shutil.copyfile(Path(config["project_root"]) / "configs" / "taxonomy.json", snapshot / "taxonomy.json")
    return runtime
