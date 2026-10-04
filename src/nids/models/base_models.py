from __future__ import annotations

import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    view: str
    standardized: bool


MODEL_SPECS = {
    "rf_Traffic": ModelSpec("rf_Traffic", "rf", "Traffic", False),
    "rf_All49": ModelSpec("rf_All49", "rf", "All49", False),
    "xgb_Protocol": ModelSpec("xgb_Protocol", "xgb", "Protocol", False),
    "xgb_All49": ModelSpec("xgb_All49", "xgb", "All49", False),
    "mlp_Behavior": ModelSpec("mlp_Behavior", "mlp", "Behavior", True),
    "mlp_All49": ModelSpec("mlp_All49", "mlp", "All49", True),
}


class GpuMemoryMonitor:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self.started = time.perf_counter()
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        with cp.cuda.Device(self.device_index):
            free, total = cp.cuda.runtime.memGetInfo()
        self.baseline_bytes = total - free
        self.peak_bytes = self.baseline_bytes
        self._thread.start()

    def _run(self) -> None:
        with cp.cuda.Device(self.device_index):
            next_message = time.perf_counter() + 30
            while not self._stop.wait(0.1):
                free, total = cp.cuda.runtime.memGetInfo()
                self.peak_bytes = max(self.peak_bytes, total - free)
                now = time.perf_counter()
                if now >= next_message:
                    print(
                        f"GPU progress: elapsed={now - self.started:.1f}s, "
                        f"used={(total - free) / 2**20:.1f} MiB, "
                        f"peak={self.peak_bytes / 2**20:.1f} MiB",
                        flush=True,
                    )
                    next_message = now + 30

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()
        with cp.cuda.Device(self.device_index):
            free, total = cp.cuda.runtime.memGetInfo()
        self.peak_bytes = max(self.peak_bytes, total - free)
        return {
            "baseline_used_mib": self.baseline_bytes / 2**20,
            "peak_used_mib": self.peak_bytes / 2**20,
            "peak_increment_mib": (self.peak_bytes - self.baseline_bytes) / 2**20,
            "measurement": "CUDA_memGetInfo_total_minus_free_sampled_at_100ms",
        }


class BinaryMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def seed_gpu(seed: int) -> None:
    np.random.seed(seed)
    cp.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    config: dict,
    spec: ModelSpec,
    train_x: cp.ndarray,
    train_y: cp.ndarray,
    artifact_path: Path,
) -> dict:
    seed = int(config["seed"])
    seed_gpu(seed)
    started = time.perf_counter()
    if spec.family == "rf":
        from cuml.ensemble import RandomForestClassifier

        params = config["models"]["rf"]
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params["min_samples_split"]),
            max_features=params["max_features"],
            n_bins=int(params["n_bins"]),
            bootstrap=bool(params["bootstrap"]),
            max_samples=float(params["max_samples"]),
            n_streams=int(params["n_streams"]),
            split_criterion=params["split_criterion"],
            random_state=seed,
            output_type="cupy",
        )
        model.fit(train_x, train_y)
        cp.cuda.Stream.null.synchronize()
        with artifact_path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        epochs = None
        final_loss = None
    elif spec.family == "xgb":
        from xgboost import XGBClassifier

        params = config["models"]["xgb"]
        model = XGBClassifier(
            device=params["device"],
            tree_method=params["tree_method"],
            objective=params["objective"],
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=float(params["min_child_weight"]),
            max_bin=int(params["max_bin"]),
            random_state=seed,
        )
        model.fit(train_x, train_y, verbose=False)
        cp.cuda.Stream.null.synchronize()
        model.save_model(artifact_path)
        epochs = None
        final_loss = None
    else:
        params = config["models"]["mlp"]
        hidden_sizes = [int(value) for value in params["hidden_sizes"]]
        model = BinaryMLP(train_x.shape[1], hidden_sizes).cuda()
        torch_x = torch.from_dlpack(train_x)
        torch_y = torch.from_dlpack(train_y).to(dtype=torch.float32)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        loss_function = nn.BCEWithLogitsLoss()
        batch_rows = int(config["batches"]["mlp_training_rows"])
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        epochs = int(params["epochs"])
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            order = torch.randperm(len(torch_x), device="cuda", generator=generator)
            loss_sum = 0.0
            model.train()
            for start in range(0, len(order), batch_rows):
                rows = order[start:start + batch_rows]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(torch_x[rows]), torch_y[rows])
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().item()) * len(rows)
            final_loss = loss_sum / len(torch_x)
            print(
                f"{spec.model_id}: epoch {epoch + 1:02d}/{epochs}, "
                f"loss={final_loss:.9f}, elapsed={time.perf_counter() - epoch_started:.3f}s",
                flush=True,
            )
        torch.cuda.synchronize()
        checkpoint = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_size": int(train_x.shape[1]),
            "hidden_sizes": hidden_sizes,
        }
        torch.save(checkpoint, artifact_path)
    return {
        "training_seconds": time.perf_counter() - started,
        "epochs": epochs,
        "final_training_bce": final_loss,
    }


def load_model(spec: ModelSpec, artifact_path: Path, device: str = "cuda:0"):
    if spec.family == "rf":
        with artifact_path.open("rb") as handle:
            return pickle.load(handle)
    if spec.family == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(device=device)
        model.load_model(artifact_path)
        model.set_params(device=device)
        model.get_booster().set_param({"device": device})
        return model
    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = BinaryMLP(checkpoint["input_size"], checkpoint["hidden_sizes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    return model


def predict_attack_probability(model, spec: ModelSpec, values: cp.ndarray) -> np.ndarray:
    if spec.family in {"rf", "xgb"}:
        probability = cp.asarray(model.predict_proba(values))[:, 1]
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(probability).astype(np.float64, copy=False)
    with torch.inference_mode():
        logits = model(torch.from_dlpack(values))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64, copy=False)
