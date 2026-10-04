from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def fit_unpenalized_logistic(
    x: np.ndarray,
    y: np.ndarray,
    solver_config: dict,
    chunk_rows: int,
) -> dict:
    device = torch.device(solver_config["device"])
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float64, device=device)
    theta = torch.zeros(x.shape[1] + 1, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=int(solver_config["max_iter"]),
        tolerance_grad=float(solver_config["tolerance_grad"]),
        tolerance_change=float(solver_config["tolerance_change"]),
        history_size=int(solver_config["history_size"]),
        line_search_fn=solver_config["line_search_fn"],
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, len(x_tensor), chunk_rows):
            stop = min(start + chunk_rows, len(x_tensor))
            logits = x_tensor[start:stop] @ theta[:-1] + theta[-1]
            part = F.binary_cross_entropy_with_logits(
                logits, y_tensor[start:stop], reduction="sum"
            ) / len(x_tensor)
            part.backward()
            objective += part.detach()
        return objective

    optimizer.step(closure)
    with torch.no_grad():
        objective = torch.zeros((), dtype=torch.float64, device=device)
        gradient = torch.zeros_like(theta)
        for start in range(0, len(x_tensor), chunk_rows):
            stop = min(start + chunk_rows, len(x_tensor))
            logits = x_tensor[start:stop] @ theta[:-1] + theta[-1]
            labels = y_tensor[start:stop]
            objective += F.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            ) / len(x_tensor)
            residual = torch.sigmoid(logits) - labels
            gradient[:-1] += x_tensor[start:stop].T @ residual / len(x_tensor)
            gradient[-1] += residual.sum() / len(x_tensor)
        state = optimizer.state[theta]
        result = {
            "coefficients": theta[:-1].cpu().numpy().tolist(),
            "intercept": float(theta[-1].item()),
            "final_objective": float(objective.item()),
            "gradient_l2_norm": float(torch.linalg.vector_norm(gradient).item()),
            "iterations": int(state["n_iter"]),
            "function_evaluations": int(state["func_evals"]),
            "closure_calls": closure_calls,
        }
    del optimizer, theta, x_tensor, y_tensor
    torch.cuda.empty_cache()
    return result


def probability_logit(probability: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float64, copy=False), epsilon, 1 - epsilon)
    return (np.log(clipped) - np.log1p(-clipped)).reshape(-1, 1)


def apply_logistic(x: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
    values = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    with torch.inference_mode():
        probability = torch.sigmoid(values[:, 0] * coefficient + intercept)
        result = probability.cpu().numpy()
    torch.cuda.empty_cache()
    return result
