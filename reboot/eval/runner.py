"""Run a prediction function over a DataLoader and write compute_metrics() JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from reboot.eval.metrics import compute_metrics


@torch.no_grad()
def predict_loader(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        p = predict_fn(x)
        ys.append(batch["y"].detach().cpu().numpy())
        ps.append(p.detach().float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_and_save(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: str | torch.device,
    out_path: str | Path,
    extra: dict | None = None,
) -> dict:
    y, p = predict_loader(predict_fn, loader, device)
    metrics = compute_metrics(y, p)
    payload = {k: v for k, v in metrics.items()}
    payload["n_test"] = int(y.size)
    if extra:
        payload.update(extra)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return metrics


def print_metrics(metrics: dict, title: str = "Eval metrics") -> None:
    print(f"{title}:")
    for k, v in metrics.items():
        if k == "error_by_bin":
            print("  error_by_bin: " + ", ".join(f"{kk.split('_')[-1]}:{vv:.4f}" for kk, vv in v.items()))
        else:
            print(f"  {k}: {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")
