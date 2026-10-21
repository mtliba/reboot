"""Re-evaluate finished checkpoints (ReBoot finetunes AND supervised baselines)
with the extended metrics, on the protocol test set, and save predictions.

    python scripts/reevaluate.py "+eval.ckpt='path/best.ckpt'" [+eval.protocol=cross_battery] [+eval.tag=name]

Writes next to the checkpoint:  eval_extended[_<protocol>].json  and  predictions[_<protocol>].npz
The checkpoint kind is detected from its state_dict keys ('model.head.' = supervised baseline).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import build_model, build_test_dataset          # noqa: E402

from reboot.baselines.supervised import build_supervised_model
from reboot.data.dataset import collate_finetune
from reboot.eval.extended import compute_extended_metrics


def _count_params(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ev = cfg.get("eval")
    ckpt = ev.get("ckpt"); assert ckpt, "set +eval.ckpt"
    protocol = ev.get("protocol", None) or str(cfg.data.splits.strategy)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    hp = state.get("hyper_parameters", {}) or {}
    # Infer the number of input channels the checkpoint was trained with (raw-only runs use 4).
    first_w = next((v for k, v in sd.items() if k.startswith("model.encoder.") and v.ndim >= 2 and ("tcn.0.weight" in k or "input_proj.weight" in k or "proj.weight" in k or "lstm.weight_ih_l0" in k or "conv.0.weight" in k or "gru.weight_ih_l0" in k)), None)
    if first_w is not None:
        ckpt_in = int(first_w.shape[1]) if first_w.ndim == 3 or "weight_ih" in str(first_w.shape) else int(first_w.shape[-1])
        if first_w.ndim == 3:
            ckpt_in = int(first_w.shape[1])
        elif first_w.ndim == 2:
            ckpt_in = int(first_w.shape[1])
        if ckpt_in == 4 and cfg.data.get("input_channels", None) is None:
            cfg.data.input_channels = [0, 1, 2, 3]
            print("checkpoint has 4 input channels -> data.input_channels=[0,1,2,3]")
    test_ds, info = build_test_dataset(cfg, strategy=protocol)
    limit = cfg.trainer.get("limit_val_batches", None)
    if limit is not None:                                     # smoke-test path
        test_ds.indices = test_ds.indices[: int(limit) * 1024]
    in_channels = test_ds.in_channels

    supervised = any(k.startswith("model.head.") for k in sd)
    if supervised:
        bcfg = hp.get("cfg", {}).get("baseline", None) if isinstance(hp.get("cfg", {}), dict) else None
        if bcfg is None and "baseline" in cfg:
            bcfg = OmegaConf.to_container(cfg.baseline, resolve=True)
        assert bcfg is not None, "supervised ckpt: pass +baseline=supervised_<enc>"
        model = build_supervised_model(bcfg["encoder_type"], in_channels, bcfg["hidden"], bcfg["encoder_kwargs"])
        model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")})
        predict = lambda x: model(x)
        kind = f"supervised_{bcfg['encoder_type']}"
    else:
        single = any(k.startswith("model.g_d.") for k in sd) is False
        cfg.model.single_latent = bool(single)
        model = build_model(cfg, in_channels, cfg.data.window_size)
        missing, unexpected = model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}, strict=False)
        assert not [k for k in missing if not k.startswith("decoder")], missing
        predict = lambda x: model.predict_soc(*model.encode(x))
        kind = "reboot"
    n_params = _count_params(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=int(cfg.data.num_workers), pin_memory=True, collate_fn=collate_finetune)
    ys, ps, ph, bi = [], [], [], []
    t0 = time.time(); n = 0
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device, non_blocking=True)
            out = predict(x)
            ys.append(b["y"].numpy()); ps.append(out.float().cpu().numpy()); ph.append(b["phase"].numpy()); bi.append(b["battery_idx"].numpy()); n += x.size(0)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    y, p, phase, bidx = map(np.concatenate, (ys, ps, ph, bi))
    m = compute_extended_metrics(y, p, phase, bidx, list(test_ds.extra["battery_ids"]))
    m.update({"kind": kind, "protocol": protocol, "checkpoint": str(ckpt), "n_params": n_params,
              "windows_per_second": n / wall, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "tag": ev.get("tag", ""), **{k: v for k, v in info.items() if k != "protocol"}})
    suffix = "" if protocol == "per_cell_random" else f"_{protocol}"
    out = Path(ckpt).parent / f"eval_extended{suffix}.json"
    json.dump(m, open(out, "w"), indent=1)
    np.savez_compressed(Path(ckpt).parent / f"predictions{suffix}.npz", y=y.astype(np.float32), p=p.astype(np.float32), phase=phase.astype(np.int8), battery_idx=bidx.astype(np.int16))
    print(f"{kind:18s} {protocol:16s} n={m['n']:,} MAE={100*m['mae']:.3f}% RMSE={100*m['rmse']:.3f}% P95={100*m['p95']:.3f}% R2={m['r2']:.4f} params={n_params:,} {m['windows_per_second']:.0f} win/s -> {out}")


if __name__ == "__main__":
    main()
