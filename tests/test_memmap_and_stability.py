"""Tests for the memmap dataset additions (clipping, next-window for L_phys) and
the training-stability pieces (warm-up schedule, StopOnNonFinite)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from reboot.data.dataset import collate_finetune
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.data.transforms import Scaler
from reboot.training.callbacks import StopOnNonFinite, warmup_cosine_lambda

W, C, STRIDE = 8, 7, 4


def _write_tiny_memmap(tmp_path, n_windows=6, outlier=1e4):
    """Two segments of 3 consecutive windows each. Window 2 -> 3 crosses a segment."""
    proc = tmp_path / "processed"
    proc.mkdir()
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_windows, W, C)).astype(np.float32)
    data[0, 2, 4] = outlier            # dV/dt glitch
    data[1, 5, 5] = -outlier           # dT/dt glitch
    # channel 0 = scaled current, channel 3 = log1p(dt)
    data[:, :, 0] = 0.5
    data[:, :, 3] = np.log1p(2.0)      # dt = 2 s per sample
    np.save(proc / "windows_data.npy", data)

    segment_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    local_t = np.array([0, STRIDE, 2 * STRIDE, 0, STRIDE, 2 * STRIDE], dtype=np.int32)
    np.savez(
        proc / "windows_meta.npz",
        phase=np.zeros(n_windows, np.int8),
        battery_idx=np.zeros(n_windows, np.int32),
        cycle_id=np.zeros(n_windows, np.int64),
        segment_id=segment_id,
        segment_length=np.full(n_windows, 100, np.int32),
        local_t=local_t,
        pseudo_soc=np.zeros(n_windows, np.float32),
        has_soc=np.ones(n_windows, bool),
        y=np.linspace(0, 1, n_windows).astype(np.float32),
        I_mean=np.full(n_windows, 0.5, np.float32),
        dt_total=np.full(n_windows, W * np.log1p(2.0), np.float32),
        split=np.zeros(n_windows, np.int8),
    )
    (proc / "windows_extra.json").write_text(json.dumps({"N": n_windows, "W": W, "C": C}))
    scaler = Scaler(I_median=1.0, I_iqr=2.0, V_median=3.3, V_iqr=0.2, has_derived=True,
                    derived_medians=np.zeros(3), derived_iqrs=np.ones(3))
    scaler_path = tmp_path / "scalers.pkl"
    scaler.save(scaler_path)
    return proc, scaler_path


def test_clip_value_bounds_every_channel(tmp_path):
    proc, _ = _write_tiny_memmap(tmp_path)
    ds = MemmapBatteryDataset(proc, mode="pretrain", clip_value=10.0, stride=STRIDE)
    x0, x1 = ds[0]["x"], ds[1]["x"]
    assert x0.abs().max() <= 10.0 and x1.abs().max() <= 10.0
    assert x0[2, 4] == 10.0 and x1[5, 5] == -10.0
    # disabling the clip returns the raw glitch
    ds_raw = MemmapBatteryDataset(proc, mode="pretrain", clip_value=None, stride=STRIDE)
    assert ds_raw[0]["x"][2, 4] == pytest.approx(1e4)


def test_next_window_respects_segment_boundary(tmp_path):
    proc, scaler_path = _write_tiny_memmap(tmp_path)
    ds = MemmapBatteryDataset(proc, mode="finetune", stride=STRIDE, scaler_path=scaler_path)
    has_next = [bool(ds[i]["has_next"]) for i in range(len(ds))]
    assert has_next == [True, True, False, True, True, False]
    # x_next of window 0 is window 1 (after clipping)
    assert torch.equal(ds[0]["x_next"], ds[1]["x"])


def test_raw_current_and_time_recovered_from_scaled_channels(tmp_path):
    proc, scaler_path = _write_tiny_memmap(tmp_path)
    ds = MemmapBatteryDataset(proc, mode="finetune", stride=STRIDE, scaler_path=scaler_path)
    item = ds[0]
    # I_scaled = 0.5 -> I_raw = 0.5 * iqr(2.0) + median(1.0) = 2.0 A
    assert item["I_raw_mean"].item() == pytest.approx(2.0, rel=1e-5)
    # dt = 2 s per sample over `stride` samples
    assert item["dt_raw_total"].item() == pytest.approx(2.0 * STRIDE, rel=1e-5)


def test_collate_finetune_stacks_next_fields(tmp_path):
    proc, scaler_path = _write_tiny_memmap(tmp_path)
    ds = MemmapBatteryDataset(proc, mode="finetune", stride=STRIDE, scaler_path=scaler_path)
    batch = collate_finetune([ds[i] for i in range(4)])
    assert batch["x_next"].shape == (4, W, C)
    assert batch["has_next"].dtype == torch.bool and batch["has_next"].tolist() == [True, True, False, True]
    assert batch["I_raw_mean"].shape == (4,) and batch["dt_raw_total"].shape == (4,)
    # eval mode without scaler does not emit the fields, and collate copes
    ds_eval = MemmapBatteryDataset(proc, mode="eval", stride=STRIDE)
    b2 = collate_finetune([ds_eval[0], ds_eval[1]])
    assert "x_next" not in b2 and "y" in b2


def test_finetune_module_uses_masked_physics_loss(tmp_path):
    from omegaconf import OmegaConf
    from reboot.models import ReBootModel
    from reboot.training.finetune_module import FinetuneModule

    proc, scaler_path = _write_tiny_memmap(tmp_path)
    ds = MemmapBatteryDataset(proc, mode="finetune", stride=STRIDE, scaler_path=scaler_path)
    batch = collate_finetune([ds[i] for i in range(6)])
    cfg = OmegaConf.create({
        "finetune": {"mode": "full_finetune", "lr_head": 1e-3, "lr_encoder": 1e-4,
                     "weight_decay": 0.0, "huber_delta": 0.05,
                     "loss_weights": {"phys": 0.01}, "losses": {"use_phys": True}},
        "trainer": {"max_epochs": 1},
    })
    model = ReBootModel(in_channels=C, W=W, encoder_type="tcn", hidden=16, d_s=4, d_d=4)
    module = FinetuneModule(model, cfg, num_batteries=1, nominal_caps_As=torch.tensor([3600.0]))
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    # physics parameters received gradient (L_phys active on the 4 has_next rows)
    assert module.phys_loss.log_Q.grad is not None
    assert module.phys_loss.eta_raw.grad is not None


def test_warmup_cosine_schedule_shape():
    f = warmup_cosine_lambda(warmup_epochs=10, max_epochs=200)
    assert f(0) == pytest.approx(0.1)                # starts at 1/warmup, NOT full LR
    assert f(9) == pytest.approx(1.0)                # end of warm-up
    assert f(10) == pytest.approx(1.0)
    assert f(105) == pytest.approx(0.5, abs=1e-2)    # half-way through cosine
    assert f(199) == pytest.approx(0.0, abs=1e-3)
    assert all(f(e) >= f(e + 1) - 1e-12 for e in range(10, 199))   # monotone decay
    g = warmup_cosine_lambda(warmup_epochs=0, max_epochs=4)
    assert g(0) == pytest.approx(1.0)


def test_stop_on_non_finite_sets_should_stop():
    class _Trainer:
        should_stop = False
        current_epoch = 3
        global_step = 42

    class _Module:
        def print(self, *a, **k):
            pass

    cb = StopOnNonFinite(patience_batches=3)
    tr, pm = _Trainer(), _Module()
    cb.on_train_batch_end(tr, pm, {"loss": torch.tensor(1.0)}, None, 0)
    assert tr.should_stop is False
    nan = {"loss": torch.tensor(float("nan"))}
    cb.on_train_batch_end(tr, pm, nan, None, 1)
    cb.on_train_batch_end(tr, pm, nan, None, 2)
    assert tr.should_stop is False                      # transient: below patience
    cb.on_train_batch_end(tr, pm, {"loss": torch.tensor(2.0)}, None, 3)   # finite resets the streak
    cb.on_train_batch_end(tr, pm, nan, None, 4)
    cb.on_train_batch_end(tr, pm, nan, None, 5)
    assert tr.should_stop is False
    cb.on_train_batch_end(tr, pm, nan, None, 6)
    assert tr.should_stop is True                       # 3 consecutive non-finite -> stop


def test_pretrain_module_freezes_phi_when_ord_disabled():
    from omegaconf import OmegaConf
    from reboot.models import ReBootModel
    from reboot.training.pretrain_module import PretrainModule

    def make(use_ord):
        cfg = OmegaConf.create({
            "pretrain": {"lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 0, "margin": 0.5, "tau": 0.1,
                         "loss_weights": {"mask": 1, "ord": 1, "refl": 0.5, "smooth": 0.1, "orth": 0.01},
                         "losses": {"use_mask": True, "use_ord": use_ord, "use_refl": True,
                                    "use_smooth": True, "use_orth": True}},
            "trainer": {"max_epochs": 2},
        })
        model = ReBootModel(in_channels=C, W=W, encoder_type="tcn", hidden=16, d_s=4, d_d=4,
                            encoder_kwargs={"kernel_size": 3, "dilations": [1, 2]})
        return PretrainModule(model, cfg)

    assert all(not p.requires_grad for p in make(False).ord_loss.parameters())
    assert all(p.requires_grad for p in make(True).ord_loss.parameters())
    # every trainable parameter must receive a gradient in an MAE-only step (DDP requirement)
    m = make(False)
    B = 6
    batch = {"x": torch.randn(B, W, C), "mask": torch.rand(B, W, C) < 0.2,
             "phase": torch.tensor([0, 0, 0, 1, 1, 1]),
             "pair_i": torch.tensor([0]), "pair_j": torch.tensor([1]),
             "consec_i": torch.tensor([0]), "consec_j": torch.tensor([1]),
             "refl_c": torch.tensor([0, 1]), "refl_d": torch.tensor([3, 4])}
    loss = m.training_step(batch, 0); loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], missing


def test_grad_guard_skips_spikes_and_nonfinite():
    from omegaconf import OmegaConf
    from reboot.models import ReBootModel
    from reboot.training.pretrain_module import PretrainModule

    cfg = OmegaConf.create({
        "pretrain": {"lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 0, "margin": 0.5, "tau": 0.1,
                     "grad_spike_factor": 10.0, "grad_spike_warmup_steps": 0,
                     "loss_weights": {"mask": 1, "ord": 1, "refl": 0.5, "smooth": 0.1, "orth": 0.01},
                     "losses": {"use_mask": True, "use_ord": True, "use_refl": True,
                                "use_smooth": True, "use_orth": True}},
        "trainer": {"max_epochs": 2},
    })
    model = ReBootModel(in_channels=C, W=W, encoder_type="tcn", hidden=16, d_s=4, d_d=4,
                        encoder_kwargs={"kernel_size": 3, "dilations": [1, 2]})
    m = PretrainModule(model, cfg)
    params = [p for p in m.parameters() if p.requires_grad]

    def set_grads(scale):
        for p in params:
            p.grad = torch.full_like(p, scale)

    set_grads(1e-3); m.on_before_optimizer_step(None)          # first call seeds the EMA
    assert m._grad_norm_ema is not None and all(p.grad is not None for p in params)
    ema0 = m._grad_norm_ema
    set_grads(1e-3); m.on_before_optimizer_step(None)          # normal step keeps grads
    assert all(p.grad is not None for p in params) and m._n_skipped_steps == 0
    set_grads(1.0); m.on_before_optimizer_step(None)           # 1000x spike -> skipped
    assert all(p.grad is None for p in params) and m._n_skipped_steps == 1
    assert m._grad_norm_ema == pytest.approx(ema0, rel=0.1)    # EMA not polluted by the spike
    set_grads(float("nan")); m.on_before_optimizer_step(None)  # non-finite -> skipped
    assert all(p.grad is None for p in params) and m._n_skipped_steps == 2
    # disabled factor: still drops non-finite, never drops finite
    m2 = PretrainModule(model, OmegaConf.merge(cfg, {"pretrain": {"grad_spike_factor": 0}}))
    set_grads(1e-3); m2.on_before_optimizer_step(None)
    set_grads(1e3); m2.on_before_optimizer_step(None)
    assert all(p.grad is not None for p in params) and m2._n_skipped_steps == 0


def test_reflective_neg_mask_removes_near_soc_negatives():
    from reboot.losses.reflective import ReflectiveContrastiveLoss
    torch.manual_seed(0)
    loss_fn = ReflectiveContrastiveLoss(temperature=0.1)
    z = torch.randn(6, 8)
    z_d = z.clone()
    z_d[1] = z[0]                                   # a hard false negative for anchor 0
    plain = loss_fn(z, z_d)
    mask = torch.zeros(6, 6, dtype=torch.bool); mask[0, 1] = True; mask[1, 0] = True
    masked = loss_fn(z, z_d, neg_mask=mask)
    assert masked < plain and torch.isfinite(masked)
    eye = torch.eye(6, dtype=torch.bool)            # positives are never removed
    assert torch.allclose(loss_fn(z, z_d, neg_mask=eye), plain)


def test_pretrain_module_builds_soc_mask_from_pseudo_soc():
    from omegaconf import OmegaConf
    from reboot.models import ReBootModel
    from reboot.training.pretrain_module import PretrainModule
    cfg = OmegaConf.create({
        "pretrain": {"lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 0, "margin": 0.5, "tau": 0.1,
                     "refl_neg_mask_eps": 0.05,
                     "loss_weights": {"mask": 1, "ord": 1, "refl": 0.5, "smooth": 0.1, "orth": 0.01},
                     "losses": {"use_mask": True, "use_ord": True, "use_refl": True, "use_smooth": True, "use_orth": True}},
        "trainer": {"max_epochs": 1},
    })
    m = PretrainModule(ReBootModel(in_channels=C, W=W, encoder_type="tcn", hidden=16, d_s=4, d_d=4,
                                   encoder_kwargs={"kernel_size": 3, "dilations": [1, 2]}), cfg)
    rc, rd = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    mask = m._refl_neg_mask({"pseudo_soc": torch.tensor([0.10, 0.50, 0.90, 0.12, 0.52, 0.88])}, rc, rd)
    assert mask.shape == (3, 3) and mask.diagonal().all()
    assert not (mask & ~torch.eye(3, dtype=torch.bool)).any()
    mask2 = m._refl_neg_mask({"pseudo_soc": torch.tensor([0.10, 0.50, 0.90, 0.12, 0.52, 0.11])}, rc, rd)
    assert mask2[0, 2]                              # charge0 (0.10) vs discharge5 (0.11) masked
    cfg.pretrain.refl_neg_mask_eps = 0.0
    assert m._refl_neg_mask({"pseudo_soc": torch.zeros(6)}, rc, rd) is None


def test_input_channels_subset(tmp_path):
    proc, scaler_path = _write_tiny_memmap(tmp_path)
    ds = MemmapBatteryDataset(proc, mode="finetune", stride=STRIDE, scaler_path=scaler_path, input_channels=[0, 1, 2, 3])
    assert ds.in_channels == 4
    x = ds[0]["x"]
    assert x.shape == (W, 4)
    full = MemmapBatteryDataset(proc, mode="eval", stride=STRIDE)[0]["x"]
    assert torch.equal(x, full[:, :4])
    # raw I/dt recovery for L_phys unchanged
    assert ds[0]["I_raw_mean"].item() == pytest.approx(2.0, rel=1e-5)
