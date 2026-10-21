"""Tests for cycle-level validation split, batched SimCLR augmentation, and the
supervised / SimCLR baseline modules."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from reboot.baselines.simclr import SimCLRModule, augment_pair, batched_mask, random_crop_resize
from reboot.baselines.supervised import SupervisedModule, build_supervised_model
from reboot.data.splits import split_indices_by_cycle, subsample_indices
from reboot.models import ReBootModel


def test_split_by_cycle_is_cycle_atomic_and_disjoint():
    # 40 cycles x 25 windows each, in a shuffled global index space
    cycle_id = np.repeat(np.arange(40), 25)
    indices = np.arange(cycle_id.size)
    tr, va = split_indices_by_cycle(indices, cycle_id, val_fraction=0.10, seed=1)
    assert np.intersect1d(tr, va).size == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), indices)
    tr_cycles, va_cycles = set(cycle_id[tr]), set(cycle_id[va])
    assert not (tr_cycles & va_cycles), "a cycle leaked across train/val"
    assert len(va_cycles) == 4                                  # 10 % of 40 cycles
    # deterministic in the seed
    tr2, va2 = split_indices_by_cycle(indices, cycle_id, val_fraction=0.10, seed=1)
    assert np.array_equal(va, va2)
    # only a subset of indices passed in -> only those are split
    sub = indices[cycle_id < 10]
    tr3, va3 = split_indices_by_cycle(sub, cycle_id, val_fraction=0.2, seed=0)
    assert set(np.concatenate([tr3, va3])) == set(sub)


def test_split_by_cycle_zero_fraction_and_subsample():
    cycle_id = np.repeat(np.arange(5), 3)
    idx = np.arange(15)
    tr, va = split_indices_by_cycle(idx, cycle_id, 0.0)
    assert va.size == 0 and np.array_equal(tr, idx)
    s = subsample_indices(idx, 0.2, seed=0)
    assert s.size == 3 and set(s) <= set(idx)
    assert np.array_equal(subsample_indices(idx, 1.0), idx)


def test_random_crop_resize_identity_when_no_crop():
    x = torch.randn(6, 32, 7)
    y = random_crop_resize(x, crop_min=1.0)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-5), "crop_min=1 must be the identity"


def test_augment_pair_shapes_and_masking():
    torch.manual_seed(0)
    x = torch.randn(16, 40, 7) + 3.0                            # strictly nonzero-ish
    v1, v2 = augment_pair(x, crop_min=0.6, mask_ratio=0.25, channel_mask_prob=0.0)
    assert v1.shape == v2.shape == x.shape
    assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
    assert not torch.allclose(v1, v2), "two views must differ"
    # exactly 25 % of timesteps are zeroed in every sample (channel masking off)
    zero_rows = (v1 == 0).all(dim=-1).sum(dim=1)
    assert torch.all(zero_rows == 10)
    # values stay within the input range (bilinear interpolation, border padding)
    assert v1.max() <= x.max() + 1e-4 and v1[v1 != 0].min() >= x.min() - 1e-4
    m = batched_mask(4, 20, 3, time_ratio=0.0, channel_prob=1.0, device="cpu")
    assert m.all()


@pytest.mark.parametrize("enc", ["lstm", "tcn", "transformer", "hybrid"])
def test_supervised_model_forward(enc):
    kw = {
        "lstm": {"num_layers": 2, "dropout": 0.1, "bidirectional": True},
        "tcn": {"kernel_size": 3, "dilations": [1, 2, 4, 8]},
        "transformer": {"nhead": 4, "num_layers": 2, "ffn_dim": 64, "max_len": 128},
        "hybrid": {"nhead": 4, "num_transformer_layers": 1, "ffn_dim": 64},
    }[enc]
    model = build_supervised_model(enc, in_channels=7, hidden=32, encoder_kwargs=kw)
    y = model(torch.randn(5, 24, 7))
    assert y.shape == (5,) and torch.all((y >= 0) & (y <= 1))
    cfg = OmegaConf.create({"finetune": {"huber_delta": 0.05, "lr_head": 1e-3, "weight_decay": 0.0},
                            "trainer": {"max_epochs": 2}})
    module = SupervisedModule(model, cfg)
    batch = {"x": torch.randn(5, 24, 7), "y": torch.rand(5)}
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    module.eval()                                   # dropout off -> deterministic
    with torch.no_grad():
        assert torch.equal(module.predict(batch["x"]), model(batch["x"]))


def test_simclr_module_trains_only_encoder_and_projector():
    cfg = OmegaConf.create({"pretrain": {"lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 1},
                            "trainer": {"max_epochs": 4}})
    model = ReBootModel(in_channels=7, W=24, encoder_type="tcn", hidden=16, d_s=4, d_d=4,
                        encoder_kwargs={"kernel_size": 3, "dilations": [1, 2]})
    module = SimCLRModule(model, cfg, temperature=0.1, crop_min=0.7, mask_ratio=0.2)
    frozen = [n for n, p in module.named_parameters() if not p.requires_grad]
    assert all(n.startswith(("model.g_s", "model.g_d", "model.decoder", "model.regression_head")) for n in frozen)
    assert all(p.requires_grad for n, p in module.named_parameters() if n.startswith(("model.encoder", "projector")))
    loss = module.training_step({"x": torch.randn(8, 24, 7)}, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for n, p in module.named_parameters() if n.startswith("projector"))
    assert model.decoder.out_proj.weight.grad is None
    opt = module.configure_optimizers()["optimizer"]
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert n_opt == n_trainable


@pytest.mark.parametrize("name", ["chemali_dnn", "chemali_lstm", "yang_gru", "hannan_transformer", "li_cnnbilstm"])
def test_literature_models_forward(name):
    from reboot.baselines.literature import LITERATURE_MODELS
    small = {"chemali_dnn": {}, "chemali_lstm": {"hidden": 16}, "yang_gru": {"hidden": 16},
             "hannan_transformer": {"d_model": 16, "nhead": 4, "num_layers": 1, "ffn_dim": 32},
             "li_cnnbilstm": {"filters": 8, "hidden": 8, "fc": 8}}[name]
    m = build_supervised_model(name, in_channels=7, hidden=16, encoder_kwargs=small)
    assert isinstance(m, LITERATURE_MODELS[name])
    y = m(torch.randn(5, 24, 7))
    assert y.shape == (5,) and torch.all((y >= 0) & (y <= 1))
    y.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
