"""Unit tests matching the Verification blocks in CLAUDE.md §6."""
from __future__ import annotations

import torch

from reboot.losses import (
    MaskedReconstructionLoss,
    OrdinalLoss,
    OrthogonalityLoss,
    PhysicsLoss,
    ReflectiveContrastiveLoss,
    SmoothnessLoss,
)


# ---------- L_mask (CLAUDE.md §6.1) ----------

def test_mask_loss_full_mask_returns_one():
    loss_fn = MaskedReconstructionLoss()
    x = torch.zeros(2, 10, 4)
    x_hat = torch.ones(2, 10, 4)
    mask = torch.ones(2, 10, 4, dtype=torch.bool)
    assert torch.allclose(loss_fn(x, x_hat, mask), torch.tensor(1.0))


def test_mask_loss_partial_mask_still_one():
    loss_fn = MaskedReconstructionLoss()
    x = torch.zeros(2, 10, 4)
    x_hat = torch.ones(2, 10, 4)
    mask = torch.zeros(2, 10, 4, dtype=torch.bool)
    mask[:, :, :2] = True
    assert torch.allclose(loss_fn(x, x_hat, mask), torch.tensor(1.0))


def test_mask_loss_zero_when_perfect():
    loss_fn = MaskedReconstructionLoss()
    x = torch.randn(3, 8, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    assert loss_fn(x, x.clone(), mask).item() == 0.0


# ---------- L_ord (CLAUDE.md §6.2) ----------

def test_ordinal_loss_runs_and_backprops():
    loss_fn = OrdinalLoss(d_s=4, margin=0.5)
    z_s = torch.tensor(
        [
            [1.0, 0, 0, 0],
            [2.0, 0, 0, 0],
            [2.0, 0, 0, 0],
            [1.0, 0, 0, 0],
        ],
        requires_grad=True,
    )
    phase = torch.tensor([0, 0, 1, 1])
    pair_i = torch.tensor([0, 2])
    pair_j = torch.tensor([1, 3])
    out = loss_fn(z_s, phase, pair_i, pair_j)
    assert out.ndim == 0
    out.backward()
    assert z_s.grad is not None


def test_ordinal_loss_empty_pairs():
    loss_fn = OrdinalLoss(d_s=4)
    z_s = torch.randn(4, 4)
    out = loss_fn(z_s, torch.zeros(4, dtype=torch.long),
                  torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    assert out.item() == 0.0


# ---------- L_refl (CLAUDE.md §6.3) ----------

def test_reflective_loss_aligned_lower_than_misaligned():
    torch.manual_seed(0)
    loss_fn = ReflectiveContrastiveLoss(temperature=0.1)
    z = torch.randn(8, 16)
    out_same = loss_fn(z, z.clone())
    out_diff = loss_fn(z, torch.randn(8, 16))
    assert out_same < out_diff


def test_reflective_loss_small_batch_returns_zero():
    loss_fn = ReflectiveContrastiveLoss()
    z = torch.randn(1, 16)
    out = loss_fn(z, z.clone())
    assert out.item() == 0.0


# ---------- L_smooth (CLAUDE.md §6.4) ----------

def test_smoothness_loss_constant_returns_zero():
    loss_fn = SmoothnessLoss()
    z_s = torch.ones(4, 16)
    out = loss_fn(z_s, torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert out.item() == 0.0


def test_smoothness_loss_increases_with_jump():
    loss_fn = SmoothnessLoss()
    z_s = torch.zeros(3, 4)
    z_s[1] = 1.0
    out = loss_fn(z_s, torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert out.item() > 0.0


# ---------- L_orth (CLAUDE.md §6.5) ----------

def test_orthogonality_independent_lower_than_correlated():
    torch.manual_seed(0)
    loss_fn = OrthogonalityLoss()
    z_s = torch.randn(128, 16)
    z_d = torch.randn(128, 16)
    out_indep = loss_fn(z_s, z_d)
    z_d_corr = z_s + 0.01 * torch.randn(128, 16)
    out_corr = loss_fn(z_s, z_d_corr)
    assert out_corr > out_indep


# ---------- L_phys (CLAUDE.md §6.6) ----------

def test_physics_loss_matches_coulomb_counting():
    nominal = torch.tensor([3600.0])                                 # 1 Ah in A·s
    loss_fn = PhysicsLoss(num_batteries=1, nominal_caps_As=nominal)

    I = torch.tensor([1.0])                                          # 1 A
    dt = torch.tensor([3600.0])                                      # 1 h
    eta = loss_fn.get_eta().item()
    y_t = torch.tensor([0.0])
    y_tp1 = torch.tensor([eta])                                      # exact
    bid = torch.tensor([0])
    out = loss_fn(y_t, y_tp1, I, dt, bid)
    assert out.item() < 1e-5


def test_physics_eta_in_range():
    loss_fn = PhysicsLoss(num_batteries=2, nominal_caps_As=torch.tensor([3600.0, 7200.0]))
    eta = loss_fn.get_eta().item()
    assert 0.9 <= eta <= 1.0
