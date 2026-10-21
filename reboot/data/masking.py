"""Masking strategy for L_mask — see CLAUDE.md §6.1."""
from __future__ import annotations

import torch


def generate_mask(
    W: int,
    C: int,
    time_mask_ratio: float = 0.15,
    channel_mask_prob: float = 0.10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a (W, C) bool mask. True = position is masked (decoder must predict it).

    Two independent maskings, combined with OR:
      1. Timestep masking: a random `time_mask_ratio` fraction of timesteps
         are fully masked across all channels.
      2. Channel masking: each channel has probability `channel_mask_prob`
         of being fully masked for this window.
    """
    assert 0.0 <= time_mask_ratio < 1.0, "time_mask_ratio must be in [0, 1)"
    assert 0.0 <= channel_mask_prob < 1.0, "channel_mask_prob must be in [0, 1)"

    mask = torch.zeros(W, C, dtype=torch.bool)

    # 1. Timestep masking
    n_time = int(W * time_mask_ratio)
    if n_time > 0:
        time_idx = torch.randperm(W, generator=generator)[:n_time]
        mask[time_idx, :] = True

    # 2. Channel masking
    if channel_mask_prob > 0.0:
        keep = torch.rand(C, generator=generator) < channel_mask_prob
        mask[:, keep] = True

    return mask
