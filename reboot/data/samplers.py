"""Cycle-grouped batch sampler for pretraining.

Why: L_refl (CLAUDE.md §6.3, boundary_symmetric) pairs a CHARGE window with the
DISCHARGE window of the SAME cycle. With a plain shuffled loader a batch of 2048
windows drawn from 2.3 M windows / 31.7 k cycles contains ~0.07 windows per cycle,
so only ~28 cross-phase pairs form (measured on the real memmap) and InfoNCE runs
over N~28 instead of N~1000. L_ord and L_smooth also need same-segment pairs.

This sampler concatenates whole cycles in a fresh random order every epoch and
chunks the sequence into fixed-size batches, so a batch holds ~30 complete cycles
(charge + discharge, all windows) plus at most two partial cycles at its edges.
Negatives in L_refl then come from other cycles in the batch, as intended.

DDP: cycles are sharded across ranks (rank r takes every world_size-th cycle of
the epoch permutation) and every rank yields exactly the same number of batches
(the minimum over ranks), so no rank blocks in the all-reduce at epoch end. The
rank / world size are read lazily at iteration time, after Lightning has set up
the process group. Use with `Trainer(use_distributed_sampler=False)`.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Sampler


def _dist_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    # Fallbacks for launchers that export the topology before init (srun / torchrun).
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", os.environ.get("LOCAL_RANK", 0))))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return rank, world


class CycleGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        cycle_ids: np.ndarray,
        batch_size: int,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        cycle_ids = np.asarray(cycle_ids)
        assert cycle_ids.ndim == 1 and cycle_ids.size > 0
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._rank, self._world = rank, world_size
        # positions (0..len(ds)-1) grouped by cycle, each group in stored order
        order = np.argsort(cycle_ids, kind="stable")
        sorted_ids = cycle_ids[order]
        bounds = np.flatnonzero(np.diff(sorted_ids)) + 1
        self.groups: list[np.ndarray] = np.split(order, bounds)
        self.group_sizes = np.array([g.size for g in self.groups], dtype=np.int64)
        self.n_cycles = len(self.groups)

    # Lightning calls this on samplers/batch_samplers at every epoch start.
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _topology(self) -> tuple[int, int]:
        if self._rank is not None and self._world is not None:
            return self._rank, self._world
        return _dist_info()

    def _perm(self) -> np.ndarray:
        return np.random.default_rng(self.seed + self.epoch).permutation(self.n_cycles)

    def _n_batches_per_rank(self, perm: np.ndarray, world: int) -> int:
        sizes = self.group_sizes[perm]
        return int(min(sizes[r::world].sum() // self.batch_size for r in range(world)))

    def __len__(self) -> int:
        _, world = self._topology()
        return self._n_batches_per_rank(self._perm(), world)

    def __iter__(self):
        rank, world = self._topology()
        perm = self._perm()
        n_batches = self._n_batches_per_rank(perm, world)
        mine = perm[rank::world]
        flat = np.concatenate([self.groups[g] for g in mine])
        flat = flat[: n_batches * self.batch_size]
        for b in range(n_batches):
            yield flat[b * self.batch_size:(b + 1) * self.batch_size].tolist()
