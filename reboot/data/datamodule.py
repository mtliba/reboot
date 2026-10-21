"""LightningDataModule wrapping windows.pkl.

Loads the pickle once, splits into train/val/test by battery id, and exposes
train/val/test DataLoaders for both pretrain and finetune modes.
"""
from __future__ import annotations

import pickle
import random
from functools import partial
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from reboot.data.dataset import BatteryDataset, Window, collate_finetune, collate_pretrain

Mode = Literal["pretrain", "finetune", "eval"]


class ReBootDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processed_dir: str | Path,
        mode: Mode,
        batch_size: int,
        stride: int = 32,
        pair_strategy: str = "boundary_symmetric",
        K_pairs: int = 8,
        mask_time_ratio: float = 0.15,
        mask_channel_prob: float = 0.10,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        val_fraction: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.mode = mode
        self.batch_size = batch_size
        self.stride = stride
        self.pair_strategy = pair_strategy
        self.K_pairs = K_pairs
        self.mask_time_ratio = mask_time_ratio
        self.mask_channel_prob = mask_channel_prob
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.val_fraction = val_fraction
        self.seed = seed

        self.train_windows: list[Window] = []
        self.val_windows: list[Window] = []
        self.test_windows: list[Window] = []
        self.battery_ids: list[str] = []
        self.capacities_As: dict[int, float] = {}

    @property
    def in_channels(self) -> int:
        if self.train_windows:
            return int(self.train_windows[0].data.shape[1])
        if self.test_windows:
            return int(self.test_windows[0].data.shape[1])
        raise RuntimeError("Datamodule has no windows; call setup() first.")

    def setup(self, stage: str | None = None) -> None:
        with open(self.processed_dir / "windows.pkl", "rb") as f:
            blob = pickle.load(f)
        windows: list[Window] = blob["windows"]
        train_ids = set(blob.get("train_ids") or [])
        test_ids = set(blob.get("test_ids") or [])
        self.battery_ids = blob.get("battery_ids") or []
        self.capacities_As = blob.get("capacities_As") or {}

        if self.mode == "pretrain":
            pool = [w for w in windows if w.battery_id in train_ids] if train_ids else windows
            for w in pool:
                w.soc = None
        else:
            pool = [w for w in windows if w.soc is not None]
            if train_ids:
                test_pool = [w for w in pool if w.battery_id in test_ids]
                pool = [w for w in pool if w.battery_id in train_ids]
                self.test_windows = test_pool

        # Carve a val slice out of the training pool.
        rng = random.Random(self.seed)
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1.0 - self.val_fraction)))
        self.train_windows = [pool[i] for i in idx[:cut]]
        self.val_windows = [pool[i] for i in idx[cut:]]

    def _make_loader(self, windows: list[Window], shuffle: bool, mode: Mode) -> DataLoader:
        ds = BatteryDataset(
            windows, mode=mode,
            mask_time_ratio=self.mask_time_ratio,
            mask_channel_prob=self.mask_channel_prob,
        )
        if mode == "pretrain":
            collate = partial(
                collate_pretrain, stride=self.stride,
                pair_strategy=self.pair_strategy, K_pairs=self.K_pairs,
            )
        else:
            collate = collate_finetune
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate, drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_windows, shuffle=True, mode=self.mode)

    def val_dataloader(self) -> DataLoader:
        if not self.val_windows:
            return self._make_loader(self.train_windows[:32], shuffle=False,
                                     mode="eval" if self.mode != "pretrain" else "pretrain")
        return self._make_loader(self.val_windows, shuffle=False,
                                 mode="eval" if self.mode != "pretrain" else "pretrain")

    def test_dataloader(self) -> DataLoader:
        ws = self.test_windows or self.val_windows
        return self._make_loader(ws, shuffle=False,
                                 mode="eval" if self.mode != "pretrain" else "pretrain")
