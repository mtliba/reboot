"""Stage 1 LightningModule — pretrain with the five SSL losses. See CLAUDE.md §8.1."""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from reboot.losses import (
    MaskedReconstructionLoss,
    OrdinalLoss,
    OrthogonalityLoss,
    ReflectiveContrastiveLoss,
    SmoothnessLoss,
)
from reboot.models import ReBootModel
from reboot.training.callbacks import warmup_cosine_lambda


class PretrainModule(pl.LightningModule):
    def __init__(self, model: ReBootModel, cfg: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cfg = cfg

        # Freeze the regression head during pretrain. This lets DDP run with the
        # vanilla `ddp` strategy — no need for `find_unused_parameters_true` (the
        # Python-side autograd walk it does is the main source of the 4× DDP
        # per-step slowdown observed at this model size).
        for p in self.model.regression_head.parameters():
            p.requires_grad_(False)

        self.mask_loss = MaskedReconstructionLoss()
        self.ord_loss = OrdinalLoss(d_s=model.d_s if not model.single_latent else model.d_s + model.d_d,
                                    margin=cfg.pretrain.margin)
        self.refl_loss = ReflectiveContrastiveLoss(temperature=cfg.pretrain.tau)
        self.smooth_loss = SmoothnessLoss()
        self.orth_loss = OrthogonalityLoss()

        # Ablations: a disabled L_ord leaves `ord_loss.phi` without gradient, which
        # vanilla DDP rejects ("parameters that were not used in producing the
        # loss") — this killed the SSL-MAE baseline job 16199. Freeze it instead.
        if not bool(getattr(cfg.pretrain.losses, "use_ord", True)):
            for p in self.ord_loss.parameters():
                p.requires_grad_(False)

        # Gradient-spike guard state (see on_before_optimizer_step).
        self._grad_norm_ema: float | None = None
        self._n_skipped_steps: int = 0

    def _encode_and_decode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["x"]                                               # (B, W, C)
        mask = batch["mask"]                                         # (B, W, C) bool
        # Masked positions are zeroed before encoding — decoder must reconstruct them.
        x_masked = x.masked_fill(mask, 0.0)
        z_s, z_d = self.model.encode(x_masked)
        x_hat = self.model.reconstruct(z_s, z_d)
        return z_s, z_d, x_hat

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        mask = batch["mask"]
        phase = batch["phase"]

        z_s, z_d, x_hat = self._encode_and_decode(batch)

        flags = self.cfg.pretrain.losses
        w = self.cfg.pretrain.loss_weights

        L_mask = self.mask_loss(x, x_hat, mask) if flags.use_mask else x.new_zeros(())
        L_ord = (
            self.ord_loss(z_s, phase, batch["pair_i"], batch["pair_j"])
            if flags.use_ord
            else x.new_zeros(())
        )

        # L_refl: use precomputed cross-phase pair indices from collate_pretrain.
        refl_c = batch.get("refl_c")
        refl_d = batch.get("refl_d")
        if flags.use_refl and refl_c is not None and refl_c.numel() >= 2:
            L_refl = self.refl_loss(z_s[refl_c], z_s[refl_d], self._refl_neg_mask(batch, refl_c, refl_d))
        else:
            L_refl = x.new_zeros(())

        L_smooth = (
            self.smooth_loss(z_s, batch["consec_i"], batch["consec_j"])
            if flags.use_smooth
            else x.new_zeros(())
        )
        L_orth = (
            self.orth_loss(z_s, z_d) if (flags.use_orth and not self.model.single_latent)
            else x.new_zeros(())
        )

        total = (
            w.mask * L_mask
            + w.ord * L_ord
            + w.refl * L_refl
            + w.smooth * L_smooth
            + w.orth * L_orth
        )

        self.log_dict(
            {
                "pretrain/loss_mask": L_mask,
                "pretrain/loss_ord": L_ord,
                "pretrain/loss_refl": L_refl,
                "pretrain/loss_smooth": L_smooth,
                "pretrain/loss_orth": L_orth,
                "pretrain/loss_total": total,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return total

    def _refl_neg_mask(self, batch: dict, refl_c, refl_d):
        """Label-free mask of near-SOC negatives for L_refl: two windows whose
        pseudo-SOC (cumulative charge, no labels) differ by less than
        pretrain.refl_neg_mask_eps are not used as negatives. eps <= 0 disables.
        Measured on the real data: ~10 % of in-batch negatives sit within 0.05
        true SOC of the anchor, i.e. plain InfoNCE pushes apart same-SOC windows."""
        eps = float(getattr(self.cfg.pretrain, "refl_neg_mask_eps", 0.0) or 0.0)
        ps = batch.get("pseudo_soc")
        if eps <= 0.0 or ps is None:
            return None
        pc, pd_ = ps[refl_c], ps[refl_d]
        return (pc[:, None] - pd_[None, :]).abs() < eps

    def on_before_optimizer_step(self, optimizer) -> None:
        """Skip the update when the (already all-reduced) gradient is non-finite or
        explodes relative to its running average.

        Both diverging pretrain runs (72178, 16191) died the same way: a healthy
        run for ~13-14k steps, then within ~100 steps loss_smooth jumped ~1000x,
        loss_mask followed (157 vs 0.15) and everything went NaN. Clipping to
        norm 1.0 bounds the step size but still lets a garbage direction in;
        dropping those few steps does not. DDP has reduced the gradients before
        this hook, so every rank sees the same norm and takes the same decision.
        Skipping = setting .grad to None, which AdamW treats as "no update".
        """
        factor = float(getattr(self.cfg.pretrain, "grad_spike_factor", 10.0) or 0.0)
        warmup_steps = int(getattr(self.cfg.pretrain, "grad_spike_warmup_steps", 200))
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return
        total = torch.stack([p.grad.detach().float().norm() for p in params]).norm()
        finite = bool(torch.isfinite(total))
        spike = (
            factor > 0.0
            and finite
            and self._grad_norm_ema is not None
            and self.global_step >= warmup_steps
            and float(total) > factor * self._grad_norm_ema
        )
        if not finite or spike:
            self._n_skipped_steps += 1
            for p in params:
                p.grad = None
            if self._trainer is None or self.trainer.is_global_zero:
                print(
                    f"[grad guard] step {self.global_step}: grad_norm={float(total):.3g} "
                    f"(ema={self._grad_norm_ema if self._grad_norm_ema is None else round(self._grad_norm_ema, 3)}) "
                    f"-> update skipped ({self._n_skipped_steps} so far)",
                    flush=True,
                )
        elif self._grad_norm_ema is None:
            self._grad_norm_ema = float(total)
        else:
            self._grad_norm_ema = 0.98 * self._grad_norm_ema + 0.02 * float(total)
        if self._trainer is not None:
            self.log_dict(
                {
                    "pretrain/grad_norm": total if finite else total.new_tensor(float("nan")),
                    "pretrain/skipped_steps": float(self._n_skipped_steps),
                },
                on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=True,
            )

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        mask = batch["mask"]
        phase = batch["phase"]
        z_s, z_d, x_hat = self._encode_and_decode(batch)

        L_mask = self.mask_loss(x, x_hat, mask)
        L_ord = self.ord_loss(z_s, phase, batch["pair_i"], batch["pair_j"])
        refl_c = batch.get("refl_c")
        refl_d = batch.get("refl_d")
        L_refl = (
            self.refl_loss(z_s[refl_c], z_s[refl_d])
            if refl_c is not None and refl_c.numel() >= 2
            else x.new_zeros(())
        )
        L_smooth = self.smooth_loss(z_s, batch["consec_i"], batch["consec_j"])
        L_orth = self.orth_loss(z_s, z_d) if not self.model.single_latent else x.new_zeros(())

        self.log_dict(
            {
                "val/loss_mask": L_mask,
                "val/loss_ord": L_ord,
                "val/loss_refl": L_refl,
                "val/loss_smooth": L_smooth,
                "val/loss_orth": L_orth,
            },
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
        )
        return L_mask + L_ord + L_refl + L_smooth + L_orth

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.pretrain.lr,
            weight_decay=self.cfg.pretrain.weight_decay,
        )
        # Linear warm-up then cosine decay (CLAUDE.md §9). The previous plain
        # CosineAnnealingLR started at the full LR on step 0, which — together
        # with un-clipped derived-channel outliers — let job 72178 diverge.
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=warmup_cosine_lambda(
                warmup_epochs=int(getattr(self.cfg.pretrain, "warmup_epochs", 0)),
                max_epochs=int(self.cfg.trainer.max_epochs),
            ),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
