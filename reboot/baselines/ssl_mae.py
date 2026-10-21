"""SSL baseline #1: pretrain with masked reconstruction (L_mask) ONLY, then finetune.

This is the ReBoot pretrain stage with L_ord / L_refl / L_smooth / L_orth all
zeroed out. We reuse PretrainModule with `losses.use_*` flags set to False
rather than duplicating code; see scripts/baseline_ssl_mae.py.
"""
