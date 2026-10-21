export LC_ALL=C
cd ${REBOOT_ROOT:-$PWD}
P=${REBOOT_PYTHON:-python}
find reboot scripts tests -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
echo "=== host: $(hostname) ==="
echo "=== pytest ==="; PYTHONPATH=. timeout 900 $P -m pytest -q -p no:warnings 2>&1 | tail -3
echo; echo "=== real-meta checks: protocol sizes + reflective pairs per batch ==="
PYTHONPATH=. timeout 600 $P - <<'PY'
import numpy as np, time
from reboot.data.memmap_dataset import MemmapBatteryDataset
from reboot.data.splits import resolve_protocol
from reboot.data.samplers import CycleGroupedBatchSampler
from reboot.data.pair_sampling import make_reflective_pairs, make_ordinal_pairs, make_consecutive_pairs
ds = MemmapBatteryDataset("data/processed", mode="pretrain", filter_split="train", clip_value=10.0, stride=32)
ids = ds.extra["battery_ids"]
for strat, kw in [("per_cell_random", {}), ("cross_battery", {"test_batteries": ["cell7-25","cell11-35","cell19-45"]}),
                  ("cross_temperature", {"holdout_temperature": 25}), ("cross_cycle", {"cycle_split_frac": 0.7})]:
    pool, test, info = resolve_protocol(strat, ds.battery_idx, ids, ds.split, ds.cycle_id, **kw)
    print(f"{strat:18s} train_pool={pool.size:>9,}  test={test.size:>9,}  {info.get('holdout_batteries','')}")
cyc = ds.cycle_id[ds.indices]
smp = CycleGroupedBatchSampler(cyc, batch_size=2048, seed=0, rank=0, world_size=2)
t = time.time(); it = iter(smp); b = np.array(next(it)); tb = time.time() - t
print(f"\ncycle sampler: {smp.n_cycles:,} cycles, {len(smp):,} batches/rank (world=2), first batch in {tb*1000:.0f} ms, distinct cycles in batch={np.unique(cyc[b]).size}")
rng = np.random.default_rng(0)
for name, idx in (("random", rng.choice(len(ds), 2048, replace=False)), ("cycle-grouped", b)):
    g = ds.indices[idx]
    t = time.time()
    rc, rd = make_reflective_pairs(phases=ds.phase[g], cycle_ids=ds.cycle_id[g], segment_ids=ds.segment_id[g], local_ts=ds.local_t[g], segment_lengths=ds.segment_length[g], strategy="boundary_symmetric")
    oi, oj = make_ordinal_pairs(ds.segment_id[g], ds.local_t[g], K=8)
    ci, cj = make_consecutive_pairs(ds.segment_id[g], ds.local_t[g], stride=32)
    print(f"  {name:14s}: refl pairs={np.asarray(rc).size:5d}  ordinal pairs={np.asarray(oi).size:5d}  consecutive pairs={np.asarray(ci).size:5d}  charge frac={np.mean(ds.phase[g]==0):.2f}  ({(time.time()-t)*1000:.0f} ms)")
PY
echo; echo "=== CPU smokes (2 batches each) ==="
SM=outputs/smoke_$(date +%s); mkdir -p $SM
COMMON="trainer.accelerator=cpu trainer.precision=32 trainer.devices=1 trainer.limit_train_batches=2 trainer.limit_val_batches=2 data.num_workers=0 finetune.batch_size=64 pretrain.batch_size=64 wandb.mode=disabled paths.checkpoint_dir=$SM hydra.run.dir=$SM/hydra"
echo "--- finetune cross_battery"; CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 900 $P scripts/finetune.py "finetune.pretrained_ckpt='outputs/checkpoints/pretrain_200_16191/pretrain/last.ckpt'" data.splits.strategy=cross_battery finetune.max_epochs=1 $COMMON 2>&1 | grep -E "^Protocol|^Validation|Best checkpoint|Traceback|Error" | head -5
echo "--- evaluate cross_battery"; CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 900 $P scripts/evaluate.py "+eval.ckpt='$SM/finetune/best.ckpt'" +eval.protocol=cross_battery data.splits.strategy=cross_battery $COMMON 2>&1 | grep -E "^Protocol|mae:|Wrote|Traceback|Error" | head -5
echo "--- evaluate cross_temperature (same ckpt)"; CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 900 $P scripts/evaluate.py "+eval.ckpt='$SM/finetune/best.ckpt'" +eval.protocol=cross_temperature $COMMON 2>&1 | grep -E "^Protocol|mae:|Wrote|Traceback|Error" | head -5
echo "--- baseline tcn cross_cycle frac 0.05"; CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 900 $P scripts/baseline_supervised.py +baseline=supervised_tcn data.splits.strategy=cross_cycle finetune.label_fraction=0.05 trainer.max_epochs=1 $COMMON 2>&1 | grep -E "^Protocol|Low-label|TEST metrics|mae:|Wrote|Traceback|Error" | head -6
echo "--- pretrain v4 recipe: cycle sampler + soc_proxy + neg mask"; CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 900 $P scripts/pretrain.py pretrain.batch_sampler=cycle data.pair_strategy=soc_proxy pretrain.refl_neg_mask_eps=0.05 trainer.max_epochs=1 $COMMON 2>&1 | grep -E "Batch sampler|Traceback|Error" | head -5; ls $SM/pretrain | head -2
echo "--- pretrain cycle sampler + cross_battery pool"; CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 900 $P scripts/pretrain.py pretrain.batch_sampler=cycle data.splits.strategy=cross_battery trainer.max_epochs=1 $COMMON 2>&1 | grep -E "^Protocol|Batch sampler|Pretrain windows|Traceback|Error" | head -5; ls $SM/pretrain
rm -rf $SM
