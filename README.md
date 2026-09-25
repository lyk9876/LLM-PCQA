# PUNet PointFilter — intentional Train/Test-overlap leakage ablation

This bundle is a self-contained snapshot of an **intentional data-leakage ablation**
for PointFilter (patch-based point-cloud denoising) on PUNet. It exists to answer a
single question: *how much do the reported numbers move when 3 Test shapes are also
in Train, and when Test itself is used to select the checkpoint?*

It is **not** a generalization benchmark. Every number produced here is optimistically
biased by construction and must not be reported as unseen-shape Test performance.

## What leaks, exactly

Original split (`shape_split_seed2026.json`, seed 2026): 27 Train / 6 Val / 7 Test shapes.

| Stage | Change | Effect |
|---|---|---|
| Split override (`shape_split_seed2026_train_test_overlap.json`) | `cup`, `dino`, `gargoyle` added to Train | **Train ∩ Test = {cup, dino, gargoyle}** (3 of the 7 Test shapes are seen during training) |
| `--validation-uses-test` | Validation replaced by the full 7-shape Test set | **Test drives checkpoint selection**; the 6 original Validation shapes are dropped from the run entirely |
| QNet retrain | `dino`/`gargoyle` patches were already in Train; `cup` added | QNet's selection set equals the Test set |

So the ablation is doubled: the teacher network (QNet) *and* the student network
(PointFilter) are both selected on the same Test shapes, and 3 of those shapes are
inside the student's training set.

## Layout

```
code/patch_filter/      PointFilter training + model + dataloader (PF student side)
code/qnet/              QNet training + model (frozen quality teacher side)
checkpoints/            5 .pth files — PF student, PF baseline, QNet, QNet init
data/manifests/         shape splits, shared-patch manifest, QNet ranking labels
data/labels/            QNet real-score labels for the Test-as-Val run
data/shared_patches/    clean/ (640 npz) and noisy/ (3200 npz) 500-pt patches
runs/                   configs, logs, split overrides, metrics, records for both runs
scripts/                exact launch commands
MANIFEST.sha256         checksums for every file in this bundle
```

## Results as measured

**PointFilter student** (`runs/pf_punet30_overlap_testasval/summary.json`) — best epoch 4,
early-stopped at epoch 14, selected on Test-as-Validation (560 patches):

- chamfer distance `0.019479`, normal error `0.181247`, curvature error `0.096779`

**Full-cloud chamfer distance, 35 cases = 7 Test shapes × 5 noise levels**
(`runs/pf_punet30_overlap_testasval/fullcloud_cd_all35.json`, vs. the pretrained
baseline; paired, same noise seed):

| Subset | cases | baseline CD | ours CD | relative |
|---|---:|---:|---:|---:|
| all | 35 | 0.013406 | 0.013382 | **+0.175%** (18/35 better) |
| Train ∩ Test 3 shapes | 15 | 0.012841 | 0.012656 | **+1.447%** (10/15 better) |
| unseen 4 shapes | 20 | 0.013829 | 0.013927 | **−0.710%** (8/20 better) |

The asymmetry is the point of the ablation: the gain is concentrated almost entirely
on the 3 shapes the student was trained on, and the unseen shapes get slightly worse.

**QNet teacher** (`runs/qnet_test_as_val_retrain/summary.json`, 5-dim scores):

| Evaluation set | PLCC ↑ | SRCC ↑ | KRCC ↑ | MSE ↓ |
|---|---:|---:|---:|---:|
| 7-shape Test (used for selection) | 0.9098 | 0.8361 | 0.6860 | 0.0741 |
| — 3-shape seen overlap | 0.9110 | 0.8773 | 0.7365 | 0.0743 |
| — 4-shape unseen | 0.9089 | 0.8067 | 0.6532 | 0.0740 |
| original 6-shape Validation (untouched holdout) | 0.7386 | 0.7497 | 0.6071 | 0.1890 |

Selecting on Test lifts the overlap 3-shape PLCC from 0.8481 (normal Validation
selection) to 0.9110 — but it does not approach 1.0, the unseen 4 shapes regress on
every rank metric, and the untouched holdout regresses across the board. The conclusion
recorded in the run notes is that this checkpoint is **not** a drop-in replacement for
a cleanly selected QNet, and should not be the default teacher for joint PF training.
The Chinese write-up is kept verbatim at `runs/qnet_test_as_val_retrain/Test_as_Validation_实验结论.md`.

## Checkpoints

| File | Role |
|---|---|
| `pointfilter_punet30_overlap_testasval_dynamic_fairsteps_seed2026_best_validation.pth` | **Main artifact** — selected student (epoch 4) |
| `pointfilter_punet30_overlap_testasval_dynamic_fairsteps_seed2026_last_state.pth` | Final-epoch state + optimizer (not the selected model) |
| `pointfilter_pretrained_baseline_model_full_ae.pth` | Pretrained PF, the "Baseline" column in every comparison |
| `patch_qnet_test_as_val_retrain_seed2026_best.pth` | Frozen QNet teacher for this ablation |
| `patch_qnet_stage1_dcf_init_seed2026_best.pth` | Stage-I D_cf init the QNet was fine-tuned from |

## Reproducing

Code paths are hard-coded to a `/data/zhangzy/zzyy`-style root layout. To run from this
bundle, place `code/patch_filter/` and `code/qnet/` together with the external data trees
listed below and adjust the `ROOT` constants at the top of each script.

Training data that is **not** in this bundle (too large), still needed for a full rerun:

- `PUNet/Gaussion/pointfilter_train_seed2026_train_test_overlap/` (212 `.npy`, 121 MB) — the PF training set for this split
- `PUNet/Gaussion/noisy/sigma_*/{shape}.xyz` (330 MB) — full noisy clouds, for online ROI sampling and full-cloud inference
- `PUNet/Gaussion/clean_normalized/` (66 MB) — reference clouds
- `PUNet/train/50000_poisson/` — upstream PUNet sources

Launch commands (both need editing for paths and GPU id):

```bash
bash scripts/launch_pf_punet30_overlap_testasval.sh   # PointFilter student
bash runs/qnet_test_as_val_retrain/run_train.sh       # QNet teacher
```

Full-cloud CD and the comparison figures were produced by helper scripts that live
outside this bundle; the recorded outputs (`fullcloud_cd_all35.json`,
`baseline_to_ours_displacement.json`, `inference_manifest.json`) are included so the
numbers can be checked without re-running inference. **No images are included.**

Verify integrity of everything here with:

```bash
sha256sum -c MANIFEST.sha256
```

The QNet checkpoint hash `d1845892739b7fb2fe9263cc0130fdfb62e2f2c64483c4a925b22de7bb9e2fbc`
matches `qnet_checkpoint_sha256_before/after` in the student run's `summary.json`, which
is how the run proves QNet stayed frozen while gradients passed through it.
