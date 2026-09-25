# Provenance

Every file in this bundle was copied verbatim from the original working tree — nothing
was regenerated or reformatted. Source roots:

| Prefix | Original location |
|---|---|
| `code/patch_filter/train_pointfilter_punet_quality_geo_ema_coverage95_fast_dynamic.py` | `patch_multiview_render/` |
| `code/patch_filter/train_pointfilter_structure_guidance_v2.py` | `patch_multiview_render/` (imported by the above as `pf_base`) |
| `code/patch_filter/edgeconv_quality_network.py` | `patch_multiview_render/` |
| `code/patch_filter/Pointfilter_*.py` | `Pointfilter-master/` |
| `code/patch_filter/visualize_pdlts.py` | `处理代码/` |
| `code/qnet/*` | `PUNet/Gaussion/patch_quality_network/test_as_val_qnet_retrain_seed2026/` |
| `checkpoints/pointfilter_punet30_*` | `PointFilter_frozen_quality_stage3/punet30_overlap_testasval_qnet_testasval_dynamic_fairsteps_seed2026/` |
| `checkpoints/pointfilter_pretrained_baseline_model_full_ae.pth` | `Pointfilter-master/Summary/pre_train_model/` |
| `checkpoints/patch_qnet_test_as_val_retrain_seed2026_best.pth` | `.../test_as_val_qnet_retrain_seed2026/run/best_patch_qnet_test_as_val_v1.pth` |
| `checkpoints/patch_qnet_stage1_dcf_init_seed2026_best.pth` | `PUNet/Gaussion/patch_quality_network/runs/stage1_dcf_specialization_v1_seed2026/` |
| `data/manifests/*` | `PUNet/Gaussion/manifests/` |
| `data/labels/d_real_mvp1032_test_as_val_seed2026.jsonl` | `.../test_as_val_qnet_retrain_seed2026/labels/` |
| `data/shared_patches/{clean,noisy}/` | `PUNet/Gaussion/shared_patches/` |
| `runs/pf_punet30_overlap_testasval/*` | root of the student run dir, except the two `test_fullcloud_visualization_*` files, which come from its `test_fullcloud_visualization/` subdir |
| `runs/qnet_test_as_val_retrain/*` | `.../test_as_val_qnet_retrain_seed2026/run/` and its parent |
| `scripts/launch_pf_*.sh` | the run dir's `launch_command.sh` |

Two renames worth knowing about:

- `runs/qnet_test_as_val_retrain/Test_as_Validation_实验结论.md` was
  `run/Test替代Validation实验结果.md`.
- `runs/pf_.../test_fullcloud_visualization_{summary.json,records.jsonl}` were
  `test_fullcloud_visualization/{summary,records}`. They are named apart from
  `summary.json` / `best_validation_records.jsonl` deliberately: both pairs exist in the
  source run dir and only one may own each basename here.

`config.json` and the `.jsonl` manifests are byte-for-byte copies and therefore still
contain the original absolute `/data/zhangzy/zzyy/...` paths, including inside the
recorded hyper-parameters. Those paths describe where the run read its data; they are
left untouched so the archived config keeps matching its own recorded hashes.
