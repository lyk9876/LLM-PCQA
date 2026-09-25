#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=4 python -u /data/zhangzy/zzyy/patch_multiview_render/train_pointfilter_punet_quality_geo_ema_coverage95_fast_dynamic.py \
  --variant quality_guided_full \
  --dynamic-guidance-ratio \
  --validation-uses-test \
  --extra-train-shapes dino,cup,gargoyle \
  --qnet-checkpoint /data/zhangzy/zzyy/PUNet/Gaussion/patch_quality_network/test_as_val_qnet_retrain_seed2026/run/best_patch_qnet_test_as_val_v1.pth \
  --quality-qnet-summary /data/zhangzy/zzyy/PUNet/Gaussion/patch_quality_network/test_as_val_qnet_retrain_seed2026/run/summary.json \
  --epochs 150 \
  --output-dir /data/zhangzy/zzyy/PointFilter_frozen_quality_stage3/punet30_overlap_testasval_qnet_testasval_dynamic_fairsteps_seed2026
