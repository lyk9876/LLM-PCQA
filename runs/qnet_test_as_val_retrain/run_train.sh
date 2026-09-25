#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=6 python -u /data/zhangzy/zzyy/PUNet/Gaussion/patch_quality_network/test_as_val_qnet_retrain_seed2026/train_patch_qnet_test_as_val.py \
  --output-dir /data/zhangzy/zzyy/PUNet/Gaussion/patch_quality_network/test_as_val_qnet_retrain_seed2026/run
