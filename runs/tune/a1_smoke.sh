#!/bin/bash
# Smoke check before runs/tune/a1.sh (2026-11-20): the one new combination is iaf_add initialised from an add
# checkpoint with its gate dropout off, trained with C's recipe and the night distillation. GPU 0 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
log start
CUDA_VISIBLE_DEVICES=0 python tools/train_dual.py --fusion iaf_add --name a1_smoke --seed 0 --smoke \
    --init-ckpt runs/step4/dual_add_s5_cleaninit/best.pt --epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 \
    --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05 --modality-drop 0 --distill 1 \
    2>&1 | grep -E "initialised|kd|Error|error|Traceback|mAP|trainable" | tail -20
log done
