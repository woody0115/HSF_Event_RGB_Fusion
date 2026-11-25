#!/bin/bash
# Smoke check of the single-factor options (2026-11-19) before phase 1: per-frame lighting of the training
# sequences, one short training run with every new option on, one quick pooled evaluation. GPU 0 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
log start
[ -f "$(python -c "from hsf.data.dataset import CFG; print(CFG['processed_root'])")/train_lighting.json" ] || \
    python tools/lighting_stats.py --split train | tail -5
CUDA_VISIBLE_DEVICES=0 python tools/train_dual.py --fusion sw2i_add --name p1_smoke --seed 0 --smoke \
    --init-ckpt runs/step12_sw/dual_sw2_add_s5/best.pt --epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 \
    --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05 --rgb-aug-depth 2 --consist 1 --distill 1 --night-weight 3 \
    2>&1 | grep -vE "^epoch .*/[0-9]+  [1-9][0-9]*00/" | tail -25
CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag p1_smoke --ckpt step12_sw/ft5c_sw2_add_s0:sw2_add \
    --no-clean --kinds gaussian_noise --severities 5 --stride 50 --extra rgb_zero \
    --pool-rgb step3/rgb_yolov8s_val/weights/best.pt 2>&1 | tail -3
log done
