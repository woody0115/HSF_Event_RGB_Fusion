#!/bin/bash
# Tuning round 1 (2026-11-17): baseline on the val split for the two use cases (RGB degradation, night).
# RGB single stream + 3 seeds of add / relp2_add / sw2_add / cafr, no overrides. GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/r1.done
log start
( CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag r1_base --ckpt step3/rgb_yolov8s_val/weights/best.pt:single_rgb
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag r1_base --family add
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag r1_base_relp2 --family relp2_add ) > runs/tune/r1_g0.log 2>&1 &
sleep 30
( CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag r1_base_sw2 --family sw2_add
  CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag r1_base_cafr --family cafr ) > runs/tune/r1_g3.log 2>&1 &
wait
log done
touch runs/tune/r1.done
