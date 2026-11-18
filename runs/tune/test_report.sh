#!/bin/bash
# Test-split report (2026-11-18) of the configuration chosen on val after tuning round 5: sw2_add fine-tune C
# (ft5c, 15 % corrupted + 5 % RGB = 0, 2 epochs, frozen backbones), against the 3 base sw2_add seeds it started
# from, plus the RGB and event single streams. Report only: nothing here feeds back into any decision.
# Same metrics as on val (tools/tune_eval.py --split test): clean all / day / dim / night + per-class, RGB
# corruption mPC on every 4th test frame (9 kinds x severities 1 / 3 / 5), RGB input = 0. GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/test_report.done
log start
T="--split test --extra rgb_zero"
( CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag test_base_sw2 --family sw2_add $T
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag test_single --split test --no-corrupt \
      --ckpt step3/event_yolov8s_val/weights/best.pt:single_evt ) > runs/tune/test_report_g0.log 2>&1 &
sleep 30
( CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag test_ft5c_sw2 $T --ckpt \
      step12_sw/ft5c_sw2_add_s0:sw2_add:step12_sw/dual_sw2_add_s5 \
      step12_sw/ft5c_sw2_add_s1:sw2_add:step12_sw/dual_sw2_add_s5_seed1 \
      step12_sw/ft5c_sw2_add_s2:sw2_add:step12_sw/dual_sw2_add_s5_seed2
  CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag test_single $T \
      --ckpt step3/rgb_yolov8s_val/weights/best.pt:single_rgb ) > runs/tune/test_report_g3.log 2>&1 &
wait
log done
touch runs/tune/test_report.done
