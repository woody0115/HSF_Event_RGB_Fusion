#!/bin/bash
# Error breakdown by lighting and class on val (2026-11-20, tools/night_errors.py): RGB single stream, the original
# add and sw2_add, and their fine-tuned versions (C, G3; add + H1), 3 seeds each. GPU 0 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
CUDA_VISIBLE_DEVICES=0 python tools/night_errors.py --tag night_err \
  --model rgb step3/rgb_yolov8s_val/weights/best.pt:single_rgb \
  --model add step4/dual_add_s5_cleaninit:add step4/dual_add_s5_seed1:add step4/dual_add_s5_seed2:add \
  --model add_H1 step4/a2h1_add_s0:add step4/a2h1_add_s1:add step4/a2h1_add_s2:add \
  --model sw2 step12_sw/dual_sw2_add_s5:sw2_add step12_sw/dual_sw2_add_s5_seed1:sw2_add step12_sw/dual_sw2_add_s5_seed2:sw2_add \
  --model sw2_C step12_sw/ft5c_sw2_add_s0:sw2_add step12_sw/ft5c_sw2_add_s1:sw2_add step12_sw/ft5c_sw2_add_s2:sw2_add \
  --model sw2_G3 step12_sw/p2g3_sw2_add_s0:sw2_add step12_sw/p2g3_sw2_add_s1:sw2_add step12_sw/p2g3_sw2_add_s2:sw2_add
touch runs/diag/night_err.done
