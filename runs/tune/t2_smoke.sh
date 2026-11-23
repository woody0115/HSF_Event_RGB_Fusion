#!/bin/bash
# Smoke test for --tta resize (2026-11-23), clean val only, G3 seed 0 and the RGB single stream.
# 'up' must reproduce t1_smoke_up (all 46.14) after the TTA refactor.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
CK=step12_sw/p2g3_sw2_add_s0:sw2_add:step12_sw/dual_sw2_add_s5
RGB=step3/rgb_yolov8s_val/weights/best.pt:single_rgb
for opt in "up $CK --tta up" "resize $CK --tta resize" "rgbresize $RGB --tta resize"; do
  set -- $opt; name=$1; ck=$2; shift 2
  rm -f runs/tune/t2_smoke_$name.json
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag t2_smoke_$name --no-corrupt --ckpt $ck "$@"
done
touch runs/tune/t2_smoke.done
