#!/bin/bash
# Smoke check of the phase-2 options (2026-11-19): localization distillation alone on all frames (to read the
# size of its loss before choosing --distill-box), then a tiny pooled WBF evaluation. GPU 0 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
log start
CUDA_VISIBLE_DEVICES=0 python tools/train_dual.py --fusion sw2i_add --name p2_smoke --seed 0 --smoke \
    --init-ckpt runs/step12_sw/dual_sw2_add_s5/best.pt --epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 \
    --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05 --distill-box 1 --distill-scope all \
    2>&1 | grep -E "kd|Error|error|Traceback|mAP" | tail -25
CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag p2_smoke --ckpt step12_sw/ft5c_sw2_add_s0:sw2_add \
    --no-corrupt --pool-rgb step3/rgb_yolov8s_val/weights/best.pt --pool-mode wbf 2>&1 | tail -3
python - <<'EOF'
import json
r = json.load(open('runs/tune/p2_smoke.json'))['step12_sw/ft5c_sw2_add_s0']
print('wbf pooled C s0: all', r['clean']['all']['mAP50-95'], 'night ped',
      r['clean']['night']['per_class']['pedestrian'], 'seconds', r['seconds'])
EOF
log done
