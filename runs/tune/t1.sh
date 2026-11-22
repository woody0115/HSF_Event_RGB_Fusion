#!/bin/bash
# Inference-only improvements on G3 (2026-11-23, docs/tuning_log.md), one at a time, no training:
#   T1 Gaussian soft-NMS, sigma 0.5          --nms soft
#   T2 TTA: + horizontally flipped pass        --tta flip
#   T3 TTA: + pass resized x1.3 (832 px)       --tta up     (night pedestrians are mostly small)
# (WBF was already tried on G3 in phase 2, p2_g3_poolwbf: worse than NMS pooling.)
# Same evaluation as the reference p2_g3_distill_all (clean + corruptions + rgb_zero), 3 seeds, paired against it.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/t1.done
log start
CK="step12_sw/p2g3_sw2_add_s0:sw2_add:step12_sw/dual_sw2_add_s5
    step12_sw/p2g3_sw2_add_s1:sw2_add:step12_sw/dual_sw2_add_s5_seed1
    step12_sw/p2g3_sw2_add_s2:sw2_add:step12_sw/dual_sw2_add_s5_seed2"
ev() {   # gpu tag args
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt $CK --extra rgb_zero "$@"
}
( ev 0 t1_softnms --nms soft
  ev 0 t1_tta_up --tta up ) > runs/tune/t1_g0.log 2>&1 &
sleep 30
( ev 3 t1_tta_flip --tta flip ) > runs/tune/t1_g3.log 2>&1 &
wait
log done
touch runs/tune/t1.done
