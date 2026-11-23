#!/bin/bash
# Where does the upscale-TTA gain (t1_tta_up: all +2.90, night pedestrian +4.41 on G3) come from?
# (2026-11-23, docs/tuning_log.md) No training; same evaluation as p2_g3_distill_all (clean + corruptions + rgb_zero).
#   U1 G3, one pass at x1.3 (832 px), nothing merged        --tta resize --tta-scale 1.3
#   U2 G3, one pass at x1.5 (960 px), nothing merged        --tta resize --tta-scale 1.5
#   U3 G3, 640 + x1.5 merged                                --tta up --tta-scale 1.5
#   R0 RGB single stream at 640 and R1 at x1.3 (fair baseline: does the single stream gain as much?)
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/t2.done
log start
CK="step12_sw/p2g3_sw2_add_s0:sw2_add:step12_sw/dual_sw2_add_s5
    step12_sw/p2g3_sw2_add_s1:sw2_add:step12_sw/dual_sw2_add_s5_seed1
    step12_sw/p2g3_sw2_add_s2:sw2_add:step12_sw/dual_sw2_add_s5_seed2"
RGB=step3/rgb_yolov8s_val/weights/best.pt:single_rgb
ev() {   # gpu tag ckpts args
  local gpu=$1 tag=$2 ck=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt $ck --extra rgb_zero "$@"
}
( ev 0 t2_resize13 "$CK" --tta resize --tta-scale 1.3
  ev 0 t2_up15 "$CK" --tta up --tta-scale 1.5 ) > runs/tune/t2_g0.log 2>&1 &
sleep 30
( ev 3 t2_resize15 "$CK" --tta resize --tta-scale 1.5
  ev 3 t2_rgb640 $RGB
  ev 3 t2_rgb_resize13 $RGB --tta resize --tta-scale 1.3 ) > runs/tune/t2_g3.log 2>&1 &
wait
log done
touch runs/tune/t2.done
