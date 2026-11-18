#!/bin/bash
# Phase 2 of the literature-based experiments (2026-11-19, docs/tuning_log.md): combinations of the phase-1
# factors that helped, plus two refinements of the best one (F5) from the second literature check.
# All on top of C (same 3 init checkpoints and seeds), evaluated on val, paired against C. GPUs 0 and 3 only.
#   G1  F5 + F4        --fusion sw2i_add --distill 1
#   G2  F5 + LD        --distill 1 --distill-box 0.25   (localization distillation, Zheng et al. CVPR 2022;
#                                                       0.25 puts it near the class term, from p2_smoke)
#   G3  F5 all frames  --distill 1 --distill-scope all  (Uni-Modal Teacher, Du et al. ICML 2023)
#   G4  F5 + F4 + LD   --fusion sw2i_add --distill 1 --distill-box 0.25
# Then F6 on every new model, clean val only (phase 1: pooling does not change the corruption metrics):
#   one NMS over both models (phase-1 F6) and per-model NMS + weighted boxes fusion (Solovyev et al. 2021).
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/p2.done
log start
C="--epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05"
INITS="step12_sw/dual_sw2_add_s5 step12_sw/dual_sw2_add_s5_seed1 step12_sw/dual_sw2_add_s5_seed2"
RGB=step3/rgb_yolov8s_val/weights/best.pt
ckpts() {   # name fusion -> run:fusion:init list
  local s=0
  for init in $INITS; do echo "step12_sw/${1}_s$s:$2:$init"; s=$((s + 1)); done
}
factor() {   # gpu name fusion tag "extra train args"
  local gpu=$1 name=$2 fusion=$3 tag=$4 args=$5 s=0
  for init in $INITS; do
    CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion $fusion --name ${name}_s$s --seed $s \
        --init-ckpt runs/$init/best.pt $C $args
    s=$((s + 1))
  done
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt $(ckpts $name $fusion) --extra rgb_zero
}
( factor 0 p2g1_sw2i_add sw2i_add p2_g1_distill_illum "--distill 1"
  factor 0 p2g3_sw2_add sw2_add p2_g3_distill_all "--distill 1 --distill-scope all" ) > runs/tune/p2_g0.log 2>&1 &
sleep 30
( factor 3 p2g2_sw2_add sw2_add p2_g2_distill_ld "--distill 1 --distill-box 0.25"
  factor 3 p2g4_sw2i_add sw2i_add p2_g4_distill_illum_ld "--distill 1 --distill-box 0.25" ) > runs/tune/p2_g3.log 2>&1 &
wait
log trained
pool() {   # gpu mode tag ckpt-list
  local gpu=$1 mode=$2 tag=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --no-corrupt --pool-rgb $RGB --pool-mode $mode \
      --ckpt "$@"
}
( for g in "p2g1_sw2i_add sw2i_add g1" "p2g2_sw2_add sw2_add g2" "p2g3_sw2_add sw2_add g3" \
           "p2g4_sw2i_add sw2i_add g4"; do
    set -- $g; pool 0 nms p2_${3}_poolnms $(ckpts $1 $2)
  done ) >> runs/tune/p2_g0.log 2>&1 &
( pool 3 wbf p2_c_poolwbf $(ckpts ft5c_sw2_add sw2_add)
  for g in "p2g1_sw2i_add sw2i_add g1" "p2g2_sw2_add sw2_add g2" "p2g3_sw2_add sw2_add g3" \
           "p2g4_sw2i_add sw2i_add g4"; do
    set -- $g; pool 3 wbf p2_${3}_poolwbf $(ckpts $1 $2)
  done ) >> runs/tune/p2_g3.log 2>&1 &
wait
log done
touch runs/tune/p2.done
