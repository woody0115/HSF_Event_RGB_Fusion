#!/bin/bash
# Literature experiments repeated on plain add (2026-11-20, docs/tuning_log.md), phase 1: the same six single
# factors as runs/tune/p1.sh, but the fusion is the original add instead of sw2_add.
#   A   baseline: add with C's recipe (frozen backbones, 15 % corrupted + 5 % RGB = 0, 2 epochs, lr 5e-4,
#       warmup 0.3 without the bias jump), from the 3 seed checkpoints of add (runs/seed_runs.sh)
#   F1 --consist 1   F2 --rgb-aug-depth 2   F3 --night-weight 3   F5 --distill 1   (same flags as p1.sh)
#   F4 --fusion iaf_add --modality-drop 0: add has no gate, so the IAF analogue is the existing IllumGate
#      (per-channel gain on the event term from the lighting descriptor + GAP of both streams, zero-initialised
#      so it starts as add); its own 15 % gate dropout is switched off to change one thing only
#   F6 A's detections pooled with the RGB single stream before one NMS (no training)
# Every factor starts from the same 3 add checkpoints as A, with the same seeds, and is paired against A on val.
# GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/a1.done
log start
C="--epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05"
INITS="step4/dual_add_s5_cleaninit step4/dual_add_s5_seed1 step4/dual_add_s5_seed2"
RGB=step3/rgb_yolov8s_val/weights/best.pt
dir_of() { case $1 in add) echo step4;; iaf_*) echo step8_iaf;; esac; }
ckpts() {   # name fusion -> run:fusion:init list
  local s=0
  for init in $INITS; do echo "$(dir_of $2)/${1}_s$s:$2:$init"; s=$((s + 1)); done
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
( factor 0 a1a_add add a1_a_base ""
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag a1_f6_pool --extra rgb_zero --pool-rgb $RGB \
      --ckpt $(ckpts a1a_add add)
  factor 0 a1f1_add add a1_f1_consist "--consist 1"
  factor 0 a1f3_add add a1_f3_night "--night-weight 3" ) > runs/tune/a1_g0.log 2>&1 &
sleep 30
( factor 3 a1f5_add add a1_f5_distill "--distill 1"
  factor 3 a1f2_add add a1_f2_compose "--rgb-aug-depth 2"
  factor 3 a1f4_iaf_add iaf_add a1_f4_illum "--modality-drop 0" ) > runs/tune/a1_g3.log 2>&1 &
wait
log done
touch runs/tune/a1.done
