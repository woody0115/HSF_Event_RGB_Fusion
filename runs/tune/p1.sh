#!/bin/bash
# Phase 1 of the literature-based experiments (2026-11-19, docs/tuning_log.md): one factor at a time on top of
# the round-5 choice C (sw2_add fine-tune: frozen backbones, 15 % corrupted + 5 % RGB = 0, 2 epochs, lr 5e-4,
# warmup 0.3 without the bias jump), same 3 init checkpoints and seeds as C, then tools/tune_eval.py on val
# (paired against C). GPUs 0 and 3 only.
#   F1 consistency  --consist 1          clean (EMA teacher) vs degraded (student) class-probability KL
#                                        (Tian et al. 2025 RGB-T consistency constraint; AugMix consistency loss)
#   F2 composition  --rgb-aug-depth 2    1-2 training corruptions in a row (AugMix-style; Michaelis et al. 2019)
#   F3 night        --night-weight 3     night frames sampled 3x as often (image re-sampling, LVIS RFS)
#   F4 illum gate   --fusion sw2i_add    lighting descriptor as extra gate input (IAF R-CNN)
#   F5 distill      --distill 1          night-only KL from the RGB single stream on clean RGB (AMFD-style
#                                        unimodal teacher; the RGB single stream is best on night pedestrians)
#   F6 late fusion  (no training)        C's detections pooled with the RGB single stream before NMS (ProbEn)
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/p1.done
log start
C="--epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05"
INITS="step12_sw/dual_sw2_add_s5 step12_sw/dual_sw2_add_s5_seed1 step12_sw/dual_sw2_add_s5_seed2"
factor() {   # gpu name fusion tag "extra train args"
  local gpu=$1 name=$2 fusion=$3 tag=$4 args=$5 ckpts=() s=0
  for init in $INITS; do
    CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion $fusion --name ${name}_s$s --seed $s \
        --init-ckpt runs/$init/best.pt $C $args
    ckpts+=("step12_sw/${name}_s$s:$fusion:$init")
    s=$((s + 1))
  done
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt "${ckpts[@]}" --extra rgb_zero
}
( factor 0 p1f1_sw2_add sw2_add p1_f1_consist "--consist 1"
  factor 0 p1f3_sw2_add sw2_add p1_f3_night "--night-weight 3"
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag p1_f6_pool --extra rgb_zero \
      --pool-rgb step3/rgb_yolov8s_val/weights/best.pt --ckpt \
      step12_sw/ft5c_sw2_add_s0:sw2_add:step12_sw/dual_sw2_add_s5 \
      step12_sw/ft5c_sw2_add_s1:sw2_add:step12_sw/dual_sw2_add_s5_seed1 \
      step12_sw/ft5c_sw2_add_s2:sw2_add:step12_sw/dual_sw2_add_s5_seed2 ) > runs/tune/p1_g0.log 2>&1 &
sleep 30
( factor 3 p1f5_sw2_add sw2_add p1_f5_distill "--distill 1"
  factor 3 p1f2_sw2_add sw2_add p1_f2_compose "--rgb-aug-depth 2"
  factor 3 p1f4_sw2i_add sw2i_add p1_f4_illum "" ) > runs/tune/p1_g3.log 2>&1 &
wait
log done
touch runs/tune/p1.done
