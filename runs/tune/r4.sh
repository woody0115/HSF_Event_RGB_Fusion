#!/bin/bash
# Tuning round 4 (2026-11-17): sw2_add fine-tuning (round 3 winner: val-mPC +13.8, RGB = 0 floor 23.4, clean -1.0,
# night pedestrians +0.2) - two variants against the clean-accuracy cost, from the same 3 seed checkpoints:
#   A  ft4a: lighter degradation, 30 % corrupted + 10 % RGB = 0 (60 % clean instead of 35 %); backbones frozen
#   B  ft4b: round-3 degradation (50 % + 15 %) with the event backbone trainable as well, so the event features
#            can adapt to carrying the detection alone (RGB backbone stays frozen)
# 2 epochs, lr 5e-4, warmup 0.3 epoch without the bias jump. Then tools/tune_eval.py on val. GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/r4.done
log start
BASE="--epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0"
INITS="step12_sw/dual_sw2_add_s5 step12_sw/dual_sw2_add_s5_seed1 step12_sw/dual_sw2_add_s5_seed2"
ft_family() {   # gpu name tag "train args"
  local gpu=$1 name=$2 tag=$3 args=$4 ckpts=() s=0
  for init in $INITS; do
    CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion sw2_add --name ${name}_s$s --seed $s \
        --init-ckpt runs/$init/best.pt $BASE $args
    ckpts+=("step12_sw/${name}_s$s:sw2_add:$init")
    s=$((s + 1))
  done
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt "${ckpts[@]}" --extra rgb_zero
}
ft_family 0 ft4a_sw2_add r4_ft_sw2_light "--freeze rgb,evt --rgb-aug 0.3 --rgb-drop 0.1" > runs/tune/r4_g0.log 2>&1 &
sleep 30
ft_family 3 ft4b_sw2_add r4_ft_sw2_evt "--freeze rgb --rgb-aug 0.5 --rgb-drop 0.15" > runs/tune/r4_g3.log 2>&1 &
wait
log done
touch runs/tune/r4.done
