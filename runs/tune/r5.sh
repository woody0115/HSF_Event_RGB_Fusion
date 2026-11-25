#!/bin/bash
# Tuning round 5 (2026-11-17): sw2_add fine-tuning, following round 4 A (30 % corrupted + 10 % RGB = 0, frozen
# backbones: val-mPC +13.9 like round 3, clean -0.5 instead of -1.0; training the event backbone did not help).
#   C  ft5c: even lighter, 15 % corrupted + 5 % RGB = 0 (80 % clean), 2 epochs - does robustness hold?
#   D  ft5d: round 4 A for 4 epochs instead of 2 (its val mAP still rose in epoch 2 for all seeds)
# lr 5e-4, warmup 0.3 epoch without the bias jump, frozen RGB and event backbones. GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/r5.done
log start
BASE="--lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 --freeze rgb,evt"
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
ft_family 0 ft5c_sw2_add r5_ft_sw2_lighter "--epochs 2 --rgb-aug 0.15 --rgb-drop 0.05" > runs/tune/r5_g0.log 2>&1 &
sleep 30
ft_family 3 ft5d_sw2_add r5_ft_sw2_light_e4 "--epochs 4 --rgb-aug 0.3 --rgb-drop 0.1" > runs/tune/r5_g3.log 2>&1 &
wait
log done
touch runs/tune/r5.done
