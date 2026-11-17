#!/bin/bash
# Tuning round 3 (2026-11-17): short fine-tuning toward RGB failure, from the 3 seed checkpoints of add (control)
# and sw2_add (its feature MLP can learn input-dependent weights once corrupted RGB appears in training).
# Round 2 showed the fused models score 0-6 mAP with the RGB input set to 0 (event single stream 21.9): the neck +
# head never learned to detect from the event term alone, which no test-time gate change can fix.
#   frozen RGB and event backbones (BatchNorm in eval mode); fusion + neck + head trained
#   per image: 15 % RGB set to 0 (FRN), 50 % one training corruption (gaussian_noise, defocus_blur, fog,
#   brightness, dark; severity 1-5), 35 % clean. Held out for evaluation: shot / impulse noise, motion blur, contrast
#   2 epochs, lr 5e-4, warmup 0.3 epoch without the 0.1 bias jump, seed = seed of the init checkpoint
# Then tools/tune_eval.py on val (clean + corruption + RGB = 0). GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/r3.done
log start
FT="--epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 --freeze rgb,evt --rgb-aug 0.5 --rgb-drop 0.15"
ft_family() {   # gpu fusion step tag init0 init1 init2
  local gpu=$1 fusion=$2 step=$3 tag=$4; shift 4
  local ckpts=() s=0
  for init in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion $fusion --name ft3_${fusion}_s$s --seed $s \
        --init-ckpt runs/$init/best.pt $FT
    ckpts+=("$step/ft3_${fusion}_s$s:$fusion:$init")
    s=$((s + 1))
  done
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt "${ckpts[@]}" --extra rgb_zero
}
ft_family 0 add step4 r3_ft_add \
    step4/dual_add_s5_cleaninit step4/dual_add_s5_seed1 step4/dual_add_s5_seed2 > runs/tune/r3_g0.log 2>&1 &
sleep 30
ft_family 3 sw2_add step12_sw r3_ft_sw2 \
    step12_sw/dual_sw2_add_s5 step12_sw/dual_sw2_add_s5_seed1 step12_sw/dual_sw2_add_s5_seed2 > runs/tune/r3_g3.log 2>&1 &
wait
log done
touch runs/tune/r3.done
