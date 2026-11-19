#!/bin/bash
# Literature experiments on plain add (2026-11-20, docs/tuning_log.md), phase 2. Phase 1 (runs/tune/a1.sh) kept
# F5 (night distillation) and F6 (pooling with the RGB single stream); F1 hurt, F2 / F3 / F4 did nothing.
#   H1 A + --distill 1 --distill-scope all                     (the G3 analogue, best on sw2_add)
#   H2 H1 + --distill-box 0.25                                  (G3 + LD, not tried on sw2_add)
# Then F6 (one NMS over the fused and the RGB single-stream detections) on F5, H1 and H2, clean val only.
# Same 3 add checkpoints and seeds as phase 1; everything is paired against A (a1_a_base) and F5. GPUs 0 and 3.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/a2.done
log start
C="--epochs 2 --lr 5e-4 --warmup 0.3 --warmup-bias-lr 0 --freeze rgb,evt --rgb-aug 0.15 --rgb-drop 0.05"
INITS="step4/dual_add_s5_cleaninit step4/dual_add_s5_seed1 step4/dual_add_s5_seed2"
RGB=step3/rgb_yolov8s_val/weights/best.pt
ckpts() {   # name -> run:fusion:init list (add runs live in step4)
  local s=0
  for init in $INITS; do echo "step4/${1}_s$s:add:$init"; s=$((s + 1)); done
}
factor() {   # gpu name tag "extra train args"
  local gpu=$1 name=$2 tag=$3 args=$4 s=0
  for init in $INITS; do
    CUDA_VISIBLE_DEVICES=$gpu python tools/train_dual.py --fusion add --name ${name}_s$s --seed $s \
        --init-ckpt runs/$init/best.pt $C $args
    s=$((s + 1))
  done
  CUDA_VISIBLE_DEVICES=$gpu python tools/tune_eval.py --tag $tag --ckpt $(ckpts $name) --extra rgb_zero
}
pool() {   # gpu tag name
  CUDA_VISIBLE_DEVICES=$1 python tools/tune_eval.py --tag $2 --no-corrupt --pool-rgb $RGB --pool-mode nms \
      --ckpt $(ckpts $3)
}
( pool 0 a2_f5_poolnms a1f5_add
  factor 0 a2h1_add a2_h1_distill_all "--distill 1 --distill-scope all"
  pool 0 a2_h1_poolnms a2h1_add ) > runs/tune/a2_g0.log 2>&1 &
sleep 30
( factor 3 a2h2_add a2_h2_distill_all_ld "--distill 1 --distill-scope all --distill-box 0.25"
  pool 3 a2_h2_poolnms a2h2_add ) > runs/tune/a2_g3.log 2>&1 &
wait
log done
touch runs/tune/a2.done
