#!/bin/bash
# Tuning round 2 (2026-11-17): training-free changes, 3 seeds each, val split, paired against round 1.
#   relp2 cnr_mode=signal   remove the noise variance from the RGB contrast (g_r rose under noise in round 1)
#   relp2 d_e P3 = 0        undo the learned P3 event damping (d_e -0.22 / -0.41 / -0.38)
#   relp2 d_e +1 all scales more event at every scale
#   sw2 bias x0.5 / bias 0  event share 0.2 -> ~0.3 / 0.5 (the MLP learned only a constant bias)
#   rgb_zero floors         every family with the RGB input set to 0 (event-only floor of the fusion model)
# GPUs 0 and 3 only.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
log() { echo "[$(date '+%F %T')] $*"; }
rm -f runs/tune/r2.done
log start
( CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag r2_relp2_cnr --family relp2_add --attr cnr_mode=signal --extra rgb_zero
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag r2_relp2_de0p3 --family relp2_add --set fusion.0.lambda_bias_e=0
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag r2_relp2_de1 --family relp2_add \
      --add fusion.0.lambda_bias_e=1 fusion.1.lambda_bias_e=1 fusion.2.lambda_bias_e=1 ) > runs/tune/r2_g0.log 2>&1 &
sleep 30
( CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag r2_sw2_half --family sw2_add \
      --add 'fusion.0.mlp.2.bias[0]=-0.5' 'fusion.0.mlp.2.bias[1]=0.5' 'fusion.1.mlp.2.bias[0]=-0.3' \
            'fusion.1.mlp.2.bias[1]=0.3' 'fusion.2.mlp.2.bias[0]=-0.3' 'fusion.2.mlp.2.bias[1]=0.3'
  CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag r2_sw2_eq --family sw2_add \
      --set fusion.0.mlp.2.bias=0 fusion.1.mlp.2.bias=0 fusion.2.mlp.2.bias=0
  for f in add sw2_add cafr; do
    CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag r2_floor --family $f --no-clean \
        --kinds gaussian_noise --severities 5 --extra rgb_zero
  done
  CUDA_VISIBLE_DEVICES=3 python tools/tune_eval.py --tag r2_floor --no-clean --kinds gaussian_noise --severities 5 \
      --ckpt step3/event_yolov8s_val/weights/best.pt:single_evt ) > runs/tune/r2_g3.log 2>&1 &
wait
log done
touch runs/tune/r2.done
