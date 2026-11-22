#!/bin/bash
# Smoke test for the 2026-11-23 inference-only options of tools/tune_eval.py (--nms soft, --tta flip / up):
# a synthetic soft-NMS check, then clean val only on G3 seed 0. 'hard' must reproduce p2_g3_distill_all seed 0.
cd ~/work/HSF_Event_RGB_Fusion && source ~/work/hsf_env.sh
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import sys; sys.path.insert(0, 'tools')
import torch
from tune_eval import soft_nms
# two identical boxes of class 0 and one far box of class 1, xywh in pixels
p = torch.zeros(1, 4 + 8, 3)
p[0, :4, 0] = torch.tensor([100., 100., 40., 80.]); p[0, 4, 0] = 0.9
p[0, :4, 1] = torch.tensor([100., 100., 40., 80.]); p[0, 4, 1] = 0.8
p[0, :4, 2] = torch.tensor([400., 300., 50., 50.]); p[0, 5, 2] = 0.7
d = soft_nms(p.cuda())[0].cpu()
print(d)
assert d.shape[0] == 3 and abs(d[0, 4] - 0.9) < 1e-6 and abs(d[1, 4] - 0.7) < 1e-6
assert abs(d[2, 4] - 0.8 * torch.exp(torch.tensor(-1 / 0.5))) < 1e-5, 'duplicate decayed by exp(-1/sigma)'
print('soft_nms ok')
PY
CK=step12_sw/p2g3_sw2_add_s0:sw2_add:step12_sw/dual_sw2_add_s5
for opt in "hard" "soft --nms soft" "flip --tta flip" "up --tta up"; do
  set -- $opt; name=$1; shift
  rm -f runs/tune/t1_smoke_$name.json
  CUDA_VISIBLE_DEVICES=0 python tools/tune_eval.py --tag t1_smoke_$name --no-corrupt --ckpt $CK "$@"
done
touch runs/tune/t1_smoke.done
