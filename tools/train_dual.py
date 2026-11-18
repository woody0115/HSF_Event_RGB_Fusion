"""Steps 4-6: train the dual-stream (event + RGB) detector.

Defaults follow the HSF paper's full-model stage (Sec. IV-B): backbones initialised from the
single-stream models of step 3, SGD lr 0.005, momentum 0.937, batch 8, 40 epochs, 640x640.

Run:  .venv\\Scripts\\python.exe tools\\train_dual.py --fusion add --epochs 20   (step 4 baseline)
      .venv\\Scripts\\python.exe tools\\train_dual.py --fusion af  --epochs 20   (step 5)
      .venv\\Scripts\\python.exe tools\\train_dual.py --fusion hsf --epochs 20   (step 6, full HSF:
            TFS + AF, trained on clips of --clip-len consecutive frames, evaluated by streaming)
      .venv\Scripts\python.exe tools\train_dual.py --fusion iaf_add --epochs 5 --lr 1e-3 --warmup 1
            (step 8: illumination-aware gate on add / af, see IllumGate in hsf/models/fusion.py)
      python tools/train_dual.py --fusion rel_add --epochs 5 --lr 1e-3 --warmup 1
            (step 11: reliability-weighted fusion, docs/fusion_model.md; relp_add = physical prior only)
      python tools/train_dual.py --fusion rel2_add --epochs 5 --lr 1e-3 --warmup 1
            (v2: reliability-gated additive fusion; relp2_add = without the learned residual)
            add --aux 0.5 for the auxiliary per-branch losses (v3), --aux-branches evt for the event branch only
      python tools/train_dual.py --fusion sw2_add --epochs 5 --lr 1e-3 --warmup 1
            (step 12: softmax modality weights from the features; sw_ = sum 1, sw2_ = sum 2)
      python tools/train_dual.py --fusion cafr --epochs 5 --lr 1e-3 --warmup 1
            (step 13: CAFR of FRN, Cao et al. ECCV 2024; cafrx = without the zero-init residual)
      add --smoke for a quick pipeline check
"""
import argparse
import os.path as osp
import random
import sys

import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.engine import start_logging, train, train_temporal  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8  # noqa: E402

STEP3 = osp.join(ROOT, 'runs', 'step3')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion', default='add')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--rgb-init', default=osp.join(STEP3, 'rgb_yolov8s', 'weights', 'best.pt'))
    ap.add_argument('--evt-init', default=osp.join(STEP3, 'event_yolov8s', 'weights', 'best.pt'))
    ap.add_argument('--name', default=None)
    ap.add_argument('--warmup', type=float, default=3, help='warmup epochs')
    ap.add_argument('--clips', type=int, default=4, help='TFS/HSF: clips per batch')
    ap.add_argument('--clip-len', type=int, default=4, help='TFS/HSF: consecutive frames per clip')
    ap.add_argument('--modality-drop', type=float, default=0.15,
                    help='iaf_* fusions: share of training images with the event contribution switched off')
    ap.add_argument('--split', default='val', choices=['val', 'legacy'],
                    help='val: train on train_core, select checkpoints on the validation sequences '
                         '(configs/splits.yaml), report test once; legacy: train on all 47, select on test (#1-#12)')
    ap.add_argument('--resume', action='store_true', help='continue from <run>/resume.pt')
    ap.add_argument('--seed', type=int, default=None,
                    help='seed python / numpy / torch RNGs (fusion init, data order, augmentation); '
                         'None = unseeded as in versions #1-#10. cuDNN stays non-deterministic')
    ap.add_argument('--eval-every', type=int, default=1,
                    help='TFS/HSF: validate every N epochs (was 2 for > 6 epochs until 2026-11-14, which skipped '
                         'epoch 1 while add/af validated every epoch)')
    ap.add_argument('--aux', type=float, default=0.0,
                    help='rel2_/relp2_ fusions: initial weight of the auxiliary per-branch detection losses '
                         '(decays linearly to 0; docs/fusion_model.md Sec. 4)')
    ap.add_argument('--aux-branches', default='rgb,evt', help='branches with an auxiliary loss: rgb,evt | evt | rgb')
    ap.add_argument('--smoke', action='store_true', help='2 short epochs on a few sequences')
    # fine-tuning toward RGB failure / night (2026-11-17, docs/tuning_log.md)
    ap.add_argument('--init-ckpt', default=None,
                    help='start from a trained dual-stream checkpoint (runs/.../best.pt) instead of the single-stream inits')
    ap.add_argument('--freeze', default='', help='comma list of frozen sub-modules: rgb, evt, model (neck + head)')
    ap.add_argument('--rgb-aug', type=float, default=0.0,
                    help='share of training images with a random RGB corruption (hsf/data/corruptions.py)')
    ap.add_argument('--rgb-aug-kinds', default='gaussian_noise,defocus_blur,fog,brightness,dark',
                    help='corruption kinds used for training; the others stay held out for evaluation')
    ap.add_argument('--rgb-drop', type=float, default=0.0, help='share of training images with the RGB input set to 0')
    ap.add_argument('--warmup-bias-lr', type=float, default=0.1,
                    help='bias lr at the start of warmup (Ultralytics 0.1); use 0 when fine-tuning a trained model')
    # single-factor options (2026-11-19, docs/tuning_log.md; hsf/engine.py train)
    ap.add_argument('--rgb-aug-depth', type=int, default=1,
                    help='corrupted images get 1..N training kinds in a row (AugMix-style composition)')
    ap.add_argument('--consist', type=float, default=0.0,
                    help='weight of the clean (EMA teacher) vs degraded (student) consistency loss')
    ap.add_argument('--distill', type=float, default=0.0,
                    help='weight of the night-only distillation from an RGB single-stream teacher')
    ap.add_argument('--distill-teacher', default=osp.join(STEP3, 'rgb_yolov8s_val', 'weights', 'best.pt'))
    ap.add_argument('--night-weight', type=float, default=1.0, help='sampling weight of night training frames')
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)

    name = a.name or f'dual_{a.fusion}' + ('_smoke' if a.smoke else '')
    step = ('step4' if a.fusion == 'add' else 'step8_iaf' if a.fusion.startswith('iaf_') else
            'step11_rel' if a.fusion.startswith('rel') else
            'step12_sw' if a.fusion.startswith('sw') else 'step13_cafr' if a.fusion.startswith('cafr') else 'step5_6')
    run_dir = osp.join(ROOT, 'runs', step, name)
    start_logging(run_dir, ROOT)
    for k, path in (('rgb', a.rgb_init), ('event', a.evt_init)):
        if not osp.isfile(path):
            print(f'NOTE: {k} init checkpoint not found ({path}); that stream starts from scratch')
    model = DualYOLOv8(nc=8, fusion=a.fusion,
                       rgb_init=a.rgb_init if osp.isfile(a.rgb_init) else None,
                       evt_init=a.evt_init if osp.isfile(a.evt_init) else None,
                       modality_drop=a.modality_drop)
    model.aux0, model.aux_branches = a.aux, tuple(a.aux_branches.split(','))
    if a.init_ckpt:
        sd = torch.load(a.init_ckpt, map_location='cpu', weights_only=False)['model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # only new zero-initialised modules may be missing (sw2i_ gate: illum_fc)
        assert not unexpected and all('illum_fc' in k for k in missing), (missing, unexpected)
        print(f'initialised from {a.init_ckpt}' + (f' (new: {missing})' if missing else ''), flush=True)
    if a.freeze:
        model.freeze([n for n in a.freeze.split(',') if n])
    model.rgb_aug_p, model.rgb_drop_p = a.rgb_aug, a.rgb_drop
    model.rgb_aug_kinds = tuple(k for k in a.rgb_aug_kinds.split(',') if k)
    model.rgb_aug_depth = a.rgb_aug_depth
    teacher = None
    if a.distill > 0:
        from hsf.models.dual_yolov8 import load_ultralytics
        teacher = load_ultralytics(a.distill_teacher)
        print(f'distillation teacher {a.distill_teacher}', flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'config: {vars(a)}  trainable params {n_train / 1e6:.2f} M', flush=True)
    smoke = dict(epochs=2, sequences_train=['zurich_city_16_a', 'zurich_city_17_a'], sequences_val=['thun_02_a'],
                 max_train_batches=150) if a.smoke else dict(epochs=a.epochs)
    splits = dict(train_split='train', val_split='test') if (a.split == 'legacy' or a.smoke) else \
        dict(train_split='train_core', val_split='val')
    smoke.update(splits, resume=a.resume)
    if model.stateful:      # TFS / full HSF: clips of consecutive frames, streaming evaluation
        train_temporal(model, run_dir, clips=a.clips, clip_len=a.clip_len, lr=a.lr, workers=a.workers,
                       eval_every=a.eval_every, warmup_epochs=a.warmup, **smoke)
    else:
        train(model, run_dir, batch=a.batch, lr=a.lr, workers=a.workers, warmup_epochs=a.warmup,
              warmup_bias_lr=a.warmup_bias_lr, consist=a.consist, distill=a.distill, teacher=teacher,
              night_weight=a.night_weight, **smoke)


if __name__ == '__main__':
    main()
