"""Evaluate models on a chosen split (val or test) to check split contamination.

Checks two suspicions about the paper-schedule runs (CHANGELOG 2026-11-14):
  1. the step-3 single-stream models (initialisation of both backbones) were trained on all 47 training
     sequences, including the 5 validation sequences -> their val score should be abnormally high;
  6. checkpoints chosen on val are all from epoch 1-2 -> compare best.pt with last.pt (epoch 40) on test.
Model keys come from tools/eval_conditions.py (MODELS); add ':last' to a dual-stream key to load last.pt
instead of best.pt. Same evaluator as training (conf 0.001, IoU 0.7, multi-label, max 300).

Run:  python tools/eval_split.py --split val --models rgb event add_paper add_paper:last
Writes runs/eval_split.json (results are merged by "<split>/<key>")
"""
import argparse
import json
import os.path as osp
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))
from eval_conditions import MODELS, RUNS, Single  # noqa: E402
from hsf.data.dataset import CFG, DualDSEC, SequenceDSEC  # noqa: E402
from hsf.engine import evaluate, evaluate_sequential  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8, load_ultralytics  # noqa: E402

NAMES = {i: n for i, n in enumerate(CFG['names'])}


def build(key, device):
    base, _, which = key.partition(':')
    kind, rel, fusion = MODELS[base]
    if which == 'last':
        rel = osp.join(osp.dirname(rel), 'last.pt')
    path = osp.join(RUNS, rel)
    if not osp.isfile(path):
        return None
    if kind.startswith('single'):
        return Single(load_ultralytics(path), kind == 'single_evt').to(device).eval()
    m = DualYOLOv8(nc=8, fusion=fusion)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['val', 'test', 'train_core'])
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    device = torch.device('cuda')
    out_json = osp.join(RUNS, 'eval_split.json')
    for key in a.models:
        model = build(key, device)
        if model is None:
            print(f'{key}: checkpoint not found, skipped', flush=True)
            continue
        t0 = time.time()
        if getattr(model, 'stateful', False):
            dl = DataLoader(SequenceDSEC(a.split), batch_size=1, num_workers=a.workers, pin_memory=True)
            res = evaluate_sequential(model, dl, device, NAMES, desc=f'{a.split} {key}')
        else:
            dl = DataLoader(DualDSEC(a.split), batch_size=16, num_workers=a.workers, pin_memory=True,
                            collate_fn=DualDSEC.collate)
            with torch.no_grad():
                res = evaluate(model, dl, device, NAMES, desc=f'{a.split} {key}')
        print(f'== {a.split} {key:16s} mAP50 {res["mAP50"] * 100:.1f}  mAP50-95 {res["mAP50-95"] * 100:.1f}'
              f'  ({time.time() - t0:.0f} s)', flush=True)
        # re-read before writing: two instances (one per GPU) may share the file
        results = json.load(open(out_json)) if osp.isfile(out_json) else {}
        results[f'{a.split}/{key}'] = res
        json.dump(results, open(out_json, 'w'), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
