"""Error breakdown by lighting and class (2026-11-20): where do the models fail at night?

For every model (one or more checkpoints, e.g. 3 seeds) the detections on the val split (same NMS as validation:
conf 0.001, IoU 0.7, multi-label, max 300) are matched to the ground truth at score >= --thr, per lighting bucket
(mean luminance of the 640x480 image, thresholds of tools/eval_conditions.py) and per class. Every GT box gets the
first outcome that applies (TIDE-like, as tools/diagnose_conditions.py, plus a split of the misses):
  found   a detection of the same class with IoU >= 0.5
  loc     same class, 0.1 <= IoU < 0.5 (mislocalised)
  cls     another class with IoU >= 0.5 (wrong class; the confusion matrix records which class)
  low     same class, IoU >= 0.5, but only below the score threshold (found with low confidence)
  miss    nothing of the above
Detections at score >= --thr whose IoU with every GT box is < 0.1 are background false positives (counted per
predicted class and per frame). Rates are averaged over the checkpoints of a model.
    -> runs/diag/<tag>.json / .md

Run:  python tools/night_errors.py --tag night_err --model rgb step3/rgb_yolov8s_val/weights/best.pt:single_rgb \
          --model g3 step12_sw/p2g3_sw2_add_s0:sw2_add step12_sw/p2g3_sw2_add_s1:sw2_add ...
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from eval_conditions import NAMES, Single, bucket_of  # noqa: E402
from hsf.data.dataset import DualDSEC  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8, load_ultralytics  # noqa: E402
from tune_eval import luminance  # noqa: E402

RUNS = osp.join(ROOT, 'runs')
BUCKETS = ('day', 'dim', 'night')
OUTCOMES = ('found', 'loc', 'cls', 'low', 'miss')
SIZES = {'small': (0, 32 ** 2), 'medium': (32 ** 2, 96 ** 2), 'large': (96 ** 2, 1e12)}
NC = len(NAMES)


def load(spec, dev):
    run, fusion = spec.rsplit(':', 1)
    if fusion in ('single_rgb', 'single_evt'):
        return Single(load_ultralytics(osp.join(RUNS, run)), fusion == 'single_evt').to(dev).eval()
    m = DualYOLOv8(nc=8, fusion=fusion)
    m.load_state_dict(torch.load(osp.join(RUNS, run, 'best.pt'), map_location='cpu', weights_only=False)['model'])
    return m.to(dev).eval()


def empty():
    return {'frames': 0, 'out': np.zeros((NC, len(OUTCOMES)), int),        # GT class x outcome
            'size': np.zeros((NC, len(SIZES), 2), int),                     # GT class x size x (gt, found)
            'conf': np.zeros((NC, NC), int),                                # GT class x predicted class (wrong class)
            'fp': np.zeros(NC, int)}                                        # background FP per predicted class


def match(acc, d, g, thr):
    """d: (N, 6) xyxy conf cls, g: (M, 5) cls xyxy, both on the same device."""
    acc['frames'] += 1
    hi = d[d[:, 4] >= thr]
    gc = g[:, 0].long()
    area = (g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2])
    iou_all = box_iou(g[:, 1:], d[:, :4]) if len(d) and len(g) else torch.zeros(len(g), len(d), device=g.device)
    keep = d[:, 4] >= thr
    iou = iou_all[:, keep]
    same = gc[:, None] == hi[None, :, 5].long()
    same_all = gc[:, None] == d[None, :, 5].long()
    for j in range(len(g)):
        c = int(gc[j])
        s = next(i for i, (lo, up) in enumerate(SIZES.values()) if lo <= float(area[j]) < up)
        acc['size'][c, s, 0] += 1
        if len(hi) and ((iou[j] >= 0.5) & same[j]).any():
            o = 'found'
            acc['size'][c, s, 1] += 1
        elif len(hi) and ((iou[j] >= 0.1) & same[j]).any():
            o = 'loc'
        elif len(hi) and (iou[j] >= 0.5).any():
            o = 'cls'
            cand = (iou[j] >= 0.5) & ~same[j]
            k = int(torch.argmax(torch.where(cand, hi[:, 4], torch.zeros_like(hi[:, 4]))))
            acc['conf'][c, int(hi[k, 5])] += 1
        elif len(d) and ((iou_all[j] >= 0.5) & same_all[j]).any():
            o = 'low'
        else:
            o = 'miss'
        acc['out'][c, OUTCOMES.index(o)] += 1
    if len(hi):
        bg = iou.amax(0) < 0.1 if len(g) else torch.ones(len(hi), dtype=torch.bool, device=hi.device)
        np.add.at(acc['fp'], hi[bg, 5].long().cpu().numpy(), 1)


@torch.no_grad()
def run_model(spec, dev, thr, workers):
    m = load(spec, dev)
    ds = DualDSEC('val')
    dl = DataLoader(ds, batch_size=16, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
    acc = {b: empty() for b in BUCKETS}
    for b in dl:
        rgb, evt = b['rgb'].to(dev), b['evt'].to(dev)
        with torch.autocast('cuda'):
            out = m(rgb, evt)
        dets = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
        lum = luminance(rgb).cpu()
        for i, d in enumerate(dets):
            sel = b['batch_idx'] == i
            g = torch.cat([b['cls'][sel], ops.xywh2xyxy(b['bboxes'][sel] * 640)], 1).to(dev).float()
            match(acc[bucket_of(float(lum[i]))], d.float(), g, thr)
    return acc


def to_json(acc):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in acc.items()}


def rates(accs, bucket):
    """Per-class outcome shares (%), mean over checkpoints; plus GT counts and FP / frame."""
    out = np.stack([np.array(a[bucket]['out'], float) for a in accs])          # (ckpt, NC, outcomes)
    gt = out.sum(2, keepdims=True)
    share = np.where(gt > 0, 100 * out / np.maximum(gt, 1), np.nan).mean(0)
    tot = out.sum(1)                                                             # (ckpt, outcomes), all classes
    share_all = (100 * tot / tot.sum(1, keepdims=True)).mean(0)
    fp = np.mean([np.array(a[bucket]['fp']).sum() / max(a[bucket]['frames'], 1) for a in accs])
    fpc = np.stack([np.array(a[bucket]['fp'], float) / max(a[bucket]['frames'], 1) for a in accs]).mean(0)
    size = np.stack([np.array(a[bucket]['size'], float) for a in accs])         # (ckpt, NC, sizes, 2)
    found_sz = np.where(size[..., 0] > 0, 100 * size[..., 1] / np.maximum(size[..., 0], 1), np.nan).mean(0)
    conf = np.stack([np.array(a[bucket]['conf'], float) for a in accs]).mean(0)
    return dict(gt=out[0].sum(1).astype(int), share=share, share_all=share_all, fp=fp, fpc=fpc,
                size_gt=size[0, ..., 0].astype(int), found_sz=found_sz, conf=conf)


def write_md(res, thr):
    f = lambda v: '–' if v != v else f'{v:.1f}'  # noqa: E731
    L = [f'# Error breakdown by lighting and class (val, score ≥ {thr})', '',
         'Share of GT boxes (%), mean over checkpoints. found = right class, IoU ≥ 0.5; loc = right class, '
         '0.1 ≤ IoU < 0.5; cls = other class, IoU ≥ 0.5; low = right class and box only below the score threshold; '
         'miss = nothing. bg FP = detections ≥ threshold with IoU < 0.1 to every GT, per frame.', '']
    labels = list(res)
    for bk in BUCKETS:
        R = {k: rates(res[k], bk) for k in labels}
        gt = R[labels[0]]['gt']
        L += [f'## {bk} ({res[labels[0]][0][bk]["frames"]} frames)', '',
              '| model | ' + ' | '.join(OUTCOMES) + ' | bg FP / frame |', '|---' * (len(OUTCOMES) + 2) + '|']
        for k in labels:
            L.append(f'| {k} | ' + ' | '.join(f(v) for v in R[k]['share_all']) + f' | {R[k]["fp"]:.2f} |')
        L += ['', '| class | GT | ' + ' | '.join(f'{k} found / cls / low / miss' for k in labels) + ' |',
              '|---' * (len(labels) + 2) + '|']
        for c in range(NC):
            if gt[c] == 0:
                continue
            cells = [' / '.join(f(R[k]['share'][c, OUTCOMES.index(o)]) for o in ('found', 'cls', 'low', 'miss'))
                     for k in labels]
            L.append(f'| {NAMES[c]} | {gt[c]} | ' + ' | '.join(cells) + ' |')
        L += ['', 'Found by size (%; GT small / medium / large in brackets):', '',
              '| class | GT S / M / L | ' + ' | '.join(labels) + ' |', '|---' * (len(labels) + 2) + '|']
        for c in range(NC):
            if gt[c] == 0:
                continue
            sg = R[labels[0]]['size_gt'][c]
            L.append(f'| {NAMES[c]} | {sg[0]} / {sg[1]} / {sg[2]} | ' +
                     ' | '.join(' / '.join(f(v) for v in R[k]['found_sz'][c]) for k in labels) + ' |')
        L += ['', 'Wrong class: GT → predicted (boxes, mean over checkpoints; ≥ 1 shown):', '']
        for k in labels:
            cf = R[k]['conf']
            pairs = sorted(((cf[i, j], i, j) for i in range(NC) for j in range(NC) if cf[i, j] >= 1), reverse=True)
            L.append(f'- {k}: ' + (', '.join(f'{NAMES[i]} → {NAMES[j]} {v:.0f}' for v, i, j in pairs[:8]) or 'none'))
        L += ['', 'Background FP per frame by predicted class:', '',
              '| model | ' + ' | '.join(NAMES[c] for c in range(NC)) + ' |', '|---' * (NC + 1) + '|']
        for k in labels:
            L.append(f'| {k} | ' + ' | '.join(f'{v:.3f}' for v in R[k]['fpc']) + ' |')
        L.append('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--model', nargs='+', action='append', required=True, metavar=('LABEL', 'RUN:FUSION'),
                    help='a label and one or more checkpoints (RUN_DIR:FUSION, or CKPT_FILE:single_rgb)')
    ap.add_argument('--thr', type=float, default=0.3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--md-only', action='store_true', help='rewrite the report from the saved json')
    a = ap.parse_args()
    out = osp.join(RUNS, 'diag', a.tag)
    os.makedirs(osp.dirname(out), exist_ok=True)
    res = json.load(open(out + '.json')) if osp.isfile(out + '.json') else {}
    if not a.md_only:
        dev = torch.device('cuda')
        for label, *specs in a.model:
            if label in res and len(res[label]) == len(specs):
                print(f'{label}: done before, skipped', flush=True)
                continue
            res[label] = []
            for spec in specs:
                t0 = time.time()
                acc = run_model(spec, dev, a.thr, a.workers)
                res[label].append({b: to_json(v) for b, v in acc.items()})
                n = acc['night']['out']
                print(f'{label} {spec}: night found {100 * n[:, 0].sum() / max(n.sum(), 1):.1f} %  '
                      f'night ped found {100 * n[0, 0] / max(n[0].sum(), 1):.1f} %  ({time.time() - t0:.0f} s)', flush=True)
            json.dump(res, open(out + '.json', 'w'))
    order = [m[0] for m in a.model if m[0] in res]
    open(out + '.md', 'w', encoding='utf-8').write(write_md({k: res[k] for k in order}, a.thr))
    print('wrote', out + '.md', flush=True)


if __name__ == '__main__':
    main()
