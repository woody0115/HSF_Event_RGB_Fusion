"""Robustness of trained models to RGB corruptions (the event input stays clean).

For every model, the test split (every --stride-th frame) is loaded once and each batch is evaluated clean and under
every corruption x severity of hsf/data/corruptions.py. Reports mAP50-95 per corruption and severity,
mPC = mean over corruptions and severities, rPC = mPC / clean (Michaelis et al. 2019), per-group means, and for
gated fusions the mean gate (event share) per corruption: a reliability-aware model should raise it as the RGB
image gets worse. Same evaluator as tools/eval_conditions.py (conf 0.001, IoU 0.7, multi-label, max 300).
Stateful models (tfs / hsf) are not supported. The event single-stream model is evaluated clean only.

Run:  python tools/eval_corruption.py --models rgb_val add_s5 --extra name=REL_PATH:FUSION ...
Writes runs/eval_corruption.json / .md (merged with earlier results).
"""
import argparse
import json
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import DetMetrics  # noqa: E402

from eval_conditions import MODELS, NAMES, build, gate_means  # noqa: E402
from hsf.data.corruptions import GROUPS, KINDS, corrupt  # noqa: E402
from hsf.data.dataset import DualDSEC  # noqa: E402
from hsf.engine import IOUV, _frame_stats  # noqa: E402

RUNS = osp.join(ROOT, 'runs')
MODELS['add_s5'] = ('dual', 'step4/dual_add_s5_cleaninit/best.pt', 'add')
MODELS['relp2_add'] = ('dual', 'step11_rel/dual_relp2_add_v2/best.pt', 'relp2_add')
MODELS['sw_add'] = ('dual', 'step12_sw/dual_sw_add_s5/best.pt', 'sw_add')
MODELS['sw2_add'] = ('dual', 'step12_sw/dual_sw2_add_s5/best.pt', 'sw2_add')


def summ(dm):
    if not dm.stats['tp']:
        return None
    dm.process()
    return round(100 * dm.box.mean_results()[3], 2)


@torch.no_grad()
def run(model, device, conds, stride, workers):
    ds = DualDSEC('test')
    sub = Subset(ds, range(0, len(ds), stride))
    dl = DataLoader(sub, batch_size=16, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
    metrics = {c: DetMetrics(names=NAMES) for c in conds}
    gates = {c: [] for c in conds}
    iouv = IOUV.to(device)
    for bi, b in enumerate(dl):
        rgb0, evt = b['rgb'].to(device), b['evt'].to(device)
        tgt = []
        for i in range(rgb0.shape[0]):
            m = b['batch_idx'] == i
            tgt.append((b['cls'][m].squeeze(-1).to(device), ops.xywh2xyxy(b['bboxes'][m].to(device) * 640)))
        for ci, cond in enumerate(conds):
            if cond == 'clean':
                rgb = rgb0
            else:
                kind, sev = cond.rsplit('@', 1)
                gen = torch.Generator(device=device).manual_seed(100003 * bi + ci)
                rgb = corrupt(rgb0, kind, int(sev), gen)
            with torch.autocast('cuda'):
                out = model(rgb, evt)
            dets = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
            g = gate_means(model)
            if g is not None:
                gates[cond] += g.tolist()
            for d, (tcls, tbox) in zip(dets, tgt):
                metrics[cond].update_stats(_frame_stats(d, tcls, tbox, iouv))
        if bi % 50 == 0:
            print(f'  batch {bi}/{len(dl)}', flush=True)
    res = {c: summ(metrics[c]) for c in conds}
    gate = {c: round(float(np.mean(v)), 4) for c, v in gates.items() if v} or None
    return res, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=['rgb_val', 'event_val', 'add_s5', 'relp2_add', 'sw_add', 'sw2_add'])
    ap.add_argument('--extra', nargs='*', default=[], metavar='KEY=REL_PATH:FUSION')
    ap.add_argument('--kinds', nargs='*', default=list(KINDS))
    ap.add_argument('--severities', nargs='*', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--stride', type=int, default=2, help='evaluate every n-th test frame')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='eval_corruption')
    a = ap.parse_args()
    for spec in a.extra:
        key, rest = spec.split('=', 1)
        rel, fusion = rest.rsplit(':', 1)
        MODELS[key] = ('dual', rel, fusion)
        a.models.append(key)
    device = torch.device('cuda')
    out_json = osp.join(RUNS, a.out + '.json')
    all_conds = ['clean'] + [f'{k}@{s}' for k in a.kinds for s in a.severities]
    for key in a.models:
        model = build(key, device)
        if model is None:
            print(f'{key}: checkpoint not found, skipped', flush=True)
            continue
        conds = ['clean'] if MODELS[key][0] == 'single_evt' else all_conds
        t0 = time.time()
        res, gate = run(model, device, conds, a.stride, a.workers)
        r = {'clean': res['clean'], 'stride': a.stride, 'corruptions': {}, 'groups': {}}
        if len(conds) > 1:
            for k in a.kinds:
                r['corruptions'][k] = [res[f'{k}@{s}'] for s in a.severities]
            vals = [v for vs in r['corruptions'].values() for v in vs]
            r['mPC'] = round(float(np.mean(vals)), 2)
            r['rPC'] = round(r['mPC'] / r['clean'], 4)
            for gname, kinds in GROUPS.items():
                vs = [v for k in kinds if k in r['corruptions'] for v in r['corruptions'][k]]
                if vs:
                    r['groups'][gname] = round(float(np.mean(vs)), 2)
        if gate:
            r['gate'] = {'clean': gate['clean']}
            for k in a.kinds:
                if f'{k}@1' in gate or len(conds) > 1:
                    r['gate'][k] = [gate.get(f'{k}@{s}') for s in a.severities]
        results = json.load(open(out_json)) if osp.isfile(out_json) else {}
        results[key] = r
        json.dump(results, open(out_json, 'w'), indent=1)
        print(f'{key:12s} clean {r["clean"]}  mPC {r.get("mPC")}  rPC {r.get("rPC")}  groups {r["groups"]}  '
              f'({time.time() - t0:.0f} s)', flush=True)
        if gate:
            print(f'{key:12s} event share: clean {gate["clean"]:.3f}  ' + '  '.join(
                f'{k} {np.mean([g for g in r["gate"][k] if g is not None]):.3f}' for k in a.kinds), flush=True)
        del model
        torch.cuda.empty_cache()

    results = json.load(open(out_json))
    groups = list(GROUPS)
    lines = ['| model | clean | mPC | rPC | ' + ' | '.join(groups) + ' |', '|---' * (4 + len(groups)) + '|']
    for key, r in results.items():
        lines.append(f'| {key} | {r["clean"]} | {r.get("mPC", "-")} | {r.get("rPC", "-")} | ' +
                     ' | '.join(str(r['groups'].get(g, '-')) for g in groups) + ' |')
    lines += ['', '| model | corruption | s1 | s2 | s3 | s4 | s5 | event share s1..s5 |', '|---|---|---|---|---|---|---|---|']
    for key, r in results.items():
        for k, vs in r['corruptions'].items():
            gs = r.get('gate', {}).get(k)
            lines.append(f'| {key} | {k} | ' + ' | '.join(str(v) for v in vs) + ' | ' +
                         (' '.join(f'{g:.2f}' for g in gs if g is not None) if gs else '-') + ' |')
    md = ('# RGB corruption robustness (DSEC-Det test, every n-th frame; event input clean)\n\n'
          'mAP50-95 (%). mPC = mean over corruptions x severities, rPC = mPC / clean. Event share = mean gate of '
          'gated fusions.\n\n' + '\n'.join(lines) + '\n')
    open(osp.join(RUNS, a.out + '.md'), 'w', encoding='utf-8').write(md)
    print('\n' + md, flush=True)


if __name__ == '__main__':
    main()
