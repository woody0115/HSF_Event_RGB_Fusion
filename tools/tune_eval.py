"""Val-split evaluation for tuning fusion models toward two use cases (2026-11-17, branch multi-seed):
RGB degradation (robustness) and night, especially pedestrians. Decisions use the val split only.

For each checkpoint (optionally with parameters overridden in memory, never written back):
  clean   full val split: mAP50-95 all / day / dim / night (bucket from the mean luminance of the
          640x480 image, same thresholds as tools/eval_conditions.py), per-class AP all and night
  corrupt every --stride-th val frame, RGB corrupted by hsf/data/corruptions.py (event clean), kinds x
          --severities -> mPC (mean over all corruption conditions), per-group means
  inside  per scale P3 / P4 / P5 and condition: event reliance r_e = |neck - rgb| / |rgb|, the RGB and
          event term norms of the neck input, and the gate statistics the fusion module exposes
          (relp2: g_r / g_e means; sw2: event share)

Overrides: --set PATH=VALUE (absolute) or --add PATH=DELTA, PATH = state-dict key, optionally indexed,
e.g. fusion.0.lambda_bias_e=-1  or  fusion.0.mlp.2.bias[1]=0.5. Applied identically to every checkpoint.
--attr NAME=VALUE sets a plain attribute on every fusion module (e.g. cnr_mode=signal for relp2).
--extra rgb_zero adds a diagnostic condition with the whole RGB input set to 0 (event-only floor of the
fusion model); extra conditions are reported but not counted in mPC or the groups.

Run:  python tools/tune_eval.py --family relp2_add --tag r1_base [--set ...] [--no-corrupt]
      --split test evaluates the test split with the same metrics, for report-only numbers of a configuration
      already chosen on val (never use it for decisions)
Appends to runs/tune/<tag>.json (one entry per checkpoint) and prints a summary.
"""
import argparse
import json
import os
import os.path as osp
import re
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

from eval_conditions import NAMES, Single, bucket_of, gate_means  # noqa: E402
from hsf.data.corruptions import GROUPS, KINDS, corrupt  # noqa: E402
from hsf.data.dataset import DualDSEC  # noqa: E402
from hsf.engine import IOUV, _frame_stats  # noqa: E402
from hsf.models.dual_yolov8 import DualYOLOv8, load_ultralytics  # noqa: E402

RUNS = osp.join(ROOT, 'runs')
BUCKETS = ('all', 'day', 'dim', 'night')
# seed 0 / 1 / 2 checkpoints of the four main fusions (runs/seed_runs.sh)
FAMILIES = {
    'add': ['step4/dual_add_s5_cleaninit', 'step4/dual_add_s5_seed1', 'step4/dual_add_s5_seed2'],
    'relp2_add': ['step11_rel/dual_relp2_add_v2', 'step11_rel/dual_relp2_add_s5_seed1', 'step11_rel/dual_relp2_add_s5_seed2'],
    'sw2_add': ['step12_sw/dual_sw2_add_s5', 'step12_sw/dual_sw2_add_s5_seed1', 'step12_sw/dual_sw2_add_s5_seed2'],
    'cafr': ['step13_cafr/dual_cafr_s5', 'step13_cafr/dual_cafr_s5_seed1', 'step13_cafr/dual_cafr_s5_seed2'],
}


def apply_overrides(model, sets, adds):
    """In-memory parameter / buffer overrides; returns the list of (path, old, new) for the record."""
    sd = dict(model.named_parameters())
    sd.update(dict(model.named_buffers()))
    done = []
    for spec, additive in [(s, False) for s in sets] + [(s, True) for s in adds]:
        path, val = spec.split('=', 1)
        m = re.fullmatch(r'(.+?)(?:\[(\d+)\])?', path)
        key, idx = m.group(1), m.group(2)
        if key not in sd:
            raise KeyError(f'{key} not in the model (e.g. {[k for k in sd if "fusion" in k][:8]})')
        t = sd[key]
        with torch.no_grad():
            view = t.view(-1)[int(idx)] if idx is not None else t
            old = view.detach().clone()
            view.copy_(old + float(val) if additive else torch.full_like(old, float(val)))
        done.append((path, old.flatten()[:4].tolist(), view.detach().flatten()[:4].tolist()))
    return done


class Inside:
    """Forward hooks on the fusion modules: per-image r_e and term norms of the neck input per scale."""

    def __init__(self, model):
        self.rows = []          # per batch: (B, 3 scales, 3) = r_e, |rgb|, |neck - rgb|
        self.cur = []
        for f in getattr(model, 'fusion', []):
            f.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        rgb = inp[1].float()
        neck = out[2].float()
        n_r = rgb.flatten(1).norm(dim=1)
        n_d = (neck - rgb).flatten(1).norm(dim=1)
        self.cur.append(torch.stack([n_d / n_r.clamp(min=1e-6), n_r, n_d], 1).cpu())

    def pop(self):
        x = torch.stack(self.cur, 1) if self.cur else None      # (B, scales, 3)
        self.cur = []
        return x


def terms(model):
    """Batch-level gate statistics the fusion modules keep after an eval forward."""
    out = {}
    for k, f in enumerate(getattr(model, 'fusion', [])):
        for name, v in (getattr(f, 'last_terms', None) or {}).items():
            out[f'P{k + 3}_{name}'] = v
    return out


def luminance(rgb):
    img = rgb[:, :, 80:560]                       # letterboxed 640x480 -> rows 80..559
    return (0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]).mean((1, 2)) * 255


def summ(dm, per_class=False):
    if not dm.stats['tp'] or not len(np.concatenate([np.atleast_1d(t) for t in dm.stats['target_cls']])):
        return None
    dm.process()
    r = {'mAP50-95': round(100 * float(dm.box.mean_results()[3]), 2)}
    if per_class:
        r['per_class'] = {NAMES[int(c)]: round(100 * float(dm.box.ap[j]), 2) for j, c in enumerate(dm.box.ap_class_index)}
    return r


@torch.no_grad()
def eval_clean(model, inside, workers, split='val'):
    ds = DualDSEC(split)
    dl = DataLoader(ds, batch_size=16, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
    dev = next(model.parameters()).device
    iouv = IOUV.to(dev)
    met = {b: DetMetrics(names=NAMES) for b in BUCKETS}
    ins = {b: [] for b in BUCKETS}
    gts = {b: np.zeros(len(NAMES), int) for b in BUCKETS}
    trm = {b: [] for b in BUCKETS}
    gate = {b: [] for b in BUCKETS}
    for b in dl:
        rgb, evt = b['rgb'].to(dev), b['evt'].to(dev)
        with torch.autocast('cuda'):
            out = model(rgb, evt)
        x = inside.pop()
        g = gate_means(model)
        t = terms(model)
        lum = luminance(rgb).cpu()
        dets = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
        bks = set()
        for i, d in enumerate(dets):
            bk = bucket_of(float(lum[i]))
            bks.add(bk)
            m = b['batch_idx'] == i
            tcls = b['cls'][m].squeeze(-1)
            st = _frame_stats(d, tcls.to(dev), ops.xywh2xyxy(b['bboxes'][m].to(dev) * 640), iouv)
            for k in ('all', bk):
                met[k].update_stats(st)
                ins[k].append(x[i] if x is not None else torch.zeros(0, 3))
                np.add.at(gts[k], tcls.long().numpy(), 1)
                if g is not None:
                    gate[k].append(float(g[i]))
        if t:   # batch-level terms: attribute to the majority bucket of the batch and to 'all'
            maj = max(bks, key=lambda k: sum(bucket_of(float(v)) == k for v in lum))
            trm['all'].append(t)
            trm[maj].append(t)
    res = {}
    for k in BUCKETS:
        r = summ(met[k], per_class=True)
        if r is None:
            continue
        r['frames'] = len(ins[k])
        r['gt'] = {NAMES[c]: int(n) for c, n in enumerate(gts[k]) if n}
        a = torch.stack(ins[k]).numpy()                       # (N, scales, 3); scales = 0 for single-stream
        r['inside'] = {f'P{s + 3}': {'r_e': round(float(a[:, s, 0].mean()), 4),
                                     'rgb_norm': round(float(a[:, s, 1].mean()), 2),
                                     'evt_term_norm': round(float(a[:, s, 2].mean()), 2)} for s in range(a.shape[1])}
        if gate[k]:
            r['gate'] = round(float(np.mean(gate[k])), 4)
        if trm[k]:
            r['terms'] = {n: round(float(np.mean([t[n] for t in trm[k] if n in t])), 4) for n in trm[k][0]}
        res[k] = r
    return res


@torch.no_grad()
def eval_corrupt(model, inside, stride, kinds, sevs, workers, extra=(), split='val'):
    ds = DualDSEC(split)
    sub = Subset(ds, range(0, len(ds), stride))
    dl = DataLoader(sub, batch_size=16, num_workers=workers, pin_memory=True, collate_fn=DualDSEC.collate)
    dev = next(model.parameters()).device
    iouv = IOUV.to(dev)
    conds = ['clean'] + [f'{k}@{s}' for k in kinds for s in sevs]
    n_main = len(conds)
    conds += list(extra)
    met = {c: DetMetrics(names=NAMES) for c in conds}
    ins = {c: [] for c in conds}
    gate = {c: [] for c in conds}
    trm = {c: [] for c in conds}
    for bi, b in enumerate(dl):
        rgb0, evt = b['rgb'].to(dev), b['evt'].to(dev)
        tgt = []
        for i in range(rgb0.shape[0]):
            m = b['batch_idx'] == i
            tgt.append((b['cls'][m].squeeze(-1).to(dev), ops.xywh2xyxy(b['bboxes'][m].to(dev) * 640)))
        for ci, c in enumerate(conds):
            if c == 'clean':
                rgb = rgb0
            elif c == 'rgb_zero':
                rgb = torch.zeros_like(rgb0)
            else:
                kind, sev = c.rsplit('@', 1)
                gen = torch.Generator(device=dev).manual_seed(100003 * bi + ci)
                rgb = corrupt(rgb0, kind, int(sev), gen)
            with torch.autocast('cuda'):
                out = model(rgb, evt)
            x = inside.pop()
            ins[c].append(x if x is not None else torch.zeros(rgb.shape[0], 0, 3))
            g = gate_means(model)
            if g is not None:
                gate[c] += g.tolist()
            t = terms(model)
            if t:
                trm[c].append(t)
            dets = ops.non_max_suppression(out[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
            for d, (tcls, tbox) in zip(dets, tgt):
                met[c].update_stats(_frame_stats(d, tcls, tbox, iouv))
    res = {'stride': stride, 'severities': sevs, 'conds': {}}
    for c in conds:
        s = summ(met[c])
        a = torch.cat(ins[c]).numpy()
        e = {'mAP50-95': s['mAP50-95'] if s else None,
                  'r_e': [round(float(a[:, k, 0].mean()), 4) for k in range(a.shape[1])]}
        if gate[c]:
            e['gate'] = round(float(np.mean(gate[c])), 4)
        if trm[c]:
            e['terms'] = {n: round(float(np.mean([t[n] for t in trm[c] if n in t])), 4) for n in trm[c][0]}
        res['conds'][c] = e
    vals = [res['conds'][c]['mAP50-95'] for c in conds[1:n_main]]
    res['mPC'] = round(float(np.mean(vals)), 2)
    res['groups'] = {}
    for gname, ks in GROUPS.items():
        v = [res['conds'][f'{k}@{s}']['mAP50-95'] for k in ks if k in kinds for s in sevs]
        if v:
            res['groups'][gname] = round(float(np.mean(v)), 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=list(FAMILIES), help='all three seeds of a fusion')
    ap.add_argument('--ckpt', nargs='*', default=[], metavar='REL_RUN_DIR:FUSION[:INIT_RUN]',
                    help='extra run dirs under runs/; INIT_RUN = the run a fine-tuned checkpoint started from '
                         '(stored as "init", used by summarize_tune.py --ref for paired deltas)')
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--add', nargs='*', default=[])
    ap.add_argument('--attr', nargs='*', default=[], help='NAME=VALUE set on every fusion module')
    ap.add_argument('--extra', nargs='*', default=[], choices=['rgb_zero'])
    ap.add_argument('--no-corrupt', action='store_true')
    ap.add_argument('--no-clean', action='store_true')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--kinds', nargs='*', default=list(KINDS))
    ap.add_argument('--severities', nargs='*', type=int, default=[1, 3, 5])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--split', default='val', choices=['val', 'test'], help='test = report only')
    a = ap.parse_args()
    jobs = [(r, a.family, None) for r in FAMILIES[a.family]] if a.family else []
    jobs += [tuple((c.split(':') + [None])[:3]) for c in a.ckpt]
    dev = torch.device('cuda')
    os.makedirs(osp.join(RUNS, 'tune'), exist_ok=True)
    out_json = osp.join(RUNS, 'tune', f'{a.tag}.json')
    for run, fusion, init in jobs:
        single = fusion in ('single_rgb', 'single_evt')      # run = checkpoint file under runs/
        path = osp.join(RUNS, run) if single else osp.join(RUNS, run, a.weights)
        if not osp.isfile(path):
            print(f'{run}: {path} not found, skipped', flush=True)
            continue
        if single:
            m = Single(load_ultralytics(path), fusion == 'single_evt').to(dev).eval()
        else:
            m = DualYOLOv8(nc=8, fusion=fusion)
            m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
            m = m.to(dev).eval()
        ov = apply_overrides(m, a.set, a.add)
        for spec in a.attr:
            name, val = spec.split('=', 1)
            for f in getattr(m, 'fusion', []):
                if not hasattr(f, name):
                    raise AttributeError(f'{type(f).__name__} has no attribute {name}')
                setattr(f, name, type(getattr(f, name))(val))
            ov.append((f'attr {name}', None, val))
        inside = Inside(m)
        t0 = time.time()
        r = {'run': run, 'fusion': fusion, 'weights': a.weights, 'overrides': ov, 'init': init, 'split': a.split}
        if not a.no_clean:
            r['clean'] = eval_clean(m, inside, a.workers, a.split)
        if not a.no_corrupt:
            r['corrupt'] = eval_corrupt(m, inside, a.stride, a.kinds, a.severities, a.workers, a.extra, a.split)
        r['seconds'] = round(time.time() - t0)
        results = json.load(open(out_json)) if osp.isfile(out_json) else {}
        results[run] = r
        json.dump(results, open(out_json, 'w'), indent=1)
        c = r.get('clean', {})
        msg = f'{run}: ' + '  '.join(f'{k} {c[k]["mAP50-95"]:.2f}' for k in BUCKETS if k in c)
        if 'night' in c:
            msg += f'  night-ped {c["night"]["per_class"].get("pedestrian", float("nan")):.2f}'
        if 'corrupt' in r:
            msg += f'  val-mPC {r["corrupt"]["mPC"]:.2f} ({r["corrupt"]["conds"]["clean"]["mAP50-95"]:.2f} clean-sub)'
            for x in a.extra:
                msg += f'  {x} {r["corrupt"]["conds"][x]["mAP50-95"]:.2f}'
        print(msg + f'  ({r["seconds"]} s)', flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
