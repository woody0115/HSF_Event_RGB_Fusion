"""Why do some fusions vary so much between test sequences? (2026-11-15)

Phase 'dump' (GPU): for every model, per-frame detections on the whole DSEC-Det test split (same evaluator as
tools/eval_conditions.py: conf 0.001, NMS IoU 0.7, multi-label, max 300) and, for dual-stream models, the event
reliance per image  r_e = ||neck_in - F_rgb|| / ||F_rgb||  (mean over P3-P5: how much the neck input differs from
the RGB-only feature). Optional ablations: event input set to "no events" (evt0) or RGB input set to the letterbox
grey (rgb0), which show how much each model depends on each stream, per sequence.
    -> runs/diag/preds/<key>.pt

Phase 'analyze' (CPU): from the dumps,
  1. per-sequence mAP50-95 with a 95 % block-bootstrap interval (blocks of 50 consecutive frames), and the same for
     the difference to a reference model: tells sampling noise from real differences;
  2. chunk level (50 consecutive frames): delta mAP (model - reference) against the scene signals of
     runs/diag/fusion_signal_stats_test.json (luminance, dark / saturated fraction, event activity, event noise floor
     n_bg, per-box RGB contrast-to-noise CNR_r and event SNR_e, box size) and against the event reliance r_e,
     Spearman rank correlation + quartile tables;
  3. error decomposition per sequence at score >= 0.3 (TIDE-like): GT found / mislocalised (same class, 0.1 <= IoU
     < 0.5) / wrong class (IoU >= 0.5) / missed, and background false positives per frame, by object size;
  4. the chunks where the fusion loses most against the reference, rendered (RGB with both models' boxes, event
     frame) for visual inspection.
    -> runs/diag/diagnose.md / .json, runs/diag/worst/*.jpg

Run:  python tools/diagnose_conditions.py dump --models rgb_val event_val add_s5 ... [--ablate evt0 rgb0]
      python tools/diagnose_conditions.py analyze --ref rgb_val --models add_s5 rel2_add ...
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import DetMetrics, box_iou  # noqa: E402

from eval_conditions import MODELS, NAMES, build  # noqa: E402
from hsf.data.dataset import DualDSEC  # noqa: E402
from hsf.engine import IOUV, _frame_stats  # noqa: E402

RUNS = osp.join(ROOT, 'runs')
DIAG = osp.join(RUNS, 'diag')
PRED = osp.join(DIAG, 'preds')
MODELS.update({
    'add_s5': ('dual', 'step4/dual_add_s5_cleaninit/best.pt', 'add'),
    'rel_add': ('dual', 'step11_rel/dual_rel_add_v1/best.pt', 'rel_add'),
    'rel2_add': ('dual', 'step11_rel/dual_rel2_add_v2/best.pt', 'rel2_add'),
    'relp2_add': ('dual', 'step11_rel/dual_relp2_add_v2/best.pt', 'relp2_add'),
    'sw_add': ('dual', 'step12_sw/dual_sw_add_s5/best.pt', 'sw_add'),
    'sw2_add': ('dual', 'step12_sw/dual_sw2_add_s5/best.pt', 'sw2_add'),
    'cafr': ('dual', 'step13_cafr/dual_cafr_s5/best.pt', 'cafr'),
    'cafrx': ('dual', 'step13_cafr/dual_cafrx_s5/best.pt', 'cafrx'),
})
CHUNK = 50
SIZES = {'small': (0, 32 ** 2), 'medium': (32 ** 2, 96 ** 2), 'large': (96 ** 2, 1e12)}


# ----------------------------------------------------------------------------------------------- dump (GPU)
class RelianceHook:
    """Records ||neck_in - rgb|| / ||rgb|| per image for every fusion module of a DualYOLOv8."""

    def __init__(self, model):
        self.vals, self.hs = [], []
        for f in getattr(model, 'fusion', []):
            self.hs.append(f.register_forward_hook(self.hook))

    def hook(self, mod, inp, out):
        rgb, neck = inp[1].float(), out[2].float()
        self.vals.append(((neck - rgb).flatten(1).norm(dim=1) / rgb.flatten(1).norm(dim=1).clamp(min=1e-6)).cpu())

    def pop(self):
        v = torch.stack(self.vals).mean(0) if self.vals else None
        self.vals = []
        return v


@torch.no_grad()
def dump(key, device, ablate, workers, limit=0):
    model = build(key, device)
    if model is None:
        print(f'{key}: checkpoint not found, skipped', flush=True)
        return
    if getattr(model, 'stateful', False):
        print(f'{key}: stateful models are not supported, skipped', flush=True)
        return
    dual = MODELS[key][0] == 'dual'
    variants = [''] + (list(ablate) if dual else [])
    hook = RelianceHook(model) if dual else None
    ds = DualDSEC('test')
    if limit:                                   # smoke test on the first frames
        ds.items = ds.items[:limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=workers, pin_memory=True,
                                     collate_fn=DualDSEC.collate)
    out = {v: {'dets': [], 'rel': []} for v in variants}
    gts, keys = [], []
    t0 = time.time()
    for bi, b in enumerate(dl):
        rgb, evt = b['rgb'].to(device), b['evt'].to(device)
        for i in range(rgb.shape[0]):
            m = b['batch_idx'] == i
            gts.append(torch.cat([b['cls'][m], ops.xywh2xyxy(b['bboxes'][m] * 640)], 1).half())
        for v in variants:
            r, e = rgb, evt
            if v == 'evt0':        # "no events" inside the frame: tensor channels [num, feat, h] = (0, 128, 128) / 255
                e = evt.clone()
                e[:, 0, 80:560] = 0.0
                e[:, 1:, 80:560] = 128 / 255
            elif v == 'rgb0':      # RGB without information: uniform letterbox grey
                r = torch.full_like(rgb, 114 / 255)
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                o = model(r, e)
            dets = ops.non_max_suppression(o[0].float(), 0.001, 0.7, multi_label=True, max_det=300)
            out[v]['dets'] += [d.half().cpu() for d in dets]
            if hook is not None:
                rel = hook.pop()
                out[v]['rel'] += rel.tolist() if rel is not None else [float('nan')] * len(dets)
        if bi % 200 == 0:
            print(f'  {key} batch {bi}/{len(dl)}', flush=True)
    for it in ds.items:
        keys.append((osp.basename(it[0]), int(it[2])))
    os.makedirs(PRED, exist_ok=True)
    for v in variants:
        name = key + (f'@{v}' if v else '')
        torch.save({'keys': keys, 'gt': gts, 'dets': out[v]['dets'], 'rel': out[v]['rel'] or None},
                   osp.join(PRED, name + '.pt'))
    print(f'{key}: dumped {len(keys)} frames, variants {variants} ({time.time() - t0:.0f} s)', flush=True)


# -------------------------------------------------------------------------------------------- analyze (CPU)
def frame_stats(dets, gts):
    iouv = IOUV
    return [_frame_stats(d.float(), g[:, 0].float(), g[:, 1:].float(), iouv) for d, g in zip(dets, gts)]


def map5095(stats):
    dm = DetMetrics(names=NAMES)
    for s in stats:
        dm.update_stats(s)
    if not dm.stats['tp'] or not len(np.concatenate([np.atleast_1d(t) for t in dm.stats['target_cls']])):
        return float('nan')
    dm.process()
    return 100 * dm.box.mean_results()[3]


def block_boot(idx, n_boot, rng):
    """Resample blocks of CHUNK consecutive frame indices (frames inside a sequence are strongly correlated)."""
    blocks = [idx[i:i + CHUNK] for i in range(0, len(idx), CHUNK)]
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        yield [j for p in pick for j in blocks[p]]


def errors(dets, gts, thr=0.3):
    """TIDE-like decomposition at score >= thr. Returns counts per size bucket and background FP."""
    res = {s: dict(gt=0, found=0, loc=0, cls=0, miss=0) for s in SIZES}
    fp_bg = 0
    for d, g in zip(dets, gts):
        d = d[d[:, 4] >= thr].float()
        g = g.float()
        area = (g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2])
        iou = box_iou(g[:, 1:], d[:, :4]) if len(d) and len(g) else torch.zeros(len(g), len(d))
        same = (g[:, :1] == d[None, :, 5]) if len(d) and len(g) else torch.zeros(len(g), len(d), dtype=torch.bool)
        for j in range(len(g)):
            s = next(k for k, (lo, hi) in SIZES.items() if lo <= area[j] < hi)
            res[s]['gt'] += 1
            if len(d) and ((iou[j] >= 0.5) & same[j]).any():
                res[s]['found'] += 1
            elif len(d) and ((iou[j] >= 0.1) & same[j]).any():
                res[s]['loc'] += 1
            elif len(d) and (iou[j] >= 0.5).any():
                res[s]['cls'] += 1
            else:
                res[s]['miss'] += 1
        if len(d):
            fp_bg += int((iou.amax(0) < 0.1).sum()) if len(g) else len(d)
    return res, fp_bg


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float('nan')
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def chunk_signals(keys):
    """Scene signals per chunk from runs/diag/fusion_signal_stats_test.json (a subset of frames per sequence)."""
    stats = json.load(open(osp.join(DIAG, 'fusion_signal_stats_test.json')))
    by = {(s['seq'], s['idx']): s for s in stats}
    sig = {}
    for (seq, c), members in chunks_of(keys).items():
        fr = [by[keys[i]] for i in members if keys[i] in by]
        if not fr:
            continue
        bx = [b for f in fr for b in f['boxes']]
        med = lambda v: float(np.median(v)) if len(v) else float('nan')     # noqa: E731
        sig[(seq, c)] = {'lum': med([f['lum'] for f in fr]), 'dark': med([f['dark'] for f in fr]),
                         'sat': float(np.mean([f['sat'] for f in fr])), 'ev_active': med([f['ev_active'] for f in fr]),
                         'n_bg': med([f['n_bg'] for f in fr]), 'cnr_r': med([b['cnr_r'] for b in bx]),
                         'snr_e': med([b['snr_e'] for b in bx]), 'box_area': med([b['area'] for b in bx])}
    return sig


def chunks_of(keys):
    ch, pos = {}, {}
    for i, (seq, idx) in enumerate(keys):
        n = pos.get(seq, 0)
        pos[seq] = n + 1
        ch.setdefault((seq, n // CHUNK), []).append(i)
    return ch


def render_worst(ref, mod, key, worst, keys, out_dir):
    import cv2
    ds = DualDSEC('test')
    pos = {(osp.basename(it[0]), int(it[2])): i for i, it in enumerate(ds.items)}
    os.makedirs(out_dir, exist_ok=True)
    for rank, (seq, c, members, dm) in enumerate(worst):
        i = members[len(members) // 2]
        r_t, e_t, _ = ds[pos[keys[i]]]
        b = {'rgb': r_t, 'evt': e_t}
        rgb = (b['rgb'].permute(1, 2, 0).numpy()[80:560, :, ::-1] * 255).astype(np.uint8).copy()
        evt = b['evt'].permute(1, 2, 0).numpy()[80:560]
        ev = ((evt[:, :, 0] * 2 + np.abs(evt[:, :, 1] - 128 / 255) * 4).clip(0, 1) * 255).astype(np.uint8)   # num + |feat|
        ev = cv2.applyColorMap(ev, cv2.COLORMAP_INFERNO)
        a, bb = rgb.copy(), rgb.copy()
        for img, d in ((a, ref['dets'][i]), (bb, mod['dets'][i])):
            for g in ref['gt'][i].float():
                x1, y1, x2, y2 = [int(v) for v in g[1:]]
                cv2.rectangle(img, (x1, y1 - 80), (x2, y2 - 80), (0, 255, 0), 1)
            for det in d.float():
                if det[4] < 0.3:
                    continue
                x1, y1, x2, y2 = [int(v) for v in det[:4]]
                cv2.rectangle(img, (x1, y1 - 80), (x2, y2 - 80), (0, 0, 255), 2)
        for img, t in ((a, 'reference'), (bb, key), (ev, 'events')):
            cv2.putText(img, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        top = np.concatenate([a, bb, ev], 1)
        cv2.putText(top, f'{seq} chunk {c}  delta {dm:+.1f}', (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)
        cv2.imwrite(osp.join(out_dir, f'{key}_{rank:02d}_{seq}_c{c}.jpg'), top, [cv2.IMWRITE_JPEG_QUALITY, 85])


def analyze(ref_key, model_keys, n_boot, n_worst):
    rng = np.random.default_rng(0)
    load = lambda k: torch.load(osp.join(PRED, k + '.pt'), weights_only=False)     # noqa: E731
    ref = load(ref_key)
    keys = [tuple(k) for k in ref['keys']]
    seqs = sorted({k[0] for k in keys})
    seq_idx = {s: [i for i, k in enumerate(keys) if k[0] == s] for s in seqs}
    chunks = chunks_of(keys)
    sig = chunk_signals(keys)
    print('frame stats ...', flush=True)
    st = {ref_key: frame_stats(ref['dets'], ref['gt'])}
    preds = {ref_key: ref}
    for k in model_keys:
        if osp.isfile(osp.join(PRED, k + '.pt')):
            preds[k] = load(k)
            st[k] = frame_stats(preds[k]['dets'], preds[k]['gt'])
        else:
            print(f'{k}: no dump, skipped', flush=True)
    names = [k for k in [ref_key] + model_keys if k in st]
    report = {'ref': ref_key, 'sequences': {}, 'chunks': {}, 'corr': {}, 'errors': {}, 'quartiles': {}}

    # 1. per sequence with bootstrap
    print('bootstrap ...', flush=True)
    for s in seqs:
        idx = seq_idx[s]
        boots = list(block_boot(idx, n_boot, rng))
        row = {}
        for k in names:
            v = map5095([st[k][i] for i in idx])
            bs = np.array([map5095([st[k][i] for i in bi]) for bi in boots])
            entry = {'mAP': round(v, 2), 'ci': [round(float(np.nanpercentile(bs, 2.5)), 2),
                                                round(float(np.nanpercentile(bs, 97.5)), 2)]}
            if k != ref_key:
                br = np.array([map5095([st[ref_key][i] for i in bi]) for bi in boots]) if 'br' not in row else row['br']
                row['br'] = br
                dv = bs - br
                entry['delta'] = round(v - row[ref_key]['mAP'], 2)
                entry['delta_ci'] = [round(float(np.nanpercentile(dv, 2.5)), 2),
                                     round(float(np.nanpercentile(dv, 97.5)), 2)]
            if preds[k].get('rel'):
                entry['rel_e'] = round(float(np.nanmean([preds[k]['rel'][i] for i in idx])), 3)
            row[k] = entry
        row.pop('br', None)
        report['sequences'][s] = {'frames': len(idx), **row}
        print(f'  {s}: ' + '  '.join(f'{k} {row[k]["mAP"]:.1f}' for k in names), flush=True)

    # 1b. per-class AP per sequence: a class with a handful of boxes weighs as much as cars in the sequence mean
    print('per class ...', flush=True)
    report['per_class'] = {}
    for s in seqs:
        idx = seq_idx[s]
        gcls = np.concatenate([ref['gt'][i][:, 0].numpy() for i in idx]).astype(int)
        row = {'gt': {NAMES[c]: int((gcls == c).sum()) for c in range(len(NAMES)) if (gcls == c).sum()}}
        for k in names:
            dm = DetMetrics(names=NAMES)
            for i in idx:
                dm.update_stats(st[k][i])
            dm.process()
            row[k] = {NAMES[c]: round(100 * float(dm.box.ap[j]), 1) for j, c in enumerate(dm.box.ap_class_index)}
        report['per_class'][s] = row

    # 2. chunks: delta vs scene signals and event reliance
    print('chunks ...', flush=True)
    ch_rows = []
    for (s, c), members in chunks.items():
        ngt = sum(len(ref['gt'][i]) for i in members)
        if ngt < 20:
            continue
        r = {'seq': s, 'chunk': c, 'n_gt': ngt, **sig.get((s, c), {})}
        r[ref_key] = map5095([st[ref_key][i] for i in members])
        for k in names[1:]:
            r[k] = map5095([st[k][i] for i in members])
            if preds[k].get('rel'):
                r[k + ':rel_e'] = float(np.nanmean([preds[k]['rel'][i] for i in members]))
        r['_members'] = members
        ch_rows.append(r)
    sig_names = ['lum', 'dark', 'sat', 'ev_active', 'n_bg', 'cnr_r', 'snr_e', 'box_area']
    for k in names[1:]:
        d = [r[k] - r[ref_key] for r in ch_rows]
        cr = {sn: round(spearman([r.get(sn, np.nan) for r in ch_rows], d), 3) for sn in sig_names}
        if any(k + ':rel_e' in r for r in ch_rows):
            cr['rel_e'] = round(spearman([r.get(k + ':rel_e', np.nan) for r in ch_rows], d), 3)
        report['corr'][k] = cr
        q = {}
        for sn in ('n_bg', 'snr_e', 'lum', 'cnr_r', 'ev_active'):
            x = np.array([r.get(sn, np.nan) for r in ch_rows], float)
            ok = np.isfinite(x)
            if ok.sum() < 8:
                continue
            edges = np.nanpercentile(x[ok], [25, 50, 75])
            bins = np.digitize(x, edges)
            q[sn] = {'edges': [round(float(e), 3) for e in edges],
                     'delta': [round(float(np.nanmean(np.array(d)[ok & (bins == b)])), 2) for b in range(4)]}
        report['quartiles'][k] = q
    report['chunks'] = [{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items() if kk != '_members'}
                        for r in ch_rows]

    # 3. error decomposition per sequence
    print('errors ...', flush=True)
    for k in names:
        report['errors'][k] = {}
        for s in seqs + ['all']:
            idx = seq_idx[s] if s != 'all' else list(range(len(keys)))
            res, fp = errors([preds[k]['dets'][i] for i in idx], [ref['gt'][i] for i in idx])
            tot = {f: sum(res[z][f] for z in SIZES) for f in ('gt', 'found', 'loc', 'cls', 'miss')}
            report['errors'][k][s] = {'fp_per_frame': round(fp / len(idx), 3), 'total': tot, 'by_size': res}

    # 4. worst chunks rendered
    for k in names[1:]:
        worst = sorted(((r['seq'], r['chunk'], r['_members'], r[k] - r[ref_key]) for r in ch_rows),
                       key=lambda t: t[3])[:n_worst]
        try:
            render_worst(preds[ref_key], preds[k], k, worst, keys, osp.join(DIAG, 'worst'))
        except Exception as e:  # rendering is optional
            print(f'render {k}: {e}', flush=True)

    json.dump(report, open(osp.join(DIAG, 'diagnose.json'), 'w'), indent=1)
    write_md(report, names, seqs)


def write_md(rep, names, seqs):
    ref = rep['ref']
    cols = list(dict.fromkeys(c for cr in rep['corr'].values() for c in cr))
    L = [f'# Per-sequence diagnosis (reference: {ref})', '',
         '## 1. mAP50-95 per sequence, 95 % block-bootstrap interval (blocks of 50 frames)', '',
         '| sequence | frames | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for s in seqs:
        r = rep['sequences'][s]
        cells = []
        for k in names:
            e = r[k]
            c = f'{e["mAP"]:.1f} [{e["ci"][0]:.0f},{e["ci"][1]:.0f}]'
            if 'delta' in e:
                sig = '*' if e['delta_ci'][0] > 0 or e['delta_ci'][1] < 0 else ''
                c += f'<br>Δ {e["delta"]:+.1f} [{e["delta_ci"][0]:+.0f},{e["delta_ci"][1]:+.0f}]{sig}'
            if 'rel_e' in e:
                c += f'<br>r_e {e["rel_e"]:.2f}'
            cells.append(c)
        L.append(f'| {s} | {r["frames"]} | ' + ' | '.join(cells) + ' |')
    L += ['', '\\* = the 95 % interval of Δ (model − reference) excludes 0. r_e = event reliance '
          '||neck_in − F_rgb|| / ||F_rgb||.', '',
          f'## 2. What explains Δ (model − {ref}) per chunk of {CHUNK} frames: Spearman ρ', '',
          '| model | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    for k, cr in rep['corr'].items():
        L.append(f'| {k} | ' + ' | '.join(f'{cr[c]:+.2f}' if c in cr and cr[c] == cr[c] else '-' for c in cols) + ' |')
    L += ['', '## 2b. AP50-95 per class and sequence (GT boxes in parentheses)', '',
          'A class with a handful of boxes counts as much as cars in the per-sequence mean.', '',
          '| sequence | model | ' + ' | '.join(NAMES.values()) + ' |', '|---|---|' + '---|' * len(NAMES)]
    for s in seqs:
        pc = rep['per_class'][s]
        L.append(f'| {s} | GT | ' + ' | '.join(str(pc['gt'].get(n, '-')) for n in NAMES.values()) + ' |')
        for k in names:
            L.append(f'| | {k} | ' + ' | '.join(str(pc[k].get(n, '-')) for n in NAMES.values()) + ' |')
    L += ['', 'Mean Δ per quartile of a scene signal (Q1 = lowest):', '',
          '| model | signal | Q1 | Q2 | Q3 | Q4 | edges |', '|---|---|---|---|---|---|---|']
    for k, q in rep['quartiles'].items():
        for sn, v in q.items():
            L.append(f'| {k} | {sn} | ' + ' | '.join(f'{x:+.1f}' for x in v['delta']) + f' | {v["edges"]} |')
    L += ['', '## 3. Error decomposition at score ≥ 0.3 (share of GT; background FP per frame)', '',
          '| model | sequence | found | mislocalised | wrong class | missed | bg FP / frame |',
          '|---|---|---|---|---|---|---|']
    for k in names:
        for s in seqs + ['all']:
            e = rep['errors'][k][s]
            t = e['total']
            g = max(t['gt'], 1)
            L.append(f'| {k} | {s} | {100 * t["found"] / g:.1f} | {100 * t["loc"] / g:.1f} | '
                     f'{100 * t["cls"] / g:.1f} | {100 * t["miss"] / g:.1f} | {e["fp_per_frame"]:.2f} |')
    L += ['', 'Worst chunks (largest loss against the reference): runs/diag/worst/*.jpg '
          '(green = GT, red = detections with score ≥ 0.3).', '']
    open(osp.join(DIAG, 'diagnose.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L[:40]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['dump', 'analyze'])
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--ablate', nargs='*', default=[], choices=['evt0', 'rgb0'])
    ap.add_argument('--ref', default='rgb_val')
    ap.add_argument('--boot', type=int, default=100)
    ap.add_argument('--worst', type=int, default=6)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='dump only the first n test frames (smoke test)')
    a = ap.parse_args()
    if a.phase == 'dump':
        dev = torch.device(a.device)
        for k in a.models:
            dump(k, dev, a.ablate, a.workers, a.limit)
            torch.cuda.empty_cache()
    else:
        analyze(a.ref, a.models, a.boot, a.worst)


if __name__ == '__main__':
    main()
