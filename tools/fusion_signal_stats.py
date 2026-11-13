"""Measure the per-modality signal quality that the reliability-weighted fusion model is built on
(docs/fusion_model.md). CPU only.

Per labelled frame (every --stride-th) of a split:
  RGB : mean luminance mu, share of dark (< 20) / saturated (> 250) pixels
  GEAR: share of pixels with events, events per active pixel (decoded from the num channel),
        share of GEAR values at the uint8 limits (tanh saturation of h / feat / num)
Per ground-truth box (frame lighting bucket, class, size):
  RGB contrast-to-noise   CNR_r = sigma_box / sqrt(mu_box + s_read^2)   (shot-noise-limited, gray units)
  event SNR               SNR_e = (n_box - n_bg) / sqrt(n_bg + 1)        (n = events per pixel in the window,
                                                                          bg = frame pixels outside all boxes)
Run:  python tools/fusion_signal_stats.py --split test --stride 5     -> runs/diag/fusion_signal_stats.json
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.data.dataset import CFG, split_filter  # noqa: E402

C_GEAR = 8.0            # GEAR gray levels per event (hsf/data/gear.py)
S_READ2 = 4.0           # read-noise variance in gray units^2 (assumed; only shifts CNR at very low mu)


def bucket_of(lum):
    return 'night' if lum < 35 else ('dim' if lum < 55 else 'day')


def decode_num(u8):
    """GEAR num channel uint8 -> events per pixel (inverse of tanh(n * c / 255) * 255)."""
    v = np.clip(u8.astype(np.float32) / 255.0, 0, 0.999)
    return np.arctanh(v) * 255.0 / C_GEAR


def frame_stats(item):
    seq_dir, rgb_dir, i = item
    g = cv2.imread(osp.join(rgb_dir, f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gear = cv2.imread(osp.join(seq_dir, 'gear', f'{i:06d}.png'))     # written by cv2 as [h, feat, num]
    n = decode_num(gear[..., 2])
    H, W = g.shape
    mu = float(g.mean())
    out = dict(seq=osp.basename(seq_dir), idx=i, lum=mu, bucket=bucket_of(mu),
               dark=float((g < 20).mean()), sat=float((g > 250).mean()),
               ev_active=float((n > 0.5).mean()), ev_per_active=float(n[n > 0.5].mean()) if (n > 0.5).any() else 0.0,
               gear_sat_h=float(((gear[..., 0] == 0) | (gear[..., 0] == 255)).mean()),
               gear_sat_feat=float(((gear[..., 1] == 0) | (gear[..., 1] == 255)).mean()),
               gear_sat_num=float((gear[..., 2] == 255).mean()), boxes=[])
    lab = osp.join(seq_dir, 'labels', f'{i:06d}.txt')
    rows = [l.split() for l in open(lab)] if osp.isfile(lab) else []
    mask = np.zeros((H, W), bool)
    rects = []
    for r in rows:
        if not r:
            continue
        c, cx, cy, w, h = int(r[0]), *map(float, r[1:5])
        x0, x1 = int(max(0, (cx - w / 2) * W)), int(min(W, (cx + w / 2) * W))
        y0, y1 = int(max(0, (cy - h / 2) * H)), int(min(H, (cy + h / 2) * H))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        rects.append((c, x0, y0, x1, y1))
        mask[y0:y1, x0:x1] = True
    n_bg = float(n[~mask].mean()) if (~mask).any() else 0.0
    out['n_bg'] = n_bg
    for c, x0, y0, x1, y1 in rects:
        pg, pn = g[y0:y1, x0:x1], n[y0:y1, x0:x1]
        mb = float(pg.mean())
        out['boxes'].append(dict(cls=c, area=(x1 - x0) * (y1 - y0), mu=mb,
                                 cnr_r=float(pg.std() / np.sqrt(mb + S_READ2)),
                                 sat=float((pg > 250).mean()), n_box=float(pn.mean()),
                                 snr_e=float((pn.mean() - n_bg) / np.sqrt(n_bg + 1.0))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    folder, keep, drop = split_filter(a.split)
    items = []
    for mpath in sorted(glob.glob(osp.join(CFG['processed_root'], folder, '*', 'meta.json'))):
        m = json.load(open(mpath))
        if (keep and m['sequence'] not in keep) or (drop and m['sequence'] in drop):
            continue
        items += [(osp.dirname(mpath), m['rgb_dir'], i) for i in m['labelled_frames'][::a.stride]]
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(frame_stats, items))
    os.makedirs(osp.join(ROOT, 'runs', 'diag'), exist_ok=True)
    out = osp.join(ROOT, 'runs', 'diag', f'fusion_signal_stats_{a.split}.json')
    json.dump(res, open(out, 'w'))

    print(f'{len(res)} frames -> {out}')
    print(f'{"bucket":6s} {"frames":>6s} {"lum":>5s} {"sat%":>5s} {"ev_act%":>7s} {"ev/act":>6s} {"n_bg":>5s} '
          f'{"Gsat_h%":>7s} {"Gsat_f%":>7s} {"Gsat_n%":>7s} | {"boxes":>6s} {"CNR_r":>6s} {"SNR_e":>6s}')
    for b in ('day', 'dim', 'night'):
        fr = [r for r in res if r['bucket'] == b]
        if not fr:
            continue
        bx = [x for r in fr for x in r['boxes']]
        med = lambda k: np.median([r[k] for r in fr])  # noqa: E731
        print(f'{b:6s} {len(fr):6d} {med("lum"):5.1f} {100 * med("sat"):5.2f} {100 * med("ev_active"):7.1f} '
              f'{med("ev_per_active"):6.2f} {med("n_bg"):5.3f} {100 * med("gear_sat_h"):7.2f} '
              f'{100 * med("gear_sat_feat"):7.2f} {100 * med("gear_sat_num"):7.3f} | {len(bx):6d} '
              f'{np.median([x["cnr_r"] for x in bx]):6.2f} {np.median([x["snr_e"] for x in bx]):6.2f}')
    names = CFG['names']
    print('\nper class (all buckets): median CNR_r / SNR_e / area')
    for c in range(len(names)):
        bx = [x for r in res for x in r['boxes'] if x['cls'] == c]
        if bx:
            print(f'  {names[c]:11s} {len(bx):6d}  {np.median([x["cnr_r"] for x in bx]):5.2f} / '
                  f'{np.median([x["snr_e"] for x in bx]):5.2f} / {np.median([x["area"] for x in bx]):7.0f}')


if __name__ == '__main__':
    main()
