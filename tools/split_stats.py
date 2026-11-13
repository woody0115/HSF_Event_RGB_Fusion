"""Per-drive statistics of the 47 training sequences, used to choose a whole-drive validation split.

A DSEC sequence name <city>_<NN>_<x> is segment <x> of drive <city>_<NN>; segments of one drive share
route, weather and lighting, so validation must hold out whole drives (see CHANGELOG 2026-11-14).
Per sequence: labelled frames, median RGB luminance (every 10th labelled frame) and boxes per class.

Run:  python tools/split_stats.py        writes runs/diag/split_stats.json and prints a per-drive table
"""
import glob
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.data.dataset import CFG  # noqa: E402

NAMES = CFG['names']


def seq_stats(mpath):
    m = json.load(open(mpath))
    seq_dir = osp.dirname(mpath)
    frames = m['labelled_frames']
    boxes = np.zeros(len(NAMES), int)
    for i in frames:
        p = osp.join(seq_dir, 'labels', f'{i:06d}.txt')
        if osp.isfile(p):
            for line in open(p):
                if line.strip():
                    boxes[int(line.split()[0])] += 1
    lum = [float(cv2.imread(osp.join(m['rgb_dir'], f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE).mean())
           for i in frames[::10]]
    return m['sequence'], {'frames': len(frames), 'lum_median': float(np.median(lum)) if lum else None,
                           'boxes': boxes.tolist()}


def main():
    metas = sorted(glob.glob(osp.join(CFG['processed_root'], 'train', '*', 'meta.json')))
    with ThreadPoolExecutor(8) as ex:
        stats = dict(ex.map(seq_stats, metas))
    os.makedirs(osp.join(ROOT, 'runs', 'diag'), exist_ok=True)
    json.dump(stats, open(osp.join(ROOT, 'runs', 'diag', 'split_stats.json'), 'w'), indent=1)

    drives = defaultdict(list)
    for s in stats:
        drives[re.sub(r'_[a-z]$', '', s)].append(s)
    total = sum(v['frames'] for v in stats.values())
    print(f'{"drive":16s} {"seqs":>4s} {"frames":>6s} {"%":>5s} {"lum":>5s}  ' + ' '.join(f'{n[:5]:>6s}' for n in NAMES))
    for d, seqs in sorted(drives.items()):
        fr = sum(stats[s]['frames'] for s in seqs)
        lum = np.median([stats[s]['lum_median'] for s in seqs])
        bx = np.sum([stats[s]['boxes'] for s in seqs], 0)
        print(f'{d:16s} {len(seqs):4d} {fr:6d} {100 * fr / total:5.1f} {lum:5.1f}  ' + ' '.join(f'{b:6d}' for b in bx))
    bx = np.sum([v['boxes'] for v in stats.values()], 0)
    print(f'{"total":16s} {len(stats):4d} {total:6d} {100:5.1f} {"":5s}  ' + ' '.join(f'{b:6d}' for b in bx))


if __name__ == '__main__':
    main()
