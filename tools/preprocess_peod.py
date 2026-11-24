"""Preprocess PEOD (Cui et al., arXiv 2511.08140) into the same per-sequence layout as tools/preprocess_dsec.py,
so hsf/data/dataset.py can read it with processed_root pointed at the output folder.

PEOD is a second benchmark, trained and reported separately from DSEC (2026-11-12 lab meeting).
Source (tools/download_peod.sh, after extraction), under --root:
    coco_0111_last/<split>/<subset>/<seq>.json   COCO labels; split train|test, subset normal|challenge|motion_blur;
                                                 every frame is labelled; file names <seq>_<NNNN>.png (train) or
                                                 <seq>_<NNNN>_test.png (test), NNNN = 1..n
    timestamp/<split>/[<subset>/]<seq>.csv       "1,<t_us>" per frame, same clock as the events (checked)
    rgb/**/<seq>/*.png                           RGB frames, 1280x720, pixel-aligned with the events
    event/**/<seq>.dat (also _sequence_024-5/)   Prophesee DAT v2: '%' header lines, 2 bytes (type 0x0C, size 8),
                                                 then 8-byte events: uint32 t_us, uint32 x | y << 14 | p << 28
Output, under --out (one folder per sequence, split = train | test):
    <split>/<seq>/rgb/000000.png      symlink to the source RGB frame (frame index = NNNN - 1)
    <split>/<seq>/labels/000000.txt   "cls xc yc w h" normalized to the 1280x720 frame, PEOD class ids:
                                      0 car, 1 person, 2 bus, 3 truck, 4 2-wheeler, 5 3-wheeler
    <split>/<seq>/boxes.npy           img_idx, cls, x0, y0, x1, y1 (px), track_id (= COCO annotation id)
    <split>/<seq>/meta.json           as DSEC, plus subset, recording date (from the DAT header) and image size
    <split>/<seq>/gear/000000.png     GEAR at the native 1280x720 (c = 8, tau = 10, window = previous -> current
                                      frame; the first frame uses the median frame interval before it)
    <split>/<seq>/gear_done.json      marker + event statistics
Safe to re-run: finished stages are skipped, sequences whose files are missing are listed and skipped.
Run (server):  python tools/preprocess_peod.py [--workers 8] [--only sequence_025] [--labels-only]
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from hsf.data.gear import GEAR  # noqa: E402

NAMES = ['car', 'person', 'bus', 'truck', '2-wheeler', '3-wheeler']
MIN_SIDE = 2.0
BOX_DTYPE = np.dtype([('img_idx', 'i4'), ('cls', 'u1'), ('x0', 'f4'), ('y0', 'f4'),
                      ('x1', 'f4'), ('y1', 'f4'), ('track_id', 'u4')])
EV_DTYPE = np.dtype([('t', '<u4'), ('d', '<u4')])


class DatReader:
    """Prophesee DAT (version 2) CD events, memory-mapped; windows by microsecond time."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            head = f.read(1 << 16)
        pos, self.header = 0, {}
        while head[pos:pos + 1] == b'%':
            nl = head.index(b'\n', pos)
            line = head[pos + 1:nl].decode('latin1').strip()
            if ' ' in line:
                k, v = line.split(' ', 1)
                self.header[k.lower()] = v
            pos = nl + 1
        ev_type, ev_size = head[pos], head[pos + 1]
        assert ev_type == 0x0C and ev_size == 8, f'{path}: unexpected event type/size {ev_type}/{ev_size}'
        n = (osp.getsize(path) - pos - 2) // 8
        self.ev = np.memmap(path, EV_DTYPE, 'r', offset=pos + 2, shape=(n,))
        self.t = self.ev['t']
        self.n = n
        # np.searchsorted on the strided memmap field would copy the whole column on every call, so search a
        # coarse in-memory index first (every 4096th timestamp; also a cheap check that times are sorted)
        self.B = 4096
        self.coarse = np.array(self.t[::self.B], dtype=np.int64)
        assert np.all(np.diff(self.coarse) >= 0), f'{path}: timestamps not sorted'

    def index_at(self, t):
        """First event index with timestamp >= t."""
        k = int(np.searchsorted(self.coarse, t, side='left'))
        lo, hi = max(0, (k - 1) * self.B), min(self.n, k * self.B + 1)
        return lo + int(np.searchsorted(np.asarray(self.t[lo:hi]), t, side='left'))

    def window(self, t0, t1):
        a, b = self.index_at(t0), self.index_at(t1)
        d = np.asarray(self.ev['d'][a:b])
        return (d & 0x3FFF).astype(np.int32), ((d >> 14) & 0x3FFF).astype(np.int32), ((d >> 28) & 1).astype(np.uint8)


def find_sources(root):
    """{(split, seq): {...paths...}} for every labelled sequence."""
    rgb_dirs = {}
    for d in glob.glob(osp.join(root, 'rgb', '**', 'sequence_*'), recursive=True):
        if osp.isdir(d):
            rgb_dirs.setdefault(osp.basename(d), []).append(d)
    dats = {}
    for f in glob.glob(osp.join(root, '**', 'sequence_*.dat'), recursive=True):
        dats.setdefault(osp.basename(f)[:-4], []).append(f)
    tss = {}
    for f in glob.glob(osp.join(root, 'timestamp', '**', '*.csv'), recursive=True):
        tss.setdefault(osp.basename(f)[:-4], []).append(f)
    out = {}
    for j in sorted(glob.glob(osp.join(root, 'coco_0111_last', '*', '*', '*.json'))):
        split, subset, seq = j.split(os.sep)[-3], j.split(os.sep)[-2], osp.basename(j)[:-5]
        pick = lambda c: (c or [None])[0] if len(c or []) <= 1 else '<ambiguous: %s>' % c  # noqa: E731
        out[(split, seq)] = {'ann': j, 'subset': subset, 'rgb': pick(rgb_dirs.get(seq)),
                             'dat': pick(dats.get(seq)), 'ts': pick(tss.get(seq))}
    return out


def frame_index(file_name):
    return int(re.findall(r'_(\d+)', file_name)[-1]) - 1


def out_dir(a, split, seq):
    return osp.join(a.out, split, seq)


def do_labels(a, split, seq, src):
    od = out_dir(a, split, seq)
    t0 = time.time()
    coco = json.load(open(src['ann']))
    ts = np.loadtxt(src['ts'], delimiter=',', dtype=np.int64, ndmin=2)[:, 1]
    imgs = {im['id']: im for im in coco['images']}
    n = len(imgs)
    assert n == len(ts), f'{seq}: {n} labelled images but {len(ts)} timestamps'
    idx_of = {k: frame_index(im['file_name']) for k, im in imgs.items()}
    assert sorted(idx_of.values()) == list(range(n)), f'{seq}: frame numbers are not 1..n'
    W, H = coco['images'][0]['width'], coco['images'][0]['height']
    os.makedirs(osp.join(od, 'labels'), exist_ok=True)
    os.makedirs(osp.join(od, 'rgb'), exist_ok=True)
    for k, im in imgs.items():                                 # symlinks: no second copy of 55 GB
        s, dst = osp.join(src['rgb'], im['file_name']), osp.join(od, 'rgb', f'{idx_of[k]:06d}.png')
        assert osp.isfile(s), f'missing RGB frame {s}'
        if not osp.lexists(dst):
            os.symlink(s, dst)
    rows, dropped = [], 0
    for an in coco['annotations']:
        if an.get('ignore', 0) or an.get('iscrowd', 0):
            dropped += 1
            continue
        x, y, w, h = an['bbox']
        x0, y0, x1, y1 = np.clip([x, y, x + w, y + h], 0, [W, H, W, H])
        if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
            dropped += 1
            continue
        rows.append((idx_of[an['image_id']], an['category_id'], x0, y0, x1, y1, an['id']))
    boxes = np.array(rows, BOX_DTYPE) if rows else np.zeros(0, BOX_DTYPE)
    boxes.sort(order=['img_idx', 'cls'])
    np.save(osp.join(od, 'boxes.npy'), boxes)
    for i in range(n):
        b = boxes[boxes['img_idx'] == i]
        lines = [f'{c} {(p + r) / 2 / W:.6f} {(q + s) / 2 / H:.6f} {(r - p) / W:.6f} {(s - q) / H:.6f}'
                 for c, p, q, r, s in zip(b['cls'], b['x0'], b['y0'], b['x1'], b['y1'])]
        with open(osp.join(od, 'labels', f'{i:06d}.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    np.savetxt(osp.join(od, 'timestamps.txt'), ts, fmt='%d')
    meta = {
        'dataset': 'PEOD', 'split': split, 'subset': src['subset'], 'sequence': seq, 'n_images': n,
        'width': W, 'height': H, 'labelled_frames': list(range(n)), 'n_boxes': int(len(boxes)),
        'n_boxes_dropped': dropped, 'names': NAMES,
        'boxes_per_class': {int(c): int(m) for c, m in zip(*np.unique(boxes['cls'], return_counts=True))},
        'rgb_dir': osp.join(od, 'rgb'), 'rgb_source': src['rgb'], 'timestamps': src['ts'], 'events_dat': src['dat'],
        'recording_date': DatReader(src['dat']).header.get('date') if src['dat'] else None,
    }
    with open(osp.join(od, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return f'labels {split}/{seq}: {n} frames, {len(boxes)} boxes ({dropped} dropped), {time.time() - t0:.0f} s'


def do_gear(a, split, seq, src):
    od = out_dir(a, split, seq)
    t0 = time.time()
    meta = json.load(open(osp.join(od, 'meta.json')))
    ts = np.loadtxt(osp.join(od, 'timestamps.txt'), dtype=np.int64)
    reader = DatReader(src['dat'])
    gear = GEAR(meta['height'], meta['width'])
    os.makedirs(osp.join(od, 'gear'), exist_ok=True)
    first_dt = int(np.median(np.diff(ts))) if len(ts) > 1 else 33_480
    n_ev = np.zeros(len(ts), np.int64)
    for i in range(len(ts)):
        x, y, p = reader.window(ts[i - 1] if i else ts[0] - first_dt, ts[i])
        n_ev[i] = len(x)
        cv2.imwrite(osp.join(od, 'gear', f'{i:06d}.png'), GEAR.to_uint8(gear(x, y, p)),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
    info = {
        'n_frames': int(len(ts)), 'events_per_frame_median': int(np.median(n_ev)),
        'frames_without_events': int((n_ev == 0).sum()),
        'events_in_file': int(len(reader.t)), 't_events': [int(reader.t[0]), int(reader.t[-1])],
        't_frames': [int(ts[0]), int(ts[-1])],
        'gear': {'c': 8, 'tau': 10, 'window': 'previous frame -> current frame', 'size': [meta['width'], meta['height']],
                 'channels': '[h, feat, num]; h/feat mid-gray=0, num black=0'},
        'seconds': round(time.time() - t0, 1),
    }
    with open(osp.join(od, 'gear_done.json'), 'w') as f:
        json.dump(info, f, indent=1)
    return f'gear   {split}/{seq}: {len(ts)} frames, {info["events_per_frame_median"]:,} ev/frame, ' \
           f'{info["frames_without_events"]} empty, {info["seconds"]:.0f} s'


def run(job):
    a, stage, split, seq, src = job
    try:
        return (do_labels if stage == 'labels' else do_gear)(a, split, seq, src)
    except Exception as e:  # report and continue with the other sequences
        return f'FAILED {stage} {split}/{seq}: {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=osp.expanduser('~/work/shared_data/PEOD'))
    ap.add_argument('--out', default=osp.expanduser('~/work/shared_data/PEOD_train'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only', nargs='*', help='sequence names to process')
    ap.add_argument('--labels-only', action='store_true')
    a = ap.parse_args()

    srcs = find_sources(a.root)
    label_jobs, gear_jobs, waiting, n_done = [], [], [], [0, 0]
    for (split, seq), src in srcs.items():
        if a.only and seq not in a.only:
            continue
        od = out_dir(a, split, seq)
        bad = [k for k in ('rgb', 'ts') if not src[k] or src[k].startswith('<')]
        if osp.isfile(osp.join(od, 'meta.json')):
            n_done[0] += 1
        elif bad:
            waiting.append(f'labels {split}/{seq}: missing {bad}')
        else:
            label_jobs.append((a, 'labels', split, seq, src))
        if a.labels_only:
            continue
        if osp.isfile(osp.join(od, 'gear_done.json')):
            n_done[1] += 1
        elif bad or not src['dat'] or src['dat'].startswith('<'):
            waiting.append(f'gear   {split}/{seq}: missing {bad or ["dat"]}')
        else:
            gear_jobs.append((a, 'gear', split, seq, src))
    print(f'{len(srcs)} labelled sequences | labels: {n_done[0]} done, {len(label_jobs)} to do | '
          f'gear: {n_done[1]} done, {len(gear_jobs)} to do | waiting: {len(waiting)}', flush=True)
    t0 = time.time()
    for jobs in (label_jobs, gear_jobs):           # gear needs the label stage's meta.json and timestamps
        if jobs:
            with Pool(min(a.workers, len(jobs))) as pool:
                for msg in pool.imap_unordered(run, jobs):
                    print('  ' + msg, flush=True)
    print(f'finished in {(time.time() - t0) / 60:.1f} min', flush=True)
    for w in waiting[:10]:
        print('  waiting: ' + w)
    if len(waiting) > 10:
        print(f'  ... and {len(waiting) - 10} more')


if __name__ == '__main__':
    main()
