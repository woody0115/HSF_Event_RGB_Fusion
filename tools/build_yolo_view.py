"""Step 3a: expose the processed data in the folder layout Ultralytics expects.

  <processed_root>/yolo/<modality>/<split>/images/<seq>_<idx>.png   hard link (no extra disk space)
  <processed_root>/yolo/<modality>/<split>/labels/<seq>_<idx>.txt   copy of the YOLO label
for modality in {rgb, event}; only labelled frames are included (DSEC-Det's 70,379 frames).
Event frames are added only for sequences whose GEAR stage is finished (gear_done.json).
Also writes configs/generated/dsec_<modality>.yaml for `YOLO.train(data=...)`.

Re-runnable: existing links are kept, newly processed sequences are added.
Run:  .venv\\Scripts\\python.exe tools\\build_yolo_view.py [--train-seqs zurich_city_16_a ...]
"""
import argparse
import glob
import json
import os
import os.path as osp
import shutil

import sys

import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from hsf.data.dataset import CFG, VAL_SEQUENCES  # noqa: E402  (CFG includes configs/paths.local.yaml)

PROC = CFG['processed_root']


def link(src, dst):
    if not osp.exists(dst):
        os.link(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-seqs', nargs='*', help='restrict train split to these sequences (smoke test)')
    ap.add_argument('--tag', default=None, help='suffix for the generated yaml / view folder, e.g. "_smoke" '
                                                '(default: "" for legacy, "_val" for --split val)')
    ap.add_argument('--split', default='legacy', choices=['legacy', 'val'],
                    help='legacy: train = all 47 sequences, val = test (step 3 of versions #1-#10); '
                         'val: train_core / val / test')
    a = ap.parse_args()
    if a.tag is None:
        a.tag = '_val' if a.split == 'val' else ''

    metas = sorted(glob.glob(osp.join(PROC, '*', '*', 'meta.json')))
    counts = {}
    for mpath in metas:
        m = json.load(open(mpath))
        split, seq = m['split'], m['sequence']
        if split == 'train' and a.train_seqs and seq not in a.train_seqs:
            continue
        if split == 'train' and a.split == 'val':
            split = 'val' if seq in VAL_SEQUENCES else 'train_core'
        seq_dir = osp.dirname(mpath)
        has_gear = osp.isfile(osp.join(seq_dir, 'gear_done.json'))
        for mod in ('rgb', 'event'):
            if mod == 'event' and not has_gear:     # events not downloaded / encoded yet
                continue
            img_dst = osp.join(PROC, f'yolo{a.tag}', mod, split, 'images')
            lab_dst = osp.join(PROC, f'yolo{a.tag}', mod, split, 'labels')
            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(lab_dst, exist_ok=True)
            for i in m['labelled_frames']:
                name = f'{seq}_{i:06d}'
                src = (osp.join(m['rgb_dir'], f'{i:06d}.png') if mod == 'rgb'
                       else osp.join(seq_dir, 'gear', f'{i:06d}.png'))
                link(src, osp.join(img_dst, name + '.png'))
                lab = osp.join(lab_dst, name + '.txt')
                if not osp.exists(lab):
                    shutil.copyfile(osp.join(seq_dir, 'labels', f'{i:06d}.txt'), lab)
            counts[(mod, split)] = counts.get((mod, split), 0) + len(m['labelled_frames'])

    os.makedirs(osp.join(ROOT, 'configs', 'generated'), exist_ok=True)
    for mod in ('rgb', 'event'):
        data = {'path': osp.join(PROC, f'yolo{a.tag}', mod).replace('\\', '/')}
        data.update({'train': 'train_core/images', 'val': 'val/images', 'test': 'test/images'} if a.split == 'val'
                    else {'train': 'train/images', 'val': 'test/images'})
        data['names'] = {i: n for i, n in enumerate(CFG['names'])}
        out = osp.join(ROOT, 'configs', 'generated', f'dsec_{mod}{a.tag}.yaml')
        with open(out, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f'{out}: ' + ', '.join(f'{k[1]} {v}' for k, v in sorted(counts.items()) if k[0] == mod) +
              ' labelled frames')


if __name__ == '__main__':
    main()
