"""Pack the checkpoints of the 10 closed-set versions (VERSIONS.md) into one zip per version for a GitHub Release.

Each zip holds the weights, the text results of that run and a README.txt saying how to load it.
.pt files are already compressed, so entries are stored without recompression.

Run:  .venv\\Scripts\\python.exe tools\\pack_release.py [--out D:\\HSF_release]
"""
import argparse
import os
import os.path as osp
import zipfile

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
RUNS = osp.join(ROOT, 'runs')

LOAD_SINGLE = "from ultralytics import YOLO\nmodel = YOLO('best.pt')"
LOAD_DUAL = ("import torch\nfrom hsf.models.dual_yolov8 import DualYOLOv8\n"
             "model = DualYOLOv8(nc=8, fusion='{fusion}')\n"
             "model.load_state_dict(torch.load('best.pt', map_location='cpu')['model'])")


# (zip name, run dir under runs/, files to include, title, result line, load snippet)
VERSIONS = [
    ('v01_rgb_single', 'step3/rgb_yolov8s', ['weights/best.pt', 'results.csv', 'args.yaml', 'console.log'],
     '#1 RGB single-stream YOLOv8s', 'mAP50-95 53.9 (day 54.1 / dim 60.6 / night 42.2)', LOAD_SINGLE),
    ('v02_event_single', 'step3/event_yolov8s', ['weights/best.pt', 'results.csv', 'args.yaml', 'console.log'],
     '#2 Event (GEAR) single-stream YOLOv8s', 'mAP50-95 32.9 (day 32.9 / dim 39.0 / night 26.7)', LOAD_SINGLE),
    ('v03_add_e20_old_event_format', 'step4/dual_add_e20', ['best.pt', 'results.csv', 'console.log'],
     '#3 dual add, 20 epochs, lr 0.005 (old event input format)', 'mAP50-95 51.9 (night 44.7)', LOAD_DUAL.format(fusion='add')),
    ('v04_af_e20_old_event_format', 'step5_6/dual_af_e20', ['best.pt', 'results.csv', 'console.log'],
     '#4 dual AF, 20 epochs, lr 0.005 (old event input format)', 'mAP50-95 52.9', LOAD_DUAL.format(fusion='af')),
    ('v05_hsf_e20_old_event_format', 'step5_6/dual_hsf_e20', ['best.pt', 'results.csv', 'console.log'],
     '#5 dual HSF (TFS + AF), 20 epochs, lr 0.005 (old event input format)', 'mAP50-95 51.5',
     LOAD_DUAL.format(fusion='hsf') + "\n# stateful: out, state = model(rgb, evt, state)"),
    ('v06_add_lr1e3', 'step4/dual_add_lr1e3_e5', ['best.pt', 'results.csv', 'console.log'],
     '#6 dual add, 5 epochs, lr 0.001', 'mAP50-95 53.0 (day 53.1 / dim 61.0 / night 45.5, best at night)',
     LOAD_DUAL.format(fusion='add')),
    ('v07_af_lr1e3', 'step5_6/dual_af_lr1e3_e5', ['best.pt', 'results.csv', 'console.log'],
     '#7 dual AF, 5 epochs, lr 0.001', 'mAP50-95 53.5 (day 53.8 / dim 62.8 / night 42.5, best when dim)',
     LOAD_DUAL.format(fusion='af')),
    ('v08_hsf_lr1e3', 'step5_6/dual_hsf_lr1e3_e5', ['best.pt', 'results.csv', 'console.log'],
     '#8 dual HSF (TFS + AF), 5 epochs, lr 0.001', 'mAP50-95 53.1 (day 53.2 / dim 61.2 / night 42.5)',
     LOAD_DUAL.format(fusion='hsf') + "\n# stateful: out, state = model(rgb, evt, state)"),
    ('v09_iaf_add', 'step8_iaf/dual_iaf_add_lr1e3_e5', ['best.pt', 'results.csv', 'console.log'],
     '#9 add + illumination gate', 'mAP50-95 53.5 (day 53.7 / dim 60.9 / night 42.5)', LOAD_DUAL.format(fusion='iaf_add')),
    ('v10_iaf_af', 'step8_iaf/dual_iaf_af_lr1e3_e5', ['best.pt', 'results.csv', 'console.log'],
     '#10 AF + illumination gate', 'mAP50-95 53.5 (day 53.5 / dim 62.1 / night 42.9)', LOAD_DUAL.format(fusion='iaf_af')),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\HSF_release')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, run, files, title, result, load in VERSIONS:
        src = osp.join(RUNS, run)
        missing = [f for f in files if not osp.isfile(osp.join(src, f))]
        if missing:
            raise FileNotFoundError(f'{name}: missing {missing}')
        readme = (f'{title}\n\nResult (DSEC-Det test): {result}\nSource run: runs/{run}\n'
                  f'Code: https://github.com/woody0115/HSF_Event_RGB_Fusion (see VERSIONS.md, CHANGELOG.md)\n\n'
                  f'Load (from the repository root):\n\n{load}\n')
        path = osp.join(a.out, f'{name}.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
            z.writestr(f'{name}/README.txt', readme)
            for f in files:
                z.write(osp.join(src, f), f'{name}/{f}')
        print(f'{name}.zip  {osp.getsize(path) / 1e6:.0f} MB', flush=True)


if __name__ == '__main__':
    main()
