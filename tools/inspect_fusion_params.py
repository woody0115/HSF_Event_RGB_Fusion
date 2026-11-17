"""Print the interpretable fusion parameters of the multi-seed checkpoints (no data needed).

relp2_add: per scale a (CNR), b (event SNR), c (saturation), d_r, d_e and the event projection P's deviation
           from identity; sw2_add: last MLP layer (bias, weight norm per output) and P's deviation;
cafr: lambda_out per scale and the norm of the CAFR output conv; add: nothing to read.
Also the per-scale weight change of each backbone relative to its single-stream init (how far fusion
training moved the streams).

Run:  python tools/inspect_fusion_params.py   Writes runs/tune/params.json
"""
import json
import os
import os.path as osp
import sys

import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(ROOT, 'tools'))
from tune_eval import FAMILIES, RUNS  # noqa: E402


def rel_dev_identity(w):
    c = w.shape[0]
    eye = torch.eye(c).view(c, c, 1, 1)
    return round(float((w - eye).norm() / eye.norm()), 4)


def main():
    init_r = torch.load(osp.join(RUNS, 'step3/rgb_yolov8s_val/weights/best.pt'), map_location='cpu', weights_only=False)
    init_e = torch.load(osp.join(RUNS, 'step3/event_yolov8s_val/weights/best.pt'), map_location='cpu', weights_only=False)
    init_r = (init_r.get('ema') or init_r['model']).float().state_dict()
    init_e = (init_e.get('ema') or init_e['model']).float().state_dict()
    out = {}
    for fam, runs in FAMILIES.items():
        for run in runs:
            sd = torch.load(osp.join(RUNS, run, 'best.pt'), map_location='cpu', weights_only=False)['model']
            r = {}
            for k in range(3):
                p = f'fusion.{k}.'
                s = {}
                if fam == 'relp2_add':
                    for n, short in [('lambda_cnr', 'a'), ('lambda_snr', 'b'), ('lambda_sat', 'c'),
                                     ('lambda_bias_r', 'd_r'), ('lambda_bias_e', 'd_e')]:
                        s[short] = round(float(sd[p + n]), 4)
                    s['P_dev'] = rel_dev_identity(sd[p + 'proj.weight'].float())
                if fam == 'sw2_add':
                    s['mlp_bias'] = [round(float(v), 4) for v in sd[p + 'mlp.2.bias']]
                    s['mlp_w_norm'] = [round(float(v), 4) for v in sd[p + 'mlp.2.weight'].norm(dim=1)]
                    s['P_dev'] = rel_dev_identity(sd[p + 'proj.weight'].float())
                if fam == 'cafr':
                    s['lambda_out'] = round(float(sd[p + 'lambda_out']), 5)
                    s['out_conv_norm'] = round(float(sd[p + 'out.conv.weight'].norm()), 3)
                r[f'P{k + 3}'] = s
            # how far fusion training moved each backbone (relative weight change, conv weights only)
            for tag, prefix, init in (('rgb', 'rgb.', init_r), ('evt', 'evt.', init_e)):
                num = den = 0.0
                for key, v in sd.items():
                    if key.startswith(prefix) and key.endswith('conv.weight'):
                        w0 = init['model.' + key[len(prefix):]]
                        num += float((v.float() - w0).norm() ** 2)
                        den += float(w0.norm() ** 2)
                r[f'{tag}_backbone_rel_change'] = round((num / den) ** 0.5, 5)
            out[run] = r
            print(run, json.dumps(r), flush=True)
    os.makedirs(osp.join(RUNS, 'tune'), exist_ok=True)
    json.dump(out, open(osp.join(RUNS, 'tune', 'params.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
