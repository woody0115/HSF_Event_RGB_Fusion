"""Summarise tools/tune_eval.py results (runs/tune/<tag>.json): mean ± std over the checkpoints of each tag.

Columns (val split): all, day, dim, night mAP50-95, night pedestrian AP, val-mPC (every 4th frame, 9 kinds x
severities 1/3/5), mPC over the training kinds and over the held-out kinds of the fine-tuning rounds, per-group
means, and inside the model: P3/P4/P5 event reliance r_e clean vs under corruption, gate (event share / g_e).

Run:  python tools/summarize_tune.py r1_base r1_base_relp2 ...   (prints markdown; --out NAME writes runs/tune/NAME.md)
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
TRAIN_KINDS = ('gaussian_noise', 'defocus_blur', 'fog', 'brightness', 'dark')     # tools/train_dual.py default
HELD_OUT = ('shot_noise', 'impulse_noise', 'motion_blur', 'contrast')


def ms(v, d=1):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    return f'{np.mean(v):.{d}f}' + (f' ± {np.std(v, ddof=1):.{d}f}' if len(v) > 1 else '')


def kind_mpc(r, kinds):
    c = r['corrupt']['conds']
    v = [e['mAP50-95'] for k, e in c.items() if k != 'clean' and k.split('@')[0] in kinds]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows, inside = [], []
    rows.append('| tag | n | all | day | dim | night | night ped | val-mPC | train kinds | held-out | noise | blur | '
                'weather | digital | light |')
    rows.append('|---' * 15 + '|')
    inside.append('| tag | r_e clean P3/P4/P5 | r_e corrupted P3/P4/P5 | gate clean | gate corrupted | gate night |')
    inside.append('|---' * 6 + '|')
    for tag in a.tags:
        p = osp.join(ROOT, 'runs', 'tune', f'{tag}.json')
        if not osp.isfile(p):
            print(f'{tag}: missing', file=sys.stderr)
            continue
        res = json.load(open(p))
        groups = {}
        for run, r in res.items():                   # several fusions may share a tag: group by fusion
            groups.setdefault(r['fusion'], []).append(r)
        for fusion, rs in groups.items():
            name = tag if len(groups) == 1 else f'{tag} ({fusion})'
            cl = lambda b: [r['clean'][b]['mAP50-95'] if 'clean' in r and b in r['clean'] else None for r in rs]  # noqa: E731
            ped = [r['clean']['night']['per_class'].get('pedestrian') if 'clean' in r and 'night' in r['clean'] else None
                   for r in rs]
            has_c = [r for r in rs if 'corrupt' in r]
            mpc = [r['corrupt']['mPC'] for r in has_c]
            grp = lambda g: [r['corrupt']['groups'].get(g) for r in has_c]  # noqa: E731
            rows.append(f'| {name} | {len(rs)} | {ms(cl("all"))} | {ms(cl("day"))} | {ms(cl("dim"))} | {ms(cl("night"))} | '
                        f'{ms(ped)} | {ms(mpc)} | {ms([kind_mpc(r, TRAIN_KINDS) for r in has_c])} | '
                        f'{ms([kind_mpc(r, HELD_OUT) for r in has_c])} | ' +
                        ' | '.join(ms(grp(g)) for g in ('noise', 'blur', 'weather', 'digital', 'light')) + ' |')
            if has_c and has_c[0]['corrupt']['conds']['clean']['r_e']:
                def re(cond_filter):
                    out = []
                    for s in range(3):
                        v = [np.mean([e['r_e'][s] for k, e in r['corrupt']['conds'].items() if cond_filter(k)])
                             for r in has_c]
                        out.append(f'{np.mean(v):.2f}')
                    return ' / '.join(out)

                def gate(cond_filter):
                    v = [np.mean([e['gate'] for k, e in r['corrupt']['conds'].items() if cond_filter(k) and 'gate' in e])
                         for r in has_c if any('gate' in e for e in r['corrupt']['conds'].values())]
                    return f'{np.mean(v):.3f}' if v else '-'
                gn = [r['clean']['night'].get('gate') for r in rs if 'clean' in r and 'night' in r['clean']]
                gn = [g for g in gn if g is not None]
                inside.append(f'| {name} | {re(lambda k: k == "clean")} | {re(lambda k: k != "clean")} | '
                              f'{gate(lambda k: k == "clean")} | {gate(lambda k: k != "clean")} | '
                              f'{np.mean(gn):.3f} |' if gn else
                              f'| {name} | {re(lambda k: k == "clean")} | {re(lambda k: k != "clean")} | - | - | - |')
    md = '\n'.join(rows) + '\n\n' + '\n'.join(inside) + '\n'
    print(md)
    if a.out:
        open(osp.join(ROOT, 'runs', 'tune', f'{a.out}.md'), 'w', encoding='utf-8').write(md)


if __name__ == '__main__':
    main()
