"""Aggregate the multi-seed runs (runs/seed_runs.sh) into mean / std / per-seed tables.

Sources per model and seed:
  val      best mAP50-95 over epochs in <run>/results.csv (the checkpoint selection criterion)
  test     runs/eval_conditions.json (all / day / dim / night, plus zurich_city_12_a and interlaken_01_a)
  RGB corr runs/eval_corruption.json (seed 0) and runs/eval_corruption_seeds_g0 / _g3.json (seeds 1, 2)

Run:  python tools/summarize_seeds.py      Writes runs/seeds_summary.md / .json
"""
import csv
import json
import os.path as osp

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
RUNS = osp.join(ROOT, 'runs')
DIRS = {'add': 'step4', 'relp2_add': 'step11_rel', 'sw2_add': 'step12_sw', 'cafr': 'step13_cafr'}
# seed 0 ran before the multi-seed driver and has its own names / evaluation keys
SEED0 = {   # fusion: (run name, eval_conditions key, eval_corruption key)
    'add': ('dual_add_s5_cleaninit', 'dual_add_s5_cleaninit', 'add_s5'),
    'relp2_add': ('dual_relp2_add_v2', 'dual_relp2_add_v2', 'relp2_add'),
    'sw2_add': ('dual_sw2_add_s5', 'dual_sw2_add_s5', 'sw2_add'),
    'cafr': ('dual_cafr_s5', 'dual_cafr_s5', 'cafr'),
}
SEEDS = [0, 1, 2]
SEQS = ['zurich_city_12_a', 'interlaken_01_a']
GROUPS = ['noise', 'blur', 'weather', 'digital', 'light']


def keys(f, s):
    if s == 0:
        return SEED0[f]
    name = f'dual_{f}_s5_seed{s}'
    return name, f'{f}_seed{s}', f'{f}_seed{s}'


def load(name):
    p = osp.join(RUNS, name)
    return json.load(open(p)) if osp.isfile(p) else {}


def val_best(f, run):
    p = osp.join(RUNS, DIRS[f], run, 'results.csv')
    if not osp.isfile(p):
        return None
    return 100 * max(float(r['mAP50-95']) for r in csv.DictReader(open(p)))


def main():
    cond = load('eval_conditions.json')
    corr = {**load('eval_corruption.json'), **load('eval_corruption_seeds_g0.json'),
            **load('eval_corruption_seeds_g3.json')}
    cols = ['val', 'test', 'day', 'dim', 'night'] + SEQS + ['clean_c', 'mPC', 'rPC'] + GROUPS
    table = {}
    for f in DIRS:
        rows = {}
        for s in SEEDS:
            run, ck, rk = keys(f, s)
            r = {'val': val_best(f, run)}
            c = cond.get(ck)
            for b, col in zip(['all', 'day', 'dim', 'night'], ['test', 'day', 'dim', 'night']):
                r[col] = c['buckets'][b]['mAP50-95'] if c and c['buckets'].get(b) else None
            for q in SEQS:
                r[q] = c['sequences'][q]['mAP50-95'] if c and c['sequences'].get(q) else None
            x = corr.get(rk)
            r['clean_c'] = x['clean'] if x else None
            r['mPC'] = x.get('mPC') if x else None
            r['rPC'] = 100 * x['rPC'] if x and x.get('rPC') is not None else None
            for g in GROUPS:
                r[g] = x['groups'].get(g) if x else None
            rows[s] = r
        table[f] = rows

    def stat(f, col):
        v = [table[f][s][col] for s in SEEDS if table[f][s][col] is not None]
        if not v:
            return None
        return {'n': len(v), 'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                'min': float(min(v)), 'max': float(max(v)), 'values': v}

    stats = {f: {c: stat(f, c) for c in cols} for f in DIRS}
    fmt = lambda st: '-' if st is None else (f'{st["mean"]:.1f} ± {st["std"]:.1f}' + ('' if st['n'] == 3 else f' (n={st["n"]})'))  # noqa: E731
    lines = ['# Multi-seed summary (seeds 0, 1, 2; short schedule)', '',
             'mAP50-95 (%), mean ± sample std over seeds. val = best epoch on the val split (selection criterion); '
             'rPC in %. Corruption columns: test every 2nd frame, RGB corrupted, event clean.', '',
             '| model | val | test | day | dim | night | zurich_city_12_a | interlaken_01_a |', '|---' * 8 + '|']
    for f in DIRS:
        lines.append(f'| {f} | ' + ' | '.join(fmt(stats[f][c]) for c in ['val', 'test', 'day', 'dim', 'night'] + SEQS) + ' |')
    lines += ['', '| model | clean | mPC | rPC | ' + ' | '.join(GROUPS) + ' |', '|---' * (4 + len(GROUPS)) + '|']
    for f in DIRS:
        lines.append(f'| {f} | ' + ' | '.join(fmt(stats[f][c]) for c in ['clean_c', 'mPC', 'rPC'] + GROUPS) + ' |')
    lines += ['', '## Per seed', '', '| model | seed | ' + ' | '.join(cols) + ' |', '|---' * (2 + len(cols)) + '|']
    for f in DIRS:
        for s in SEEDS:
            lines.append(f'| {f} | {s} | ' + ' | '.join('-' if table[f][s][c] is None else f'{table[f][s][c]:.2f}'
                                                       for c in cols) + ' |')
    # differences to add, seed-matched (same seed -> same data order) and unpaired
    lines += ['', '## Difference to add (val and mPC)', '',
              '| model | Δval mean | Δval per seed | ΔmPC mean | ΔmPC per seed |', '|---|---|---|---|---|']
    for f in [k for k in DIRS if k != 'add']:
        cells = []
        for c in ['val', 'mPC']:
            d = [table[f][s][c] - table['add'][s][c] for s in SEEDS
                 if table[f][s][c] is not None and table['add'][s][c] is not None]
            cells += [f'{np.mean(d):+.2f}' if d else '-', ' '.join(f'{x:+.2f}' for x in d) or '-']
        lines.append(f'| {f} | ' + ' | '.join(cells) + ' |')
    md = '\n'.join(lines) + '\n'
    open(osp.join(RUNS, 'seeds_summary.md'), 'w', encoding='utf-8').write(md)
    json.dump({'per_seed': table, 'stats': stats}, open(osp.join(RUNS, 'seeds_summary.json'), 'w'), indent=1)
    print(md)


if __name__ == '__main__':
    main()
