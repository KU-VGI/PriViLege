# Merges the per-machine results files (runs.csv from Colab, runs_runpod*.csv from the pods) into results/all_runs.csv:
# one row per (experiment, dataset, seed), in the order of the paper's tables. The results notebook
# (colab/PriViLege_results.ipynb) reads only that file.
#
#   python colab/merge_results.py results/raw/*.csv          (writes results/all_runs.csv)
#
# Checks every row (session list length, A_Base / A_Last / A_Avg against the session list), drops smoke tests, keeps one
# row when two machines finished the same run (their numbers must agree) and lists any planned run that is missing.
import csv, json, os, sys

# Experiments in table order: name -> (datasets, seeds). abl_Baseline is the released -ft option, kept for reference
# (Table 3 uses abl_FT, see colab/make_notebook.py section 6).
PLAN = {
    'PriViLege':       (['cifar100', 'cub200', 'mini_imagenet'], [1, 2, 3, 4, 5]),
    'abl_FT':          (['cub200'], [1, 2, 3, 4, 5]),
    'abl_PKT':         (['cub200'], [1, 2, 3, 4, 5]),
    'abl_PKT_ED':      (['cub200'], [1, 2, 3, 4, 5]),
    'abl_PKT_SKD':     (['cub200'], [1, 2, 3, 4, 5]),
    **{f'layers_{L}':  (['cub200'], [1, 2, 3, 4, 5]) for L in [0, 5, 7, 10, 12]},
    'clip_zeroshot':   (['cifar100'], [1]),
    'clip_PKT':        (['cifar100'], [1, 2, 3, 4, 5]),
    'clip_PKT_ED':     (['cifar100'], [1, 2, 3, 4, 5]),
    'clip_PKT_ED_SKD': (['cifar100'], [1, 2, 3, 4, 5]),
    'abl_Baseline':    (['cub200'], [1, 2, 3, 4, 5]),
}
SESSIONS = {'cub200': 11, 'cifar100': 9, 'mini_imagenet': 9}   # base session + incremental sessions
FIELDS = ['name', 'dataset', 'seed', 'A_Base', 'A_Last', 'A_Avg', 'sessions', 'flags', 'commit', 'gpu', 'batch',
          'minutes', 'resumed', 'date', 'source']


def check(r):
    acc = json.loads(r['sessions'])
    want = SESSIONS[r['dataset']]
    problems = []
    if len(acc) != want:
        problems.append(f'{len(acc)} sessions, expected {want}')
    for m, v in [('A_Base', acc[0]), ('A_Last', acc[-1]), ('A_Avg', sum(acc) / len(acc))]:
        if abs(float(r[m]) - v) > 0.006:
            problems.append(f'{m} {r[m]} but the session list gives {v:.3f}')
    return problems


def main(paths, out='results/all_runs.csv'):
    runs, bad = {}, 0
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        print(f'{path}: {len(rows)} rows')
        for r in rows:
            if r['name'] == 'smoke':
                continue
            r['source'] = os.path.basename(path)
            key = (r['name'], r['dataset'], int(r['seed']))
            if r['name'] not in PLAN:
                print('  not in the plan, kept:', *key)
            problems = check(r)
            if problems:
                bad += 1
                print('  CHECK FAILED', *key, '|', '; '.join(problems))
            if key in runs:
                old = runs[key]
                same = all(abs(float(old[m]) - float(r[m])) < 0.006 for m in ['A_Base', 'A_Last', 'A_Avg'])
                print(f"  duplicate {key[0]} {key[1]} seed {key[2]}: in {old['source']} and {r['source']}, numbers "
                      + ('identical, kept the first' if same else 'DIFFER, kept the first: check both logs'))
                continue
            runs[key] = r

    order = {n: i for i, n in enumerate(PLAN)}
    ds_order = {'cifar100': 0, 'cub200': 1, 'mini_imagenet': 2}
    keys = sorted(runs, key=lambda k: (order.get(k[0], len(order)), ds_order[k[1]], k[2]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for k in keys:
            w.writerow(runs[k])

    planned = [(n, ds, s) for n, (dss, seeds) in PLAN.items() for ds in dss for s in seeds]
    missing = [k for k in planned if k not in runs]
    print(f'\nwrote {out}: {len(keys)} runs | {len(planned) - len(missing)}/{len(planned)} planned runs present'
          + (f' | {bad} rows failed the check' if bad else ' | every row passed the check'))
    for k in missing:
        print('  missing:', *k)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__ or 'usage: python colab/merge_results.py <runs csv> [<runs csv> ...]')
    main(sys.argv[1:])
