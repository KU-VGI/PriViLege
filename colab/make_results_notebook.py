# Builds colab/PriViLege_results.ipynb: the reproduction's results next to the paper's, read only from
# results/all_runs.csv (made by colab/merge_results.py). No training, no GPU, no Drive.
#
#   python colab/make_results_notebook.py colab/PriViLege_results.ipynb          (write the notebook)
#   python colab/make_results_notebook.py colab/PriViLege_results.ipynb --run    (also run it and save the outputs)
import json, os, sys

cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md(r'''
# PriViLege (CVPR 2024): reproduction results

Our results for **Table 1**, **Table 3**, **Table 4** and **Figure 3**, each next to the paper's. Every number comes from
one file, `results/all_runs.csv`: one row per experiment, dataset and seed, with the accuracy after every session.
Nothing is trained here, so this runs anywhere in seconds (VS Code, Jupyter, or Colab on a CPU runtime).

- **How the runs were made:** `colab/PriViLege_colab.ipynb` (the authors' code, branch `t4-fixes`), 5 seeds per
  experiment (1 for CLIP zero-shot, which has no training), 5 base epochs and 3 epochs per incremental session.
- **How the file was made:** `python colab/merge_results.py results/raw/*.csv` merges the results files of every
  machine (`results/raw/`) and checks each row against its session list.
- **Metrics:** A_Base = accuracy after the base session, A_Last = after the last session, A_Avg = mean over all sessions.
  Ours are mean ± standard deviation over the seeds; Δ = ours − paper (green within 1 point, orange within 3, red beyond).
''')

md('## 1. Load the results')
code(r'''
import json, os, urllib.request
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Opened from the repo (colab/ or the repo root) it reads the local file; anywhere else it downloads it from GitHub.
URL = 'https://raw.githubusercontent.com/AkilaDissanayake/PriViLege/t4-fixes/results/all_runs.csv'
CSV = next((p for p in ['../results/all_runs.csv', 'results/all_runs.csv'] if os.path.exists(p)), None)
if CSV is None:
    CSV = urllib.request.urlretrieve(URL, 'all_runs.csv')[0]
runs = pd.read_csv(CSV)
runs['sessions'] = runs['sessions'].map(json.loads)
print(f'{len(runs)} runs read from {CSV}')
''')

md(r'''
## 2. Paper numbers and helpers
Paper values are (mean, std) of A_Base, A_Last and A_Avg. Figure 3 has no table: its values were read off the plot
(accuracy per session, within about ± 0.3), so it has no std.
''')
code(r'''
PAPER = {
    'Table 1': {('PriViLege', 'cifar100'):      [(90.88, .20), (86.06, .32), (88.08, .20)],
                ('PriViLege', 'cub200'):        [(82.21, .35), (75.08, .52), (77.50, .33)],
                ('PriViLege', 'mini_imagenet'): [(96.68, .06), (94.10, .13), (95.27, .11)]},
    'Table 3': {('abl_FT', 'cub200'):      [(84.21, .13), (3.79, 1.47), (21.60, 1.32)],
                ('abl_PKT', 'cub200'):     [(79.06, .77), (70.81, .76), (73.36, .77)],
                ('abl_PKT_ED', 'cub200'):  [(80.31, .54), (72.70, .45), (75.04, .40)],
                ('abl_PKT_SKD', 'cub200'): [(82.10, .57), (73.44, .40), (76.27, .30)],
                ('PriViLege', 'cub200'):   [(82.21, .35), (75.08, .52), (77.50, .33)]},
    'Table 4': {('layers_0', 'cub200'):  [(76.07, .56), (60.19, 1.11), (67.08, .71)],
                ('abl_PKT', 'cub200'):   [(79.06, .77), (70.81, .76), (73.36, .77)],
                ('layers_5', 'cub200'):  [(78.42, .84), (68.52, .84), (71.99, .84)],
                ('layers_7', 'cub200'):  [(76.96, .74), (63.15, 2.73), (68.06, 1.54)],
                ('layers_10', 'cub200'): [(74.95, .78), (57.78, 2.30), (64.19, 1.59)],
                ('layers_12', 'cub200'): [(73.62, 2.72), (56.02, 1.47), (62.71, 2.14)]},
}
FIG3 = {   # Figure 3, accuracy per session (base, 1..8), read off the plot
    'clip_zeroshot':   [78.8, 76.6, 75.2, 73.1, 72.5, 72.4, 72.7, 72.3, 71.0],
    'clip_PKT':        [82.3, 79.5, 77.6, 75.4, 74.0, 73.2, 73.3, 72.6, 71.4],
    'clip_PKT_ED':     [83.7, 81.2, 79.8, 77.6, 76.7, 76.3, 76.5, 76.2, 74.7],
    'clip_PKT_ED_SKD': [84.2, 82.1, 81.0, 79.0, 78.8, 78.4, 78.6, 78.3, 77.1],
}
PAPER['Figure 3'] = {(n, 'cifar100'): [(v[0], None), (v[-1], None), (round(float(np.mean(v)), 2), None)]
                     for n, v in FIG3.items()}

LABELS = {'PriViLege': 'PriViLege (Ours)', 'abl_FT': 'Baseline (fine-tune + proto)', 'abl_PKT': 'PKT',
          'abl_PKT_ED': 'PKT + L_ED', 'abl_PKT_SKD': 'PKT + L_SKD', 'layers_0': '0 layers', 'layers_5': '5 layers',
          'layers_7': '7 layers', 'layers_10': '10 layers', 'layers_12': '12 layers',
          'clip_zeroshot': 'CLIP zero-shot', 'clip_PKT': 'CLIP-PKT', 'clip_PKT_ED': 'CLIP-PKT + L_ED',
          'clip_PKT_ED_SKD': 'CLIP-PKT + L_ED + L_SKD',
          'abl_Baseline': 'Baseline, released -ft (not used)'}
DS = {'cub200': 'CUB200', 'cifar100': 'CIFAR-100', 'mini_imagenet': 'miniImageNet'}
METRICS = ['A_Base', 'A_Last', 'A_Avg']

def of(name, ds):
    return runs[(runs['name'] == name) & (runs['dataset'] == ds)]

def ms(x):
    return f'{np.mean(x):.2f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.2f}'

def label(table, name, ds):
    if table == 'Table 1':
        return DS[ds]
    if table == 'Table 4' and name == 'abl_PKT':
        return '2 layers (= PKT)'
    return LABELS[name]

def color(v):
    if pd.isna(v):
        return ''
    return 'color: green' if abs(v) <= 1 else ('color: orange' if abs(v) <= 3 else 'color: red')

def compare(table):
    rows = []
    for (name, ds), paper in PAPER[table].items():
        r = of(name, ds)
        row = {'': label(table, name, ds), 'seeds': len(r)}
        for m, (pm, ps) in zip(METRICS, paper):
            row[f'{m} ours'] = ms(r[m]) if len(r) else '—'
            row[f'{m} paper'] = f'{pm:.2f}' + (f' ± {ps:.2f}' if ps is not None else ' (fig.)')
            row[f'Δ {m}'] = round(r[m].mean() - pm, 2) if len(r) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index('')
    display(Markdown(f'**{table}: ours vs. paper**'))
    display(df.style.map(color, subset=[f'Δ {m}' for m in METRICS]).format(precision=2, na_rep='—'))
    return df

def plot_sessions(table, paper_curves=None):
    plt.figure(figsize=(7, 4))
    for i, (name, ds) in enumerate(PAPER[table]):
        r, c, lab = of(name, ds), f'C{i}', label(table, name, ds)
        if len(r):
            acc = np.array(r['sessions'].tolist())
            x = np.arange(acc.shape[1])
            plt.plot(x, acc.mean(0), '-o', color=c, ms=4, label=f'{lab} (ours, n={len(r)})')
            if len(r) > 1:
                sd = acc.std(0, ddof=1)
                plt.fill_between(x, acc.mean(0) - sd, acc.mean(0) + sd, color=c, alpha=.15)
        if paper_curves and name in paper_curves:
            plt.plot(paper_curves[name], '--*', color=c, alpha=.7, label=f'{lab} (paper)')
    plt.xlabel('session (0 = base)'); plt.ylabel('accuracy (%)'); plt.title(f'{table}: accuracy per session')
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

def show(table):
    compare(table)
    plot_sessions(table, FIG3 if table == 'Figure 3' else None)
''')

md(r'''
## 3. What is in the file
Seeds per experiment and the GPU time of each run (NVIDIA A40). A run that was interrupted and resumed only records the
time of its last part, so those minutes are too low.
''')
code(r'''
summary = (runs.groupby(['name', 'dataset'], sort=False)
               .agg(seeds=('seed', lambda s: ', '.join(map(str, sorted(s)))),
                    minutes_per_run=('minutes', 'median'), resumed=('resumed', lambda s: (s == 'yes').sum()))
               .rename(index=LABELS, level=0).rename(index=DS, level=1))
display(summary)
print(f"{len(runs)} runs | about {runs['minutes'].sum() / 60:.1f} GPU hours recorded")
''')

md(r'''
## 4. Table 1: PriViLege on the three benchmarks
CUB200: 100 base classes + 10 sessions of 10-way 5-shot. CIFAR-100 and miniImageNet: 60 base classes + 8 sessions of
5-way 5-shot.
''')
code("show('Table 1')")

md(r'''
## 5. Table 3: ablation on CUB200
- **Baseline**: fine-tuning the pre-trained ViT (blocks 0–1 and the classifier in the base session, every parameter in
  the incremental sessions), classified with the prototype classifier; no prompts (`-ft -plain_ft`).
- **PKT**, **PKT + L_ED**, **PKT + L_SKD**: the method without one or both losses. **Ours** is the Table 1 CUB200 run.
''')
code("show('Table 3')")
md(r'''
The released `-ft` option alone (`abl_Baseline`) is not the paper's Baseline: it keeps the classifier frozen at random
weights, runs the images through prompts left at random values, and builds the prototypes from a different feature
than the test images. It is listed here only to show the difference; Table 3 above uses `-ft -plain_ft`.
''')
code(r'''
PAPER['Baseline variants'] = {('abl_Baseline', 'cub200'): PAPER['Table 3'][('abl_FT', 'cub200')],
                              ('abl_FT', 'cub200'):       PAPER['Table 3'][('abl_FT', 'cub200')]}
show('Baseline variants')
''')

md(r'''
## 6. Table 4: number of tuned layers (CUB200)
Without the two losses, as in the paper; the 2-layer row is the PKT row of Table 3.
''')
code("show('Table 4')")

md(r'''
## 7. Figure 3: PriViLege on CLIP (CIFAR-100)
CLIP ViT-B/16 (LAION-2B). Dashed lines: the paper's curves, read off Figure 3.
''')
code("show('Figure 3')")

md('## 8. Every comparison in one table')
code(r'''
allrows = []
for t in ['Table 1', 'Table 3', 'Table 4', 'Figure 3']:
    for (name, ds), paper in PAPER[t].items():
        r = of(name, ds)
        row = {'table': t, 'row': label(t, name, ds), 'seeds': len(r)}
        for m, (pm, ps) in zip(METRICS, paper):
            row[m] = round(r[m].mean(), 2) if len(r) else np.nan
            row[f'Δ {m}'] = round(r[m].mean() - pm, 2) if len(r) else np.nan
        allrows.append(row)
alldf = pd.DataFrame(allrows).set_index(['table', 'row'])
display(alldf.style.map(color, subset=[f'Δ {m}' for m in METRICS]).format(precision=2, na_rep='—'))
''')

md(r'''
## 9. Every run, seed by seed
''')
code(r'''
cols = ['name', 'dataset', 'seed', 'A_Base', 'A_Last', 'A_Avg', 'minutes', 'resumed', 'commit', 'flags']
display(runs[cols].style.hide(axis='index').format(precision=2))
''')

nb = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
for c in nb['cells']:   # store source as a list of lines, like Jupyter does
    c['source'] = [l + '\n' for l in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')

out = sys.argv[1]
if '--run' in sys.argv:   # run it (from the notebook's folder, like Jupyter) and keep the outputs
    import nbformat
    from nbclient import NotebookClient
    nbo = nbformat.reads(json.dumps(nb), as_version=4)
    NotebookClient(nbo, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': os.path.dirname(os.path.abspath(out))}}).execute()
    nbformat.write(nbo, out)
else:
    json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
print('wrote', out, len(cells), 'cells' + (' (with outputs)' if '--run' in sys.argv else ''))
