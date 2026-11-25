# Builds colab/PriViLege_colab.ipynb (run with any Python 3).
#
#   python colab/make_notebook.py colab/PriViLege_colab.ipynb          (write the notebook)
#   python colab/make_notebook.py colab/PriViLege_colab.ipynb --run    (also run Part A, the results, and keep its outputs)
import json, os, sys

cells = []
def md(s, part=None):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
def code(s, part=None):
    cells.append({'cell_type': 'code', 'metadata': {'part': part} if part else {}, 'execution_count': None,
                  'outputs': [], 'source': s.strip('\n')})

# =====================================================================================================================
md(r'''
# PriViLege (CVPR 2024): reproduction

Our reproduction of **Table 1** (CUB200, CIFAR-100, miniImageNet), **Tables 3 and 4** (ablations on CUB200) and
**Figure 3** (PriViLege on CLIP, CIFAR-100) with the authors' code: branch `t4-fixes` of `AkilaDissanayake/PriViLege`,
their code plus the fixes needed on today's PyTorch and 16 GB GPUs.

- **Part A: Results** (any machine, no GPU, a few seconds). Every finished run is in one file, `results/all_runs.csv`
  (5 seeds per experiment, 1 for CLIP zero-shot). Part A reads it and shows each table and plot next to the paper's.
- **Part B: Training** (GPU: Colab or RunPod). Sets up data and code and runs experiments. Every planned run is already
  in `results/all_runs.csv`, so Part B skips them: it is only needed to repeat a run or add new ones.

**Metrics.** A_Base = accuracy after the base session, A_Last = after the last session, A_Avg = mean over all sessions.
Ours are mean ± standard deviation over the seeds; Δ = ours − paper (green within 1 point, orange within 3, red beyond).
''')

# =====================================================================================================================
md(r'''
# Part A: Results
## A1. Load the results
Reads `results/all_runs.csv` from the repo (or from GitHub when the notebook is opened on its own). After new runs in
Part B, run this cell again: their rows are added, and a repeated run replaces the earlier one.
''')
code(r'''
import csv, glob, json, os, urllib.request
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown

RESULTS_URL = 'https://raw.githubusercontent.com/AkilaDissanayake/PriViLege/t4-fixes/results/all_runs.csv'

def results_files():
    # results/all_runs.csv, then the results files of runs made with Part B (their rows win: newer)
    repo = globals().get('REPO')
    local = [p for p in ([f'{repo}/results/all_runs.csv'] if repo else []) + ['../results/all_runs.csv', 'results/all_runs.csv']
             if os.path.exists(p)]
    base = local[0] if local else urllib.request.urlretrieve(RESULTS_URL, 'all_runs.csv')[0]
    return [base] + (sorted(glob.glob(f"{globals()['RUNS']}/results/runs*.csv")) if 'RUNS' in globals() else [])

def load_runs():
    rows = {}
    for path in results_files():
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['name'] in EXPERIMENTS:   # skips the smoke test and anything not in the paper's tables
                    rows[(r['name'], r['dataset'], int(r['seed']))] = r
    df = pd.DataFrame(list(rows.values()))
    df['seed'] = df['seed'].astype(int)
    for c in ['A_Base', 'A_Last', 'A_Avg', 'minutes']:
        df[c] = df[c].astype(float)
    df['sessions'] = df['sessions'].map(json.loads)
    order = {n: i for i, n in enumerate(EXPERIMENTS)}
    return df.sort_values(['name', 'dataset', 'seed'], key=lambda s: s.map(order) if s.name == 'name' else s,
                          ignore_index=True)

# Experiments in the paper's tables, in table order, with their row labels
EXPERIMENTS = {'PriViLege': 'PriViLege (Ours)', 'abl_FT': 'Baseline (fine-tune + proto)', 'abl_PKT': 'PKT',
               'abl_PKT_ED': 'PKT + L_ED', 'abl_PKT_SKD': 'PKT + L_SKD',
               'layers_0': '0 layers', 'layers_5': '5 layers', 'layers_7': '7 layers', 'layers_10': '10 layers',
               'layers_12': '12 layers', 'clip_zeroshot': 'CLIP zero-shot', 'clip_PKT': 'CLIP-PKT',
               'clip_PKT_ED': 'CLIP-PKT + L_ED', 'clip_PKT_ED_SKD': 'CLIP-PKT + L_ED + L_SKD'}
DS = {'cub200': 'CUB200', 'cifar100': 'CIFAR-100', 'mini_imagenet': 'miniImageNet'}
METRICS = ['A_Base', 'A_Last', 'A_Avg']

runs = load_runs()
print(f'{len(runs)} runs from', ', '.join(results_files()))
''', part='A')

md(r'''
## A2. Paper numbers and helpers
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

def of(name, ds):
    return runs[(runs['name'] == name) & (runs['dataset'] == ds)]

def ms(x):
    return f'{np.mean(x):.2f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.2f}'

def label(table, name, ds):
    if table == 'Table 1':
        return DS[ds]
    if table == 'Table 4' and name == 'abl_PKT':
        return '2 layers (= PKT)'
    return EXPERIMENTS[name]

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

def plot_sessions(table):
    curves = FIG3 if table == 'Figure 3' else {}
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
        if name in curves:
            plt.plot(curves[name], '--*', color=c, alpha=.7, label=f'{lab} (paper)')
    plt.xlabel('session (0 = base)'); plt.ylabel('accuracy (%)'); plt.title(f'{table}: accuracy per session')
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

def show(table):
    compare(table)
    plot_sessions(table)
''', part='A')

md(r'''
## A3. What is in the file
Seeds and GPU time per experiment (NVIDIA A40). A run that was interrupted and resumed only records the time of its
last part.
''')
code(r'''
summary = (runs.groupby(['name', 'dataset'], sort=False)
               .agg(seeds=('seed', lambda s: ', '.join(map(str, sorted(s)))),
                    minutes_per_run=('minutes', 'median'), resumed=('resumed', lambda s: (s == 'yes').sum()))
               .rename(index=EXPERIMENTS, level=0).rename(index=DS, level=1))
display(summary)
print(f"{len(runs)} runs | about {runs['minutes'].sum() / 60:.1f} GPU hours recorded")
''', part='A')

md(r'''
## A4. Table 1: PriViLege on the three benchmarks
CUB200: 100 base classes + 10 sessions of 10-way 5-shot. CIFAR-100 and miniImageNet: 60 base classes + 8 sessions of
5-way 5-shot.
''')
code("show('Table 1')", part='A')

md(r'''
## A5. Table 3: ablation on CUB200
- **Baseline**: fine-tuning the pre-trained ViT with the prototype classifier, no prompts (`-ft -plain_ft`): ViT blocks
  0–1 and the classifier are trained in the base session, every parameter in the incremental sessions, and the
  prototypes and the test images use the same ViT CLS feature.
- **PKT**, **PKT + L_ED**, **PKT + L_SKD**: the method without one or both losses. **Ours** is the Table 1 CUB200 run.
''')
code("show('Table 3')", part='A')

md(r'''
## A6. Table 4: number of tuned layers (CUB200)
Without the two losses, as in the paper; the 2-layer row is the PKT row of Table 3.
''')
code("show('Table 4')", part='A')

md(r'''
## A7. Figure 3: PriViLege on CLIP (CIFAR-100)
CLIP ViT-B/16 (LAION-2B). Dashed lines: the paper's curves, read off Figure 3.
''')
code("show('Figure 3')", part='A')

md('## A8. Every comparison in one table')
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
''', part='A')

md('## A9. Every run, seed by seed')
code(r'''
cols = ['name', 'dataset', 'seed', 'A_Base', 'A_Last', 'A_Avg', 'minutes', 'resumed', 'commit', 'flags']
display(runs[cols].style.hide(axis='index').format(precision=2))
''', part='A')

# =====================================================================================================================
md(r'''
# Part B: Training
Only needed to repeat a run or add new ones: every planned run is in `results/all_runs.csv` and is skipped.

### Once only
1. **Drive.** The folder owner shares https://drive.google.com/drive/folders/1UuLlbEoUMMv8ShyRU-x117Ir3ePqYPeq with your
   Google account as **Editor**. Colab: signed in as that account, open the link, right-click the folder →
   **Organize → Add shortcut → My Drive**.
2. **RunPod only.** On your PC: install rclone (`winget install Rclone.Rclone`), run `rclone authorize "drive"`, sign in
   with that same account and copy the token it prints (one line, `{...}`). B1b asks for it once per pod.

### Every session
- **Colab:** Runtime → Change runtime type → **T4 GPU** or better.
- **RunPod:** a PyTorch template on an A40, RTX 4090 or better (the code runs in fp32, where an H100 is no faster), with
  **150 GB container disk**. No network volume is needed: everything is kept in the shared Drive folder. In a
  JupyterLab terminal:
  `wget https://raw.githubusercontent.com/AkilaDissanayake/PriViLege/t4-fixes/colab/PriViLege_colab.ipynb` and open it.

Run **B1a → B1f**, then **B2** and **B3**. Every finished run is copied to the shared Drive folder (`runs_v2/`) at once,
and on RunPod the checkpoints of running jobs every 10 minutes: after any stop, a new session continues each run from
its last checkpoint. Training stops on its own at the first NaN loss (a `FloatingPointError`).

## B1. Setup
''')

code(r'''
# B1a. Where datasets, logs and results live. Colab: the shared Drive folder, mounted. RunPod: a local folder kept in
# step with the same Drive folder by rclone (B1b).
DRIVE_FOLDER_ID = '1UuLlbEoUMMv8ShyRU-x117Ir3ePqYPeq'
DRIVE_FOLDER    = 'PriViLege'   # fallback: a folder with this name directly in My Drive (e.g. for the owner)
import os, glob, socket, uuid
try:
    import google.colab
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

def find_drive_folder():
    # A shortcut in My Drive exposes the shared folder here, whatever the shortcut is called or where it sits
    hits = [p for p in glob.glob(f'/content/drive/.shortcut-targets-by-id/{DRIVE_FOLDER_ID}/*') if os.path.isdir(p)]
    if hits:
        return hits[0]
    own = f'/content/drive/MyDrive/{DRIVE_FOLDER}'
    return own if os.path.isdir(own) else None

def usable(folder):
    # RunPod network volumes refuse chmod, which rclone's install and git need: use the container disk instead
    try:
        os.makedirs(folder, exist_ok=True)
        probe = f'{folder}/.probe_{uuid.uuid4().hex[:8]}'
        open(probe, 'w').close(); os.chmod(probe, 0o755); os.remove(probe)
        return True
    except OSError:
        return False

if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA, REPO, DRIVE = '/content/data', '/content/PriViLege', find_drive_folder()
    MACHINE = 'colab'
else:
    BASE = '/workspace' if usable('/workspace') else os.path.expanduser('~/work')
    DATA, REPO, DRIVE = f'{BASE}/data', f'{BASE}/PriViLege', f'{BASE}/privilege_drive'
    MACHINE = 'runpod_' + os.environ.get('RUNPOD_POD_ID', socket.gethostname())   # each pod writes its own results file
    os.makedirs(DRIVE, exist_ok=True)
if DRIVE is None:
    raise FileNotFoundError('The shared Drive folder is not visible to this account. Open '
                            f'https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID} while signed in as the account '
                            'you mounted, right-click the folder -> Organize -> Add shortcut -> My Drive, then run this '
                            'cell again. If the link says you need access, ask the owner to share it with you as Editor.')
probe = f'{DRIVE}/.write_test_{uuid.uuid4().hex[:8]}'
try:
    open(probe, 'w').close(); os.remove(probe)
except OSError as e:
    raise PermissionError(f'{DRIVE} is read-only for this account: ask the folder owner to share it as Editor.') from e
RUN_SET = 'runs_v2'                 # logs/, results/ and resume/ of this reproduction in the shared Drive folder
RUNS = f'{DRIVE}/{RUN_SET}'
for d in [f'{DRIVE}/datasets', f'{RUNS}/logs', f'{RUNS}/results', f'{RUNS}/resume', DATA]:
    os.makedirs(d, exist_ok=True)
print('Running on', MACHINE, '| data and runs in', DRIVE)
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')

code(r'''
# B1b. RunPod only (does nothing on Colab): link to the shared Drive folder with rclone and copy its datasets/ and
# runs_v2/ to the pod. drive_push() copies logs and results back after every run.
import subprocess, shutil

def drive_push(resume=False):
    pass   # Colab writes straight into the mounted Drive folder

def drive_drop_resume(name, dataset, seed):
    pass

if not ON_COLAB:
    import zipfile, getpass
    RCLONE      = f'{BASE}/bin/rclone'
    RCLONE_CONF = f'{BASE}/rclone.conf'   # contains your Drive token: do not share it
    if not os.path.exists(RCLONE):
        os.makedirs(f'{BASE}/bin', exist_ok=True)
        zpath = f'{BASE}/bin/rclone.zip'
        url = 'https://downloads.rclone.org/v1.68.2/rclone-v1.68.2-linux-amd64.zip'
        # curl, as in rclone's own install script (the server refuses Python's urllib); wget if there is no curl
        if not any(shutil.which(c[0]) and subprocess.run(c).returncode == 0
                   for c in (['curl', '-fsSL', '-o', zpath, url], ['wget', '-q', '-O', zpath, url])):
            raise RuntimeError(f'Could not download rclone from {url}: install it by hand into {RCLONE}.')
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith('/rclone'))
            with z.open(member) as src, open(RCLONE, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        os.chmod(RCLONE, 0o755); os.remove(zpath)

    def rclone(*args):
        return subprocess.run([RCLONE, '--config', RCLONE_CONF, *args], text=True, capture_output=True)

    if 'gdrive:' not in rclone('listremotes').stdout:
        token = getpass.getpass('Paste the token printed by `rclone authorize "drive"` on your PC: ').strip()
        r = rclone('config', 'create', 'gdrive', 'drive', 'scope=drive', f'root_folder_id={DRIVE_FOLDER_ID}',
                   f'token={token}', '--non-interactive')
        if r.returncode != 0:
            raise RuntimeError('rclone config failed: ' + r.stderr[-500:])
    r = rclone('lsf', 'gdrive:', '--max-depth', '1')
    if r.returncode != 0:
        raise RuntimeError('Cannot read the shared Drive folder with rclone. Check that the account has Editor access, '
                           f'or delete {RCLONE_CONF} and run this cell again. ' + r.stderr[-500:])
    # resume/: checkpoints of unfinished runs from an earlier pod or Colab, so this pod continues them
    for d in ['datasets', f'{RUN_SET}/logs', f'{RUN_SET}/results', f'{RUN_SET}/resume']:
        if rclone('lsf', f'gdrive:{d}', '--max-depth', '1').returncode != 0:
            continue   # not created yet
        print(f'Copying Drive {d}/ to the pod ...')
        get_ipython().system(f'{RCLONE} --config {RCLONE_CONF} copy gdrive:{d} "{DRIVE}/{d}" --update --stats-one-line -P')

    def drive_push(resume=False):
        # logs and results after every run; resume=True: also the checkpoints of running jobs
        for d in [f'{RUN_SET}/logs', f'{RUN_SET}/results'] + ([f'{RUN_SET}/resume'] if resume else []):
            # running jobs' logs grow during the upload, so their checksum cannot match: upload them as they are
            grow = ['--local-no-check-updated', '--ignore-checksum'] if d.endswith('/logs') else []
            r = rclone('copy', f'{DRIVE}/{d}', f'gdrive:{d}', '--update', '--exclude', '*.tmp', *grow)
            if r.returncode != 0:
                print(f'!! Could not copy {d}/ to Drive (the files are safe on the pod):', r.stderr[-300:])

    def drive_drop_resume(name, dataset, seed):
        rclone('purge', f'gdrive:{RUN_SET}/resume/{name}/{dataset}/seed{seed}')   # a finished run's checkpoint
''')

code(r'''
# B1c. Code: the t4-fixes branch. The commit id is recorded with every result.
BRANCH = 't4-fixes'
if not os.path.exists(REPO):
    !git clone -q -b {BRANCH} https://github.com/AkilaDissanayake/PriViLege.git {REPO}
else:
    !git -C {REPO} fetch -q origin && git -C {REPO} checkout -q {BRANCH} && git -C {REPO} pull -q --ff-only || echo "!! git pull failed: run  git -C {REPO} status  in a terminal"
%cd {REPO}
COMMIT = !git rev-parse --short HEAD
COMMIT = COMMIT[0]
!git log -1 --oneline
''')

code(r'''
# B1d. Packages (not requirements.txt, a full Anaconda dump that breaks Colab). timm 0.6.13 loads the same ViT-B/16
# checkpoint as the authors' 0.6.12, which fails to import on Python 3.12. open_clip: CLIP experiments (Figure 3).
!pip install -q timm==0.6.13 open_clip_torch gdown transformers pandas scikit-learn matplotlib tqdm
import timm, torch, transformers, open_clip
print('timm', timm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__,
      '| open_clip', open_clip.__version__)
assert timm.__version__ == '0.6.13', 'Restart the session (Colab: Runtime -> Restart session) and run B1a-B1d again'
assert torch.cuda.is_available(), 'No GPU: Colab: Runtime -> Change runtime type -> GPU. RunPod: start a GPU pod.'
GPU = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
GRAD_CKPT = GPU_GB < 20   # 16 GB GPUs (T4): batch 128 only fits with gradient checkpointing (same results, slower)
print(f'GPU: {GPU} ({GPU_GB:.0f} GB)' + (' -> -grad_ckpt on' if GRAD_CKPT else ''))
''')

code(r'''
# B1e. Datasets: copy one archive from Drive and extract it on the local disk (many small files from Drive are slow).
# Archives missing from Drive are downloaded from the CEC folder used by FSCIL papers:
# https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN
GDRIVE_IDS = {'cub200': '1EtKb1Bxd9ZtK-WUFk5dIhoOdhLeohA1q',          # CUB_200_2011.tgz
              'mini_imagenet': '1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U'}   # miniimagenet.tar
ARCHIVES = {   # dataset -> (archive in Drive, folder after extraction)
    'cifar100':      ('cifar-100-python.tar.gz', 'cifar-100-python'),
    'cub200':        ('CUB_200_2011.tgz',        'CUB_200_2011'),
    'mini_imagenet': ('miniimagenet.tar',        'miniimagenet'),
}

def prepare(dataset):
    archive, folder = ARCHIVES[dataset]
    src = f'{DRIVE}/datasets/{archive}'
    if not os.path.isdir(f'{DATA}/{folder}'):
        if not os.path.exists(src):
            if dataset == 'cifar100':
                print('CIFAR-100 is not in Drive yet: train.py downloads it (copied to Drive afterwards).')
            else:
                download_to_drive(dataset)
        if os.path.exists(src):
            print(f'Extracting {archive} ...')
            get_ipython().system(f'tar -xf "{src}" -C {DATA} --no-same-owner')
    if dataset == 'mini_imagenet':
        # the repo's split files: the standard FSCIL split the loader and the session lists expect
        os.makedirs(f'{DATA}/miniimagenet/split', exist_ok=True)
        get_ipython().system(f'cp {REPO}/data/index_list/mini_imagenet/*.csv {DATA}/miniimagenet/split/')

def download_to_drive(dataset):
    import gdown
    archive = ARCHIVES[dataset][0]
    tmp, dst = f'{DATA}/{archive}.download', f'{DRIVE}/datasets/{archive}'
    print(f'{archive} is not in Drive: downloading it from the CEC folder ...')
    gdown.download(id=GDRIVE_IDS[dataset], output=tmp, quiet=False)
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 10_000_000:
        raise RuntimeError(f'Download of {archive} failed (Google Drive quota?). Try again later, or put it in '
                           f'{DRIVE}/datasets/ by hand from the CEC folder.')
    shutil.copy(tmp, dst + '.part'); os.replace(dst + '.part', dst); os.remove(tmp)

def save_cifar_to_drive():
    src, dst = f'{DATA}/cifar-100-python.tar.gz', f'{DRIVE}/datasets/cifar-100-python.tar.gz'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
''')

code(r'''
# B1f. Pretrained weights, downloaded once so a network problem shows up now instead of mid-run: ViT-B/16
# (ImageNet-21k -> 1k AugReg, as the authors), BERT, and CLIP ViT-B/16 (LAION-2B).
m = timm.create_model('vit_base_patch16_224', pretrained=True)
assert 'i21k' in m.default_cfg['url'], m.default_cfg['url']
del m
from transformers import BertTokenizer, BertModel
BertTokenizer.from_pretrained('bert-base-cased'); BertModel.from_pretrained('bert-base-cased')
open_clip.create_model_and_transforms('ViT-B/16', pretrained='laion2b_s34b_b88k')
print('ViT-B/16, bert-base-cased and CLIP ViT-B/16 (LAION-2B) ready')
''')

md(r'''
## B2. Experiments and runner
`run()` calls `train.py` with the official arguments (`run_script/vit_run_pretrain.py`) plus each experiment's flags. A
run is skipped when it is in `results/all_runs.csv` or in a results file of this reproduction (`runs_v2/results/`);
an interrupted run resumes from its last saved epoch or session. Each finished run adds one row (A_Base, A_Last, A_Avg
and every session's accuracy) to this machine's results file, `runs_v2/results/runs_<machine>.csv`.
''')
code(r'''
import ast, csv, json, time, collections, threading, queue
import numpy as np, pandas as pd
from IPython.display import display

LOG_CSV = f'{RUNS}/results/runs_{MACHINE}.csv'
RESUME  = f'{RUNS}/resume'
SEEDS   = [1, 2, 3, 4, 5]
VIT     = '-vit -comp_out 1'

# name -> datasets, train.py flags, epoch overrides. Every experiment has its own name (= -out), because the checkpoint
# path does not include the -ED / -SKD / -taskblock / -ft flags.
EXP = {
    'smoke':           dict(datasets=['cifar100'], flags=f'{VIT} -LT -ED -SKD', epochs_base=1, epochs_new=1),
    # Table 1 (and the 'Ours' row of Table 3)
    'PriViLege':       dict(datasets=['cifar100', 'cub200', 'mini_imagenet'], flags=f'{VIT} -LT -ED -SKD'),
    # Table 3: Baseline = fine-tuning + prototype classifier, no prompts; then the method without one or both losses
    'abl_FT':          dict(datasets=['cub200'], flags=f'{VIT} -ft -plain_ft'),
    'abl_PKT':         dict(datasets=['cub200'], flags=f'{VIT} -LT'),
    'abl_PKT_ED':      dict(datasets=['cub200'], flags=f'{VIT} -LT -ED'),
    'abl_PKT_SKD':     dict(datasets=['cub200'], flags=f'{VIT} -LT -SKD'),
    # Table 4: number of tuned layers, without the two losses (2 layers = abl_PKT)
    **{f'layers_{L}':  dict(datasets=['cub200'], flags=f'{VIT} -LT -taskblock {L}') for L in [0, 5, 7, 10, 12]},
    # Figure 3: CLIP ViT-B/16 (LAION-2B) on CIFAR-100. Zero-shot has no training, so one run is enough.
    'clip_zeroshot':   dict(datasets=['cifar100'], flags='-clip', epochs_base=0, seeds=[1]),
    'clip_PKT':        dict(datasets=['cifar100'], flags=f'{VIT} -pret_clip -LT'),
    'clip_PKT_ED':     dict(datasets=['cifar100'], flags=f'{VIT} -pret_clip -LT -ED'),
    'clip_PKT_ED_SKD': dict(datasets=['cifar100'], flags=f'{VIT} -pret_clip -LT -ED -SKD'),
}
PLAN = [(n, ds, s) for n, e in EXP.items() if n != 'smoke' for ds in e['datasets'] for s in e.get('seeds', SEEDS)]

def finished_runs():
    # results/all_runs.csv first, then this reproduction's results files (a repeated run's newer row wins)
    done = {}
    for path in [f'{REPO}/results/all_runs.csv'] + sorted(glob.glob(f'{RUNS}/results/runs*.csv')):
        if os.path.exists(path):
            with open(path, newline='') as f:
                done.update({(r['name'], r['dataset'], int(r['seed'])): r for r in csv.DictReader(f)})
    return done

def describe_state(path):
    s = json.load(open(path))
    where = (f"base epoch {s['epoch'] + 1}/{s['epochs_base']}" if s['phase'] == 'base'
             else f"session {s['session']}/{s['sessions'] - 1}")
    return f"saved after {where} at {s['saved_at']}"

_LOCK = threading.Lock()   # parallel runs share the results file, the CIFAR archive copy and the Drive upload
CKPT_UPLOAD_MIN = 10       # RunPod: upload the checkpoints of running jobs to Drive this often (minutes)
_UPLOADER = []

def start_checkpoint_upload():
    # RunPod only (Colab saves checkpoints straight into Drive): a lost pod loses at most CKPT_UPLOAD_MIN of training
    if ON_COLAB or _UPLOADER:
        return
    def loop():
        while True:
            time.sleep(CKPT_UPLOAD_MIN * 60)
            with _LOCK:
                drive_push(resume=True)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    _UPLOADER.append(t)

def run(name, dataset, seed, force=False, batch_size=128, gpu=None, quiet=False):
    # force: run again even if finished. gpu: GPU index (multi-GPU pods). quiet: output only to the log (run_parallel).
    e = EXP[name]
    start_checkpoint_upload()
    epochs_base, epochs_new = e.get('epochs_base', 5), e.get('epochs_new', 3)
    done = finished_runs().get((name, dataset, seed))
    ckpt = f'{RESUME}/{name}/{dataset}/seed{seed}'
    if done and not force:
        print(f"== {name} | {dataset} | seed {seed}: already finished (A_Base {done['A_Base']}, "
              f"A_Last {done['A_Last']}, A_Avg {done['A_Avg']}). Skipping.")
        return True
    if force:
        shutil.rmtree(ckpt, ignore_errors=True)
    prepare(dataset)
    state_json = f'{ckpt}/resume_state.json'
    resumed = os.path.exists(state_json)
    tag = f'[GPU {gpu}] ' if gpu is not None else ''
    print(f'{tag}>> {name} | {dataset} | seed {seed}: ' + (f'RESUMING, {describe_state(state_json)}' if resumed else 'starting'),
          flush=True)
    log = f'{RUNS}/logs/{name}_{dataset}_seed{seed}.log'
    cmd = (f'python train.py -project base -dataset {dataset} -base_mode ft_dot -new_mode avg_cos -gamma 0.1 '
           f'-lr_base 2e-4 -lr_new 2e-4 -decay 0.0005 -epochs_base {epochs_base} -epochs_new {epochs_new} '
           f'-schedule Cosine -milestones 20 30 45 -gpu 0 -temperature 16 -start_session 0 '
           f'-batch_size_base {batch_size} -seed {seed} {e["flags"]} -out {name} -dataroot {DATA} -ckpt_dir {ckpt}'
           + (' -grad_ckpt' if GRAD_CKPT else ''))
    if gpu is not None:   # pin the process to one GPU before it starts CUDA
        cmd = f'CUDA_VISIBLE_DEVICES={gpu} ' + cmd
    if not quiet:
        print(cmd, '\n')
    with open(log, 'a') as f:
        f.write(f"\n===== {'resume' if resumed else 'start'} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n{cmd}\n")
    t0 = time.time()
    if quiet:
        subprocess.run(['bash', '-c', f'cd {REPO} && {cmd} >> {log} 2>&1'])
    else:
        get_ipython().system(f'cd {REPO} && {cmd} 2>&1 | tee -a {log}')
    minutes = (time.time() - t0) / 60
    if dataset == 'cifar100':
        with _LOCK:
            save_cifar_to_drive()
    # This run's own results.txt (another run with the same seed may finish at the same time). The -clip zero-shot
    # trainer ignores -out and writes to checkpoint/Ours_Baseline.
    out_dir = 'Ours_Baseline' if '-clip ' in e['flags'] + ' ' else name
    hits = sorted((p for p in glob.glob(f'{REPO}/checkpoint/{out_dir}/{dataset}/**/*seed_{seed}*/results.txt', recursive=True)
                   if os.path.getmtime(p) > t0), key=os.path.getmtime)
    if not hits:
        msg = describe_state(state_json) if os.path.exists(state_json) else 'no checkpoint saved yet'
        print(f'\n{tag}!! {name} | {dataset} | seed {seed} stopped before finishing ({msg}). Re-run to continue. Log: {log}',
              flush=True)
        with _LOCK:
            drive_push(resume=True)   # keep its latest checkpoint in Drive too
        return False
    dst = f'{RUNS}/results/{name}/{dataset}/seed{seed}'
    os.makedirs(dst, exist_ok=True)
    shutil.copy(hits[-1], f'{dst}/results.txt')
    for pth in glob.glob(f'{os.path.dirname(hits[-1])}/*.pth'):   # model checkpoints (0.35-1 GB each): not needed
        os.remove(pth)
    accs = ast.literal_eval(open(hits[-1]).read().strip().splitlines()[-1])   # accuracy of every session
    a_base, a_last, a_avg = accs[0], accs[-1], sum(accs) / len(accs)
    with _LOCK:
        new_file = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['name', 'dataset', 'seed', 'flags', 'batch', 'gpu', 'commit', 'A_Base', 'A_Last', 'A_Avg',
                            'sessions', 'minutes', 'resumed', 'date'])
            w.writerow([name, dataset, seed, e['flags'] + (' -grad_ckpt' if GRAD_CKPT else ''), batch_size, GPU, COMMIT,
                        f'{a_base:.2f}', f'{a_last:.2f}', f'{a_avg:.2f}', json.dumps(accs), f'{minutes:.1f}',
                        'yes' if resumed else 'no', time.strftime('%Y-%m-%d %H:%M')])
        shutil.rmtree(ckpt, ignore_errors=True)   # finished: its resume state is no longer needed
        drive_push()
        drive_drop_resume(name, dataset, seed)
    print(f'{tag}== {name} | {dataset} | seed {seed}: A_Base {a_base:.2f} | A_Last {a_last:.2f} | A_Avg {a_avg:.2f} | '
          f'{minutes:.1f} min', flush=True)
    return True

def run_all(name, datasets=None, seeds=None, force=False):
    e = EXP[name]
    for ds in datasets or e['datasets']:
        for s in seeds or e.get('seeds', SEEDS):
            if not run(name, ds, s, force=force):   # stop at the first failure
                raise SystemExit(f'Stopped: {name} | {ds} | seed {s} did not finish. Fix the error above, then re-run.')

def run_parallel(jobs=None, seeds=None, gpus=None):
    # One run per GPU at a time (gpus=[0, 0] runs two side by side on one large GPU) until all are done, longest first.
    # jobs: list of (name, dataset, seed), default PLAN; seeds: keep only these seeds. Finished runs are skipped,
    # interrupted ones resume. A failed run does not stop the others; failures are listed at the end.
    gpus = list(range(torch.cuda.device_count())) if gpus is None else gpus
    jobs = [j for j in (jobs or PLAN) if seeds is None or j[2] in seeds or EXP[j[0]].get('seeds')]
    done = finished_runs()
    todo = sorted((j for j in jobs if j not in done), key=lambda j: -{'cifar100': 3, 'mini_imagenet': 3, 'cub200': 1}[j[1]])
    print(f'{len(jobs) - len(todo)} of {len(jobs)} runs already finished; {len(todo)} to run on GPUs {gpus}.')
    for ds in dict.fromkeys(j[1] for j in todo):   # extract each dataset once, before the parallel runs start
        prepare(ds)
    q = queue.Queue()
    for j in todo:
        q.put(j)
    failed, t0 = [], time.time()
    def worker(gpu):
        while True:
            try:
                name, ds, seed = q.get_nowait()
            except queue.Empty:
                return
            try:
                ok = run(name, ds, seed, gpu=gpu, quiet=True)
            except Exception as ex:   # keep the other GPUs busy
                print(f'[GPU {gpu}] !! {name} | {ds} | seed {seed}: {ex}', flush=True)
                ok = False
            if not ok:
                failed.append((name, ds, seed))
            print(f'   {q.qsize()} runs waiting | {(time.time() - t0) / 3600:.1f} h elapsed', flush=True)
    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'\nDone in {(time.time() - t0) / 3600:.1f} h. Failed: {failed or "none"}')
    if failed:
        print('Their logs are in', f'{RUNS}/logs/', '(the last lines show the error); then run run_parallel() again.')

def status():
    done = finished_runs()
    rows = []
    for n, ds, s in PLAN:
        state_json = f'{RESUME}/{n}/{ds}/seed{s}/resume_state.json'
        rows.append((n, ds, 'done' if (n, ds, s) in done else ('resumable' if os.path.exists(state_json) else 'not started')))
    df = pd.DataFrame(rows, columns=['experiment', 'dataset', 'state'])
    display(df.groupby(['experiment', 'dataset'], sort=False)['state'].value_counts().unstack(fill_value=0))
    print(f"{sum(st == 'done' for *_, st in rows)}/{len(rows)} planned runs done")

status()
''')

md(r'''
## B3. Run
Pick one line. Everything finished is skipped, so after any stop just run the same cell again. Watch a run live in a
terminal: `tail -f <RUNS>/logs/<name>_<dataset>_seed<k>.log`.
''')
code(r'''
run_parallel()                                    # every planned run not finished yet, one per GPU
# run('smoke', 'cifar100', 1)                     # quick environment check: 1 epoch per session
# run_all('abl_FT')                               # one experiment, all its seeds
# run('layers_12', 'cub200', 3, force=True)       # repeat one finished run
''')

md(r'''
## B4. Add the new runs to `results/all_runs.csv`
Copies the results files of `runs_v2/results/` into `results/raw/` of this machine's repo and merges everything with
`colab/merge_results.py` (checks every row, keeps one row per run, lists missing runs). Then run Part A again, and
download `results/` or commit it so the next session starts from the new file.
''')
code(r'''
for p in glob.glob(f'{RUNS}/results/runs*.csv'):
    shutil.copy(p, f'{REPO}/results/raw/{os.path.basename(p)}')
!cd {REPO} && python colab/merge_results.py results/raw/*.csv
''')

md(r'''
## B5. Free disk space (optional)
Finished runs delete their `.pth` checkpoints; this removes any left by interrupted runs.
''')
code("!find {REPO}/checkpoint -name '*.pth' -delete\n!df -h {DATA} | tail -1")

# =====================================================================================================================
nb = {'cells': cells, 'metadata': {'colab': {'provenance': [], 'gpuType': 'T4'},
                                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                   'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
      'nbformat': 4, 'nbformat_minor': 0}
for c in nb['cells']:   # store source as a list of lines, like Jupyter does
    c['source'] = [l + '\n' for l in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')

out = sys.argv[1]
if '--run' in sys.argv:
    # Run Part A (no GPU needed) from the notebook's folder, like Jupyter, and keep its outputs in the notebook
    import nbformat
    from nbclient import NotebookClient
    part_a = [i for i, c in enumerate(nb['cells']) if c['metadata'].get('part') == 'A']
    sub = nbformat.reads(json.dumps(dict(nb, cells=[nb['cells'][i] for i in part_a])), as_version=4)
    NotebookClient(sub, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': os.path.dirname(os.path.abspath(out))}}).execute()
    for i, c in zip(part_a, sub.cells):
        nb['cells'][i]['outputs'] = json.loads(json.dumps(c.outputs))
        nb['cells'][i]['execution_count'] = c.execution_count
        for o in nb['cells'][i]['outputs']:   # Jupyter's on-disk form: text as a list of lines
            for k in ('text',):
                if isinstance(o.get(k), str):
                    o[k] = o[k].splitlines(True)
for c in nb['cells']:
    c['metadata'].pop('part', None)
json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
print('wrote', out, len(cells), 'cells' + (' (Part A with outputs)' if '--run' in sys.argv else ''))
