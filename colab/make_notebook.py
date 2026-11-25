# Builds colab/PriViLege_colab.ipynb (run with any Python 3).
import json, sys

cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md(r'''
# PriViLege (CVPR 2024): reproduction on Colab or RunPod

Runs the authors' code (branch `t4-fixes` of `AkilaDissanayake/PriViLege`: their code plus the fixes needed on today's
PyTorch and on a 16 GB T4) for the paper's results: **Table 1** (CUB200, CIFAR-100, miniImageNet), **Tables 3 and 4**
(ablations on CUB200) and **Figure 3** (PriViLege on CLIP, CIFAR-100). Every experiment cell ends with our numbers next to
the paper's.

**Fresh start.** Every log, result and resume state of this reproduction lives in `PriViLege/runs_v2/` in the shared Drive
folder. Older runs (made before the L_ED NaN fix) stay untouched in `PriViLege/logs`, `results` and `resume` and are never
read, so every experiment here starts from the beginning. Datasets in `PriViLege/datasets/` are shared and reused.

### Once only
1. **Drive.** The folder owner shares https://drive.google.com/drive/folders/1UuLlbEoUMMv8ShyRU-x117Ir3ePqYPeq with your
   Google account as **Editor**. Signed in as that account, open the link, right-click the folder → **Organize → Add
   shortcut → My Drive** (Colab can only see a shared folder through a shortcut).
2. **RunPod only.** On your PC (PowerShell): `winget install Rclone.Rclone`, open a new terminal, run
   `rclone authorize "drive"`, sign in with that same account and copy the token it prints (one line, `{...}`).

### Every session (Colab or RunPod)
Run **1a → 1a-2 → 1b → 1c → 1d → 1e**, then the two cells of section 2, then **3. Status**.
- Colab: Runtime → Change runtime type → **T4 GPU** (or better). If 1c stops at the timm check: Runtime → **Restart
  session**, then run 1a–1c again.
- RunPod: a PyTorch template and a **100 GB volume** on `/workspace`. GPU: RTX 4090 or 5090 (24/32 GB: no `-grad_ckpt`).
  For speed take **several of them in one pod** (e.g. 8 x RTX 4090) and use section 4c: the code runs in fp32, where an
  H100/A100 is no faster than a 4090.
  In a JupyterLab terminal: `cd /workspace && wget https://raw.githubusercontent.com/AkilaDissanayake/PriViLege/t4-fixes/colab/PriViLege_colab.ipynb`,
  open it, and paste the rclone token when 1a-2 asks (once per volume).

### Order of the experiments
1. **4. Smoke test** (T4 ~15 min). Must end with `== smoke | cifar100 | seed 1: A_Base …`.
2. **4b. Quick checks** (T4 ~45 min): one real CUB200 run and CLIP zero-shot, printed next to the paper's numbers.
   If both are close, the code is confirmed: move to RunPod for the rest.
3. **5. Table 1** CUB200 first, then **6. Table 3** and **7. Table 4** (all CUB200, cheap), then Table 1 CIFAR-100 and
   miniImageNet (about 2 h per run on a T4), then **8. Figure 3** (CLIP; run one `clip_PKT_ED_SKD` seed first), then **9**.

### If anything stops
Re-run the setup cells and the same experiment cell: finished runs are skipped, an interrupted run resumes from its last
saved epoch or session. Training stops on its own at the first NaN loss (a `FloatingPointError`): send the last 30 lines of
the output. On RunPod every finished run is copied to the shared Drive folder (`runs_v2/`) at once, and the checkpoints of
running jobs every 10 minutes (`runs_v2/resume/`). A new pod's setup cell downloads them, so even after terminating a pod
the next one (or Colab) continues each unfinished run from its last uploaded checkpoint.
''')

md(r'''
## 1. Setup
''')

code(r'''
# 1a. Where datasets, logs and results live.
# Colab: the shared Drive folder, mounted. RunPod / other machines: a folder on the persistent volume, synced with the
# same shared Drive folder by rclone (cell 1a-2).
# The shared project folder: https://drive.google.com/drive/folders/1UuLlbEoUMMv8ShyRU-x117Ir3ePqYPeq
DRIVE_FOLDER_ID = '1UuLlbEoUMMv8ShyRU-x117Ir3ePqYPeq'
DRIVE_FOLDER    = 'PriViLege'   # fallback: a folder with this name directly in My Drive (e.g. for the owner)
import os, glob, uuid
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

if ON_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA  = '/content/data'
    REPO  = '/content/PriViLege'
    DRIVE = find_drive_folder()
else:
    BASE  = '/workspace' if os.path.isdir('/workspace') else os.path.expanduser('~')   # RunPod's persistent volume
    DATA  = f'{BASE}/data'
    REPO  = f'{BASE}/PriViLege'
    DRIVE = f'{BASE}/privilege_drive'   # local copy of the shared Drive folder
    os.makedirs(DRIVE, exist_ok=True)
# Stop here instead of quietly creating an empty folder in the wrong Drive and downloading everything again.
if DRIVE is None:
    raise FileNotFoundError('The shared Drive folder is not visible to this account. Open '
                            f'https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID} while signed in as the account '
                            'you mounted, right-click the folder -> Organize -> Add shortcut -> My Drive, then run this cell '
                            'again. If the link says you need access, ask the owner to share it with you as Editor.')
probe = f'{DRIVE}/.write_test_{uuid.uuid4().hex[:8]}'
try:
    open(probe, 'w').close(); os.remove(probe)
except OSError as e:
    raise PermissionError(f'{DRIVE} is read-only for this account: ask the folder owner to share it with you as Editor.') from e
# Every log, result and resume state of this reproduction goes to RUNS. Runs made before the NaN fix are in DRIVE/logs,
# DRIVE/results and DRIVE/resume and are never read, so nothing is resumed or skipped because of them.
RUN_SET = 'runs_v2'
RUNS = f'{DRIVE}/{RUN_SET}'
os.makedirs(f'{DRIVE}/datasets', exist_ok=True)
for d in ['logs', 'results', 'resume']:
    os.makedirs(f'{RUNS}/{d}', exist_ok=True)
os.makedirs(DATA, exist_ok=True)
print('Running on', 'Colab' if ON_COLAB else 'RunPod / local Jupyter', '| folder:', DRIVE, '| contents:', sorted(os.listdir(DRIVE)))
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')

code(r'''
# 1a-2. RunPod only (does nothing on Colab): link to the shared Drive folder with rclone, then copy its datasets/, logs/
# and results/ to the pod. drive_push() copies logs/ and results/ back after every run.
import subprocess, shutil

def drive_push(resume=False):
    pass   # Colab writes straight into the mounted Drive folder (logs, results and resume states)

def drive_drop_resume(name, dataset, seed):
    pass

if not ON_COLAB:
    import zipfile, getpass
    RCLONE      = f'{BASE}/bin/rclone'
    RCLONE_CONF = f'{BASE}/rclone.conf'   # on the volume: survives pod restarts. Contains your Drive token: do not share it.
    if not os.path.exists(RCLONE):        # install rclone on the volume once
        os.makedirs(f'{BASE}/bin', exist_ok=True)
        zpath = f'{BASE}/bin/rclone.zip'
        url = 'https://downloads.rclone.org/v1.68.2/rclone-v1.68.2-linux-amd64.zip'
        # curl, as in rclone's own install script (the server refuses Python's urllib with 403 Forbidden); wget if no curl
        ok = False
        for cmd in (['curl', '-fsSL', '-o', zpath, url], ['wget', '-q', '-O', zpath, url]):
            if shutil.which(cmd[0]) and subprocess.run(cmd).returncode == 0:
                ok = True
                break
        if not ok or not os.path.exists(zpath):
            raise RuntimeError(f'Could not download rclone from {url}. Install it by hand into {RCLONE} (see the header), '
                               'then run this cell again.')
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
        raise RuntimeError('Cannot read the shared Drive folder with rclone. Check that the account you signed in with has '
                           f'Editor access, or delete {RCLONE_CONF} and run this cell again. ' + r.stderr[-500:])
    print('Shared Drive folder contents:', r.stdout.split())
    # resume/: checkpoints of unfinished runs (from Colab or an earlier pod), so this pod continues them
    SYNC = ['datasets', f'{RUN_SET}/logs', f'{RUN_SET}/results', f'{RUN_SET}/resume']
    for d in SYNC:   # --update: never replace a newer file on the pod with an older one
        if rclone('lsf', f'gdrive:{d}', '--max-depth', '1').returncode != 0:
            print(f'Drive {d}/: nothing there yet (it is created when the first run finishes)')
            continue
        print(f'Copying Drive {d}/ to the pod ...')
        get_ipython().system(f'{RCLONE} --config {RCLONE_CONF} copy gdrive:{d} "{DRIVE}/{d}" --update --stats-one-line -P')

    def drive_push(resume=False):
        # logs and results after every run; resume=True: also the checkpoints of running jobs (every few minutes)
        for d in [f'{RUN_SET}/logs', f'{RUN_SET}/results'] + ([f'{RUN_SET}/resume'] if resume else []):
            r = rclone('copy', f'{DRIVE}/{d}', f'gdrive:{d}', '--update', '--exclude', '*.tmp')
            if r.returncode != 0:
                print(f'!! Could not copy {d}/ to Drive (the files are safe on the pod volume):', r.stderr[-300:])

    def drive_drop_resume(name, dataset, seed):
        # a finished run's checkpoint is no longer needed in Drive
        rclone('purge', f'gdrive:{RUN_SET}/resume/{name}/{dataset}/seed{seed}')
''')

code(r'''
# 1b. Code: the t4-fixes branch (colab-fixes + T4 memory and NaN fixes). The commit id is logged with every result.
BRANCH = 't4-fixes'
if not os.path.exists(REPO):
    !git clone -b {BRANCH} https://github.com/AkilaDissanayake/PriViLege.git {REPO}
else:
    !git -C {REPO} fetch -q origin && git -C {REPO} checkout -q {BRANCH} && git -C {REPO} pull -q
%cd {REPO}
COMMIT = !git rev-parse --short HEAD
COMMIT = COMMIT[0]
!git log -1 --oneline
''')

code(r'''
# 1c. Packages. Not requirements.txt (a full Anaconda dump that breaks Colab).
# timm 0.6.13 loads the same ViT-B/16 checkpoint as the authors' 0.6.12 (which fails to import on Python 3.12);
# open_clip is only needed for the CLIP experiments (Figure 3).
# transformers, pandas, scikit-learn, matplotlib and tqdm come with Colab but not with RunPod's PyTorch image
# (pip leaves packages that are already installed as they are).
!pip install -q timm==0.6.13 open_clip_torch gdown transformers pandas scikit-learn matplotlib tqdm
import timm, torch, transformers, pandas, open_clip
print('timm', timm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__,
      '| pandas', pandas.__version__, '| open_clip', open_clip.__version__)
assert timm.__version__ == '0.6.13', 'Restart the session (Runtime -> Restart session / VS Code Restart) and run 1a-1c again'
assert torch.cuda.is_available(), ('No GPU: this is a CPU runtime and every run would fail. Colab: Runtime -> Change runtime '
                                   'type -> GPU. VS Code: Select Kernel -> Colab -> New Colab Server -> GPU. RunPod: '
                                   'start a GPU pod.')
GPU = torch.cuda.get_device_name(0)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
GRAD_CKPT = GPU_GB < 20   # T4 (16 GB): batch 128 only fits with gradient checkpointing (same results, slower).
                          # 24 GB and up (RTX 3090/4090, L40S, A100): not needed.
print(f'GPU: {GPU} ({GPU_GB:.0f} GB)' + (' -> -grad_ckpt on' if GRAD_CKPT else ''))
''')

code(r'''
# 1d. Datasets: copy one archive from Drive and extract it on the local disk (reading many small files from Drive is slow).
import shutil

# Google Drive file ids of the archives in the CEC folder used by FSCIL papers
# https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN
GDRIVE_IDS = {
    'cub200':        '1EtKb1Bxd9ZtK-WUFk5dIhoOdhLeohA1q',   # CUB_200_2011.tgz
    'mini_imagenet': '1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U',   # miniimagenet.tar
}
ARCHIVES = {   # dataset -> (archive in Drive, folder after extraction)
    'cifar100':      ('cifar-100-python.tar.gz', 'cifar-100-python'),
    'cub200':        ('CUB_200_2011.tgz',        'CUB_200_2011'),
    'mini_imagenet': ('miniimagenet.tar',        'miniimagenet'),
}

def prepare(dataset):
    archive, folder = ARCHIVES[dataset]
    src = f'{DRIVE}/datasets/{archive}'
    if os.path.isdir(f'{DATA}/{folder}'):
        pass
    elif os.path.exists(src):
        print(f'Extracting {archive} ...')
        get_ipython().system(f'tar -xf "{src}" -C {DATA}')
    elif dataset == 'cifar100':
        print('CIFAR-100 is not in Drive yet; train.py downloads it (copied to Drive afterwards).')
    else:
        download_to_drive(dataset)
        print(f'Extracting {archive} ...')
        get_ipython().system(f'tar -xf "{src}" -C {DATA}')
    if dataset == 'mini_imagenet':
        # Always the repo's split files: the standard FSCIL split (100 classes x 500 train / 100 test) that the loader and
        # the session lists expect, even if the archive ships a different split/ folder.
        os.makedirs(f'{DATA}/miniimagenet/split', exist_ok=True)
        get_ipython().system(f'cp {REPO}/data/index_list/mini_imagenet/*.csv {DATA}/miniimagenet/split/')

def download_to_drive(dataset):
    import gdown
    archive = ARCHIVES[dataset][0]
    tmp, dst = f'{DATA}/{archive}.download', f'{DRIVE}/datasets/{archive}'   # DATA exists on Colab and RunPod
    print(f'{archive} is not in Drive: downloading it from the CEC Google Drive folder ...')
    gdown.download(id=GDRIVE_IDS[dataset], output=tmp, quiet=False)
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 10_000_000:
        raise RuntimeError(f'Download of {archive} failed (Google Drive quota?). Try again later, or put it in '
                           f'{DRIVE}/datasets/ by hand from the CEC folder.')
    shutil.copy(tmp, dst + '.part'); os.replace(dst + '.part', dst); os.remove(tmp)
    print(f'Saved {archive} to {dst} ({os.path.getsize(dst) / 1e9:.2f} GB)')

def save_cifar_to_drive():
    src, dst = f'{DATA}/cifar-100-python.tar.gz', f'{DRIVE}/datasets/cifar-100-python.tar.gz'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst); print('Saved CIFAR-100 archive to Drive')
''')

code(r'''
# 1e. Pretrained weights: ViT-B/16 (ImageNet-21k -> 1k AugReg, as the authors), BERT, and CLIP ViT-B/16 (LAION-2B).
# Downloads them once, so a network problem shows up now instead of mid-run.
m = timm.create_model('vit_base_patch16_224', pretrained=True)
assert 'i21k' in m.default_cfg['url'], m.default_cfg['url']
del m
from transformers import BertTokenizer, BertModel
BertTokenizer.from_pretrained('bert-base-cased'); BertModel.from_pretrained('bert-base-cased')
open_clip.create_model_and_transforms('ViT-B/16', pretrained='laion2b_s34b_b88k')
print('ViT-B/16, bert-base-cased and CLIP ViT-B/16 (LAION-2B) ready')
''')

md(r'''
## 2. Experiments, runner and paper targets

`run()` calls `train.py` with the official arguments (`run_script/vit_run_pretrain.py`) plus each experiment's flags.
It skips runs already in `PriViLege/results/runs.csv` (in Drive), resumes interrupted runs, appends the log to Drive and records
A_Base / A_Last / A_Avg and every session's accuracy. In the comparison tables Δ = ours − paper:
green |Δ| ≤ 1, orange ≤ 3, red > 3.
''')

code(r'''
import ast, csv, glob, json, time, collections, threading, queue, subprocess
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# One results file per kind of machine, so a RunPod sync never overwrites rows written from Colab (and vice versa)
LOG_CSV = f'{RUNS}/results/runs.csv' if ON_COLAB else f'{RUNS}/results/runs_runpod.csv'
RESUME  = f'{RUNS}/resume'
SEEDS   = [1, 2, 3, 4, 5]
VIT     = '-vit -comp_out 1'

# name -> datasets, train.py flags, epoch overrides.  Every experiment has its own name (= -out), because the
# checkpoint path does not include the -ED / -SKD / -taskblock / -ft flags.
EXP = {
    'smoke':           dict(datasets=['cifar100'], flags=f'{VIT} -LT -ED -SKD', epochs_base=1, epochs_new=1),
    # Table 1 (and the 'Ours' row of Table 3)
    'PriViLege':       dict(datasets=['cifar100', 'cub200', 'mini_imagenet'], flags=f'{VIT} -LT -ED -SKD'),
    # Table 3: Baseline = fine-tuning + prototype classifier (the released -ft option), then the PKT ablations
    'abl_Baseline':    dict(datasets=['cub200'], flags=f'{VIT} -ft'),
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

def parse_results(path):
    accs = ast.literal_eval(open(path).read().strip().splitlines()[-1])   # accuracy of every session
    return accs

def finished_runs():
    done = {}
    for path in sorted(glob.glob(f'{RUNS}/results/runs*.csv')):   # runs.csv (Colab) and runs_runpod.csv (RunPod)
        with open(path, newline='') as f:
            done.update({(r['name'], r['dataset'], int(r['seed'])): r for r in csv.DictReader(f)})
    return done

def describe_state(path):
    s = json.load(open(path))
    where = (f"base epoch {s['epoch'] + 1}/{s['epochs_base']}" if s['phase'] == 'base'
             else f"session {s['session']}/{s['sessions'] - 1}")
    return f"saved after {where} at {s['saved_at']}"

_LOCK = threading.Lock()   # parallel runs share runs.csv, the CIFAR archive copy and the Drive upload
CKPT_UPLOAD_MIN = 10       # RunPod: upload the checkpoints of running jobs to Drive this often (minutes)
_UPLOADER = []

def start_checkpoint_upload():
    # RunPod only (Colab saves checkpoints straight into Drive). Every CKPT_UPLOAD_MIN minutes, copy runs_v2/resume/ to
    # the shared Drive folder, so a stopped or lost pod loses at most that much training: a new pod's setup cell
    # downloads the checkpoints and the same cell continues the runs.
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
    print(f'Checkpoints of running jobs are copied to Drive every {CKPT_UPLOAD_MIN} min.')

def run(name, dataset, seed, force=False, batch_size=128, gpu=None, quiet=False):
    # gpu: physical GPU index for this run (multi-GPU pods); quiet: log only to the log file (used by run_parallel)
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
    print(f'{tag}>> {name} | {dataset} | seed {seed}: ' + (f'RESUMING, {describe_state(state_json)}' if resumed else 'starting'), flush=True)
    log = f'{RUNS}/logs/{name}_{dataset}_seed{seed}.log'
    cmd = (f'python train.py -project base -dataset {dataset} -base_mode ft_dot -new_mode avg_cos -gamma 0.1 '
           f'-lr_base 2e-4 -lr_new 2e-4 -decay 0.0005 -epochs_base {epochs_base} -epochs_new {epochs_new} '
           f'-schedule Cosine -milestones 20 30 45 -gpu 0 -temperature 16 -start_session 0 '
           f'-batch_size_base {batch_size} -seed {seed} {e["flags"]} -out {name} -dataroot {DATA} -ckpt_dir {ckpt}'
           + (' -grad_ckpt' if GRAD_CKPT else ''))
    if gpu is not None:   # pin the process to one GPU before it starts CUDA (train.py keeps this setting)
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
            drive_push(resume=True)   # RunPod: keep its latest checkpoint in Drive too
        return False
    dst = f'{RUNS}/results/{name}/{dataset}/seed{seed}'
    os.makedirs(dst, exist_ok=True)
    shutil.copy(hits[-1], f'{dst}/results.txt')
    for pth in glob.glob(f'{os.path.dirname(hits[-1])}/*.pth'):   # model checkpoints (0.35-1 GB each): not needed for results
        os.remove(pth)
    accs = parse_results(hits[-1])
    a_base, a_last, a_avg = accs[0], accs[-1], sum(accs) / len(accs)
    with _LOCK:   # one run at a time writes runs.csv and uploads to Drive
        new_file = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, 'a', newline='') as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(['name', 'dataset', 'seed', 'flags', 'batch', 'gpu', 'commit', 'A_Base', 'A_Last', 'A_Avg',
                            'sessions', 'minutes', 'resumed', 'date'])
            w.writerow([name, dataset, seed, e['flags'] + (' -grad_ckpt' if GRAD_CKPT else ''), batch_size, GPU, COMMIT, f'{a_base:.2f}', f'{a_last:.2f}',
                        f'{a_avg:.2f}', json.dumps(accs), f'{minutes:.1f}', 'yes' if resumed else 'no',
                        time.strftime('%Y-%m-%d %H:%M')])
        shutil.rmtree(ckpt, ignore_errors=True)   # finished: free the Drive space of its resume state
        drive_push()                               # RunPod: copy the log and results to the shared Drive folder
        drive_drop_resume(name, dataset, seed)     # RunPod: and delete its uploaded checkpoint
    print(f'{tag}== {name} | {dataset} | seed {seed}: A_Base {a_base:.2f} | A_Last {a_last:.2f} | A_Avg {a_avg:.2f} | '
          f'{minutes:.1f} min', flush=True)
    return True

def run_all(name, datasets=None, seeds=None):
    e = EXP[name]
    for ds in datasets or e['datasets']:
        for s in seeds or e.get('seeds', SEEDS):
            if not run(name, ds, s):   # stop at the first failure instead of failing every remaining seed
                raise SystemExit(f'Stopped: {name} | {ds} | seed {s} did not finish. Fix the error above, then re-run.')

PLAN = [(n, ds, s) for n, e in EXP.items() if n != 'smoke'
        for ds in e['datasets'] for s in e.get('seeds', SEEDS)]

def run_parallel(jobs=None, seeds=None, gpus=None):
    # Runs experiments side by side, one per GPU, until all are done. jobs: list of (name, dataset, seed), default every
    # experiment in PLAN; seeds: keep only these seeds (e.g. [1, 2, 3] to save time). Finished runs are skipped,
    # interrupted ones resume. A failed run does not stop the others; failures are listed at the end.
    import torch
    gpus = list(range(torch.cuda.device_count())) if gpus is None else gpus
    jobs = [j for j in (jobs or PLAN) if seeds is None or j[2] in seeds or EXP[j[0]].get('seeds')]
    done = finished_runs()
    todo = [j for j in jobs if j not in done]
    weight = {'cifar100': 3, 'mini_imagenet': 3, 'cub200': 1}   # longest runs first, so the GPUs finish together
    todo.sort(key=lambda j: -weight[j[1]])
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
        print('Check their logs in', f'{RUNS}/logs/', '(the last lines show the error), then run run_parallel() again.')

def status():
    done = finished_runs()
    rows, mins = [], collections.defaultdict(list)
    for (n, ds, s), r in done.items():
        mins[(n, ds)].append(float(r['minutes']))
    for n, ds, s in PLAN:
        state_json = f'{RESUME}/{n}/{ds}/seed{s}/resume_state.json'
        state = 'done' if (n, ds, s) in done else ('resumable' if os.path.exists(state_json) else 'not started')
        rows.append((n, ds, state))
    df = pd.DataFrame(rows, columns=['experiment', 'dataset', 'state'])
    table = df.groupby(['experiment', 'dataset'], sort=False)['state'].value_counts().unstack(fill_value=0)
    display(table)
    left = [(n, ds) for n, ds, st in rows if st != 'done']
    per = {k: np.mean(v) for k, v in mins.items()}
    by_ds = collections.defaultdict(list)
    for (n, ds), v in mins.items():
        by_ds[ds] += v
    guess = [per.get(k, np.mean(by_ds[k[1]]) if by_ds[k[1]] else None) for k in left]
    est, unknown = sum(g for g in guess if g is not None), sum(g is None for g in guess)
    used = sum(sum(v) for v in mins.values())
    print(f'{len(rows) - len(left)}/{len(rows)} runs done | GPU time used {used / 60:.1f} h | '
          f'estimated time left {est / 60:.1f} h' +
          (f' + {unknown} runs on datasets with no finished run yet' if unknown else ''))
''')

code(r'''
# Paper numbers: (mean, std) of A_Base, A_Last, A_Avg. Figure 3 has no table: those values were read off the plot
# (per session, accuracy within about +/- 0.3).
PAPER = {
    'Table 1': {('PriViLege', 'cub200'):        [(82.21, .35), (75.08, .52), (77.50, .33)],
                ('PriViLege', 'cifar100'):      [(90.88, .20), (86.06, .32), (88.08, .20)],
                ('PriViLege', 'mini_imagenet'): [(96.68, .06), (94.10, .13), (95.27, .11)]},
    'Table 3': {('abl_Baseline', 'cub200'): [(84.21, .13), (3.79, 1.47), (21.60, 1.32)],
                ('abl_PKT', 'cub200'):      [(79.06, .77), (70.81, .76), (73.36, .77)],
                ('abl_PKT_ED', 'cub200'):   [(80.31, .54), (72.70, .45), (75.04, .40)],
                ('abl_PKT_SKD', 'cub200'):  [(82.10, .57), (73.44, .40), (76.27, .30)],
                ('PriViLege', 'cub200'):    [(82.21, .35), (75.08, .52), (77.50, .33)]},
    'Table 4': {('layers_0', 'cub200'):  [(76.07, .56), (60.19, 1.11), (67.08, .71)],
                ('abl_PKT', 'cub200'):   [(79.06, .77), (70.81, .76), (73.36, .77)],
                ('layers_5', 'cub200'):  [(78.42, .84), (68.52, .84), (71.99, .84)],
                ('layers_7', 'cub200'):  [(76.96, .74), (63.15, 2.73), (68.06, 1.54)],
                ('layers_10', 'cub200'): [(74.95, .78), (57.78, 2.30), (64.19, 1.59)],
                ('layers_12', 'cub200'): [(73.62, 2.72), (56.02, 1.47), (62.71, 2.14)]},
}
FIG3 = {   # Figure 3, accuracy per session (Base, 1..8)
    'clip_zeroshot':   [78.8, 76.6, 75.2, 73.1, 72.5, 72.4, 72.7, 72.3, 71.0],
    'clip_PKT':        [82.3, 79.5, 77.6, 75.4, 74.0, 73.2, 73.3, 72.6, 71.4],
    'clip_PKT_ED':     [83.7, 81.2, 79.8, 77.6, 76.7, 76.3, 76.5, 76.2, 74.7],
    'clip_PKT_ED_SKD': [84.2, 82.1, 81.0, 79.0, 78.8, 78.4, 78.6, 78.3, 77.1],
}
PAPER['Figure 3'] = {(n, 'cifar100'): [(v[0], None), (v[-1], None), (round(float(np.mean(v)), 2), None)]
                     for n, v in FIG3.items()}
LABELS = {'PriViLege': 'PriViLege (Ours)', 'abl_Baseline': 'Baseline (fine-tune + proto)', 'abl_PKT': 'PKT',
          'abl_PKT_ED': 'PKT + L_ED', 'abl_PKT_SKD': 'PKT + L_SKD', 'layers_0': '0 layers', 'layers_5': '5 layers',
          'layers_7': '7 layers', 'layers_10': '10 layers', 'layers_12': '12 layers',
          'clip_zeroshot': 'CLIP zero-shot', 'clip_PKT': 'CLIP-PKT', 'clip_PKT_ED': 'CLIP-PKT + L_ED',
          'clip_PKT_ED_SKD': 'CLIP-PKT + L_ED + L_SKD'}
DS = {'cub200': 'CUB200', 'cifar100': 'CIFAR-100', 'mini_imagenet': 'miniImageNet'}
METRICS = ['A_Base', 'A_Last', 'A_Avg']

def runs_of(name, dataset):
    return [r for (n, ds, s), r in sorted(finished_runs().items()) if n == name and ds == dataset]

def ms(xs):
    return f'{np.mean(xs):.2f} ± {np.std(xs, ddof=1) if len(xs) > 1 else 0:.2f}'

def compare(table):
    rows = []
    for (name, ds), paper in PAPER[table].items():
        label = LABELS[name] if name != 'abl_PKT' or table != 'Table 4' else '2 layers (= PKT)'
        if table == 'Table 1':
            label = DS[ds]
        rs = runs_of(name, ds)
        row = {'': label, 'seeds': len(rs)}
        for i, m in enumerate(METRICS):
            ours = [float(r[m]) for r in rs]
            pm, ps = paper[i]
            row[f'{m} ours'] = ms(ours) if ours else '—'
            row[f'{m} paper'] = f'{pm:.2f}' + (f' ± {ps:.2f}' if ps is not None else ' (fig.)')
            row[f'Δ {m}'] = round(np.mean(ours) - pm, 2) if ours else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index('')
    def color(v):
        if pd.isna(v): return ''
        return 'color: green' if abs(v) <= 1 else ('color: orange' if abs(v) <= 3 else 'color: red')
    display(Markdown(f'**{table}: ours vs. paper**'))
    display(df.style.map(color, subset=[f'Δ {m}' for m in METRICS]).format(precision=2, na_rep='—'))

def plot_sessions(items, title, paper_curves=None):
    plt.figure(figsize=(7, 4))
    for i, (name, ds) in enumerate(items):
        rs = runs_of(name, ds)
        c = f'C{i}'
        label = DS[ds] if title.startswith('Table 1') else LABELS[name]
        if rs:
            acc = np.array([json.loads(r['sessions']) for r in rs])
            x = np.arange(acc.shape[1])
            plt.plot(x, acc.mean(0), '-o', color=c, ms=4, label=f'{label} (ours, n={len(rs)})')
            if len(rs) > 1:
                plt.fill_between(x, acc.mean(0) - acc.std(0, ddof=1), acc.mean(0) + acc.std(0, ddof=1), color=c, alpha=.15)
        if paper_curves and name in paper_curves:
            plt.plot(paper_curves[name], '--*', color=c, alpha=.7, label=f'{label} (paper)')
    plt.xlabel('session (0 = base)'); plt.ylabel('accuracy (%)'); plt.title(title)
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

def show(table):
    compare(table)
    items = list(PAPER[table])
    plot_sessions(items, f'{table}: accuracy per session', FIG3 if table == 'Figure 3' else None)
''')

md(r'''
## 3. Status
What is finished, resumable or not started, GPU time used so far and an estimate of the time left.
''')
code('status()')

md(r'''
## 4. Smoke test (1 base epoch + 1 epoch per session)
Catches environment, data and code problems before the long runs. It passes if the log reaches `Incremental session: [8]`
and the last line prints A_Base / A_Last / A_Avg. The numbers themselves are not meaningful.
''')
code("assert run('smoke', 'cifar100', seed=1), 'Smoke test failed: see the error above'")

md(r'''
## 4b. Quick checks on real data (before the full tables)
One real CUB200 run (Table 1, seed 1; T4 ~30 min) and CLIP zero-shot (Figure 3, no training). Within about 1-2 points of the
paper means the code is working: continue with the tables (on RunPod if you like). Far off: stop and check the log.
''')

code(r'''
assert run('PriViLege', 'cub200', seed=1), 'CUB200 run failed: see the error above'
assert run('clip_zeroshot', 'cifar100', seed=1), 'CLIP zero-shot failed: see the error above'
done = finished_runs()
for name, ds, paper in [('PriViLege', 'cub200', PAPER['Table 1'][('PriViLege', 'cub200')]),
                        ('clip_zeroshot', 'cifar100', PAPER['Figure 3'][('clip_zeroshot', 'cifar100')])]:
    r = done[(name, ds, 1)]
    ours = [float(r[m]) for m in METRICS]
    print(f"{name:14s} {ds:9s} A_Base / A_Last / A_Avg  ours {ours[0]:.2f} / {ours[1]:.2f} / {ours[2]:.2f}"
          f"  paper {paper[0][0]:.2f} / {paper[1][0]:.2f} / {paper[2][0]:.2f}"
          f"  | diff {ours[0] - paper[0][0]:+.2f} / {ours[1] - paper[1][0]:+.2f} / {ours[2] - paper[2][0]:+.2f}")
''')

md(r'''
## 4c. Everything at once on a multi-GPU pod (RunPod)
Runs every experiment of Tables 1, 3, 4 and Figure 3, one per GPU at the same time, longest first. With 8 x RTX 4090 the
full plan (5 seeds) takes roughly 2.5-3 hours (4 x RTX 4090: 4-5 hours; `seeds=[1, 2, 3]` cuts either by 40%). Each run writes only to its own log
(`runs_v2/logs/`); this cell prints one line per start and finish. Finished runs are skipped and interrupted ones resume,
so after any stop just run the cell again. Then run section 9 for the comparison tables. (On a single GPU this also works,
one run after another.) To watch a run live, in a terminal: `tail -f /workspace/privilege_drive/runs_v2/logs/<name>_<dataset>_seed<k>.log`
''')
code(r'''
run_parallel()                    # all experiments, 5 seeds
# run_parallel(seeds=[1, 2, 3])   # faster: 3 seeds
''')

md(r'''
## 5. Table 1: PriViLege on the three benchmarks (5 seeds each)
CUB200: 100 base classes + 10 sessions of 10-way 5-shot. CIFAR-100 and miniImageNet: 60 base classes + 8 sessions of 5-way 5-shot.
''')
code("run_all('PriViLege', ['cifar100'])\nshow('Table 1')")
code("run_all('PriViLege', ['cub200'])\nshow('Table 1')")
code("run_all('PriViLege', ['mini_imagenet'])\nshow('Table 1')")

md(r'''
## 6. Table 3: ablation on CUB200 (5 seeds each)
- **Baseline**: fine-tuning the pre-trained ViT with the prototype classifier ψ. This uses the released `-ft` option (ViT blocks 0–1
  trained in the base session, every parameter trained in the incremental sessions). The authors' compiled "Baseline" file
  is an unfinished experiment that does not match the paper's description, so it is not used.
- **PKT**, **PKT + L_ED**, **PKT + L_SKD**: the method without one or both losses. **Ours** reuses the Table 1 CUB200 runs.
''')
code("for name in ['abl_Baseline', 'abl_PKT', 'abl_PKT_ED', 'abl_PKT_SKD']:\n    run_all(name)\nshow('Table 3')")

md(r'''
## 7. Table 4: number of tuned layers (CUB200, 5 seeds each)
Run without the two losses, as in the paper; the 2-layer row is the PKT row of Table 3. `-taskblock` sets how many of the first
ViT blocks are trained; the B-Prompt stays in blocks 0–1.
''')
code("for L in [0, 5, 7, 10, 12]:\n    run_all(f'layers_{L}')\nshow('Table 4')")

md(r'''
## 8. Figure 3: PriViLege on CLIP (CIFAR-100, 5 seeds each)
Official open_clip CLIP ViT-B/16 pre-trained on LAION-2B. Only the vision encoder is trained, with the B-Prompt and VL-Prompt added
on top of the official modules (`models/CLIP_prompt.py`); the frozen CLIP text encoder gives the class-name embeddings.
The authors did not release this code: `models/base/ViT_Network_CLIP.py` and `models/base/CLIP_*.py` were rebuilt from their
compiled files. The paper's values are read off the figure.
''')
code("for name in ['clip_zeroshot', 'clip_PKT', 'clip_PKT_ED', 'clip_PKT_ED_SKD']:\n    run_all(name)\nshow('Figure 3')")

md(r'''
## 9. All results vs. the paper
''')
code("for t in ['Table 1', 'Table 3', 'Table 4', 'Figure 3']:\n    compare(t)")

md(r'''
## 10. Free disk space (optional)
Finished runs delete their `.pth` checkpoints automatically; this removes any left by interrupted or older runs.
''')
code("!find {REPO}/checkpoint -name '*.pth' -delete\n!df -h /content | tail -1")

nb = {'cells': cells, 'metadata': {'colab': {'provenance': [], 'gpuType': 'T4'},
                                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                                   'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
      'nbformat': 4, 'nbformat_minor': 0}
for c in nb['cells']:   # store source as a list of lines, like Jupyter does
    c['source'] = [l + '\n' for l in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')
json.dump(nb, open(sys.argv[1], 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)   # utf-8: also works on Windows
print('wrote', sys.argv[1], len(cells), 'cells')
