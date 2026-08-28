"""Final C4 run used for the manuscript: physical-location LOLO at G=2 s.

Protocol
--------
- current common G2 cohort: 1138 train + 241 validation pairs
- five 4-digit physical locations: 0000, 0002, 0400, 0401, 0500
- frozen matched pairs; no fold-wise rematching
- eval pair only when both positive and control are in the held-out location
- train pair only when neither side is in the held-out location
- one-sided/cross-boundary pairs are dropped
- 85/15 deterministic inner split inside non-heldout pairs
- corrected camera-view video resolution
- Motion TCN, 10 canonical seeds
- no test split

The script intentionally contains structural assertions so that a changed cohort or
pair construction fails loudly instead of silently producing a different result.
"""

from pathlib import Path
from datetime import datetime
import zipfile, pickle, re, copy, random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import binomtest

# -----------------------------------------------------------------------------
# Paths from the Colab run used for the paper.
# Override with environment-specific paths before rerunning elsewhere.
# -----------------------------------------------------------------------------
MY = Path('/content/drive/MyDrive')
ROOT = MY / 'VIRAT_Project'
PKG_PATH = MY / 'virat_gap_common_cohort.pkl'
ZIP_PATH = ROOT / 'gap_experiment_recovered_20260820.zip'
OBS_RES = ROOT / 'results/CAMERA_VIEW_RESOLUTION_MAP/20260826_154914/actual_resolution_by_view.csv'
REC_RES = ROOT / 'results/MISSING_VIEW_VIDEO_RECOVERY/20260826_155125/recovery_summary.csv'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = ROOT / 'results/TRUE_PHYSICAL_LOCATION_LOLO_G2_10SEED' / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
TMP = Path('/content/c4_lolo_10seed')
TMP.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1337, 2024, 7, 17, 73, 101, 314, 777, 1729]
LOCS = ['0000', '0002', '0400', '0401', '0500']
OLD_STRUCTURE = {
    '0000': {'train':1090, 'eval':203, 'drop':86},
    '0002': {'train': 935, 'eval':384, 'drop':60},
    '0400': {'train':1043, 'eval':277, 'drop':59},
    '0401': {'train': 986, 'eval':282, 'drop':111},
    '0500': {'train':1283, 'eval': 54, 'drop':42},
}

HIDDEN = 80
DROPOUT = .20
LR = 1e-3
WD = 1e-4
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0
INNER_SPLIT_SEED = 20260820
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for p in [PKG_PATH, ZIP_PATH, OBS_RES, REC_RES]:
    assert p.exists(), p

# -----------------------------------------------------------------------------
# Verified video resolution map
# -----------------------------------------------------------------------------
def parse_res(x):
    m = re.fullmatch(r'(\d+)x(\d+)', str(x).strip())
    assert m, x
    return int(m.group(1)), int(m.group(2))

view_res = {}
a = pd.read_csv(OBS_RES, dtype={'view': str})
a['view'] = a['view'].astype(str).str.zfill(6)
for _, r in a.iterrows():
    assert int(r['n_unique_resolutions']) == 1
    view_res[r['view']] = parse_res(r['resolutions'])

b = pd.read_csv(REC_RES, dtype={'view': str})
b['view'] = b['view'].astype(str).str.zfill(6)
for _, r in b.iterrows():
    assert int(r['n_direct_videos']) >= 1
    res = parse_res(r['direct_resolutions'])
    if r['view'] in view_res:
        assert view_res[r['view']] == res
    view_res[r['view']] = res
assert len(view_res) == 21

VIEW6_RE = re.compile(r'VIRAT_S_(\d{6})')
LOC4_RE = re.compile(r'VIRAT_S_(\d{4})')

def view_of(clip):
    m = VIEW6_RE.search(str(clip)); assert m, clip
    return m.group(1)

def loc_of(clip):
    m = LOC4_RE.search(str(clip)); assert m, clip
    return m.group(1)

def fs_of(clip):
    v = view_of(clip); assert v in view_res, (clip, v)
    return view_res[v]

# -----------------------------------------------------------------------------
# Load G2 train + validation only
# -----------------------------------------------------------------------------
with open(PKG_PATH, 'rb') as f:
    pkg = pickle.load(f)
assert 2 in pkg and 'test' not in pkg[2]
tr, va = pkg[2]['train'], pkg[2]['val']
assert len(tr['event_id']) == 1138 and len(va['event_id']) == 241

def cat(key):
    return np.concatenate([np.asarray(tr[key]), np.asarray(va[key])], axis=0)

event_id = cat('event_id').astype(str)
target_class = cat('target_class').astype(str)
pos_xy = cat('pos_xy').astype(np.float32)
ctrl_xy = cat('ctrl_xy').astype(np.float32)
assert len(event_id) == 1379
assert pos_xy.shape == ctrl_xy.shape and pos_xy.shape[:2] == (1379, 90)

# -----------------------------------------------------------------------------
# Align frozen pair metadata
# -----------------------------------------------------------------------------
member = 'gap_experiment/gap_G2s_matched_pairs.parquet'
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    assert member in z.namelist()
    z.extract(member, TMP)
pairs = pd.read_parquet(TMP / member)
pairs = pairs[pairs['split'].isin(['train', 'val'])].copy()
pairs['event_key'] = (
    pairs['pos_clip'].astype(str) + '__' +
    pairs['pos_event_onset'].astype(str) + '__' +
    pairs['target_class'].astype(str) + '__' +
    pairs['pos_actor_id'].astype(str)
)
assert not pairs['event_key'].duplicated().any()
pairs = pairs.set_index('event_key').loc[event_id].reset_index()
assert pairs['event_key'].astype(str).tolist() == event_id.tolist()
assert np.array_equal(pairs['target_class'].astype(str).to_numpy(), target_class)

pos_loc = pairs['pos_clip'].map(loc_of).to_numpy()
ctrl_loc = pairs['ctrl_clip'].map(loc_of).to_numpy()
pos_fs = np.asarray([fs_of(x) for x in pairs['pos_clip']], dtype=np.float32)
ctrl_fs = np.asarray([fs_of(x) for x in pairs['ctrl_clip']], dtype=np.float32)

# -----------------------------------------------------------------------------
# Motion-6 features
# -----------------------------------------------------------------------------
def motion6(xy, fs):
    xy = np.asarray(xy, dtype=np.float32)
    fs = np.asarray(fs, dtype=np.float32)
    fw, fh = np.maximum(fs[:,0:1], 1.0), np.maximum(fs[:,1:2], 1.0)
    cx, cy = xy[:,:,0] / fw, xy[:,:,1] / fh
    dcx = np.diff(cx, axis=1, prepend=cx[:,:1])
    dcy = np.diff(cy, axis=1, prepend=cy[:,:1])
    step = np.sqrt(dcx**2 + dcy**2)
    speed = step.copy()
    d_speed = np.diff(speed, axis=1, prepend=speed[:,:1])
    accel = np.abs(d_speed)
    d_accel = np.diff(accel, axis=1, prepend=accel[:,:1])
    heading = np.arctan2(dcy, dcx)
    dh = np.diff(heading, axis=1, prepend=heading[:,:1])
    heading_rate = np.angle(np.exp(1j * dh)).astype(np.float32)
    curvature = heading_rate / (speed + 1e-8)
    dx0, dy0 = cx - cx[:,:1], cy - cy[:,:1]
    displacement = np.sqrt(dx0**2 + dy0**2)
    out = np.stack([d_speed, d_accel, heading_rate, curvature, displacement, step], axis=-1).astype(np.float32)
    assert np.isfinite(out).all()
    return out

P, C = motion6(pos_xy, pos_fs), motion6(ctrl_xy, ctrl_fs)

# -----------------------------------------------------------------------------
# Motion TCN
# -----------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
    def forward(self, x):
        return x if self.n == 0 else x[:,:,:-self.n].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.net(x) + self.res(x)

class MotionTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(TCNBlock(6,HIDDEN,1), TCNBlock(HIDDEN,HIDDEN,2), TCNBlock(HIDDEN,HIDDEN,4))
        self.head = nn.Linear(HIDDEN,1)
    def forward(self, x):
        x = self.blocks(x.transpose(1,2)).mean(dim=2)
        return self.head(x).squeeze(1)

probe = MotionTCN()
assert sum(p.numel() for p in probe.parameters() if p.requires_grad) == 98561
del probe

# -----------------------------------------------------------------------------
# Training / metrics
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def norm_stats(Px, Cx):
    x = np.concatenate([Px,Cx], axis=0)
    mu = x.mean(axis=(0,1), keepdims=True)
    sd = x.std(axis=(0,1), keepdims=True); sd[sd < 1e-9] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def predict(model, P0, C0, mu, sd):
    model.eval(); lp, lc = [], []
    with torch.no_grad():
        for st in range(0, len(P0), 256):
            en = min(st+256, len(P0))
            xp = torch.tensor((P0[st:en]-mu)/sd, dtype=torch.float32, device=DEVICE)
            xc = torch.tensor((C0[st:en]-mu)/sd, dtype=torch.float32, device=DEVICE)
            lp.append(model(xp).cpu().numpy()); lc.append(model(xc).cpu().numpy())
    return np.concatenate(lp), np.concatenate(lc)

def metrics(pl, cl, cls):
    pp = 1/(1+np.exp(-np.clip(pl,-40,40))); cp = 1/(1+np.exp(-np.clip(cl,-40,40)))
    y = np.r_[np.ones(len(pp)), np.zeros(len(cp))]; pr = np.r_[pp,cp]
    macro = [np.mean(pl[cls==c] > cl[cls==c]) for c in sorted(np.unique(cls))]
    return {
        'AP': float(average_precision_score(y,pr)),
        'AUC': float(roc_auc_score(y,pr)),
        'PW': float(np.mean(pl>cl)),
        'MacroPW': float(np.mean(macro)),
    }

def train_one(seed, fit_idx, val_idx, eval_idx):
    set_seed(seed)
    mu, sd = norm_stats(P[fit_idx], C[fit_idx])
    X = np.concatenate([(P[fit_idx]-mu)/sd, (C[fit_idx]-mu)/sd], axis=0).astype(np.float32)
    y = np.r_[np.ones(len(fit_idx),dtype=np.float32), np.zeros(len(fit_idx),dtype=np.float32)]
    gen = torch.Generator(); gen.manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)), batch_size=BATCH, shuffle=True, generator=gen)
    model = MotionTCN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.BCEWithLogitsLoss()
    best_ap, best_state, best_epoch, wait = -1, None, None, 0
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for xb,yb in dl:
            xb,yb = xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
        vpl,vcl = predict(model,P[val_idx],C[val_idx],mu,sd)
        vm = metrics(vpl,vcl,target_class[val_idx])
        if vm['AP'] > best_ap + 1e-10:
            best_ap = vm['AP']; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    assert best_state is not None
    model.load_state_dict(best_state)
    pl,cl = predict(model,P[eval_idx],C[eval_idx],mu,sd)
    return metrics(pl,cl,target_class[eval_idx]), pl, cl, best_epoch

# -----------------------------------------------------------------------------
# Fold construction with exact structural gates
# -----------------------------------------------------------------------------
folds = {}
for loc in LOCS:
    eval_idx = np.where((pos_loc==loc) & (ctrl_loc==loc))[0]
    train_idx = np.where((pos_loc!=loc) & (ctrl_loc!=loc))[0]
    drop_idx = np.where((pos_loc==loc) ^ (ctrl_loc==loc))[0]
    ref = OLD_STRUCTURE[loc]
    assert len(train_idx)==ref['train'] and len(eval_idx)==ref['eval'] and len(drop_idx)==ref['drop']
    fit_idx, inner_val_idx = train_test_split(train_idx, test_size=.15, random_state=INNER_SPLIT_SEED, stratify=target_class[train_idx])
    folds[loc] = {'train':train_idx,'fit':np.asarray(fit_idx),'val':np.asarray(inner_val_idx),'eval':eval_idx,'drop':drop_idx}
assert sum(len(folds[x]['eval']) for x in LOCS) == 1200

# -----------------------------------------------------------------------------
# Run all folds and seeds
# -----------------------------------------------------------------------------
rows, pred_rows = [], []
for loc in LOCS:
    f = folds[loc]
    for seed in SEEDS:
        mm,pl,cl,ep = train_one(seed,f['fit'],f['val'],f['eval'])
        rows.append({'holdout_location':loc,'seed':seed,'train_pairs':len(f['train']),'inner_fit_pairs':len(f['fit']),'inner_val_pairs':len(f['val']),'eval_pairs':len(f['eval']),'dropped_one_sided':len(f['drop']),'selected_epoch':ep,**mm})
        for eid,cls,pv,cv in zip(event_id[f['eval']],target_class[f['eval']],pl,cl):
            pred_rows.append({'holdout_location':loc,'seed':seed,'event_id':eid,'target_class':cls,'pos_logit':float(pv),'ctrl_logit':float(cv)})

raw_df = pd.DataFrame(rows); pred_df = pd.DataFrame(pred_rows)
raw_df.to_csv(OUT/'fold_seed_results_10seeds.csv',index=False)
pred_df.to_csv(OUT/'fold_predictions_10seeds.csv',index=False)

pooled_seed_rows = []
for seed in SEEDS:
    s = pred_df[pred_df.seed==seed]; assert len(s)==1200
    pooled_seed_rows.append({'seed':seed,'N_eval_pairs':len(s),**metrics(s.pos_logit.to_numpy(),s.ctrl_logit.to_numpy(),s.target_class.astype(str).to_numpy())})
pd.DataFrame(pooled_seed_rows).to_csv(OUT/'pooled_seed_results_10seeds.csv',index=False)

# -----------------------------------------------------------------------------
# Formal pair-level inference on logits averaged across 10 seeds
# -----------------------------------------------------------------------------
def bootstrap_pw_ci(win, B=50000, seed=20260828):
    win = np.asarray(win,dtype=np.float64); rng = np.random.default_rng(seed); n=len(win)
    vals=np.empty(B,dtype=np.float64); st=0
    while st<B:
        m=min(5000,B-st); ix=rng.integers(0,n,size=(m,n)); vals[st:st+m]=win[ix].mean(axis=1); st+=m
    lo,hi=np.quantile(vals,[.025,.975]); return float(lo),float(hi)

def formal_stats(df, seed):
    pos=df.pos_logit.to_numpy(); ctrl=df.ctrl_logit.to_numpy()
    wins=pos>ctrl; losses=pos<ctrl; ties=~(wins|losses)
    ci=bootstrap_pw_ci(wins.astype(float),seed=seed)
    nw,nl=int(wins.sum()),int(losses.sum())
    p=float(binomtest(nw,nw+nl,p=.5,alternative='two-sided').pvalue)
    return {'N':len(df),'PW':float(wins.mean()),'CI95_low':ci[0],'CI95_high':ci[1],'wins':nw,'losses':nl,'ties':int(ties.sum()),'p_two_sided':p}

avg = pred_df.groupby(['holdout_location','event_id','target_class'],as_index=False).agg(pos_logit=('pos_logit','mean'),ctrl_logit=('ctrl_logit','mean'),n_seeds=('seed','nunique'))
assert (avg.n_seeds==10).all() and len(avg)==1200
formal_rows=[]
for j,loc in enumerate(LOCS):
    formal_rows.append({'location':loc,**formal_stats(avg[avg.holdout_location==loc],20260828+j)})
formal_rows.append({'location':'POOLED',**formal_stats(avg,20260928)})
pd.DataFrame(formal_rows).to_csv(OUT/'formal_10seed_pairwise.csv',index=False)
avg.to_csv(OUT/'seed_averaged_pair_logits.csv',index=False)

print(pd.DataFrame(formal_rows).to_string(index=False))
print('Saved:',OUT)
