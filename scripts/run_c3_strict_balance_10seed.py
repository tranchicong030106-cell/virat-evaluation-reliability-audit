"""Final C3 run used for the manuscript: strict matched-balance sensitivity at G=2 s.

Protocol
--------
- reconstruct the original greedy spatial-only balance filter
- prune separately by target_class x split
- matching variables: cx, cy, bbox_w, bbox_h
- stop when max |SMD| < 0.10, with max 25% drop allowed
- expected structural fingerprint: train=1138, val=236, 5 removed validation pairs
- corrected camera-view resolution
- same Motion TCN and 10 canonical seeds as the main gap experiment
- no test split

The structural assertions are intentional: a changed cohort or filter must fail
rather than silently become a different experiment.
"""

from pathlib import Path
from datetime import datetime
import zipfile, pickle, re, math, copy, random, json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import binomtest

MY = Path('/content/drive/MyDrive')
ROOT = MY / 'VIRAT_Project'
PKG_PATH = MY / 'virat_gap_common_cohort.pkl'
ZIP_PATH = ROOT / 'gap_experiment_recovered_20260820.zip'
OBS_RES = ROOT / 'results/CAMERA_VIEW_RESOLUTION_MAP/20260826_154914/actual_resolution_by_view.csv'
REC_RES = ROOT / 'results/MISSING_VIEW_VIDEO_RECOVERY/20260826_155125/recovery_summary.csv'
STANDARD_DIR = ROOT / 'results/FINAL_AUDITED_RUN_10SEED/20260826_160240'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = ROOT / 'results/STRICT_BALANCE_G2_10SEED' / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
TMP = Path('/content/c3_strict_balance')
TMP.mkdir(parents=True, exist_ok=True)

TARGETS = ['vehicle_starting','vehicle_stopping','vehicle_turning_left','vehicle_turning_right','Opening','Closing','Entering','Interacts']
SPATIAL = ['cx','cy','bbox_w','bbox_h']
SPLITS = ('train','val')
TARGET_SMD = 0.10
MAX_DROP_PCT = 0.25
SEEDS = [42,1337,2024,7,17,73,101,314,777,1729]
HIDDEN = 80
DROPOUT = .20
LR = 1e-3
WD = 1e-4
BATCH = 64
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for p in [PKG_PATH,ZIP_PATH,OBS_RES,REC_RES,STANDARD_DIR/'all_gaps_10seed_metrics.csv',STANDARD_DIR/'formal_pairwise_results.csv']:
    assert p.exists(), p

# -----------------------------------------------------------------------------
# Current G2 train + validation package only
# -----------------------------------------------------------------------------
with open(PKG_PATH,'rb') as f:
    pkg = pickle.load(f)
assert 2 in pkg and 'test' not in pkg[2]
tr,va = pkg[2]['train'],pkg[2]['val']
assert len(tr['event_id'])==1138 and len(va['event_id'])==241
train_ids=np.asarray(tr['event_id']).astype(str)
val_ids=np.asarray(va['event_id']).astype(str)
common_keys=set(train_ids.tolist()+val_ids.tolist())
assert len(common_keys)==1379

# -----------------------------------------------------------------------------
# Load G2 matched pairs in original parquet order
# -----------------------------------------------------------------------------
member='gap_experiment/gap_G2s_matched_pairs.parquet'
with zipfile.ZipFile(ZIP_PATH,'r') as z:
    assert member in z.namelist(); z.extract(member,TMP)
g2=pd.read_parquet(TMP/member)
g2['event_key']=(g2['pos_clip'].astype(str)+'__'+g2['pos_event_onset'].astype(str)+'__'+g2['target_class'].astype(str)+'__'+g2['pos_actor_id'].astype(str))
g2=g2[g2['event_key'].isin(common_keys)&g2['split'].isin(SPLITS)].reset_index(drop=True)
assert len(g2)==1379
for c in ['pos_cx','pos_cy','pos_bbox_w','pos_bbox_h','ctrl_cx','ctrl_cy','ctrl_bbox_w','ctrl_bbox_h']:
    assert c in g2.columns,c

# -----------------------------------------------------------------------------
# Exact SMD helpers and greedy pruning used by the earlier sensitivity script
# -----------------------------------------------------------------------------
def smd_vec(pos_mat,ctrl_mat):
    if len(pos_mat)<2: return np.full(pos_mat.shape[1],np.nan)
    mp,mc=pos_mat.mean(0),ctrl_mat.mean(0)
    vp,vc=pos_mat.var(0,ddof=1),ctrl_mat.var(0,ddof=1)
    denom=np.sqrt((vp+vc)/2); denom=np.where(denom<1e-9,1e-9,denom)
    return (mp-mc)/denom

def max_abs_smd(pos_mat,ctrl_mat):
    return float(np.nanmax(np.abs(smd_vec(pos_mat,ctrl_mat))))

def greedy_prune(pos_vals,ctrl_vals):
    N=len(pos_vals); kept=np.ones(N,dtype=bool)
    min_size=max(2,math.ceil(N*(1-MAX_DROP_PCT)))
    while True:
        P,C=pos_vals[kept],ctrl_vals[kept]
        max_s=max_abs_smd(P,C)
        if max_s<TARGET_SMD: return kept,False,'balanced'
        if kept.sum()<=min_size: return kept,True,'max_drop'
        indices=np.where(kept)[0]; best_score=max_s; best_i=-1
        for i in indices:
            kept[i]=False
            if kept.sum()>=2:
                score=max_abs_smd(pos_vals[kept],ctrl_vals[kept])
                if score<best_score: best_score,best_i=score,i
            kept[i]=True
        if best_i==-1:
            s_vec=smd_vec(P,C); worst_cov=int(np.nanargmax(np.abs(s_vec)))
            diffs=pos_vals[:,worst_cov]-ctrl_vals[:,worst_cov]
            best_i=indices[np.argmax(diffs[indices])] if s_vec[worst_cov]>0 else indices[np.argmin(diffs[indices])]
        kept[best_i]=False

# -----------------------------------------------------------------------------
# Reconstruct strict keep-list
# -----------------------------------------------------------------------------
global_keep=np.zeros(len(g2),dtype=bool); post_rows=[]
for cls in TARGETS:
    for sp in SPLITS:
        sub_mask=(g2['target_class']==cls)&(g2['split']==sp)
        sub_idx=np.where(sub_mask.values)[0]
        if len(sub_idx)==0: continue
        sub=g2.iloc[sub_idx]
        P4=sub[['pos_cx','pos_cy','pos_bbox_w','pos_bbox_h']].to_numpy(np.float64)
        C4=sub[['ctrl_cx','ctrl_cy','ctrl_bbox_w','ctrl_bbox_h']].to_numpy(np.float64)
        keep_local,stopped,reason=greedy_prune(P4,C4)
        global_keep[sub_idx[keep_local]]=True
        s2=smd_vec(P4[keep_local],C4[keep_local])
        post_rows.append({'class':cls,'split':sp,'N_before':len(sub_idx),'N_after':int(keep_local.sum()),'dropped':int(len(sub_idx)-keep_local.sum()),'cx':float(s2[0]),'cy':float(s2[1]),'bbox_w':float(s2[2]),'bbox_h':float(s2[3]),'max_abs_smd':float(np.nanmax(np.abs(s2))),'stopped':stopped,'reason':reason})

strict_df=g2[global_keep].copy().reset_index(drop=True)
removed_df=g2[~global_keep].copy().reset_index(drop=True)
post_df=pd.DataFrame(post_rows)
n_train=int((strict_df['split']=='train').sum()); n_val=int((strict_df['split']=='val').sum())
MAX_POST=float(post_df['max_abs_smd'].max())
removed_counts=removed_df['target_class'].value_counts().to_dict()

assert n_train==1138 and n_val==236 and len(strict_df)==1374 and len(removed_df)==5
assert (removed_df['split']=='val').all()
assert removed_counts=={'vehicle_stopping':2,'Closing':2,'vehicle_starting':1},removed_counts
assert MAX_POST<.10 and abs(MAX_POST-.0911)<.0010
assert not (post_df['max_abs_smd']>=.10).any() and not post_df['stopped'].any()

strict_df[['event_key','split','target_class']].to_csv(OUT/'strict_balanced_keep_ids.csv',index=False)
removed_df[['event_key','split','target_class']].to_csv(OUT/'strict_balanced_removed_ids.csv',index=False)
post_df.to_csv(OUT/'strict_balanced_smd_report.csv',index=False)

# -----------------------------------------------------------------------------
# Map strict keep-list back to the package
# -----------------------------------------------------------------------------
keep_keys=set(strict_df['event_key'].astype(str))
tr_mask=np.asarray([e in keep_keys for e in train_ids]); va_mask=np.asarray([e in keep_keys for e in val_ids])
assert tr_mask.sum()==1138 and va_mask.sum()==236 and tr_mask.all()
PXY_TR=np.asarray(tr['pos_xy'])[tr_mask]; CXY_TR=np.asarray(tr['ctrl_xy'])[tr_mask]
PXY_VA=np.asarray(va['pos_xy'])[va_mask]; CXY_VA=np.asarray(va['ctrl_xy'])[va_mask]
CLS_VA=np.asarray(va['target_class']).astype(str)[va_mask]
EID_VA=np.asarray(va['event_id']).astype(str)[va_mask]

# -----------------------------------------------------------------------------
# Verified camera-view resolutions
# -----------------------------------------------------------------------------
def parse_res(s):
    m=re.fullmatch(r'(\d+)x(\d+)',str(s).strip()); assert m,s
    return int(m.group(1)),int(m.group(2))
view_res={}
x=pd.read_csv(OBS_RES,dtype={'view':str}); x['view']=x['view'].astype(str).str.zfill(6)
for _,r in x.iterrows():
    assert int(r['n_unique_resolutions'])==1; view_res[r['view']]=parse_res(r['resolutions'])
x=pd.read_csv(REC_RES,dtype={'view':str}); x['view']=x['view'].astype(str).str.zfill(6)
for _,r in x.iterrows():
    assert int(r['n_direct_videos'])>=1
    res=parse_res(r['direct_resolutions'])
    if r['view'] in view_res: assert view_res[r['view']]==res
    view_res[r['view']]=res
assert len(view_res)==21
VIEW_RE=re.compile(r'VIRAT_S_(\d{6})')
def view_of(clip):
    m=VIEW_RE.search(str(clip)); assert m,clip; return m.group(1)
def fs_of(clip):
    v=view_of(clip); assert v in view_res; return view_res[v]
meta=g2.set_index('event_key')
def fs_for_ids(ids,which):
    clips=meta.loc[ids,f'{which}_clip'].astype(str).tolist()
    return np.asarray([fs_of(c) for c in clips],dtype=np.float32)
POS_FS_TR=fs_for_ids(train_ids[tr_mask],'pos'); CTRL_FS_TR=fs_for_ids(train_ids[tr_mask],'ctrl')
POS_FS_VA=fs_for_ids(val_ids[va_mask],'pos'); CTRL_FS_VA=fs_for_ids(val_ids[va_mask],'ctrl')

# -----------------------------------------------------------------------------
# Motion-6 features
# -----------------------------------------------------------------------------
def motion6(xy,fs):
    xy=np.asarray(xy,dtype=np.float32); fs=np.asarray(fs,dtype=np.float32)
    fw,fh=np.maximum(fs[:,0:1],1.),np.maximum(fs[:,1:2],1.)
    cx,cy=xy[:,:,0]/fw,xy[:,:,1]/fh
    dcx=np.diff(cx,axis=1,prepend=cx[:,:1]); dcy=np.diff(cy,axis=1,prepend=cy[:,:1])
    step=np.sqrt(dcx**2+dcy**2); speed=step.copy()
    d_speed=np.diff(speed,axis=1,prepend=speed[:,:1]); accel=np.abs(d_speed)
    d_accel=np.diff(accel,axis=1,prepend=accel[:,:1])
    heading=np.arctan2(dcy,dcx); dh=np.diff(heading,axis=1,prepend=heading[:,:1])
    heading_rate=np.angle(np.exp(1j*dh)).astype(np.float32)
    curvature=heading_rate/(speed+1e-8)
    dx0,dy0=cx-cx[:,:1],cy-cy[:,:1]
    displacement=np.sqrt(dx0**2+dy0**2)
    feat=np.stack([d_speed,d_accel,heading_rate,curvature,displacement,step],axis=-1).astype(np.float32)
    assert np.isfinite(feat).all(); return feat

P_TR=motion6(PXY_TR,POS_FS_TR); C_TR=motion6(CXY_TR,CTRL_FS_TR)
P_VA=motion6(PXY_VA,POS_FS_VA); C_VA=motion6(CXY_VA,CTRL_FS_VA)
assert P_TR.shape==(1138,90,6) and P_VA.shape==(236,90,6)

# -----------------------------------------------------------------------------
# Motion TCN
# -----------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self,n): super().__init__(); self.n=n
    def forward(self,x): return x if self.n==0 else x[:,:,:-self.n].contiguous()
class TCNBlock(nn.Module):
    def __init__(self,in_ch,out_ch,dilation):
        super().__init__(); pad=2*dilation
        self.net=nn.Sequential(nn.Conv1d(in_ch,out_ch,3,padding=pad,dilation=dilation),Chomp1d(pad),nn.ReLU(),nn.Dropout(DROPOUT),nn.Conv1d(out_ch,out_ch,3,padding=pad,dilation=dilation),Chomp1d(pad),nn.ReLU(),nn.Dropout(DROPOUT))
        self.res=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x): return self.net(x)+self.res(x)
class MotionTCN(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=nn.Sequential(TCNBlock(6,HIDDEN,1),TCNBlock(HIDDEN,HIDDEN,2),TCNBlock(HIDDEN,HIDDEN,4)); self.head=nn.Linear(HIDDEN,1)
    def forward(self,x): return self.head(self.blocks(x.transpose(1,2)).mean(dim=2)).squeeze(1)
probe=MotionTCN(); assert sum(p.numel() for p in probe.parameters() if p.requires_grad)==98561; del probe

# -----------------------------------------------------------------------------
# Training and metrics
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def norm_stats(P,C):
    x=np.concatenate([P,C],axis=0); mu=x.mean(axis=(0,1),keepdims=True); sd=x.std(axis=(0,1),keepdims=True); sd[sd<1e-9]=1.
    return mu.astype(np.float32),sd.astype(np.float32)
def predict(model,P,C,mu,sd):
    model.eval(); lp,lc=[],[]
    with torch.no_grad():
        for st in range(0,len(P),256):
            en=min(st+256,len(P)); xp=torch.tensor((P[st:en]-mu)/sd,dtype=torch.float32,device=DEVICE); xc=torch.tensor((C[st:en]-mu)/sd,dtype=torch.float32,device=DEVICE)
            lp.append(model(xp).cpu().numpy()); lc.append(model(xc).cpu().numpy())
    return np.concatenate(lp),np.concatenate(lc)
def metric_dict(pl,cl):
    pp=1/(1+np.exp(-np.clip(pl,-40,40))); cp=1/(1+np.exp(-np.clip(cl,-40,40)))
    y=np.r_[np.ones(len(pp)),np.zeros(len(cp))]; pr=np.r_[pp,cp]
    class_pw=[float(np.mean(pl[CLS_VA==cls]>cl[CLS_VA==cls])) for cls in sorted(np.unique(CLS_VA))]
    return {'AP':float(average_precision_score(y,pr)),'AUC':float(roc_auc_score(y,pr)),'PW':float(np.mean(pl>cl)),'MacroPW':float(np.mean(class_pw))}
def run_one(seed):
    set_seed(seed); mu,sd=norm_stats(P_TR,C_TR)
    X=np.concatenate([(P_TR-mu)/sd,(C_TR-mu)/sd],axis=0).astype(np.float32)
    y=np.r_[np.ones(len(P_TR),dtype=np.float32),np.zeros(len(C_TR),dtype=np.float32)]
    gen=torch.Generator(); gen.manual_seed(seed)
    dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=BATCH,shuffle=True,generator=gen)
    model=MotionTCN().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); loss_fn=nn.BCEWithLogitsLoss()
    best_ap,best_state,best_epoch,wait=-1,None,None,0
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step()
        pl,cl=predict(model,P_VA,C_VA,mu,sd); mm=metric_dict(pl,cl)
        if mm['AP']>best_ap+1e-10:
            best_ap=mm['AP']; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; wait=0
        else:
            wait+=1
            if wait>=PATIENCE: break
    assert best_state is not None; model.load_state_dict(best_state)
    pl,cl=predict(model,P_VA,C_VA,mu,sd)
    return metric_dict(pl,cl),pl,cl,best_epoch

rows=[]; pred_rows=[]
for seed in SEEDS:
    mm,pl,cl,ep=run_one(seed)
    rows.append({'seed':seed,'N_val':len(EID_VA),'best_epoch':ep,**mm})
    for eid,cls,pv,cv in zip(EID_VA,CLS_VA,pl,cl):
        pred_rows.append({'seed':seed,'event_id':eid,'target_class':cls,'pos_logit':float(pv),'ctrl_logit':float(cv)})
strict_metrics=pd.DataFrame(rows); strict_pred=pd.DataFrame(pred_rows)
strict_metrics.to_csv(OUT/'strict_g2_10seed_metrics.csv',index=False)
strict_pred.to_csv(OUT/'strict_g2_10seed_predictions.csv',index=False)

# -----------------------------------------------------------------------------
# Direct seed-wise contrast against the final standard G2 result
# -----------------------------------------------------------------------------
std=pd.read_csv(STANDARD_DIR/'all_gaps_10seed_metrics.csv')
std=std[std['gap']==2].copy(); assert len(std)==10 and set(std['seed'].astype(int))==set(SEEDS)
std=std.set_index('seed').loc[SEEDS].reset_index(); strict_metrics=strict_metrics.set_index('seed').loc[SEEDS].reset_index()
strict_metrics['standard_PW']=std['safe_PW'].to_numpy()
strict_metrics['delta_PW_strict_minus_standard']=strict_metrics['PW']-strict_metrics['standard_PW']
strict_metrics.to_csv(OUT/'strict_vs_standard_10seed.csv',index=False)

# -----------------------------------------------------------------------------
# Formal pair-level inference on 10-seed averaged logits
# -----------------------------------------------------------------------------
avg=strict_pred.groupby(['event_id','target_class'],as_index=False).agg(pos_logit=('pos_logit','mean'),ctrl_logit=('ctrl_logit','mean'),n_seeds=('seed','nunique'))
assert len(avg)==236 and (avg['n_seeds']==10).all()
pos,ctrl=avg['pos_logit'].to_numpy(),avg['ctrl_logit'].to_numpy(); wins=pos>ctrl; losses=pos<ctrl; ties=~(wins|losses)
formal_pw=float(wins.mean())
rng=np.random.default_rng(20260828); B=50000; n=len(wins); boot=np.empty(B,dtype=np.float64); st=0
while st<B:
    m=min(5000,B-st); ix=rng.integers(0,n,size=(m,n)); boot[st:st+m]=wins[ix].mean(axis=1); st+=m
ci_lo,ci_hi=np.quantile(boot,[.025,.975]); nw,nl=int(wins.sum()),int(losses.sum())
p_two=float(binomtest(nw,nw+nl,p=.5,alternative='two-sided').pvalue)
formal={'N':236,'PW':formal_pw,'CI95_low':float(ci_lo),'CI95_high':float(ci_hi),'wins':nw,'losses':nl,'ties':int(ties.sum()),'p_two_sided':p_two}
with open(OUT/'strict_g2_formal.json','w') as f: json.dump(formal,f,indent=2)
avg.to_csv(OUT/'strict_g2_seed_averaged_logits.csv',index=False)

print('Structural fingerprint:',{'train':n_train,'val':n_val,'max_abs_smd':MAX_POST,'removed':removed_counts})
print('Seed-wise strict PW mean/std:',strict_metrics['PW'].mean(),strict_metrics['PW'].std(ddof=1))
print('Seed-wise standard PW mean/std:',strict_metrics['standard_PW'].mean(),strict_metrics['standard_PW'].std(ddof=1))
print('Formal strict:',formal)
print('Saved:',OUT)
