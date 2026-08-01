from pathlib import Path
import json,numpy as np
from numba import njit,prange
D=Path('/mnt/data/v1_work/dataset');R=Path('/mnt/data/v1_work/results');B=Path('/mnt/data/v1_work/results/v2_type_meta');O=Path('/mnt/data/v3_work/rich_meta');O.mkdir(exist_ok=True)
names=json.load(open(D/'feature_names.json'));idx={n:i for i,n in enumerate(names)}
TOP=[n for n in ['candidate_target_max_hp','candidate_target_special_energy','candidate_area','candidate_inplay_area','candidate_option_reverse_position','candidate_raw_index','candidate_raw_inplay_index','candidate_hand_cost','candidate_net_hand_delta','candidate_total_draw_count','candidate_evolution_draw_count','candidate_enriching_draw_count','post_action_hand_count','post_action_powerful_hand_damage','post_action_ko_estimate','attack_lethal_estimate','breaks_current_ko_estimate','preserves_current_ko_estimate','candidate_fills_energy','candidate_fills_missing_abra','candidate_fills_stage2','candidate_is_route_card','candidate_is_psychic_energy','candidate_is_rare_candy'] if n in idx]
AGG=[n for n in ['attack_lethal_estimate','breaks_current_ko_estimate','preserves_current_ko_estimate','post_action_ko_estimate','post_action_hand_count','post_action_powerful_hand_damage','candidate_hand_cost','candidate_net_hand_delta','candidate_total_draw_count','candidate_evolution_draw_count','candidate_enriching_draw_count','candidate_fills_energy','candidate_fills_missing_abra','candidate_fills_stage2','candidate_is_route_card'] if n in idx]
top_idx=np.array([idx[n] for n in TOP],np.int32);agg_idx=np.array([idx[n] for n in AGG],np.int32);ai=idx['action_type_id'];posi=idx['candidate_option_position'];cardi=idx['candidate_card_id'];targeti=idx['candidate_target_id']
PER=len(TOP)+2*len(AGG)+4;print({'top':len(TOP),'agg':len(AGG),'per':PER,'total_extra':12*PER},flush=True)
folds=[np.load(R/f'oof/fold_{i}_scores.npy',mmap_mode='r') for i in range(4)];oof=np.full(folds[0].shape,np.nan,np.float32)
for a in folds:oof[np.isfinite(a)]=a[np.isfinite(a)]
base=np.load(R/'v1_baseline_scores.npz');SM={'train':oof,'validation':base['validation'],'test':base['test']}
@njit(parallel=True)
def make_extra(X,scores,starts,ends,top_idx,agg_idx,ai,posi,cardi,targeti):
 n=len(starts);per=len(top_idx)+2*len(agg_idx)+4;out=np.zeros((n,12*per),np.float32)
 for d in prange(n):
  st=starts[d];en=ends[d]
  for t in range(12):
   basecol=t*per;top=-1;topscore=-1e30;cnt=0;minpos=1e9;sumpos=0.0
   # top and counts
   for r in range(st,en):
    if int(X[r,ai])==t:
     cnt+=1;v=scores[r]
     if v>topscore:topscore=v;top=r
     p=X[r,posi];sumpos+=p
     if p<minpos:minpos=p
   if cnt==0:
    for j in range(len(top_idx)):out[d,basecol+j]=-1
    continue
   c=basecol
   for j in range(len(top_idx)):out[d,c+j]=X[top,top_idx[j]]
   c+=len(top_idx)
   for j in range(len(agg_idx)):
    mx=-1e30;sm=0.0
    for r in range(st,en):
     if int(X[r,ai])==t:
      v=X[r,agg_idx[j]];sm+=v
      if v>mx:mx=v
    out[d,c]=mx;out[d,c+1]=sm;c+=2
   out[d,c]=minpos;out[d,c+1]=sumpos/cnt
   # unique card/target small nested
   uc=0;ut=0
   for r in range(st,en):
    if int(X[r,ai])!=t:continue
    seen=False
    for q in range(st,r):
     if int(X[q,ai])==t and X[q,cardi]==X[r,cardi]:seen=True;break
    if not seen:uc+=1
    seen=False
    for q in range(st,r):
     if int(X[q,ai])==t and X[q,targeti]==X[r,targeti]:seen=True;break
    if not seen:ut+=1
   out[d,c+2]=uc;out[d,c+3]=ut
 return out
for split in ['train','validation','test']:
 X=np.asarray(np.load(D/f'{split}_features.npy',mmap_mode='r'),np.float32);g=np.load(D/f'{split}_groups.npy').astype(np.int64);ends=np.cumsum(g);starts=np.r_[0,ends[:-1]].astype(np.int64);extra=make_extra(X,np.asarray(SM[split],np.float32),starts,ends,top_idx,agg_idx,ai,posi,cardi,targeti);old=np.load(B/f'{split}_X.npy',mmap_mode='r');rich=np.concatenate([np.asarray(old,np.float32),extra],axis=1);np.save(O/f'{split}_X.npy',rich);np.save(O/f'{split}_y.npy',np.load(B/f'{split}_y.npy'));print(split,rich.shape,flush=True)
json.dump({'base_features':356,'top_fields':TOP,'agg_fields':AGG,'per_type_extra':PER,'total_features':356+12*PER},open(O/'spec.json','w'),indent=2)
