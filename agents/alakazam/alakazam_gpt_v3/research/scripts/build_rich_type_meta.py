from pathlib import Path
import json,numpy as np
D=Path('/mnt/data/v1_work/dataset');R=Path('/mnt/data/v1_work/results');O=Path('/mnt/data/v3_work/rich_meta');O.mkdir(exist_ok=True)
names=json.load(open(D/'feature_names.json'));idx={n:i for i,n in enumerate(names)}
spec=json.load(open('/mnt/data/v3_work/alakazam_gpt_v2/type_runtime_spec.json'));state_names=spec['state_names'];state_idx=np.array([idx[n] for n in state_names],np.int32);ai=idx['action_type_id']
# candidate fields of top-scoring option, all runtime-computable
TOP_FIELDS=[n for n in [
'candidate_card_id','candidate_attack_id','candidate_target_id','candidate_target_hp','candidate_target_max_hp','candidate_target_energy','candidate_target_special_energy','candidate_area','candidate_inplay_area','candidate_option_position','candidate_option_reverse_position','candidate_raw_index','candidate_raw_inplay_index','candidate_hand_cost','candidate_net_hand_delta','candidate_total_draw_count','candidate_evolution_draw_count','candidate_enriching_draw_count','post_action_hand_count','post_action_powerful_hand_damage','post_action_ko_estimate','attack_lethal_estimate','breaks_current_ko_estimate','preserves_current_ko_estimate','candidate_fills_energy','candidate_fills_missing_abra','candidate_fills_stage2','candidate_is_route_card','candidate_is_psychic_energy','candidate_is_rare_candy','candidate_target_is_active','candidate_target_is_bench','candidate_same_action_preceding','candidate_same_card_preceding','same_action_option_count','same_card_option_count'] if n in idx]
# aggregate numeric/boolean fields across a type
AGG_FIELDS=[n for n in [
'attack_lethal_estimate','breaks_current_ko_estimate','preserves_current_ko_estimate','post_action_ko_estimate','post_action_hand_count','post_action_powerful_hand_damage','candidate_hand_cost','candidate_net_hand_delta','candidate_total_draw_count','candidate_evolution_draw_count','candidate_enriching_draw_count','candidate_fills_energy','candidate_fills_missing_abra','candidate_fills_stage2','candidate_is_route_card','candidate_is_psychic_energy','candidate_is_rare_candy','candidate_target_is_active','candidate_target_is_bench','candidate_enrich_cycle_target','candidate_enrich_draw_safe'] if n in idx]
BASE_STATS=14 # count/max/second/gap/mean/std/min/range/near.1/.25/.5/toplocal/minpos/meanpos
BLOCK=BASE_STATS+len(TOP_FIELDS)+len(AGG_FIELDS)*3+2 # agg max/mean/sum + unique card/target
print({'state':len(state_names),'top':len(TOP_FIELDS),'agg':len(AGG_FIELDS),'block':BLOCK},flush=True)
# score arrays
folds=[np.load(R/f'oof/fold_{i}_scores.npy',mmap_mode='r') for i in range(4)];oof=np.full(folds[0].shape,np.nan,np.float32)
for a in folds:oof[np.isfinite(a)]=a[np.isfinite(a)]
assert np.isfinite(oof).all();base=np.load(R/'v1_baseline_scores.npz');SM={'train':oof,'validation':base['validation'],'test':base['test']}
feature_names=[f'state__{n}' for n in state_names]
for t in range(12):
 feature_names += [f't{t}__{x}' for x in ['count','score_max','score_second','score_gap','score_mean','score_std','score_min','score_range','near_01','near_025','near_05','top_local_index','min_option_pos','mean_option_pos']]
 feature_names += [f't{t}__top__{n}' for n in TOP_FIELDS]
 for n in AGG_FIELDS: feature_names += [f't{t}__max__{n}',f't{t}__mean__{n}',f't{t}__sum__{n}']
 feature_names += [f't{t}__unique_card_count',f't{t}__unique_target_count']
# global per type mask, rank, delta, and summary
GLOBAL_PER=['legal','count','top_score','score_delta_best','score_rank','first_option_pos']
feature_names += [f'global_t{t}__{n}' for t in range(12) for n in GLOBAL_PER]
feature_names += ['global_best_type','global_second_type','global_third_type','global_best_score','global_second_score','global_third_score','global_gap12','global_gap23','global_entropy','global_legal_types','global_base_top_type','global_base_second_type','global_base_top_gap']
F=len(feature_names);print('F',F,flush=True)

def build(split):
 X=np.load(D/f'{split}_features.npy',mmap_mode='r');g=np.load(D/f'{split}_groups.npy');meta=np.load(D/f'{split}_decision_meta.npy',mmap_mode='r');sc=np.asarray(SM[split],np.float32)
 out=np.zeros((len(g),F),np.float32);y=meta[:,3].astype(np.int32);off=0
 top_idx=np.array([idx[n] for n in TOP_FIELDS],np.int32);agg_idx=np.array([idx[n] for n in AGG_FIELDS],np.int32);card_i=idx['candidate_card_id'];target_i=idx['candidate_target_id'];pos_i=idx['candidate_option_position']
 for di,z0 in enumerate(g):
  z=int(z0);e=off+z;a=np.asarray(X[off:e],np.float32);s=sc[off:e];p=0;out[di,p:p+len(state_idx)]=a[0,state_idx];p+=len(state_idx)
  type_count=np.zeros(12,np.float32);type_top=np.full(12,-99,np.float32);first_pos=np.full(12,99,np.float32)
  for t in range(12):
   ids=np.flatnonzero(a[:,ai].astype(np.int32)==t)
   block=np.zeros(BLOCK,np.float32)
   if len(ids):
    vals=s[ids];order=np.argsort(-vals,kind='stable');top=ids[order[0]];second=float(vals[order[1]]) if len(ids)>1 else float(vals.max()-5);mx=float(vals.max());mn=float(vals.min());positions=a[ids,pos_i]
    basef=[len(ids),mx,second,mx-second,float(vals.mean()),float(vals.std()),mn,mx-mn,float(np.sum(vals>=mx-.1)),float(np.sum(vals>=mx-.25)),float(np.sum(vals>=mx-.5)),float(order[0]),float(positions.min()),float(positions.mean())]
    q=0;block[q:q+BASE_STATS]=basef;q+=BASE_STATS
    block[q:q+len(top_idx)]=a[top,top_idx];q+=len(top_idx)
    av=a[ids][:,agg_idx]
    for j in range(len(AGG_FIELDS)):
     block[q:q+3]=[float(av[:,j].max()),float(av[:,j].mean()),float(av[:,j].sum())];q+=3
    block[q]=len(np.unique(a[ids,card_i]));block[q+1]=len(np.unique(a[ids,target_i]));
    type_count[t]=len(ids);type_top[t]=mx;first_pos[t]=positions.min()
   else:
    # missing types use explicit sentinel in score/top identity slots
    block[1:8]=[-99,-99,0,-99,0,-99,0]
    q=BASE_STATS
    for j,n in enumerate(TOP_FIELDS):block[q+j]=-1
   out[di,p:p+BLOCK]=block;p+=BLOCK
  legal=(type_count>0).astype(np.float32);order=np.argsort(-type_top,kind='stable');ranks=np.empty(12,np.float32);ranks[order]=np.arange(12)
  best=float(type_top[order[0]])
  for t in range(12):
   out[di,p:p+6]=[legal[t],type_count[t],type_top[t],type_top[t]-best,ranks[t],first_pos[t]];p+=6
  valid=type_top[type_top>-90];soft=np.exp(np.clip(valid-valid.max(),-20,20));soft/=soft.sum();entropy=float(-(soft*np.log(soft+1e-9)).sum())
  out[di,p:p+13]=[order[0],order[1],order[2],type_top[order[0]],type_top[order[1]],type_top[order[2]],type_top[order[0]]-type_top[order[1]],type_top[order[1]]-type_top[order[2]],entropy,len(valid),int(a[np.argmax(s),ai]),int(a[np.argsort(-s,kind='stable')[1],ai]),float(s[np.argmax(s)]-s[np.argsort(-s,kind='stable')[1]])]
  off=e
 np.save(O/f'{split}_X.npy',out);np.save(O/f'{split}_y.npy',y);print(split,out.shape,flush=True)
for s in ['train','validation','test']:build(s)
json.dump({'feature_names':feature_names,'state_names':state_names,'top_fields':TOP_FIELDS,'agg_fields':AGG_FIELDS,'block':BLOCK},open(O/'spec.json','w'),indent=2)
