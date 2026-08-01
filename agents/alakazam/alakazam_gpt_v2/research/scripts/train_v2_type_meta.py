from __future__ import annotations
import json, numpy as np, lightgbm as lgb
from pathlib import Path
D=Path('/mnt/data/v1_work/dataset'); R=Path('/mnt/data/v1_work/results'); O=R/'v2_type_meta'; O.mkdir(parents=True,exist_ok=True)
names=json.load(open(D/'feature_names.json')); idx={n:i for i,n in enumerate(names)}
# combine OOF train score arrays
folds=[np.load(R/f'oof/fold_{i}_scores.npy',mmap_mode='r') for i in range(4)]
train_scores=np.full(folds[0].shape,np.nan,dtype=np.float32)
for a in folds:
 m=~np.isnan(a); train_scores[m]=a[m]
assert not np.isnan(train_scores).any()
base=np.load(R/'v1_baseline_scores.npz')
score_by_split={'train':train_scores,'validation':base['validation'],'test':base['test']}
# top invariant state features by baseline model importance + mandatory sequence/state
booster=lgb.Booster(model_file=str(R/'v1_baseline_lgb.txt'))
imp=dict(zip(booster.feature_name(),booster.feature_importance('gain')))
mandatory=[n for n in names if n.startswith('seq_') or n in {
 'turn','turn_action_count','legal_option_count','self_hand_count','self_deck_count','self_prize_count','self_board_count','opp_hand_count','opp_deck_count','opp_prize_count','opp_board_count','self_active_id','opp_active_id','current_powerful_hand_damage','current_ko_estimate','has_ready_active_alakazam','has_ready_backup_alakazam','has_alakazam_anywhere','has_abra_anywhere','dudunsparce_engine_count','dunsparce_engine_count','supporter_played','stadium_played','energy_attached','early_game','deck_runway_margin','deck_pressure_risk','visible_psychic_energy_count'}]
# invariant within candidate group based on known varying list from sample
X0=np.load(D/'train_features.npy',mmap_mode='r');g0=np.load(D/'train_groups.npy');var=np.zeros(len(names),bool);off=0
for z in g0[:5000]:
 e=off+int(z);a=np.asarray(X0[off:e],np.float32);var|=(np.ptp(a,axis=0)!=0);off=e
inv_names=[n for i,n in enumerate(names) if not var[i]]
ranked=sorted(inv_names,key=lambda n:-imp.get(n,0.0))
state_names=[]
for n in mandatory+ranked:
 if n in inv_names and n not in state_names: state_names.append(n)
state_names=state_names[:180]
state_idx=np.asarray([idx[n] for n in state_names],dtype=np.int64)
# candidate details per type
cand_fields=['candidate_card_id','candidate_attack_id','candidate_target_id','candidate_target_hp','candidate_target_energy','candidate_option_position','fallback_policy_score','legacy_ranker_score','v29_ranker_score']
cf_idx={n:idx[n] for n in cand_fields if n in idx}
action_idx=idx['action_type_id']

def build(split):
 X=np.load(D/f'{split}_features.npy',mmap_mode='r');g=np.load(D/f'{split}_groups.npy');meta=np.load(D/f'{split}_decision_meta.npy',mmap_mode='r');sc=np.asarray(score_by_split[split],np.float32)
 F=len(state_idx)+12*(8+len(cf_idx))+8
 out=np.zeros((len(g),F),dtype=np.float32); y=np.asarray(meta[:,3],np.int32); off=0
 for di,z in enumerate(g):
  e=off+int(z); a=np.asarray(X[off:e],np.float32); s=sc[off:e]
  out[di,:len(state_idx)]=a[0,state_idx]
  pos=len(state_idx)
  type_tops=[]
  for t in range(12):
   m=np.flatnonzero(a[:,action_idx].astype(np.int32)==t)
   if len(m):
    vals=s[m]; order=m[np.argsort(-vals,kind='stable')]; top=order[0]; second=vals[np.argsort(-vals,kind='stable')[1]] if len(m)>1 else vals.max()-5
    feats=[len(m),vals.max(),second,vals.max()-second,vals.mean(),vals.std(),float(top),float(np.sum(vals>vals.max()-0.25))]
    type_tops.append(vals.max())
    for n,j in cf_idx.items(): feats.append(a[top,j])
   else:
    feats=[0,-99,-99,0,-99,0,-1,0]+[-1]*len(cf_idx);type_tops.append(-99)
   out[di,pos:pos+len(feats)]=feats;pos+=len(feats)
  tt=np.asarray(type_tops,np.float32); ordt=np.argsort(-tt)
  probs=np.exp(np.clip(tt-tt.max(),-20,20));probs/=probs.sum()
  out[di,pos:pos+8]=[ordt[0],ordt[1],tt[ordt[0]],tt[ordt[1]],tt[ordt[0]]-tt[ordt[1]],-(probs*np.log(probs+1e-9)).sum(),(tt>-90).sum(),float(a[np.argmax(s),action_idx])]
  off=e
 np.save(O/f'{split}_X.npy',out);np.save(O/f'{split}_y.npy',y)
 return out,y

sets={}
for s in ['train','validation','test']:
 p=O/f'{s}_X.npy'; sets[s]=(np.load(p,mmap_mode='r'),np.load(O/f'{s}_y.npy')) if p.exists() else build(s)
X,y=sets['train'];Xv,yv=sets['validation'];Xt,yt=sets['test']
print('data',X.shape,flush=True)
configs=[
 ('m31',300,.04,31,35,.9),
 ('m63',400,.03,63,30,.85),
 ('m127',450,.025,127,35,.8),
]
res=[]
for name,trees,lr,leaves,child,col in configs:
 print('fit',name,flush=True)
 model=lgb.LGBMClassifier(objective='multiclass',num_class=12,n_estimators=trees,learning_rate=lr,num_leaves=leaves,min_child_samples=child,colsample_bytree=col,subsample=.9,subsample_freq=1,reg_alpha=.3,reg_lambda=3,random_state=3507,n_jobs=8,verbosity=-1)
 model.fit(X,y,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(50)])
 pv=model.predict_proba(Xv,num_iteration=model.best_iteration_);pt=model.predict_proba(Xt,num_iteration=model.best_iteration_)
 av=float((pv.argmax(1)==yv).mean());at=float((pt.argmax(1)==yt).mean())
 print('RESULT',name,model.best_iteration_,av,at,flush=True)
 model.booster_.save_model(str(O/f'{name}.txt'),num_iteration=model.best_iteration_)
 np.savez_compressed(O/f'{name}_probs.npz',validation=pv.astype(np.float32),test=pt.astype(np.float32))
 res.append({'name':name,'iter':model.best_iteration_,'validation':av,'test':at})
json.dump({'state_names':state_names,'results':res},open(O/'summary.json','w'),indent=2)
