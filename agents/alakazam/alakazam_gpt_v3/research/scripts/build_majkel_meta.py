import numpy as np,json,lightgbm as lgb
from pathlib import Path
Z=np.load('/mnt/data/v3_work/majkel_dataset.npz',allow_pickle=False)
X=Z['features'].astype(np.float32); g=Z['groups'].astype(int); meta=Z['decision_meta']; names=Z['feature_names'].astype(str).tolist(); idx={n:i for i,n in enumerate(names)}
spec=json.load(open('/mnt/data/v3_work/alakazam_gpt_v2/type_runtime_spec.json')); state_names=spec['state_names']; cand_fields=spec['candidate_fields'];state_idx=np.array([idx[n] for n in state_names]);cf_idx={n:idx[n] for n in cand_fields};ai=idx['action_type_id']
bo=lgb.Booster(model_file='/mnt/data/v1_work/results/v1_baseline_lgb.txt');scores=bo.predict(X).astype(np.float32)
F=len(state_idx)+12*(8+len(cf_idx))+8;out=np.zeros((len(g),F),np.float32);y=meta[:,3].astype(np.int32);off=0
for di,z in enumerate(g):
 e=off+z;a=X[off:e];s=scores[off:e];out[di,:len(state_idx)]=a[0,state_idx];pos=len(state_idx);tops=[]
 for t in range(12):
  m=np.flatnonzero(a[:,ai].astype(int)==t)
  if len(m):
   vals=s[m];o=np.argsort(-vals,kind='stable');top=m[o[0]];second=vals[o[1]] if len(m)>1 else vals.max()-5
   feats=[len(m),vals.max(),second,vals.max()-second,vals.mean(),vals.std(),float(top),float(np.sum(vals>vals.max()-0.25))]
   tops.append(vals.max());feats.extend(a[top,j] for j in cf_idx.values())
  else: feats=[0,-99,-99,0,-99,0,-1,0]+[-1]*len(cf_idx);tops.append(-99)
  out[di,pos:pos+len(feats)]=feats;pos+=len(feats)
 tt=np.array(tops,np.float32);ordt=np.argsort(-tt);p=np.exp(np.clip(tt-tt.max(),-20,20));p/=p.sum();out[di,pos:pos+8]=[ordt[0],ordt[1],tt[ordt[0]],tt[ordt[1]],tt[ordt[0]]-tt[ordt[1]],-(p*np.log(p+1e-9)).sum(),(tt>-90).sum(),float(a[np.argmax(s),ai])];off=e
np.save('/mnt/data/v3_work/majkel_meta_X.npy',out);np.save('/mnt/data/v3_work/majkel_meta_y.npy',y)
print(out.shape,dict(zip(*np.unique(y,return_counts=True))))
