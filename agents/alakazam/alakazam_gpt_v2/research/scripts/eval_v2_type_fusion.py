import numpy as np,json
from pathlib import Path
D=Path('/mnt/data/v1_work/dataset');R=Path('/mnt/data/v1_work/results');O=R/'v2_type_meta'
names=json.load(open(D/'feature_names.json'));ai=names.index('action_type_id')
classes=np.unique(np.load(O/'train_y.npy'));probs=np.load(O/'xgb_probs.npz');base=np.load(R/'v1_baseline_scores.npz')
def eval_split(split,alpha,mode='add',conf=0):
 X=np.load(D/f'{split}_features.npy',mmap_mode='r');y=np.load(D/f'{split}_labels.npy',mmap_mode='r');g=np.load(D/f'{split}_groups.npy');p=probs[split];s=base[split];off=0;hit=0
 for di,z in enumerate(g):
  e=off+int(z);types=X[off:e,ai].astype(int);bs=np.asarray(s[off:e],float);pr=np.full(int(z),1e-9)
  for ci,c in enumerate(classes):pr[types==c]=p[di,ci]
  if mode=='restrict':
   t=classes[int(np.argmax(p[di]))]
   if p[di].max()>=conf and np.any(types==t): score=np.where(types==t,bs,-1e9)
   else:score=bs
  else:
   zz=(bs-bs.mean())/max(bs.std(),1e-5);score=zz+alpha*np.log(np.maximum(pr,1e-9))
  hit+=int(y[off+int(np.argmax(score))]);off=e
 return hit/len(g)
best=(0,None)
for a in np.linspace(0,4,81):
 v=eval_split('validation',a)
 if v>best[0]:best=(v,a)
print('add best',best,'test',eval_split('test',best[1]))
for c in [0,.3,.4,.5,.6,.7,.8,.85,.9,.95,.98,.99]:
 print('restrict',c,eval_split('validation',0,'restrict',c),eval_split('test',0,'restrict',c))
