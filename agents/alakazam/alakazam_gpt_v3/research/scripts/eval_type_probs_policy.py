import numpy as np,json,sys
from pathlib import Path
model=sys.argv[1]
D=Path('/mnt/data/v1_work/dataset');R=Path('/mnt/data/v1_work/results');O=Path('/mnt/data/v1_work/results/v2_type_meta')
names=json.load(open(D/'feature_names.json'));ai=names.index('action_type_id');classes=np.load(O/'final/xgb_final_probs.npz')['classes'].astype(int);base=np.load(R/'v1_baseline_scores.npz')
if model=='rich': pv=np.load('/mnt/data/v3_work/rich_models/rich80.npz')['validation'];pt=np.load('/mnt/data/v3_work/rich_models/rich80_test.npz')['test']
elif model=='stack': z=np.load('/mnt/data/v3_work/stack/probs.npz');pv=z['validation'];pt=z['test']
elif model=='base':pv=np.load(O/'xgb_probs.npz')['validation'];pt=np.load(O/'final/xgb_final_probs.npz')['test']
P={'validation':pv,'test':pt}
def ev(split,alpha,restrict=False,conf=0):
 X=np.load(D/f'{split}_features.npy',mmap_mode='r');y=np.load(D/f'{split}_labels.npy',mmap_mode='r');g=np.load(D/f'{split}_groups.npy');p=P[split];s=base[split];off=0;hit=0
 for di,z0 in enumerate(g):
  e=off+int(z0);types=X[off:e,ai].astype(int);bs=np.asarray(s[off:e],float);pr=np.full(int(z0),1e-9)
  for ci,c in enumerate(classes):pr[types==c]=p[di,ci]
  if restrict:
   t=classes[p[di].argmax()];score=np.where((types==t)&(p[di].max()>=conf),bs,bs if p[di].max()<conf else -1e9)
  else:
   zz=(bs-bs.mean())/max(bs.std(),1e-5);score=zz+alpha*np.log(np.maximum(pr,1e-9))
  hit+=int(y[off+int(np.argmax(score))]);off=e
 return hit/len(g)
best=(0,None)
for a in np.linspace(0,2,81):
 v=ev('validation',a)
 if v>best[0]:best=(v,float(a))
print(model,'add',best,'test',ev('test',best[1]))
for c in np.linspace(.3,.9,13):print('r',c,ev('validation',0,True,c),ev('test',0,True,c))
