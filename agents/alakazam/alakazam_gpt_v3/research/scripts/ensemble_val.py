import numpy as np,itertools,json
from pathlib import Path
O=Path('/mnt/data/v1_work/results/v2_type_meta'); y=np.load(O/'validation_y.npy');classes=np.load(O/'final/xgb_final_probs.npz')['classes'].astype(int)
mods={
'base':np.load(O/'xgb_probs.npz')['validation'],
'rich':np.load('/mnt/data/v3_work/rich_models/rich80.npz')['validation'],
'aug':np.load('/mnt/data/v3_work/aug_xgb/w0p03.npz')['validation'],
'cat':np.load('/mnt/data/v3_work/cat_type/d6_val.npz')['p'],
}
for k,p in mods.items():print(k,float(np.mean(classes[p.argmax(1)]==y)))
# oracle
preds={k:classes[p.argmax(1)] for k,p in mods.items()}
print('oracle',float(np.mean(np.any(np.stack([v==y for v in preds.values()]),axis=0))))
best=[]
keys=list(mods)
# convex grids step .05 for base/rich/aug; cat optional
for wb in np.arange(0,1.001,.05):
 for wr in np.arange(0,1.001-wb,.05):
  for wa in np.arange(0,1.001-wb-wr,.05):
   wc=1-wb-wr-wa
   if wc<-1e-6:continue
   p=wb*mods['base']+wr*mods['rich']+wa*mods['aug']+wc*mods['cat'];acc=float(np.mean(classes[p.argmax(1)]==y));best.append((acc,wb,wr,wa,wc))
best.sort(reverse=True)
for r in best[:20]:print(r)
json.dump({'best':best[0],'top':best[:20]},open('/mnt/data/v3_work/ensemble_val.json','w'),indent=2)
