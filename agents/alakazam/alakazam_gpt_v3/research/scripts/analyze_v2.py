import numpy as np, json, collections
from pathlib import Path
D=Path('/mnt/data/v1_work/dataset'); O=Path('/mnt/data/v1_work/results/v2_type_meta')
classes=np.load(O/'final/xgb_final_probs.npz')['classes'].astype(int)
for split in ['validation','test']:
 y=np.load(O/f'{split}_y.npy').astype(int)
 p=np.load(O/'final/xgb_final_probs.npz')[split]
 pred=classes[p.argmax(1)]
 c=collections.Counter(zip(y.tolist(),pred.tolist()))
 print('\n',split,'acc',np.mean(y==pred), 'n',len(y))
 print('per class')
 for t in classes:
  m=y==t
  print(t,int(m.sum()), float(np.mean(pred[m]==t)) if m.any() else None, 'topconf', sorted(((n,b) for (a,b),n in c.items() if a==t and b!=t),reverse=True)[:4])
 print('errors')
 for (a,b),n in c.most_common(30):
  if a!=b: print(n,a,'->',b)
