import numpy as np,json
from pathlib import Path
from sklearn.feature_selection import f_classif
R=Path('/mnt/data/v3_work/rich_meta');O=Path('/mnt/data/v1_work/results/v2_type_meta');OUT=Path('/mnt/data/v3_work/rich_models');OUT.mkdir(exist_ok=True)
X=np.load(R/'train_X.npy',mmap_mode='r');y=np.load(R/'train_y.npy');oldcols=np.load(O/'xgb_probs.npz')['cols'].astype(int)
extra=np.asarray(X[:,356:],np.float32);f,p=f_classif(extra,y);f=np.nan_to_num(f,nan=0,posinf=1e9,neginf=0);rank=np.argsort(-f)
for n in [80,160,240,360,500]:
 cols=np.unique(np.r_[oldcols,356+rank[:n]]).astype(np.int32);np.save(OUT/f'cols_{n}.npy',cols);print(n,len(cols),float(f[rank[n-1]]))
np.save(OUT/'extra_fscore.npy',f)
