import sys,json,numpy as np,importlib.util
from pathlib import Path
A=Path('/mnt/data/v3_work/alakazam_gpt_v3');sys.path.insert(0,str(A))
spec=importlib.util.spec_from_file_location('stubtest',A/'test_v11_runtime_logic.py');mod=importlib.util.module_from_spec(spec);sys.modules['stubtest']=mod;spec.loader.exec_module(mod);mod.install_cg_stub()
from policy_runtime import HybridRanker
D=Path('/mnt/data/v1_work/dataset');R=Path('/mnt/data/v1_work/results');M=Path('/mnt/data/v3_work/rich_meta')
names=json.load(open(D/'feature_names.json'));X=np.load(D/'validation_features.npy',mmap_mode='r');g=np.load(D/'validation_groups.npy');s=np.load(R/'v1_baseline_scores.npz')['validation'];target=np.load(M/'validation_X.npy',mmap_mode='r')
r=HybridRanker(attacks={});off=0;mx=0.;bad=0
for di,z0 in enumerate(g[:300]):
 z=int(z0);e=off+z;features=[{n:float(v) for n,v in zip(names,np.asarray(row,np.float32))} for row in X[off:e]];row=np.asarray(r._type_meta_row(features,list(range(z)),s[off:e].astype(float).tolist()),np.float32);d=float(np.max(np.abs(row-target[di])));mx=max(mx,d);bad+=int(d>1e-5);off=e
print({'decisions':300,'max_abs_error':mx,'bad':bad,'row_len':len(row),'target_len':target.shape[1]});assert bad==0
