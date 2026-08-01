import sys,importlib.util,numpy as np
from pathlib import Path
A=Path('/mnt/data/v3_work/alakazam_gpt_v3');sys.path.insert(0,str(A));spec=importlib.util.spec_from_file_location('stubtest2',A/'test_v11_runtime_logic.py');mod=importlib.util.module_from_spec(spec);sys.modules['stubtest2']=mod;spec.loader.exec_module(mod);mod.install_cg_stub();from policy_runtime import HybridRanker
X=np.load('/mnt/data/v3_work/rich_meta/test_X.npy',mmap_mode='r');z=np.load('/mnt/data/v3_work/rich_models/rich_final_probs.npz');p=z['test'];classes=z['classes'].astype(int);r=HybridRanker(attacks={});bad=0;mx=0
for i in range(500):
 t,c=r._predict_action_type(X[i].astype(float).tolist());j=int(np.argmax(p[i]));bad+=int(t!=classes[j]);mx=max(mx,abs(c-float(p[i,j])))
print({'rows':500,'argmax_mismatch':bad,'confidence_max_abs_error':mx});assert bad==0 and mx<2e-5
