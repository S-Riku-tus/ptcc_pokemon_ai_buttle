import json, re, numpy as np
from pathlib import Path
src=Path('/mnt/data/v1_work/results/v2_type_meta/final/xgb_final.json')
out=Path('/mnt/data/v1_work/results/v2_type_meta/final/type_model_compact.json')
j=json.load(open(src))
learner=j['learner']; model=learner['gradient_booster']['model']
base_raw=learner['learner_model_param']['base_score']
base=[float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?',base_raw)]
classes=np.load('/mnt/data/v1_work/results/v2_type_meta/final/xgb_final_probs.npz')['classes'].astype(int).tolist()
cols=np.load('/mnt/data/v1_work/results/v2_type_meta/final/xgb_final_probs.npz')['cols'].astype(int).tolist()
trees=[]
for cls,t in zip(model['tree_info'],model['trees']):
    trees.append({
      'c':int(cls),'l':t['left_children'],'r':t['right_children'],
      'f':t['split_indices'],'v':t['split_conditions'],'d':t['default_left'],
      'w':t['base_weights'],
    })
payload={'format':'v2_xgb_type_v1','classes':classes,'cols':cols,'base':base,'trees':trees,'threshold':0.5}
json.dump(payload,open(out,'w'),separators=(',',':'))
print(out,out.stat().st_size,len(trees),len(cols),classes,base)
