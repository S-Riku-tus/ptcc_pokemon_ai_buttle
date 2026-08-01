import json,re,numpy as np
from pathlib import Path
src=Path('/mnt/data/v3_work/rich_models/rich_final.json');j=json.load(open(src));learner=j['learner'];model=learner['gradient_booster']['model'];base_raw=learner['learner_model_param']['base_score'];base=[float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?',base_raw)]
z=np.load('/mnt/data/v3_work/rich_models/rich_final_probs.npz');classes=z['classes'].astype(int).tolist();cols=z['cols'].astype(int).tolist();trees=[]
for cls,t in zip(model['tree_info'],model['trees']):trees.append({'c':int(cls),'l':t['left_children'],'r':t['right_children'],'f':t['split_indices'],'v':t['split_conditions'],'d':t['default_left'],'w':t['base_weights']})
payload={'format':'v3_xgb_rich_type_v1','classes':classes,'cols':cols,'base':base,'trees':trees,'threshold':0.5}
out=Path('/mnt/data/v3_work/type_model_v3.json');json.dump(payload,open(out,'w'),separators=(',',':'));print(out.stat().st_size,len(trees),len(cols),base)
