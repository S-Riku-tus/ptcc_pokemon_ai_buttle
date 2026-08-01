import json,re,numpy as np
from pathlib import Path
src=Path('/mnt/data/v37_work/experts/rmy.json');j=json.load(open(src));learner=j['learner'];model=learner['gradient_booster']['model'];base_raw=learner['learner_model_param']['base_score'];base=[float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?',base_raw)]
classes=np.load('/mnt/data/v37_work/experts/rmy.npz')['classes'].astype(int).tolist();cols=json.load(open('/mnt/data/v36_work/alakazam_ml_v36/type_model.json'))['cols'];trees=[]
for cls,t in zip(model['tree_info'],model['trees']):trees.append({'c':int(cls),'l':t['left_children'],'r':t['right_children'],'f':t['split_indices'],'v':t['split_conditions'],'d':t['default_left'],'w':t['base_weights']})
payload={'format':'v37_xgb_rmy_type_expert_v1','classes':classes,'cols':cols,'base':base,'trees':trees,'weight':0.15}
out=Path('/mnt/data/v37_work/alakazam_ml_v37/rmy_type_model.json');json.dump(payload,open(out,'w'),separators=(',',':'));print(out.stat().st_size,len(trees),len(cols),base)
