from pathlib import Path
import json,numpy as np
D=Path('/mnt/data/v1_work/dataset');B=Path('/mnt/data/v1_work/results')
classes=np.array(json.load(open('/mnt/data/v3_work/alakazam_gpt_v3/type_model.json'))['classes'],int)
pb=np.load('/mnt/data/v3_work/rich_models/rich_final_probs.npz')['test'];pr=np.load('/mnt/data/v4_work/experts/rmy.npz')['test'];p=.85*pb+.15*pr;th=.45
names=json.load(open(D/'feature_names.json'));ai=names.index('action_type_id');X=np.load(D/'test_features.npy',mmap_mode='r');lab=np.load(D/'test_labels.npy',mmap_mode='r');g=np.load(D/'test_groups.npy');bs=np.load(B/'v1_baseline_scores.npz')['test'];yt=np.load('/mnt/data/v3_work/rich_meta/test_y.npy')
def run(P,threshold):
 off=0;hit=0
 for di,g0 in enumerate(g):
  n=int(g0);e=off+n;types=X[off:e,ai].astype(int);sc=np.asarray(bs[off:e],float)
  if P[di].max()>=threshold:
   t=classes[P[di].argmax()]
   if np.any(types==t):sc=np.where(types==t,sc,-1e30)
  hit+=int(lab[off+int(np.argmax(sc))]);off=e
 return hit
base=run(pb,.5);new=run(p,th)
res={'base_correct':base,'base_top1':base/len(g),'v4_correct':new,'v4_top1':new/len(g),'gain':new-base,'blend':{'v3':.85,'rmy_expert':.15},'confidence_threshold':th,'base_type_accuracy':float(np.mean(classes[pb.argmax(1)]==yt)),'v4_type_accuracy':float(np.mean(classes[p.argmax(1)]==yt)),'rmy_expert_type_accuracy':float(np.mean(classes[pr.argmax(1)]==yt))}
print(json.dumps(res,indent=2));json.dump(res,open('/mnt/data/v4_work/v4_final_test.json','w'),indent=2)
