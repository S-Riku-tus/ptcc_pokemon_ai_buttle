from pathlib import Path
import json,numpy as np
D=Path('/mnt/data/v1_work/dataset');B=Path('/mnt/data/v1_work/results')
classes=np.array(json.load(open('/mnt/data/v3_work/alakazam_gpt_v3/type_model.json'))['classes'],int)
pb=np.load('/mnt/data/v3_work/rich_models/rich80.npz')['validation']
pe={n:np.load(f'/mnt/data/v4_work/experts/{n}.npz')['validation'] for n in ['yushin','majkel','rmy']}
yt=np.load('/mnt/data/v3_work/rich_meta/validation_y.npy')
names=json.load(open(D/'feature_names.json'));ai=names.index('action_type_id');X=np.load(D/'validation_features.npy',mmap_mode='r');lab=np.load(D/'validation_labels.npy',mmap_mode='r');g=np.load(D/'validation_groups.npy');bs=np.load(B/'v1_baseline_scores.npz')['validation']
# precompute candidate correctness by action type and ungated base
H=np.zeros((len(g),12),np.int8);HB=np.zeros(len(g),np.int8);off=0
for di,g0 in enumerate(g):
 n=int(g0);e=off+n;types=X[off:e,ai].astype(int);sc=np.asarray(bs[off:e],float);yy=lab[off:e];HB[di]=yy[int(np.argmax(sc))]
 for t in range(12):
  ids=np.flatnonzero(types==t)
  if len(ids):H[di,t]=yy[ids[int(np.argmax(sc[ids]))]]
 off=e
def policy(p,th=.5):
 pred=classes[p.argmax(1)];gate=p.max(1)>=th;return int(np.sum(np.where(gate,H[np.arange(len(g)),pred],HB)))
res=[]
for name,p in pe.items():
 for a in np.arange(.01,.51,.01):
  q=(1-a)*pb+a*p
  for th in [.35,.4,.45,.5,.55,.6]:res.append((policy(q,th),f'linear_{name}',float(a),th,float(np.mean(classes[q.argmax(1)]==yt))))
 for a in np.arange(.01,.31,.01):
  q=np.exp((1-a)*np.log(np.maximum(pb,1e-9))+a*np.log(np.maximum(p,1e-9)));q/=q.sum(1,keepdims=True)
  for th in [.35,.4,.45,.5,.55,.6]:res.append((policy(q,th),f'log_{name}',float(a),th,float(np.mean(classes[q.argmax(1)]==yt))))
for ay in [.01,.02,.05,.1,.15,.2]:
 for am in [0,.01,.02,.05]:
  for ar in [0,.01,.02,.05]:
   s=ay+am+ar
   q=(1-s)*pb+ay*pe['yushin']+am*pe['majkel']+ar*pe['rmy']
   for th in [.35,.4,.45,.5,.55,.6]:res.append((policy(q,th),'mix',ay,am,ar,th,float(np.mean(classes[q.argmax(1)]==yt))))
for name,p in pe.items():
 bc=pb.max(1);ec=p.max(1);bt=pb.argmax(1);et=p.argmax(1)
 for bmax in [.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
  for emin in [.5,.6,.7,.8,.9]:
   for margin in [0,.05,.1,.2]:
    mask=(bc<=bmax)&(ec>=emin)&(ec-bc>=margin)&(bt!=et);q=pb.copy();q[mask]=p[mask]
    res.append((policy(q,.5),f'override_{name}',bmax,emin,margin,float(mask.mean()),float(np.mean(classes[q.argmax(1)]==yt))))
res.sort(reverse=True,key=lambda x:x[0])
print('base',policy(pb,.5),np.mean(classes[pb.argmax(1)]==yt))
for r in res[:50]:print(r)
json.dump([list(r) for r in res[:200]],open('/mnt/data/v4_work/expert_blend_validation.json','w'),indent=2)
np.savez_compressed('/mnt/data/v4_work/validation_policy_lookup.npz',H=H,HB=HB)
