from pathlib import Path
import numpy as np,torch,json
from torch import nn
O=Path('/mnt/data/v1_work/results/v2_type_meta');D=Path('/mnt/data/v1_work/dataset');classes=np.load(O/'final/xgb_final_probs.npz')['classes'].astype(int);cmap={int(v):i for i,v in enumerate(classes)}
val=[np.load(O/'xgb_probs.npz')['validation'],np.load('/mnt/data/v3_work/rich_models/rich80.npz')['validation'],np.load('/mnt/data/v3_work/aug_xgb/w0p03.npz')['validation'],np.load('/mnt/data/v3_work/cat_type/d6_val.npz')['p']]
test=[np.load(O/'final/xgb_final_probs.npz')['test'],np.load('/mnt/data/v3_work/rich_models/rich80_test.npz')['test'],np.load('/mnt/data/v3_work/aug_xgb/w0p03_test.npz')['test'],np.load('/mnt/data/v3_work/cat_type/d6_test.npz')['test']]
V=np.stack(val,1).astype(np.float32);T=np.stack(test,1).astype(np.float32);yv_raw=np.load(O/'validation_y.npy');yt_raw=np.load(O/'test_y.npy');yv=np.array([cmap[int(x)] for x in yv_raw]);yt=np.array([cmap[int(x)] for x in yt_raw]);meta=np.load(D/'validation_decision_meta.npy');eps=meta[:,0];ue=np.unique(eps);foldmap={int(e):i%4 for i,e in enumerate(ue)};fold=np.array([foldmap[int(e)] for e in eps])
class Stack(nn.Module):
 def __init__(self):super().__init__();self.w=nn.Parameter(torch.ones(V.shape[1],V.shape[2])/V.shape[1]);self.b=nn.Parameter(torch.zeros(V.shape[2]))
 def forward(self,p):return (torch.log(p.clamp_min(1e-7))*self.w.unsqueeze(0)).sum(1)+self.b

def fit(train_idx,lam,steps=600):
 m=Stack();opt=torch.optim.Adam(m.parameters(),lr=.025);x=torch.from_numpy(V[train_idx]);y=torch.from_numpy(yv[train_idx])
 for _ in range(steps):
  opt.zero_grad();z=m(x);loss=nn.functional.cross_entropy(z,y)+lam*((m.w-.25)**2).mean()+lam*.1*(m.b**2).mean();loss.backward();opt.step()
 return m
lams=[0,.001,.003,.01,.03,.1,.3,1.]
cv=[]
for lam in lams:
 acc=[];ll=[]
 for f in range(4):
  tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f);m=fit(tr,lam,450)
  with torch.no_grad():z=m(torch.from_numpy(V[va]));p=z.softmax(1).numpy();acc.append(np.mean(p.argmax(1)==yv[va]));ll.append(float(nn.functional.cross_entropy(z,torch.from_numpy(yv[va]))))
 cv.append({'lambda':lam,'accuracy':float(np.mean(acc)),'logloss':float(np.mean(ll))});print(cv[-1],flush=True)
best=min(cv,key=lambda r:(-r['accuracy'],r['logloss']));m=fit(np.arange(len(yv)),best['lambda'],900)
with torch.no_grad():pv=m(torch.from_numpy(V)).softmax(1).numpy();pt=m(torch.from_numpy(T)).softmax(1).numpy()
res={'cv':cv,'selected_lambda':best['lambda'],'validation_accuracy':float(np.mean(pv.argmax(1)==yv)),'test_accuracy':float(np.mean(pt.argmax(1)==yt)),'weights':m.w.detach().numpy().tolist(),'bias':m.b.detach().numpy().tolist()};print(json.dumps({k:v for k,v in res.items() if k not in ['weights','bias','cv']},indent=2),flush=True);Path('/mnt/data/v3_work/stack').mkdir(exist_ok=True);json.dump(res,open('/mnt/data/v3_work/stack/summary.json','w'),indent=2);np.savez_compressed('/mnt/data/v3_work/stack/probs.npz',validation=pv.astype(np.float32),test=pt.astype(np.float32),classes=classes)
