from pathlib import Path
import numpy as np, torch, json, time
from torch import nn
O=Path('/mnt/data/v1_work/results/v2_type_meta');D=Path('/mnt/data/v1_work/dataset');OUT=Path('/mnt/data/v3_work/multitask');OUT.mkdir(exist_ok=True);torch.set_num_threads(6)
cols=np.load(O/'xgb_probs.npz')['cols']; classes=np.load(O/'final/xgb_final_probs.npz')['classes'].astype(int); cmap={int(c):i for i,c in enumerate(classes)}

def labels(split):
 meta=np.load(D/f'{split}_decision_meta.npy'); y=meta[:,3].astype(int); n=len(y)
 nxt=np.full(n,len(classes),np.int64);term=np.empty(n,np.int64);rem=np.empty(n,np.int64);atk=np.zeros(n,np.float32);mask=np.zeros((n,len(classes)),np.float32)
 # groups by contiguous episode+turn
 st=0
 while st<n:
  ep,turn=meta[st,0],meta[st,2];e=st+1
  while e<n and meta[e,0]==ep and meta[e,2]==turn:e+=1
  seq=y[st:e]
  terminal=seq[-1]
  terminal_class=cmap.get(int(terminal),len(classes))
  for i in range(st,e):
   loc=i-st; future=seq[loc+1:]
   if len(future): nxt[i]=cmap.get(int(future[0]),len(classes))
   term[i]=terminal_class; rem[i]=min(5,e-i-1)
   atk[i]=float(np.any(seq[loc:] == 1))
   for t in np.unique(seq[loc+1:]):
    if int(t) in cmap:mask[i,cmap[int(t)]]=1
  st=e
 return np.array([cmap[int(v)] for v in y],np.int64),nxt,term,rem,atk,mask
sets={}
for split in ['train','validation','test']:
 X=np.asarray(np.load(O/f'{split}_X.npy',mmap_mode='r')[:,cols],np.float32);lab=labels(split);sets[split]=(X,*lab)
mean=sets['train'][0].mean(0);std=sets['train'][0].std(0);std[std<1e-5]=1
for s in sets:sets[s]=(np.clip((sets[s][0]-mean)/std,-10,10),*sets[s][1:])
class Net(nn.Module):
 def __init__(self,seed,drop=.12):
  super().__init__();torch.manual_seed(seed);d=len(cols)
  self.trunk=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.LayerNorm(512),nn.Dropout(drop),nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(drop),nn.Linear(384,256),nn.GELU())
  self.main=nn.Linear(256,len(classes));self.nxt=nn.Linear(256,len(classes)+1);self.term=nn.Linear(256,len(classes)+1);self.rem=nn.Linear(256,6);self.atk=nn.Linear(256,1);self.mask=nn.Linear(256,len(classes))
 def forward(self,x):
  h=self.trunk(x);return self.main(h),self.nxt(h),self.term(h),self.rem(h),self.atk(h).squeeze(1),self.mask(h)
def batches(split,bs,shuffle=False):
 X,y,nx,te,re,at,ma=sets[split];ids=np.arange(len(y));
 if shuffle:np.random.shuffle(ids)
 for st in range(0,len(ids),bs):
  q=ids[st:st+bs]; yield tuple(torch.from_numpy(a[q]) for a in (X,y.astype(np.int64),nx,te,re,at,ma))
def evalm(m,split):
 m.eval();pp=[]
 with torch.no_grad():
  for b in batches(split,2048):pp.append(m(b[0])[0].softmax(1).numpy())
 p=np.concatenate(pp);pred=classes[p.argmax(1)];true=classes[sets[split][1]];return float(np.mean(pred==true)),p
configs=[(3601,.2,.2,.15,.1,.1),(3607,.35,.25,.2,.15,.15),(3613,.15,.35,.1,.1,.2)]
res=[]
for seed,wn,wt,wr,wa,wm in configs:
 m=Net(seed);opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=5e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.55,patience=2);best=(-1,None,0);wait=0;t0=time.time()
 for ep in range(1,31):
  m.train();losses=[]
  for x,y,nx,te,re,at,ma in batches('train',1024,True):
   opt.zero_grad();a,b,c,d,e,f=m(x);loss=nn.functional.cross_entropy(a,y,label_smoothing=.01)+wn*nn.functional.cross_entropy(b,nx)+wt*nn.functional.cross_entropy(c,te)+wr*nn.functional.cross_entropy(d,re)+wa*nn.functional.binary_cross_entropy_with_logits(e,at)+wm*nn.functional.binary_cross_entropy_with_logits(f,ma);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
  av,_=evalm(m,'validation');sched.step(av);print(seed,ep,round(np.mean(losses),4),av,flush=True)
  if av>best[0]+1e-5:best=(av,{k:v.detach().clone() for k,v in m.state_dict().items()},ep);wait=0
  else:wait+=1
  if wait>=6:break
 m.load_state_dict(best[1]);av,pv=evalm(m,'validation');r={'seed':seed,'epoch':best[2],'validation':av,'seconds':time.time()-t0,'weights':[wn,wt,wr,wa,wm]};print('RESULT',r,flush=True);torch.save({'state':m.state_dict(),'mean':mean,'std':std,'cols':cols,'classes':classes,'config':r},OUT/f'model_{seed}.pt');np.savez_compressed(OUT/f'probs_{seed}.npz',validation=pv.astype(np.float32),classes=classes);res.append(r)
json.dump(res,open(OUT/'summary.json','w'),indent=2)
