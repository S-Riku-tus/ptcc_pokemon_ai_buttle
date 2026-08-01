from pathlib import Path
import numpy as np,torch,json,time
from torch import nn
P=Path('/mnt/data/v1_work/results/v2_type_deepset');O=Path('/mnt/data/v3_work/typeset_fast');O.mkdir(exist_ok=True);torch.set_num_threads(6)
classes=np.array([0,1,2,3,4,5,6,7,9,10,11],np.int64);cmap={int(v):i for i,v in enumerate(classes)};T=len(classes)
sets={}
for s in ['train','validation','test']:
 z=np.load(P/f'{s}.npz');sets[s]=(torch.from_numpy(z['cont']),torch.from_numpy(z['cats']),torch.from_numpy(z['mask']),torch.from_numpy(np.array([cmap[int(v)] for v in z['y']],np.int64)))
c,ca,m,y=sets['train'];flat=c[m][:250000].float();mean=flat.mean(0);std=flat.std(0);std[std<1e-4]=1
class TypeSet(nn.Module):
 def __init__(self,seed,h=80,drop=.08):
  super().__init__();torch.manual_seed(seed);sizes=[int(ca[:,:,j].max())+1 for j in range(ca.shape[2])]
  self.emb=nn.ModuleList([nn.Embedding(sz,6,padding_idx=0) for sz in sizes]);inp=c.shape[2]+6*len(sizes)
  self.enc=nn.Sequential(nn.Linear(inp,160),nn.GELU(),nn.LayerNorm(160),nn.Dropout(drop),nn.Linear(160,h),nn.GELU())
  self.type_emb=nn.Embedding(T,12);self.score=nn.Sequential(nn.Linear(h*4+16,160),nn.GELU(),nn.Dropout(drop),nn.Linear(160,1))
 def forward(self,cont,cats,mask):
  x=((cont.float()-mean)/std).clamp(-10,10);ee=torch.cat([e(cats[:,:,j].long()) for j,e in enumerate(self.emb)],-1);h=self.enc(torch.cat([x,ee],-1));B,N,H=h.shape
  M=mask.unsqueeze(-1);gmean=(h*M).sum(1)/M.sum(1).clamp_min(1);neg=torch.full_like(h,-1e4);gmax=torch.where(M,h,neg).max(1).values
  typ=(cats[:,:,0].long()-1).clamp(min=0,max=T-1);valid=(cats[:,:,0]>0)&mask;ind=typ.unsqueeze(-1).expand(-1,-1,H)
  tsum=torch.zeros(B,T,H,device=h.device);tsum.scatter_add_(1,ind,h*valid.unsqueeze(-1));cnt=torch.zeros(B,T,device=h.device);cnt.scatter_add_(1,typ,valid.float());tmean=tsum/cnt.unsqueeze(-1).clamp_min(1)
  tmax=torch.full((B,T,H),-1e4,device=h.device);tmax.scatter_reduce_(1,ind,torch.where(valid.unsqueeze(-1),h,torch.full_like(h,-1e4)),reduce='amax',include_self=True)
  # normalized max baseline score per type
  bs=cont[:,:,-1].float();bs=(bs-bs.masked_fill(~mask,0).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1))/bs.masked_fill(~mask,0).std(1,keepdim=True).clamp_min(1e-4)
  bmax=torch.full((B,T),-20.,device=h.device);bmax.scatter_reduce_(1,typ,torch.where(valid,bs,torch.full_like(bs,-20.)),reduce='amax',include_self=True)
  gb=torch.arange(T,device=h.device).unsqueeze(0).expand(B,-1);te=self.type_emb(gb);gm=gmean.unsqueeze(1).expand(-1,T,-1);gx=gmax.unsqueeze(1).expand(-1,T,-1);stats=torch.stack([cnt/32,bmax,(bmax-bmax.max(1,keepdim=True).values),torch.log1p(cnt)],-1)
  feat=torch.cat([tmean,tmax,gm,gx,te,stats],-1);log=self.score(feat).squeeze(-1);return torch.where(cnt>0,log,torch.full_like(log,-1e4))
def batches(split,bs,shuffle=False):
 c,ca,m,y=sets[split];ids=torch.randperm(len(y)) if shuffle else torch.arange(len(y))
 for st in range(0,len(y),bs):q=ids[st:st+bs];yield c[q],ca[q],m[q],y[q]
def evalm(model,split):
 model.eval();pp=[]
 with torch.no_grad():
  for b in batches(split,2048):pp.append(model(*b[:3]).softmax(1).cpu())
 p=torch.cat(pp).numpy();y=sets[split][3].numpy();return float(np.mean(p.argmax(1)==y)),p
res=[]
for seed,h,drop in [(3621,64,.06),(3627,96,.1)]:
 model=TypeSet(seed,h,drop);opt=torch.optim.AdamW(model.parameters(),lr=1.8e-3,weight_decay=4e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2);best=(-1,None,0);wait=0;t0=time.time()
 for ep in range(1,26):
  model.train();ls=[]
  for c0,ca0,m0,y0 in batches('train',1536,True):
   opt.zero_grad();log=model(c0,ca0,m0);loss=nn.functional.cross_entropy(log,y0,label_smoothing=.01);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ls.append(float(loss.detach()))
  av,_=evalm(model,'validation');sched.step(av);print(seed,ep,round(np.mean(ls),4),av,flush=True)
  if av>best[0]+1e-5:best=(av,{k:v.detach().clone() for k,v in model.state_dict().items()},ep);wait=0
  else:wait+=1
  if wait>=6:break
 model.load_state_dict(best[1]);av,pv=evalm(model,'validation');r={'seed':seed,'hidden':h,'dropout':drop,'epoch':best[2],'validation':av,'seconds':time.time()-t0};print('RESULT',r,flush=True);torch.save({'state':model.state_dict(),'mean':mean,'std':std,'classes':classes,'config':r},O/f'model_{seed}.pt');np.savez_compressed(O/f'probs_{seed}.npz',validation=pv.astype(np.float32),classes=classes);res.append(r)
json.dump(res,open(O/'summary.json','w'),indent=2)
