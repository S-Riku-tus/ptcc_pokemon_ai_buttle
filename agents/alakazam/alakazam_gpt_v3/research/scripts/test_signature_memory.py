from pathlib import Path
import numpy as np, xgboost as xgb, collections, json, math
O=Path('/mnt/data/v1_work/results/v2_type_meta')
X=np.load(O/'train_X.npy',mmap_mode='r'); y=np.load(O/'train_y.npy').astype(int)
Xv=np.load(O/'validation_X.npy',mmap_mode='r'); yv=np.load(O/'validation_y.npy').astype(int)
Xt=np.load(O/'test_X.npy',mmap_mode='r'); yt=np.load(O/'test_y.npy').astype(int)
z=np.load(O/'xgb_probs.npz');cols=z['cols'].astype(int);pv=z['validation'];
zf=np.load(O/'final/xgb_final_probs.npz');pt=zf['test'];classes=zf['classes'].astype(int)
# final xgb feature importance indexed by f0.. corresponding selected cols
bo=xgb.Booster();bo.load_model(O/'xgb.json');imp=bo.get_score(importance_type='gain')
rank=sorted(range(len(cols)),key=lambda j:-imp.get(f'f{j}',0.0))
# transformed selected matrices
A=np.asarray(X[:,cols],np.float32);Av=np.asarray(Xv[:,cols],np.float32);At=np.asarray(Xt[:,cols],np.float32)
# robust scale/cardinality
sample=A[:min(20000,len(A))]; std=np.std(sample,axis=0); uniq=[len(np.unique(sample[:,j])) for j in range(A.shape[1])]
base_pred=classes[pv.argmax(1)];test_base=classes[pt.argmax(1)]
print('base',np.mean(base_pred==yv),np.mean(test_base==yt))

def codes(M,sel,q):
 out=np.empty((len(M),len(sel)),np.int32)
 for k,j in enumerate(sel):
  if uniq[j] <= 80 and np.max(np.abs(sample[:,j]-np.round(sample[:,j])))<1e-4:
   out[:,k]=np.round(M[:,j]).astype(np.int32)
  else:
   w=max(float(std[j])*q,1e-3)
   out[:,k]=np.round(M[:,j]/w).astype(np.int32)
 return out

def keyrows(C):
 # bytes key
 return [row.tobytes() for row in np.ascontiguousarray(C)]
results=[]
for n in [6,8,10,12,16,20,24,32]:
 sel=rank[:n]
 for q in [.1,.2,.35,.5,.75,1.0]:
  tr=keyrows(codes(A,sel,q)); va=keyrows(codes(Av,sel,q))
  mem={}
  for k,t in zip(tr,y):
   c=mem.get(k)
   if c is None:c=collections.Counter();mem[k]=c
   c[int(t)]+=1
  stats={k:(c.most_common(1)[0][0],c.most_common(1)[0][1],sum(c.values())) for k,c in mem.items()}
  for minsup in [2,3,5,8,12]:
   for conf in [.7,.8,.9,1.0]:
    pred=base_pred.copy();cov=0
    for i,k in enumerate(va):
     z0=stats.get(k)
     if z0 and z0[2]>=minsup and z0[1]/z0[2]>=conf:
      pred[i]=z0[0];cov+=1
    acc=float(np.mean(pred==yv));results.append((acc,cov/len(yv),n,q,minsup,conf,sel))
results.sort(reverse=True,key=lambda r:(r[0],r[1]));
for r in results[:20]:print('val',r[:6])
best=results[0];_,_,n,q,minsup,conf,sel=best
tr=keyrows(codes(A,sel,q));te=keyrows(codes(At,sel,q));mem={}
for k,t in zip(tr,y):mem.setdefault(k,collections.Counter())[int(t)]+=1
stats={k:(c.most_common(1)[0][0],c.most_common(1)[0][1],sum(c.values())) for k,c in mem.items()};pred=test_base.copy();cov=0
for i,k in enumerate(te):
 z0=stats.get(k)
 if z0 and z0[2]>=minsup and z0[1]/z0[2]>=conf:pred[i]=z0[0];cov+=1
print('BEST',best[:6],'test',float(np.mean(pred==yt)),'cov',cov/len(yt),'cols',cols[np.array(sel)].tolist())
json.dump({'validation':best[0],'coverage':best[1],'n':n,'q':q,'min_support':minsup,'confidence':conf,'test':float(np.mean(pred==yt)),'test_coverage':cov/len(yt),'meta_columns':cols[np.array(sel)].tolist()},open('/mnt/data/v3_work/signature_summary.json','w'),indent=2)
