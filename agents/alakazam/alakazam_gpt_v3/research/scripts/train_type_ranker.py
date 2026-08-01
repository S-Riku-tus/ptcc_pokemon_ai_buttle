from pathlib import Path
import numpy as np, json, time
import lightgbm as lgb
from xgboost import XGBRanker
O=Path('/mnt/data/v1_work/results/v2_type_meta'); OUT=Path('/mnt/data/v3_work/type_ranker'); OUT.mkdir(exist_ok=True)
state_len=180; block=14; ntypes=12; global_len=8

def build(split):
    X=np.load(O/f'{split}_X.npy',mmap_mode='r'); y=np.load(O/f'{split}_y.npy')
    rows=[]; labels=[]; groups=[]; decs=[]
    # full global vector + selected type block + deltas and onehot
    for i in range(len(y)):
        legal=[]
        blocks=[]
        for t in range(ntypes):
            b=np.asarray(X[i,state_len+t*block:state_len+(t+1)*block],np.float32)
            if b[0]>0:
                legal.append(t); blocks.append(b)
        if int(y[i]) not in legal: continue
        # global top statistics from all type blocks
        tops=np.array([b[1] for b in blocks],np.float32)
        best=float(tops.max()); mean=float(tops.mean()); std=float(tops.std())
        for t,b in zip(legal,blocks):
            one=np.zeros(ntypes,np.float32);one[t]=1
            # state + all original type summaries/global + selected block + deltas + onehot
            row=np.concatenate([
                np.asarray(X[i,:state_len],np.float32),
                np.asarray(X[i,state_len:],np.float32),
                b,
                np.array([t,b[1]-best,b[1]-mean,(b[1]-mean)/max(std,1e-5),len(legal)],np.float32),
                one,
            ])
            rows.append(row); labels.append(int(t==int(y[i]))); decs.append(i)
        groups.append(len(legal))
    return np.asarray(rows,np.float32),np.asarray(labels,np.int8),groups,np.asarray(decs,np.int32),y

sets={}
for split in ['train','validation','test']:
 p=OUT/f'{split}.npz'
 if p.exists():
  z=np.load(p);sets[split]=(z['X'],z['y'],z['groups'].tolist(),z['decs'],np.load(O/f'{split}_y.npy'))
 else:
  sets[split]=build(split);X,y,g,d,_=sets[split];np.savez_compressed(p,X=X,y=y,groups=np.asarray(g),decs=d)
 print(split,sets[split][0].shape,len(sets[split][2]),flush=True)

def acc(scores, groups, labels):
 off=0;hit=0
 for g in groups:
  e=off+g; hit+=int(labels[off+int(np.argmax(scores[off:e]))]);off=e
 return hit/len(groups)
X,y,g,_,_=sets['train'];Xv,yv,gv,_,_=sets['validation'];Xt,yt,gt,_,_=sets['test']
# Try multiple lgb configs
res=[]
for name,obj,leaves,child,lr,trees in [
 ('lgb_l31','lambdarank',31,40,.03,700),
 ('lgb_l63','lambdarank',63,50,.025,800),
 ('lgb_l127','lambdarank',127,60,.02,900),
 ('lgb_x63','rank_xendcg',63,50,.025,800),
]:
 print('fit',name,flush=True);t=time.time()
 m=lgb.LGBMRanker(objective=obj,metric='ndcg',n_estimators=trees,learning_rate=lr,num_leaves=leaves,min_child_samples=child,max_depth=-1,subsample=.9,subsample_freq=1,colsample_bytree=.8,reg_alpha=.5,reg_lambda=5,random_state=3601,n_jobs=8,verbosity=-1)
 m.fit(X,y,group=g,eval_set=[(Xv,yv)],eval_group=[gv],callbacks=[lgb.early_stopping(60,verbose=False)])
 pv=m.predict(Xv,num_iteration=m.best_iteration_);pt=m.predict(Xt,num_iteration=m.best_iteration_)
 r={'name':name,'iter':int(m.best_iteration_),'validation':acc(pv,gv,yv),'test':acc(pt,gt,yt),'seconds':time.time()-t};print(r,flush=True)
 m.booster_.save_model(str(OUT/f'{name}.txt'),num_iteration=m.best_iteration_);np.savez_compressed(OUT/f'{name}_scores.npz',validation=pv.astype(np.float32),test=pt.astype(np.float32));res.append(r)
json.dump(res,open(OUT/'summary.json','w'),indent=2)
