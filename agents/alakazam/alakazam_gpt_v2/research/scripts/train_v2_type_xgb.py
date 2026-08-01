import numpy as np, xgboost as xgb, json, time
from pathlib import Path
O=Path('/mnt/data/v1_work/results/v2_type_meta')
X=np.load(O/'train_X.npy',mmap_mode='r');y=np.load(O/'train_y.npy');Xv=np.load(O/'validation_X.npy',mmap_mode='r');yv=np.load(O/'validation_y.npy');Xt=np.load(O/'test_X.npy',mmap_mode='r');yt=np.load(O/'test_y.npy')
# select 220 columns via variance + direct meta tail; avoid huge expensive full matrix
# retain last 176 type summaries and top variance state cols
var=np.var(np.asarray(X[:10000],np.float32),axis=0)
state=np.argsort(-var[:180])[:100]
cols=np.unique(np.r_[state,np.arange(max(0,X.shape[1]-176),X.shape[1])]).astype(np.int32)
X=np.asarray(X[:,cols],np.float32);Xv=np.asarray(Xv[:,cols],np.float32);Xt=np.asarray(Xt[:,cols],np.float32)
print('shape',X.shape,flush=True)
classes=np.unique(y); cmap={int(c):i for i,c in enumerate(classes)}; y2=np.array([cmap[int(v)] for v in y],dtype=np.int32); yv2=np.array([cmap.get(int(v),-1) for v in yv],dtype=np.int32); yt2=np.array([cmap.get(int(v),-1) for v in yt],dtype=np.int32)
model=xgb.XGBClassifier(objective='multi:softprob',num_class=len(classes),n_estimators=450,max_depth=8,learning_rate=.05,subsample=.9,colsample_bytree=.85,min_child_weight=3,reg_alpha=.2,reg_lambda=2,tree_method='hist',n_jobs=5,random_state=3517,eval_metric='mlogloss',early_stopping_rounds=40)
model.fit(X,y2,eval_set=[(Xv,yv2)],verbose=25)
pv=model.predict_proba(Xv);pt=model.predict_proba(Xt)
predv=classes[pv.argmax(1)];predt=classes[pt.argmax(1)]
res={'best_iteration':int(model.best_iteration),'validation':float((predv==yv).mean()),'test':float((predt==yt).mean()),'features':len(cols)}
print(res,flush=True)
model.save_model(O/'xgb.json');np.savez_compressed(O/'xgb_probs.npz',validation=pv.astype(np.float32),test=pt.astype(np.float32),cols=cols)
json.dump(res,open(O/'xgb_summary.json','w'),indent=2)
