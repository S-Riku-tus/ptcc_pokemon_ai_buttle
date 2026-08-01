import numpy as np, xgboost as xgb, json, time
from pathlib import Path
O=Path('/mnt/data/v1_work/results/v2_type_meta'); F=O/'final';F.mkdir(exist_ok=True)
X1=np.load(O/'train_X.npy',mmap_mode='r');y1=np.load(O/'train_y.npy');X2=np.load(O/'validation_X.npy',mmap_mode='r');y2=np.load(O/'validation_y.npy');Xt=np.load(O/'test_X.npy',mmap_mode='r');yt=np.load(O/'test_y.npy')
cols=np.load(O/'xgb_probs.npz')['cols']; X=np.concatenate([np.asarray(X1[:,cols],np.float32),np.asarray(X2[:,cols],np.float32)]); y=np.r_[y1,y2]; Xt=np.asarray(Xt[:,cols],np.float32)
classes=np.unique(y); cmap={int(c):i for i,c in enumerate(classes)}; y2c=np.array([cmap[int(v)] for v in y],dtype=np.int32)
model=xgb.XGBClassifier(objective='multi:softprob',num_class=len(classes),n_estimators=220,max_depth=8,learning_rate=.05,subsample=.9,colsample_bytree=.85,min_child_weight=3,reg_alpha=.2,reg_lambda=2,tree_method='hist',n_jobs=7,random_state=3517,eval_metric='mlogloss')
t=time.time(); model.fit(X,y2c,verbose=False); pt=model.predict_proba(Xt); pred=classes[pt.argmax(1)];
rec={'train_decisions':len(y),'iterations':220,'test_type_accuracy':float((pred==yt).mean()),'seconds':time.time()-t};print(json.dumps(rec),flush=True)
model.save_model(F/'xgb_final.json');np.savez_compressed(F/'xgb_final_probs.npz',test=pt.astype(np.float32),cols=cols,classes=classes);json.dump(rec,open(F/'summary.json','w'),indent=2)
