from pathlib import Path
import numpy as np,xgboost as xgb,json,time
R=Path('/mnt/data/v3_work/rich_meta');M=Path('/mnt/data/v3_work/rich_models');cols=np.load(M/'cols_80.npy')
X=np.concatenate([np.asarray(np.load(R/'train_X.npy',mmap_mode='r')[:,cols],np.float32),np.asarray(np.load(R/'validation_X.npy',mmap_mode='r')[:,cols],np.float32)]);y=np.r_[np.load(R/'train_y.npy'),np.load(R/'validation_y.npy')]
Xt=np.asarray(np.load(R/'test_X.npy',mmap_mode='r')[:,cols],np.float32);yt=np.load(R/'test_y.npy');classes=np.unique(y);c={int(v):i for i,v in enumerate(classes)};cv=lambda a:np.array([c[int(v)] for v in a],np.int32)
m=xgb.XGBClassifier(objective='multi:softprob',num_class=len(classes),n_estimators=227,max_depth=8,learning_rate=.045,subsample=.9,colsample_bytree=.82,min_child_weight=3,reg_alpha=.25,reg_lambda=2.5,tree_method='hist',n_jobs=4,random_state=3621,eval_metric='mlogloss')
t=time.time();m.fit(X,cv(y),verbose=False);p=m.predict_proba(Xt);pred=classes[p.argmax(1)];res={'train_decisions':len(y),'features':len(cols),'iterations':227,'test_type_accuracy':float(np.mean(pred==yt)),'seconds':time.time()-t};print(json.dumps(res),flush=True);m.save_model(M/'rich_final.json');np.savez_compressed(M/'rich_final_probs.npz',test=p.astype(np.float32),classes=classes,cols=cols);json.dump(res,open(M/'rich_final_summary.json','w'),indent=2)
