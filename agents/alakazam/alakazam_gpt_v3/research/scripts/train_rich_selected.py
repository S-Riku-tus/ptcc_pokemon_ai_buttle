import sys,time,json
from pathlib import Path
import numpy as np,xgboost as xgb
n=int(sys.argv[1]);name=f'rich{n}';R=Path('/mnt/data/v3_work/rich_meta');M=Path('/mnt/data/v3_work/rich_models');cols=np.load(M/f'cols_{n}.npy')
X=np.asarray(np.load(R/'train_X.npy',mmap_mode='r')[:,cols],np.float32);y=np.load(R/'train_y.npy');Xv=np.asarray(np.load(R/'validation_X.npy',mmap_mode='r')[:,cols],np.float32);yv=np.load(R/'validation_y.npy')
classes=np.unique(y);c={int(v):i for i,v in enumerate(classes)};cv=lambda a:np.array([c[int(v)] for v in a],np.int32)
m=xgb.XGBClassifier(objective='multi:softprob',num_class=len(classes),n_estimators=650,max_depth=8,learning_rate=.045,subsample=.9,colsample_bytree=.82,min_child_weight=3,reg_alpha=.25,reg_lambda=2.5,tree_method='hist',n_jobs=4,random_state=3621,eval_metric='mlogloss',early_stopping_rounds=50)
t=time.time();m.fit(X,cv(y),eval_set=[(Xv,cv(yv))],verbose=False);p=m.predict_proba(Xv);pred=classes[p.argmax(1)];r={'n_extra':n,'features':len(cols),'iter':int(m.best_iteration)+1,'validation':float(np.mean(pred==yv)),'sec':time.time()-t};print(json.dumps(r),flush=True);m.save_model(M/f'{name}.json');np.savez_compressed(M/f'{name}.npz',validation=p.astype(np.float32),classes=classes,cols=cols);json.dump(r,open(M/f'{name}_summary.json','w'),indent=2)
