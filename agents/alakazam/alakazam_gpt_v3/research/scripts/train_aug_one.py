import sys,gc,time,json
from pathlib import Path
import numpy as np,xgboost as xgb
w=float(sys.argv[1]);O=Path('/mnt/data/v1_work/results/v2_type_meta');OUT=Path('/mnt/data/v3_work/aug_xgb');OUT.mkdir(exist_ok=True)
cols=np.load(O/'xgb_probs.npz')['cols'];X=np.asarray(np.load(O/'train_X.npy',mmap_mode='r')[:,cols],np.float32);y=np.load(O/'train_y.npy');Xm=np.asarray(np.load('/mnt/data/v3_work/majkel_meta_X.npy',mmap_mode='r')[:,cols],np.float32);ym=np.load('/mnt/data/v3_work/majkel_meta_y.npy');Xv=np.asarray(np.load(O/'validation_X.npy',mmap_mode='r')[:,cols],np.float32);yv=np.load(O/'validation_y.npy')
classes=np.unique(y);c={int(v):i for i,v in enumerate(classes)};cv=lambda a:np.array([c[int(v)] for v in a],np.int32)
XX=np.r_[X,Xm];yy=np.r_[cv(y),cv(ym)];sw=np.r_[np.ones(len(y),np.float32),np.full(len(ym),w,np.float32)]
m=xgb.XGBClassifier(objective='multi:softprob',num_class=len(classes),n_estimators=500,max_depth=8,learning_rate=.05,subsample=.9,colsample_bytree=.85,min_child_weight=3,reg_alpha=.2,reg_lambda=2,tree_method='hist',n_jobs=8,random_state=3611,eval_metric='mlogloss',early_stopping_rounds=45)
t=time.time();m.fit(XX,yy,sample_weight=sw,eval_set=[(Xv,cv(yv))],verbose=False);p=m.predict_proba(Xv);pred=classes[p.argmax(1)];r={'weight':w,'iter':int(m.best_iteration)+1,'validation':float(np.mean(pred==yv)),'sec':time.time()-t};print(json.dumps(r),flush=True);tag=str(w).replace('.','p');m.save_model(OUT/f'w{tag}.json');np.savez_compressed(OUT/f'w{tag}.npz',validation=p.astype(np.float32),classes=classes,cols=cols);json.dump(r,open(OUT/f'w{tag}_summary.json','w'),indent=2)
