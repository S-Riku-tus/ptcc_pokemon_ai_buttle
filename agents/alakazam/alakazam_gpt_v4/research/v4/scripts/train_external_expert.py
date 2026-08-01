from pathlib import Path
import argparse,json,time
import numpy as np,xgboost as xgb
R=Path('/mnt/data/v3_work/rich_meta');OUT=Path('/mnt/data/v4_work/experts');OUT.mkdir(exist_ok=True)
classes=np.array([0,1,2,3,4,5,6,7,9,10,11],np.int32);c={int(v):i for i,v in enumerate(classes)};cv=lambda y:np.array([c[int(v)] for v in y],np.int32)
cols=np.array(json.load(open('/mnt/data/v3_work/alakazam_gpt_v3/type_model.json'))['cols'],np.int32)
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
z=np.load(a.source);X=np.asarray(z['X'],np.float32);y=z['y'];Xv=np.asarray(np.load(R/'validation_X.npy',mmap_mode='r')[:,cols],np.float32);yv=np.load(R/'validation_y.npy');Xt=np.asarray(np.load(R/'test_X.npy',mmap_mode='r')[:,cols],np.float32)
m=xgb.XGBClassifier(objective='multi:softprob',num_class=len(classes),n_estimators=450,max_depth=7,learning_rate=.04,subsample=.9,colsample_bytree=.85,min_child_weight=3,reg_alpha=.25,reg_lambda=3,tree_method='hist',n_jobs=4,random_state=3711,eval_metric='mlogloss',early_stopping_rounds=40)
t=time.time();m.fit(X,cv(y),eval_set=[(Xv,cv(yv))],verbose=False);pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);res={'source':a.source,'rows':len(y),'best_iteration':int(m.best_iteration)+1,'validation_type':float(np.mean(classes[pv.argmax(1)]==yv)),'seconds':time.time()-t};print(json.dumps(res));m.save_model(OUT/f'{a.name}.json');np.savez_compressed(OUT/f'{a.name}.npz',validation=pv.astype(np.float32),test=pt.astype(np.float32),classes=classes);json.dump(res,open(OUT/f'{a.name}_summary.json','w'),indent=2)
