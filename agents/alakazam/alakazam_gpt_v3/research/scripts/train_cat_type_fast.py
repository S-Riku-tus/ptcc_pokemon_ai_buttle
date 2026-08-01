from pathlib import Path
import numpy as np,json,time
from catboost import CatBoostClassifier
O=Path('/mnt/data/v1_work/results/v2_type_meta'); OUT=Path('/mnt/data/v3_work/cat_type');OUT.mkdir(exist_ok=True)
z=np.load(O/'xgb_probs.npz');cols=z['cols']
X=np.asarray(np.load(O/'train_X.npy',mmap_mode='r')[:,cols],np.float32);y=np.load(O/'train_y.npy');Xv=np.asarray(np.load(O/'validation_X.npy',mmap_mode='r')[:,cols],np.float32);yv=np.load(O/'validation_y.npy')
classes=np.unique(y);cmap={int(c):i for i,c in enumerate(classes)};yy=np.array([cmap[int(v)] for v in y]);yyv=np.array([cmap[int(v)] for v in yv])
for name,depth,lr,it in [('d6',6,.06,600),('d8',8,.05,500)]:
 print('fit',name,flush=True);t=time.time()
 m=CatBoostClassifier(loss_function='MultiClass',eval_metric='Accuracy',iterations=it,depth=depth,learning_rate=lr,l2_leaf_reg=5,random_seed=3603,random_strength=.3,bootstrap_type='Bernoulli',subsample=.9,thread_count=8,verbose=50,od_type='Iter',od_wait=50,allow_writing_files=False)
 m.fit(X,yy,eval_set=(Xv,yyv),use_best_model=True)
 p=m.predict_proba(Xv);pred=classes[p.argmax(1)];print({'name':name,'best':m.get_best_iteration()+1,'val':float(np.mean(pred==yv)),'sec':time.time()-t},flush=True)
 m.save_model(OUT/f'{name}.cbm');np.savez_compressed(OUT/f'{name}_val.npz',p=p.astype(np.float32),classes=classes,cols=cols)
