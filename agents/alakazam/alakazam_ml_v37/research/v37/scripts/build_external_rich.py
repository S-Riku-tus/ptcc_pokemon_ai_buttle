from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import lightgbm as lgb
import numpy as np

SEMANTIC = ['option_type','candidate_card_id','candidate_attack_id','candidate_target_id','candidate_target_hp','candidate_target_max_hp','candidate_target_energy','candidate_target_special_energy','candidate_inplay_area']

def softmax_entropy(peaks):
    a=np.asarray(peaks,np.float64); m=float(a.max()); p=np.exp(np.clip(a-m,-50,50)); p/=p.sum(); return float(-(p*np.log(p+1e-9)).sum())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pattern',required=True); ap.add_argument('--output',required=True); ap.add_argument('--teacher',required=True)
    a=ap.parse_args(); paths=sorted(glob.glob(a.pattern));
    if not paths: raise SystemExit('no chunks')
    spec=json.load(open('/mnt/data/v36_work/alakazam_ml_v36/type_runtime_spec.json'))
    model=json.load(open('/mnt/data/v36_work/alakazam_ml_v36/type_model.json'))
    cols=np.asarray(model['cols'],np.int32)
    booster=lgb.Booster(model_file='/mnt/data/v34_work/results/v34_baseline_lgb.txt')
    all_rows=[]; all_y=[]; all_meta=[]; total_raw=total_sem=total_dec=0
    feature_names=None
    for path in paths:
        z=np.load(path,allow_pickle=False); X=np.asarray(z['features'],np.float32); groups=z['groups'].astype(np.int64); meta=z['decision_meta']; names=z['feature_names'].astype(str).tolist()
        if feature_names is None: feature_names=names
        elif feature_names!=names: raise RuntimeError('feature mismatch')
        idx={n:i for i,n in enumerate(names)}; sem_idx=[idx[n] for n in SEMANTIC]
        scores=np.asarray(booster.predict(X),np.float32)
        off=0
        for di,g0 in enumerate(groups):
            g=int(g0); end=off+g; A=X[off:end]; S=scores[off:end]
            seen=set(); reps=[]
            for r in range(g):
                key=tuple(float(A[r,j]) for j in sem_idx)
                if key not in seen: seen.add(key); reps.append(r)
            if len(reps)<2:
                off=end; continue
            R=A[reps]; RS=S[reps]; first=R[0]
            row=[float(first[idx[n]]) for n in spec['state_names']]
            type_peaks=[]; type_locals=[]; type_tops=[]
            ai=idx['action_type_id']
            for t in range(12):
                local=np.flatnonzero(R[:,ai].astype(np.int32)==t).tolist(); type_locals.append(local)
                if local:
                    vals=RS[local]; ranked=sorted(local,key=lambda q:float(RS[q]),reverse=True); top=ranked[0]; type_tops.append(top)
                    tv=float(RS[top]); sec=float(RS[ranked[1]]) if len(ranked)>1 else tv-5.0; mean=float(vals.mean()); std=float(vals.std())
                    out=[len(local),tv,sec,tv-sec,mean,std,float(top),float(np.sum(vals>tv-.25))]
                    out += [float(R[top,idx[n]]) for n in spec['candidate_fields']]; type_peaks.append(tv)
                else:
                    type_tops.append(None); out=[0,-99,-99,0,-99,0,-1,0]+[-1]*len(spec['candidate_fields']); type_peaks.append(-99.)
                row.extend(out)
            order=sorted(range(12),key=lambda t:type_peaks[t],reverse=True)
            row += [float(order[0]),float(order[1]),type_peaks[order[0]],type_peaks[order[1]],type_peaks[order[0]]-type_peaks[order[1]],softmax_entropy(type_peaks),float(sum(v>-90 for v in type_peaks)),float(R[int(np.argmax(RS)),ai])]
            for t in range(12):
                local=type_locals[t]; top=type_tops[t]
                if local and top is not None:
                    row += [float(R[top,idx[n]]) for n in spec['rich_top_fields']]
                    for n in spec['rich_agg_fields']:
                        vals=R[local,idx[n]].astype(np.float64); row += [float(vals.max()),float(vals.sum())]
                    pos=R[local,idx['candidate_option_position']].astype(np.float64)
                    cards=set(int(R[q,idx['candidate_card_id']]) for q in local); targets=set(int(R[q,idx['candidate_target_id']]) for q in local)
                    row += [float(pos.min()),float(pos.mean()),float(len(cards)),float(len(targets))]
                else:
                    row += [-1.0]*len(spec['rich_top_fields']) + [0.0]*(2*len(spec['rich_agg_fields'])+4)
            if len(row)!=1052: raise RuntimeError(len(row))
            all_rows.append(np.asarray(row,np.float32)[cols]); all_y.append(int(meta[di,3])); all_meta.append(meta[di].copy())
            total_raw+=g; total_sem+=len(reps); total_dec+=1; off=end
        z.close(); print(Path(path).name,'done',len(groups),flush=True)
    XX=np.asarray(all_rows,np.float32); yy=np.asarray(all_y,np.int32); mm=np.asarray(all_meta,np.int32)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,X=XX,y=yy,meta=mm,cols=cols,teacher=np.asarray([a.teacher]))
    print(json.dumps({'teacher':a.teacher,'chunks':len(paths),'decisions':len(yy),'raw_candidates':total_raw,'semantic_candidates':total_sem,'features':XX.shape[1],'class_counts':dict(zip(*[x.tolist() for x in np.unique(yy,return_counts=True)])),'output':a.output}),flush=True)
if __name__=='__main__':main()
