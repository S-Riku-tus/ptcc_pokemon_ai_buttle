import csv, json, sys, statistics, os
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
sys.path.insert(0,'agents/grimmsnarl/grimmsnarl_ml_v8')
import ml_features as mf
from analyze_grimmsnarl_matchup_ceiling import family
from ml.core.replay_io import deck_hash
from analyze_grimmsnarl_v10_tempo import landmarks, turn_order
OUR='9714ab5c3996f6cc'

def collect(path, seat):
    rep=json.load(open(path,encoding='utf-8'))
    steps=rep.get('steps') or []
    decks=[None,None]
    if len(steps)>1:
        for s in (0,1):
            a=(steps[1][s] or {}).get('action')
            if isinstance(a,list) and len(a)==60: decks[s]=[int(v) for v in a]
    if decks[seat] is None or deck_hash(decks[seat])!=OUR: return None
    if family(decks[1-seat])!='Alakazam': return None
    if turn_order(rep,seat) is not False: return None
    rewards=rep.get('rewards') or [None,None]
    other=rewards[1-seat]
    lm=landmarks(rep,seat)
    board={}
    for step in steps:
        if seat>=len(step): continue
        cur=((step[seat] or {}).get('observation') or {}).get('current') or {}
        pl=cur.get('players') or []
        if len(pl)<2: continue
        board[int(cur.get('turn',-1))]=len(mf._in_play(pl[seat]))
    return dict(won=bool((rewards[seat] or 0)>(other if other is not None else 0)),
                bodies_t2=board.get(2), bodies_t4=board.get(4), bodies_t6=board.get(6), **lm)

field=[]; seen=set()
for raw in csv.DictReader(open('data/kaggle_grimmsnarl_top50/indexes/episodes.csv',encoding='utf-8-sig')):
    if raw.get('download_status')!='success' or raw.get('deck_hash')!=OUR: continue
    if raw.get('episode_type')!='EPISODE_TYPE_PUBLIC': continue
    k=(raw['episode_id'],raw['seat_index'])
    if k in seen: continue
    seen.add(k)
    p=f"data/kaggle_grimmsnarl_top50/replays/episode_{raw['episode_id']}.json"
    if not os.path.exists(p): continue
    try: r=collect(p,int(raw['seat_index']))
    except Exception: continue
    if r: field.append(r)

ours=[]
for raw in csv.DictReader(open('data/submissions/submission_55317804/episodes.csv',encoding='utf-8-sig')):
    a0,a1=raw['agent_0_submission_id'],raw['agent_1_submission_id']
    if raw['episode_type']!='EPISODE_TYPE_PUBLIC' or a0==a1: continue
    e=raw['episode_id']
    p=f"data/submissions/submission_55317804/episodes/{e}/replay/episode_{e}.json"
    if not os.path.exists(p): continue
    r=collect(p, 0 if a0=='55317804' else 1)
    if r: ours.append(r)

def desc(rows,key):
    v=[r[key] for r in rows if r.get(key) is not None]
    return (f"{statistics.median(v):.1f}/{statistics.fmean(v):.2f} "
            f"(n={len(v)},miss={len(rows)-len(v)})") if v else "-"
out=[f"Alakazam going SECOND: field n={len(field)}  v8 n={len(ours)}"]
for key in ('first_grimmsnarl_turn','first_attack_turn','own_turns','bodies_t2','bodies_t4','bodies_t6'):
    out.append(f"  {key:22s} field {desc(field,key):30s} v8 {desc(ours,key)}")
fw=[r for r in field if r['won']]; fl=[r for r in field if not r['won']]
out.append(f"  [field WIN  n={len(fw)}] grimm {desc(fw,'first_grimmsnarl_turn')} attack {desc(fw,'first_attack_turn')} t4bodies {desc(fw,'bodies_t4')}")
out.append(f"  [field LOSS n={len(fl)}] grimm {desc(fl,'first_grimmsnarl_turn')} attack {desc(fl,'first_attack_turn')} t4bodies {desc(fl,'bodies_t4')}")
print("\n".join(out))
json.dump({'field':field,'v8':ours}, open('experiments/grimmsnarl_ml_v10_safe_residual/alakazam_second_tempo.json','w'), indent=1)
