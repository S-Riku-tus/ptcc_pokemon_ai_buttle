"""Pass 2: classify WHY each of our turns in LOST games ended without an attack,
plus inspect the quick losses (turns<=3) and Mist/energy states.
"""
import sys, json, csv, os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/sessions/charming-exciting-franklin/mnt/ptcc_pokemon_ai_buttle')
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'agents' / 'alakazam741_v3'))
os.chdir(ROOT / 'agents' / 'alakazam741_v3')
import main as v3
from cg.api import (OptionType, SelectContext, to_observation_class, all_card_data)
os.chdir(ROOT)
CARD = {c.cardId: c for c in all_card_data()}
RUN = ROOT / 'data/runs/20260711_222717_alakazam741_v3_latest_sub54557078'

manifest = {}
with open(RUN / 'manifest.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        manifest[row['episode_id']] = int(row['detected_submission_agent_index'])

LOSSES = ['85317719','85319428','85320529','85321067','85322665','85323707',
          '85325864','85327557','85328616','85328678','85329674','85332822',
          '85333401','85333420','85333966','85334490','85335088','85337617',
          '85339122','85339627','85343306']
ALAKAZAM_IDS = {743, 245}

def name(cid):
    return CARD[cid].name if cid in CARD else str(cid)

reasons_total = Counter()
per_ep = {}
for epid in LOSSES:
    epdir = RUN / 'episodes' / epid
    ep = json.loads((epdir/'replay'/f'episode_{epid}.json').read_text())
    seat = manifest.get(epid, 0)
    steps = ep['steps']
    statuses = ep.get('statuses')
    # per-turn: last MAIN observation state + whether attack chosen
    turn_attack = {}
    turn_info = {}   # turn -> dict
    for si, st in enumerate(steps):
        o = st[seat]['observation']
        if not o.get('current') or not o.get('select'): continue
        cur, sel = o['current'], o['select']
        if sel.get('context') != 0: continue  # MAIN only
        action = steps[si+1][seat].get('action') if si+1 < len(steps) else None
        try:
            obs = to_observation_class(o); pol = v3.AlakazamPolicy(obs)
        except Exception:
            continue
        t = cur['turn']
        me = pol.me; opp = pol.opponent
        act = me.active[0] if me.active else None
        opp_act = opp.active[0] if opp.active else None
        info = dict(
            active=name(act.id) if act else None,
            act_energy=len(act.energies) if act else 0,
            opp_active=name(opp_act.id) if opp_act else None,
            opp_hp=opp_act.hp if opp_act else 0,
            hand=me.handCount, deck=me.deckCount,
            alakazam_in_play=any(p is not None and p.id in ALAKAZAM_IDS for p in me.active+me.bench),
            powered_alakazam=any(p is not None and p.id in ALAKAZAM_IDS and len(p.energies)>=1 for p in me.active+me.bench),
            can_attack_active=(act is not None and pol._can_attack(act)),
            attack_offered=any(x.get('type')==13 for x in sel['option']),
            mist=pol._effect_prevented(opp_act) if opp_act else False,
            p_energy_hand=pol._psychic_in_hand(),
            my_prizes=len(me.prize),
        )
        turn_info[t] = info
        if isinstance(action, list):
            for i in action:
                if 0 <= i < len(sel['option']) and sel['option'][i].get('type')==13:
                    turn_attack[t] = sel['option'][i].get('attackId')
    rows = []
    for t in sorted(turn_info):
        info = turn_info[t]
        atk = turn_attack.get(t)
        if atk:
            dmgnote = ''
            rows.append((t, 'ATTACK', atk, info))
            continue
        # classify
        if not info['alakazam_in_play']:
            r = 'no_alakazam_built'
        elif not info['powered_alakazam']:
            r = 'alakazam_no_energy' + ('' if info['p_energy_hand'] else '(no_P_in_hand)')
        elif not info['can_attack_active']:
            r = 'powered_on_bench_stuck_active=' + str(info['active'])
        elif info['mist']:
            r = 'mist_blocked'
        elif not info['attack_offered']:
            r = 'attack_not_offered(status?)'
        else:
            r = 'attack_offered_but_not_chosen'
        reasons_total[r.split('=')[0]] += 1
        rows.append((t, 'NO-ATTACK', r, info))
    per_ep[epid] = (statuses, rows)

for epid, (statuses, rows) in per_ep.items():
    print(f'--- {epid} statuses={statuses}')
    for t, kind, detail, info in rows:
        if kind == 'ATTACK':
            print(f'  T{t:2d} ATTACK {detail} active={info["active"]}({info["act_energy"]}E) hand={info["hand"]} deck={info["deck"]} opp={info["opp_active"]}({info["opp_hp"]}hp) mist={info["mist"]}')
        else:
            print(f'  T{t:2d} **{detail}** active={info["active"]}({info["act_energy"]}E) hand={info["hand"]} deck={info["deck"]} opp={info["opp_active"]}({info["opp_hp"]}hp) Pinhand={info["p_energy_hand"]}')
print()
print('=== NO-ATTACK reason totals (losses) ===')
for k, v in reasons_total.most_common():
    print(f'{k:45s} {v}')
