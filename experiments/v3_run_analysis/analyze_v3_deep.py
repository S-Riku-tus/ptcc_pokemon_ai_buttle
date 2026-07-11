"""Deep analysis of v3 ladder run: replay every decision of our seat through the
v3 policy classes to detect concrete blunders.

Detections per episode:
  A. lethal_end     : a MAIN select offered a lethal attack, but the turn ended
                      without ANY attack being chosen that turn.
  B. boss_no_ko     : Boss's Orders played on a turn where we took no prize.
  C. enrich_on_line : Enriching Energy attached to Abra/Kadabra/Alakazam.
  D. lowdeck_spend  : optional draw/search (ability yes / poffin / pokepad /
                      hilda / dawn / dudun ability) chosen with deckCount<=10.
  E. stuck_active   : our turn ends with active in {Dunsparce, Dudunsparce,
                      Abra, Kadabra} while a powered Alakazam sits on bench.
  F. mist_zero      : we chose Powerful Hand into an effect-prevented target (0 dmg).
  G. boss_bad_active: Boss played while our active is a non-attacker.
  H. no_attack_turns: fraction of our turns with no attack at all.
"""
import sys, json, csv, os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/sessions/charming-exciting-franklin/mnt/ptcc_pokemon_ai_buttle')
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'agents' / 'alakazam741_v3'))
os.chdir(ROOT / 'agents' / 'alakazam741_v3')

import main as v3  # the actual v3 agent module
from cg.api import (OptionType, SelectContext, to_observation_class,
                    all_card_data, CardType)

CARD = {c.cardId: c for c in all_card_data()}
RUN = ROOT / 'data/runs/20260711_222717_alakazam741_v3_latest_sub54557078'

def archetype(deck):
    pokes = Counter(cid for cid in deck if CARD.get(cid) and CARD[cid].cardType == 0)
    if not pokes: return 'unknown'
    def key(item):
        cid, n = item; c = CARD[cid]
        return (c.stage2, c.megaEx or c.ex, c.stage1, n, c.hp)
    return CARD[max(pokes.items(), key=key)[0]].name

manifest = {}
with open(RUN / 'manifest.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        manifest[row['episode_id']] = int(row['detected_submission_agent_index'])

LINE_IDS = {741, 742, 743, 245}
ENGINE_IDS = {305, 66}
ALAKAZAM_IDS = {743, 245}
ENRICHING = 13
POWERFUL_HAND = 1072

results = []
blunders = defaultdict(list)   # kind -> [(ep, turn, detail)]

for epdir in sorted((RUN / 'episodes').iterdir()):
    epid = epdir.name
    rp = epdir / 'replay' / f'episode_{epid}.json'
    if not rp.exists(): continue
    ep = json.loads(rp.read_text(encoding='utf-8'))
    seat = manifest.get(epid, 0)
    steps = ep['steps']
    rew = (ep.get('rewards') or [None, None])[seat]
    # decks
    decks = [None, None]
    if len(steps) > 1:
        for pi in (0, 1):
            act = steps[pi and 1 or 1][pi].get('action')
            if isinstance(act, list) and len(act) == 60:
                decks[pi] = act
    opp_arch = archetype(decks[1-seat]) if decks[1-seat] else 'unknown'

    # per-turn tracking of MY turns
    turn_had_attack = {}
    turn_had_lethal_offer = {}
    turn_last_state = {}       # turn -> (active_id, bench_ids, deckCount)
    my_prizes_left_by_turn = {}
    ep_blunders = Counter()
    last_cur = None
    mist_zero = 0

    for si, st in enumerate(steps):
        ag = st[seat]
        obs_d = ag.get('observation') or {}
        cur = obs_d.get('current'); sel = obs_d.get('select')
        if cur: last_cur = cur
        if not cur or not sel: continue
        # VERIFIED: the response to the observation at step i is recorded at step i+1
        action = steps[si+1][seat].get('action') if si + 1 < len(steps) else None
        try:
            obs = to_observation_class(obs_d)
        except Exception:
            continue
        if obs.select is None or obs.current is None: continue
        try:
            pol = v3.AlakazamPolicy(obs)
        except Exception:
            continue
        turn = cur['turn']
        me = cur['players'][cur['yourIndex']]
        chosen = []
        if isinstance(action, list):
            chosen = [sel['option'][i] for i in action if isinstance(i, int) and 0 <= i < len(sel['option'])]
        ctx = sel.get('context')

        if ctx == SelectContext.MAIN:
            act_id = pol.me.active[0].id if pol.me.active else None
            bench_ids = [p.id for p in pol.me.bench if p is not None]
            turn_last_state[turn] = (act_id, bench_ids, me['deckCount'],
                                     pol._bench_attacker_ready())
            my_prizes_left_by_turn[turn] = len(me.get('prize') or [])
            try:
                if pol._lethal_attack_offered():
                    turn_had_lethal_offer[turn] = True
            except Exception:
                pass
            for o in chosen:
                ot = o.get('type')
                if ot == OptionType.ATTACK:
                    turn_had_attack[turn] = True
                    # F: powerful hand into prevented target
                    if o.get('attackId') == POWERFUL_HAND:
                        opp_a = pol.opponent.active[0] if pol.opponent.active else None
                        if opp_a is not None and pol._effect_prevented(opp_a):
                            mist_zero += 1
                            ep_blunders['mist_zero'] += 1
                if ot == OptionType.PLAY:
                    idx = o.get('index')
                    hand = me.get('hand') or []
                    cid = hand[idx]['id'] if idx is not None and idx < len(hand) else None
                    if cid == 1182:  # Boss's Orders
                        blunders['boss_played'].append((epid, turn, act_id))
                        if act_id not in ALAKAZAM_IDS:
                            ep_blunders['boss_bad_active'] += 1
                            blunders['boss_bad_active'].append((epid, turn, act_id))
                    if cid in (1086, 1152, 1225, 1231) and me['deckCount'] <= 10:
                        ep_blunders['lowdeck_spend'] += 1
                        blunders['lowdeck_spend'].append((epid, turn, cid, me['deckCount']))
                if ot == OptionType.ABILITY and me['deckCount'] <= 10:
                    ep_blunders['lowdeck_ability'] += 1
                    blunders['lowdeck_ability'].append((epid, turn, me['deckCount']))
                if ot in (OptionType.ENERGY, OptionType.ATTACH):
                    # which energy attached to what?
                    idx = o.get('index'); hand = me.get('hand') or []
                    cid = hand[idx]['id'] if idx is not None and idx < len(hand) else None
                    tgt_area, tgt_idx = o.get('inPlayArea'), o.get('inPlayIndex')
                    tgt = None
                    if tgt_area == 3 and pol.me.active: tgt = pol.me.active[0]
                    elif tgt_area == 4 and tgt_idx is not None and tgt_idx < len(pol.me.bench):
                        tgt = pol.me.bench[tgt_idx]
                    if cid == ENRICHING and tgt is not None and tgt.id in LINE_IDS:
                        ep_blunders['enrich_on_line'] += 1
                        blunders['enrich_on_line'].append((epid, turn, tgt.id))
        elif ctx == SelectContext.ATTACH_TO:
            cc = sel.get('contextCard') or {}
            if cc.get('id') == ENRICHING:
                for o in chosen:
                    tgt_area, tgt_idx = o.get('inPlayArea'), o.get('inPlayIndex')
                    tgt = None
                    if tgt_area == 3 and pol.me.active: tgt = pol.me.active[0]
                    elif tgt_area == 4 and tgt_idx is not None and tgt_idx < len(pol.me.bench):
                        tgt = pol.me.bench[tgt_idx]
                    if tgt is not None and tgt.id in LINE_IDS:
                        ep_blunders['enrich_on_line'] += 1
                        blunders['enrich_on_line'].append((epid, turn, tgt.id, 'ATTACH_TO'))

    # post-episode: turn-level blunders
    my_turns = sorted(turn_last_state)
    lethal_end, stuck = 0, 0
    for t in my_turns:
        if turn_had_lethal_offer.get(t) and not turn_had_attack.get(t):
            lethal_end += 1
            blunders['lethal_end'].append((epid, t))
        aid, bench, dk, ready = turn_last_state[t]
        if aid in (305, 66, 741) and ready and not turn_had_attack.get(t):
            stuck += 1
            blunders['stuck_active'].append((epid, t, aid))
    # boss_no_ko: boss played turn t, prizes left did not drop by t+2 (next my turn)
    boss_no_ko = 0
    for (e2, t, act) in blunders['boss_played']:
        if e2 != epid: continue
        pl_now = my_prizes_left_by_turn.get(t)
        nxt = [pp for tt, pp in my_prizes_left_by_turn.items() if tt > t]
        if pl_now is not None and nxt and min([pl_now] + [my_prizes_left_by_turn[tt] for tt in my_prizes_left_by_turn if t < tt <= t+2] or [pl_now]) >= pl_now:
            boss_no_ko += 1
            blunders['boss_no_ko'].append((epid, t))

    fin_act = None; end_reason = ''
    if last_cur:
        mep = last_cur['players'][seat if last_cur.get('yourIndex') is None else last_cur['yourIndex']]
        # careful: last_cur is from OUR observation so yourIndex is ours
        yi = last_cur['yourIndex']
        mep = last_cur['players'][yi]; opp = last_cur['players'][1-yi]
        fin_act = (mep['active'][0]['id'] if mep['active'] else None)
        if mep['deckCount'] == 0 and rew == -1: end_reason = 'deckout(me)'
        elif opp['deckCount'] == 0 and rew == 1: end_reason = 'deckout(opp)'
        elif rew == 1 and not mep.get('prize'): end_reason = 'prizes(win)'
        elif rew == -1 and not opp.get('prize'): end_reason = 'prizes(loss)'
        else: end_reason = 'other'
    n_turns = len(my_turns)
    n_attack_turns = sum(1 for t in my_turns if turn_had_attack.get(t))
    results.append(dict(ep=epid, win=rew==1, opp=opp_arch, turns=n_turns,
                        attack_turns=n_attack_turns, lethal_end=lethal_end,
                        stuck=stuck, boss_no_ko=boss_no_ko,
                        fin_active=CARD[fin_act].name if fin_act in CARD else fin_act,
                        deck_left=last_cur['players'][last_cur['yourIndex']]['deckCount'] if last_cur else None,
                        my_prizes_left=len(last_cur['players'][last_cur['yourIndex']].get('prize') or []) if last_cur else None,
                        opp_prizes_left=len(last_cur['players'][1-last_cur['yourIndex']].get('prize') or []) if last_cur else None,
                        end=end_reason, **{k: v for k, v in ep_blunders.items()}))

import pandas as pd
df = pd.DataFrame(results)
pd.set_option('display.width', 250)
print('=== W/L by opponent ===')
print(df.groupby('opp').agg(games=('win','size'), wins=('win','sum')).assign(wr=lambda d: (d.wins/d.games).round(2)))
print()
print('=== losses detail ===')
loss = df[~df.win]
cols = ['ep','opp','turns','attack_turns','lethal_end','stuck','boss_no_ko','fin_active','deck_left','my_prizes_left','opp_prizes_left','end']
print(loss[cols].to_string(index=False))
print()
print('=== blunder totals (all games) ===')
for k in ('lethal_end','stuck_active','boss_played','boss_no_ko','boss_bad_active','enrich_on_line','lowdeck_spend','lowdeck_ability','mist_zero'):
    eps = blunders[k]
    in_loss = sum(1 for x in eps if not df[df.ep==x[0]].win.iloc[0]) if len(eps) else 0
    print(f'{k:18s} total={len(eps):3d} in-losses={in_loss}')
print()
print('=== attack rate ===')
print('win  games attack-turn ratio:', (df[df.win].attack_turns.sum() / max(1,df[df.win].turns.sum())).round(2))
print('loss games attack-turn ratio:', (loss.attack_turns.sum() / max(1,loss.turns.sum())).round(2))
df.to_csv('/sessions/charming-exciting-franklin/mnt/outputs/v3_deep.csv', index=False)
import pickle
pickle.dump(dict(blunders), open('/sessions/charming-exciting-franklin/mnt/outputs/v3_blunders.pkl','wb'))
                                                                                                                                                                