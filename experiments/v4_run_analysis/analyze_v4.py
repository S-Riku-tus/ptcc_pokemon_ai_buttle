"""v4 ladder run (sub54576108) full-episode analysis.

Methodology (established in experiments/v3_run_analysis):
  - The response to the observation at steps[i] is recorded at steps[i+1].action.
  - Damage is verified from ground truth (HP by serial across observations),
    NOT from the policy's own damage model (which is what we are auditing).

Detections / metrics per episode:
  outcome, seat, went_first, opponent archetype, end reason
  A. dudun_last_body : Dudunsparce ability used while our board has <=1 Pokemon
  B. ph_zero         : Powerful Hand chosen and the target's HP did not change
  C. boss usage      : played Boss's Orders -> attacked same turn? prize within
                       our next turn?
  D. xerosic usage
  E. bodies at the end of our 2nd own turn
  F. non-attack turns (excl. our first turn) + whether an Alakazam was in play
  G. deck-out losses / deck floor behaviour
  H. kadabra_active  : turns our active is Kadabra that cannot evolve now,
                       with a Dunsparce-line body available on bench
  I. opponent tech   : Mist/Rock energy, global-protector (Articuno 414),
                       Lillie's Determination usage by opponent
"""
import sys, json, csv, os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'agents' / 'alakazam741_v4'))
os.chdir(ROOT / 'agents' / 'alakazam741_v4')

import main as v4  # the actual v4 agent module
from cg.api import (OptionType, SelectContext, to_observation_class,
                    all_card_data, CardType)

CARD = {c.cardId: c for c in all_card_data()}
RUN = ROOT / 'data/runs/20260712_115550_alakazam741_v4_latest_sub54576108'

POWERFUL_HAND = 1072
BOSS = 1182
XEROSIC = 1197
LINE_IDS = {741, 742, 743, 245}
ALAKAZAM_IDS = {743, 245}
DUDUN = 66

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

# validation episodes (agent vs itself) are excluded from ladder aggregates
EXCLUDE = set()
with open(RUN / 'episodes.csv', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row.get('episode_type') != 'EPISODE_TYPE_PUBLIC':
            EXCLUDE.add(row['episode_id'])

# board-wide effect protectors (e.g. Team Rocket's Articuno 414, Rabsca 74)
GLOBAL_PROTECTORS = {414, 74}

# find interesting card ids by name
NAME = {cid: c.name for cid, c in CARD.items()}
LILLIE = [cid for cid, c in CARD.items() if 'Lillie' in (c.name or '')]
ARTICUNO = [cid for cid, c in CARD.items() if 'Articuno' in (c.name or '')]

rows = []
events = defaultdict(list)

for epdir in sorted((RUN / 'episodes').iterdir()):
    epid = epdir.name
    if epid in EXCLUDE: continue
    rp = epdir / 'replay' / f'episode_{epid}.json'
    if not rp.exists(): continue
    ep = json.loads(rp.read_text(encoding='utf-8'))
    seat = manifest.get(epid, 0)
    opp = 1 - seat
    steps = ep['steps']
    rew = (ep.get('rewards') or [None, None])[seat]

    decks = [None, None]
    if len(steps) > 1:
        for pi in (0, 1):
            act = steps[1][pi].get('action')
            if isinstance(act, list) and len(act) == 60:
                decks[pi] = act
    opp_deck = decks[opp] or []
    opp_arch = archetype(opp_deck) if opp_deck else 'unknown'

    went_first = None
    last_cur = None
    my_turns = []                 # ordered distinct turn numbers where we saw MAIN
    turn_attack = {}              # turn -> attackId chosen
    turn_state = {}               # turn -> dict at last MAIN obs of that turn
    ph_attacks = []               # (si, turn, serial, cid, hp, opp_protector_in_play)
    ph_total = 0
    boss_plays = []               # (turn, si)
    xerosic_plays = []
    dudun_last_body = []          # (turn, area, bodies)
    kadabra_active = []           # (turn, chose)
    opp_lillie = 0
    my_prizes_seq = []            # (turn, prizes_left)

    # serial -> list of (si, hp) for opponent pokemon, ground truth for damage
    opp_hp_track = defaultdict(list)
    opp_board_snapshots = []      # (si, {serial}) for every obs with a current state

    for si, st in enumerate(steps):
        ag = st[seat]
        obs_d = ag.get('observation') or {}
        cur = obs_d.get('current'); sel = obs_d.get('select')
        if cur:
            last_cur = cur
            if cur.get('turn', 0) >= 1 and cur.get('firstPlayer') is not None:
                went_first = (cur['firstPlayer'] == cur['yourIndex'])
            snap = set()
            for p in (cur['players'][1 - cur['yourIndex']].get('active') or []) + \
                     (cur['players'][1 - cur['yourIndex']].get('bench') or []):
                if p:
                    opp_hp_track[p['serial']].append((si, p['hp']))
                    snap.add(p['serial'])
            opp_board_snapshots.append((si, snap))
        # opponent logs: Lillie played by opponent
        for lg in (obs_d.get('logs') or []):
            if lg.get('cardId') in LILLIE and lg.get('playerIndex') == (1 - (cur or {}).get('yourIndex', seat)):
                opp_lillie += 1
        if not cur or not sel: continue
        action = steps[si+1][seat].get('action') if si + 1 < len(steps) else None
        chosen = []
        if isinstance(action, list):
            chosen = [sel['option'][i] for i in action
                      if isinstance(i, int) and 0 <= i < len(sel['option'])]
        ctx = sel.get('context')
        turn = cur['turn']
        me = cur['players'][cur['yourIndex']]
        op = cur['players'][1 - cur['yourIndex']]

        if ctx == SelectContext.MAIN:
            if turn not in my_turns:
                my_turns.append(turn)
            bodies = len([p for p in (me.get('active') or []) if p]) + \
                     len([p for p in (me.get('bench') or []) if p])
            hand_ids = [c['id'] for c in (me.get('hand') or [])]
            turn_state[turn] = dict(
                bodies=bodies,
                active=(me['active'][0]['id'] if me.get('active') and me['active'] else None),
                bench=[p['id'] for p in (me.get('bench') or []) if p],
                deck=me['deckCount'], hand=len(hand_ids),
                alakazam_in_play=any(p and p['id'] in ALAKAZAM_IDS
                                     for p in (me.get('active') or []) + (me.get('bench') or [])),
                prizes=len(me.get('prize') or []),
            )
            my_prizes_seq.append((turn, len(me.get('prize') or [])))
            for o in chosen:
                ot = o.get('type')
                if ot == OptionType.ATTACK:
                    turn_attack[turn] = o.get('attackId')
                    if o.get('attackId') == POWERFUL_HAND:
                        ph_total += 1
                        tgt = (op.get('active') or [None])[0]
                        if tgt:
                            prot = any(p and p['id'] in GLOBAL_PROTECTORS
                                       for p in (op.get('active') or []) + (op.get('bench') or []))
                            ph_attacks.append((si, turn, tgt['serial'], tgt['id'],
                                               tgt['hp'], prot))
                elif ot == OptionType.ABILITY:
                    card = None
                    area, idx = o.get('area'), o.get('index')
                    from cg.api import AreaType as AT
                    src = (me.get('active') if area == AT.ACTIVE else me.get('bench')) or []
                    if idx is not None and 0 <= idx < len(src) and src[idx]:
                        card = src[idx]
                    if card and card['id'] == DUDUN and area == AT.ACTIVE and bodies <= 1:
                        dudun_last_body.append((turn, 'ACTIVE', bodies))
                elif ot == OptionType.PLAY:
                    idx = o.get('index')
                    hand = me.get('hand') or []
                    if idx is not None and 0 <= idx < len(hand):
                        cid = hand[idx]['id']
                        if cid == BOSS: boss_plays.append((turn, si))
                        if cid == XEROSIC: xerosic_plays.append((turn, si))
            # H: Kadabra active, cannot evolve now, line body available on bench
            act0 = (me.get('active') or [None])[0]
            if act0 and act0['id'] == 742 and 743 not in hand_ids and 245 not in hand_ids:
                if any(p and p['id'] in (305, 66) for p in (me.get('bench') or [])):
                    kadabra_active.append((turn,
                        'attack' if turn_attack.get(turn) else 'other'))

    # post-process: resolve zero-damage Powerful Hands from the full HP history
    # zero-damage = at the FIRST subsequent observation the target is still on
    # the board with unchanged HP. If it vanished from the board, it was KO'd
    # (or bounced) — Night-Stretcher re-benching the same serial at full HP must
    # not be miscounted as a zero.
    ph_zero = []
    for (si, turn, serial, cid, hp, prot) in ph_attacks:
        nxt = next(((s2, snap) for (s2, snap) in opp_board_snapshots if s2 > si), None)
        if nxt is None or serial not in nxt[1]:
            continue  # target gone -> KO / bounce, damage clearly happened
        h2 = next(h for (s2, h) in opp_hp_track[serial] if s2 == nxt[0])
        if h2 >= hp:
            ph_zero.append((turn, cid, hp, prot))

    attacked_turns = set(turn_attack)
    nonattack = [t for t in my_turns[1:] if t not in attacked_turns]
    nonattack_no_alk = [t for t in nonattack if not turn_state[t]['alakazam_in_play']]
    boss_res = []
    for (bt, bsi) in boss_plays:
        same_turn_attack = bt in attacked_turns
        # prize taken by end of our NEXT own turn?
        p_before = turn_state[bt]['prizes']
        later = [p for (t, p) in my_prizes_seq if t > bt]
        prize_gain = bool(later) and min(later) < p_before
        boss_res.append((bt, same_turn_attack, prize_gain))

    t2 = my_turns[1] if len(my_turns) >= 2 else None
    bodies_t2 = turn_state[t2]['bodies'] if t2 else None

    me_end = last_cur['players'][manifest_seat] if False else None
    # end state from our seat's last observation
    endme = last_cur['players'][last_cur['yourIndex']] if last_cur and last_cur['yourIndex'] == None else None
    # simpler: walk last_cur assuming our seat's obs
    reason = ''
    if last_cur:
        yi = last_cur['yourIndex']
        m = last_cur['players'][yi]; o2 = last_cur['players'][1 - yi]
        if rew != 1:
            if m.get('deckCount') == 0: reason = 'deck_out'
            elif len(o2.get('prize') or []) == 0: reason = 'prizes'
            elif not any((m.get('active') or []) + (m.get('bench') or [])): reason = 'board_wipe'
            else: reason = 'other'
        else:
            if o2.get('deckCount') == 0: reason = 'win_opp_deck_out'
            elif len(m.get('prize') or []) == 0: reason = 'win_prizes'
            else: reason = 'win_other'

    has_mist = 11 in opp_deck
    has_rock = 20 in opp_deck
    has_articuno = any(a in opp_deck for a in ARTICUNO)

    rows.append(dict(
        ep=epid, win=(rew == 1), first=went_first, opp=opp_arch, reason=reason,
        turns=(last_cur or {}).get('turn'), bodies_t2=bodies_t2,
        ph_total=ph_total, ph_zero=len(ph_zero),
        boss=len(boss_plays), boss_ok=sum(1 for b in boss_res if b[2]),
        boss_atk=sum(1 for b in boss_res if b[1]),
        xerosic=len(xerosic_plays),
        nonattack=len(nonattack), nonattack_no_alk=len(nonattack_no_alk),
        dudun_last=len(dudun_last_body),
        kad_active=len(kadabra_active),
        opp_mist=has_mist, opp_rock=has_rock, opp_articuno=has_articuno,
        opp_lillie=opp_lillie,
        my_deck_end=(last_cur['players'][last_cur['yourIndex']].get('deckCount')
                     if last_cur else None),
        my_prizes_end=(len(last_cur['players'][last_cur['yourIndex']].get('prize') or [])
                       if last_cur else None),
        opp_prizes_end=(len(last_cur['players'][1 - last_cur['yourIndex']].get('prize') or [])
                        if last_cur else None),
    ))
    if dudun_last_body: events['dudun_last'].append((epid, rew == 1, dudun_last_body))
    if ph_zero: events['ph_zero'].append((epid, rew == 1, opp_arch, ph_zero))
    if boss_res: events['boss'].append((epid, rew == 1, opp_arch, boss_res))
    if kadabra_active: events['kadabra_active'].append((epid, rew == 1, opp_arch, kadabra_active))

out = RUN.parent.parent / 'summaries'
out.mkdir(exist_ok=True)
with open(out / 'v4_episode_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- aggregate report ----
W = [r for r in rows]; wins = sum(1 for r in W if r['win'])
print(f"episodes={len(W)} wins={wins} losses={len(W)-wins} wr={wins/len(W):.1%}")
for f_ in (True, False):
    sub = [r for r in W if r['first'] is f_]
    if sub:
        print(f"  {'first' if f_ else 'second'}: {len(sub)} games, "
              f"{sum(1 for r in sub if r['win'])}/{len(sub)} = "
              f"{sum(1 for r in sub if r['win'])/len(sub):.1%}")
print("\n-- by opponent --")
by = defaultdict(list)
for r in W: by[r['opp']].append(r)
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    w2 = sum(1 for r in v if r['win'])
    ff = sum(1 for r in v if r['first'])
    print(f"  {k:32s} {len(v):2d}g {w2:2d}W-{len(v)-w2:2d}L {w2/len(v):5.1%} (first {ff})")
print("\n-- loss reasons --")
print(Counter(r['reason'] for r in W if not r['win']))
print("-- win reasons --")
print(Counter(r['reason'] for r in W if r['win']))
print("\n-- bodies at end of our 2nd turn vs outcome --")
for thr in (5,):
    hi = [r for r in W if (r['bodies_t2'] or 0) >= thr]
    lo = [r for r in W if (r['bodies_t2'] or 0) < thr]
    for name, grp in (('>=5', hi), ('<=4', lo)):
        if grp:
            print(f"  {name}: {len(grp)}g wr={sum(1 for r in grp if r['win'])/len(grp):.1%}")
print("\n-- powerful hand zero-damage --")
print(f"total PH={sum(r['ph_total'] for r in W)}, zero={sum(r['ph_zero'] for r in W)} "
      f"in {sum(1 for r in W if r['ph_zero'])} games")
for e in events['ph_zero']:
    print("  ", e[0], 'WIN' if e[1] else 'LOSS', e[2], f"n={len(e[3])}",
          [(t, NAME.get(cid, cid), 'PROT' if prot else '') for (t, cid, hp, prot) in e[3][:8]])
print("\n-- dudunsparce last-body ability --")
for e in events['dudun_last']:
    print("  ", e[0], 'WIN' if e[1] else 'LOSS', e[2])
print("\n-- boss orders --")
tot = sum(len(e[3]) for e in events['boss'])
ok = sum(1 for e in events['boss'] for b in e[3] if b[2])
atk = sum(1 for e in events['boss'] for b in e[3] if b[1])
print(f"plays={tot} prize_by_next_turn={ok} same_turn_attack={atk}")
for e in events['boss']:
    bad = [b for b in e[3] if not b[2]]
    if bad: print("  no-prize:", e[0], 'WIN' if e[1] else 'LOSS', e[2], bad)
print("\n-- xerosic --")
print(f"plays={sum(r['xerosic'] for r in W)}")
print("\n-- non-attack turns (excl. our T1) --")
lw = [r for r in W if not r['win']]; ww = [r for r in W if r['win']]
print(f"  losses: {sum(r['nonattack'] for r in lw)} turns, "
      f"no-alakazam {sum(r['nonattack_no_alk'] for r in lw)}")
print(f"  wins:   {sum(r['nonattack'] for r in ww)} turns, "
      f"no-alakazam {sum(r['nonattack_no_alk'] for r in ww)}")
print("\n-- kadabra active stuck (no evo in hand, line body on bench) --")
for e in events['kadabra_active'][:15]:
    print("  ", e[0], 'WIN' if e[1] else 'LOSS', e[2], e[3])
print("\n-- opp tech --")
print("  mist:", [(r['ep'], r['win']) for r in W if r['opp_mist']])
print("  rock:", [(r['ep'], r['win']) for r in W if r['opp_rock']])
print("  articuno:", [(r['ep'], r['win']) for r in W if r['opp_articuno']])
print("\n-- deck-out details --")
for r in W:
    if r['reason'] == 'deck_out':
        print("  ", r['ep'], r['opp'], f"ph_zero={r['ph_zero']}",
              f"my_prizes_left={r['my_prizes_end']}")
