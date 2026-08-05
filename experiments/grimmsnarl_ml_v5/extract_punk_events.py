"""How many Dark Energy do the teachers actually search off Punk Up, and what
board fact predicts it?  Emits one row per ctx22 (ATTACH_TO / effect Grimmsnarl)
event over the same-60 corpus, plus the v4 run for comparison."""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v4'))
import ml_features as mf

DARK, MUNK, IMPI, MORG, GRIM = 7, 112, 646, 647, 648
RARE_CANDY, NIGHT_STRETCHER = 1079, 1097
MARNIE = {IMPI, MORG, GRIM}
DECK_HASH = '9714ab5c3996f6cc'


def nested_card(v):
    if isinstance(v, dict):
        if 'id' in v or 'cardId' in v:
            try:
                cid = int(v.get('id', v.get('cardId', -1)))
            except Exception:
                cid = -1
            try:
                ser = int(v.get('serial', -1))
            except Exception:
                ser = -1
            return cid, ser
        for x in v.values():
            r = nested_card(x)
            if r[0] >= 0:
                return r
    elif isinstance(v, list):
        for x in v:
            r = nested_card(x)
            if r[0] >= 0:
                return r
    return -1, -1


def scan(payload):
    path, seat, tag = payload
    try:
        rep = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return []
    steps = rep.get('steps') or []
    out = []
    for i, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[i + 1]):
            continue
        rec = step[seat] or {}
        if rec.get('status') != 'ACTIVE':
            continue
        obs = rec.get('observation') or {}
        sel = obs.get('select') or {}
        if int(sel.get('context', -1)) != 22:
            continue
        eff_id, eff_serial = nested_card(sel.get('effect'))
        if eff_id != GRIM:
            continue
        cur = obs.get('current') or {}
        players = cur.get('players') or [{}, {}]
        your = int(cur.get('yourIndex', seat))
        if your >= len(players):
            continue
        me = players[your]
        opp = players[1 - your] if len(players) > 1 else {}
        opts = sel.get('option') or []
        act = (steps[i + 1][seat] or {}).get('action')
        if not isinstance(act, list):
            continue
        searched = len([a for a in act if isinstance(a, int) and 0 <= a < len(opts)])
        maximum = int(sel.get('maxCount', 0) or 0)

        active_card = (mf._cards(me, 'active') or [None])[0]
        active_serial = -1 if active_card is None else int(active_card.get('serial', -1))
        board = []
        for c in mf._in_play(me):
            cid = int(c.get('id', -1))
            if cid not in MARNIE:
                continue
            board.append({
                'id': cid,
                'serial': int(c.get('serial', -1)),
                'e': mf._dark_energy_count(c),
                'active': int(int(c.get('serial', -1)) == active_serial),
            })
        hand = Counter()
        for c in (me.get('hand') or []):
            try:
                hand[int(c.get('id', -1))] += 1
            except Exception:
                pass
        trigger = next((b for b in board if b['serial'] == eff_serial), None)
        out.append({
            'tag': tag,
            'turn': int(cur.get('turn', -1)),
            'searched': searched,
            'max': maximum,
            'trigger_e': -1 if trigger is None else trigger['e'],
            'trigger_active': -1 if trigger is None else trigger['active'],
            'board': board,
            'hand_candy': hand[RARE_CANDY],
            'hand_grim': hand[GRIM],
            'hand_morg': hand[MORG],
            'hand_dark': hand[DARK],
            'deck': int(me.get('deckCount', 0) or 0),
            'mirror': int(any(int(c.get('id', -1)) in MARNIE for c in mf._in_play(opp))),
        })
    return out


def jobs():
    index = ROOT / 'data' / 'kaggle_grimmsnarl_top50' / 'indexes' / 'replay_index.csv'
    rows = list(csv.DictReader(open(index, encoding='utf-8-sig')))
    base = ROOT / 'data' / 'kaggle_grimmsnarl_top50'
    seen = set()
    for r in rows:
        if r['deck_hash'] != DECK_HASH or r['download_status'] != 'success':
            continue
        key = (r['episode_id'], r['seat_index'])
        if key in seen:
            continue
        seen.add(key)
        p = base / Path(r['replay_path'].replace(chr(92), '/'))
        if p.exists():
            yield (str(p), int(r['seat_index']), 't{}|{}'.format(r['team_id'], r['submission_score']))


def v4_jobs():
    run = ROOT / 'data' / 'runs' / 'grimmsnarl' / '20260805_grimmsnarl_ml_v4_sub55253296'
    sub = '55253296'
    for r in csv.DictReader(open(run / 'episodes.csv', encoding='utf-8-sig')):
        seat = 0 if r['agent_0_submission_id'] == sub else 1
        p = run / 'episodes' / r['episode_id'] / 'replay' / 'episode_{}.json'.format(r['episode_id'])
        if p.exists():
            yield (str(p), seat, 'v4')


if __name__ == '__main__':
    all_jobs = list(jobs()) + list(v4_jobs())
    print('replays', len(all_jobs), flush=True)
    events = []
    with ProcessPoolExecutor() as ex:
        for res in ex.map(scan, all_jobs, chunksize=16):
            events.extend(res)
    print('punk events', len(events))
    Path('.tmp/_v5_punk_events.json').write_text(json.dumps(events), encoding='utf-8')
