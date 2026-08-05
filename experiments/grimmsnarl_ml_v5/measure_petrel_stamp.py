"""Unfair Stamp is an Item that can only be played if one of our Pokemon was
Knocked Out during the opponent's last turn. Do the top pilots search it off
Petrel only when it is playable?"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v5'))
import ml_features as mf

STAMP, PETREL, BOSS = 1080, 1219, 1182
GRIM = 648
DECK_HASH = '9714ab5c3996f6cc'


def nested_id(v):
    if isinstance(v, dict):
        if 'id' in v or 'cardId' in v:
            try:
                return int(v.get('id', v.get('cardId', -1)))
            except Exception:
                return -1
        for x in v.values():
            r = nested_id(x)
            if r >= 0:
                return r
    elif isinstance(v, list):
        for x in v:
            r = nested_id(x)
            if r >= 0:
                return r
    return -1


def scan(payload):
    path, seat, tag = payload
    try:
        rep = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return Counter()
    steps = rep.get('steps') or []
    c = Counter()
    # opponent prize count at the end of each of our turns
    opp_prize_by_turn: dict[int, int] = {}
    for i, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[i + 1]):
            continue
        rec = step[seat] or {}
        if rec.get('status') != 'ACTIVE':
            continue
        obs = rec.get('observation') or {}
        sel = obs.get('select') or {}
        if not sel:
            continue
        cur = obs.get('current') or {}
        players = cur.get('players') or [{}, {}]
        your = int(cur.get('yourIndex', seat))
        if your >= len(players) or len(players) < 2:
            continue
        me, opp = players[your], players[1 - your]
        turn = int(cur.get('turn', -1))
        opp_prize = len(opp.get('prize') or [])
        # the earliest reading inside this turn is the state we inherited
        if turn not in opp_prize_by_turn:
            opp_prize_by_turn[turn] = opp_prize

        if int(sel.get('context', -1)) != 7:
            continue
        if nested_id(sel.get('effect')) != PETREL:
            continue
        opts = sel.get('option') or []
        act = (steps[i + 1][seat] or {}).get('action')
        if not isinstance(act, list):
            continue
        ids = []
        for o in opts:
            card, _, _ = mf.resolve_option(cur, sel, o)
            ids.append(int((card or {}).get('id', -1)))
        picked = [ids[a] for a in act if isinstance(a, int) and 0 <= a < len(ids)]
        hand = Counter()
        for x in (me.get('hand') or []):
            try:
                hand[int(x.get('id', -1))] += 1
            except Exception:
                pass
        if STAMP not in ids or hand[STAMP]:
            continue
        # Did they take a prize since our previous turn?  Prizes are only taken
        # on a knockout, so that is exactly "one of ours was Knocked Out".
        previous = [t for t in opp_prize_by_turn if t < turn]
        prior = opp_prize_by_turn[max(previous)] if previous else 6
        ko_last_turn = opp_prize < prior
        key = 'playable' if ko_last_turn else 'dead'
        took = int(STAMP in picked)
        c['stamp_off_' + key] += 1
        c['stamp_take_' + key] += took
        if BOSS in ids:
            c['boss_off_' + key] += 1
            c['boss_take_' + key] += int(BOSS in picked)
    return c


def jobs():
    index = ROOT / 'data' / 'kaggle_grimmsnarl_top50' / 'indexes' / 'replay_index.csv'
    base = ROOT / 'data' / 'kaggle_grimmsnarl_top50'
    seen = set()
    for r in csv.DictReader(open(index, encoding='utf-8-sig')):
        if r['deck_hash'] != DECK_HASH or r['download_status'] != 'success':
            continue
        key = (r['episode_id'], r['seat_index'])
        if key in seen:
            continue
        seen.add(key)
        p = base / Path(r['replay_path'].replace(chr(92), '/'))
        if p.exists():
            score = float(r['submission_score'])
            tag = 'top' if score >= 1100 else ('mid' if score >= 1060 else 'low')
            yield (str(p), int(r['seat_index']), tag)


def v4_jobs():
    run = ROOT / 'data' / 'runs' / 'grimmsnarl' / '20260805_grimmsnarl_ml_v4_sub55253296'
    sub = '55253296'
    for r in csv.DictReader(open(run / 'episodes.csv', encoding='utf-8-sig')):
        seat = 0 if r['agent_0_submission_id'] == sub else 1
        p = run / 'episodes' / r['episode_id'] / 'replay' / 'episode_{}.json'.format(r['episode_id'])
        if p.exists():
            yield (str(p), seat, 'v4')


def pct(c, a, b):
    return '{:5.1f}% ({:4d}/{:4d})'.format(100 * c[a] / c[b], c[a], c[b]) if c[b] else '     -       '


if __name__ == '__main__':
    all_jobs = list(jobs()) + list(v4_jobs())
    tags = [j[2] for j in all_jobs]
    totals = defaultdict(Counter)
    with ProcessPoolExecutor() as ex:
        for tag, res in zip(tags, ex.map(scan, all_jobs, chunksize=16)):
            totals[tag].update(res)
    print('Petrel, a fresh Unfair Stamp offered; "playable" = they took a prize '
          'since our previous turn')
    for t in ('top', 'mid', 'low', 'v4'):
        c = totals[t]
        print('{:5s} stamp taken: playable {}  unplayable {}   |  boss taken: playable {}  unplayable {}'.format(
            t,
            pct(c, 'stamp_take_playable', 'stamp_off_playable'),
            pct(c, 'stamp_take_dead', 'stamp_off_dead'),
            pct(c, 'boss_take_playable', 'boss_off_playable'),
            pct(c, 'boss_take_dead', 'boss_off_dead')))
