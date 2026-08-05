"""Does stripping the deck with Punk Up starve the hand attachment?

Per own turn: was a Dark Energy in hand, was the once-per-turn attachment made,
and how many Dark Energy are left in the deck."""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v4'))
import ml_features as mf

DARK, MUNK = 7, 112
IMPI, MORG, GRIM = 646, 647, 648
MARNIE = {IMPI, MORG, GRIM}
DECK_TOTAL_DARK = 10
DECK_HASH = '9714ab5c3996f6cc'


def scan(payload):
    path, seat, tag = payload
    try:
        rep = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return Counter()
    steps = rep.get('steps') or []
    c = Counter()
    turns = {}
    for i, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[i + 1]):
            continue
        rec = step[seat] or {}
        if rec.get('status') != 'ACTIVE':
            continue
        obs = rec.get('observation') or {}
        sel = obs.get('select') or {}
        if int(sel.get('context', -1)) != 0:
            continue
        cur = obs.get('current') or {}
        players = cur.get('players') or [{}, {}]
        your = int(cur.get('yourIndex', seat))
        if your >= len(players) or len(players) < 2:
            continue
        me = players[your]
        opts = sel.get('option') or []
        act = (steps[i + 1][seat] or {}).get('action')
        if not isinstance(act, list) or not act:
            continue
        pick = act[0]
        if not 0 <= pick < len(opts):
            continue
        turn = int(cur.get('turn', -1))
        row = turns.setdefault(turn, {'offered': False, 'took': False,
                                      'hand_dark': 0, 'deck_dark': 0, 'seen': 0})
        hand_dark = sum(1 for x in (me.get('hand') or [])
                        if int(x.get('id', -1)) == DARK)
        # Dark energy still unseen (deck + prizes): total minus what is visible.
        visible = hand_dark
        for x in mf._in_play(me):
            visible += mf._dark_energy_count(x)
        for x in (me.get('discard') or []):
            if int(x.get('id', -1)) == DARK:
                visible += 1
        row['hand_dark'] = max(row['hand_dark'], hand_dark)
        row['deck_dark'] = DECK_TOTAL_DARK - visible
        kinds = [mf.action_type(cur, o, sel) for o in opts]
        if any(k == 'energy' for k in kinds):
            row['offered'] = True
        if kinds[pick] == 'energy':
            row['took'] = True
    for turn, row in turns.items():
        c['own_turns'] += 1
        c['turns_attach_offered'] += int(row['offered'])
        c['turns_attach_taken'] += int(row['took'])
        c['turns_hand_dark'] += int(row['hand_dark'] > 0)
        c['deck_dark_sum'] += max(0, row['deck_dark'])
        if turn >= 5:
            c['late_turns'] += 1
            c['late_deck_dark_sum'] += max(0, row['deck_dark'])
            c['late_hand_dark'] += int(row['hand_dark'] > 0)
            c['late_attach_taken'] += int(row['took'])
            c['late_attach_offered'] += int(row['offered'])
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
            tag = 'elite' if score >= 1120 else ('mid' if score >= 1070 else 'low')
            yield (str(p), int(r['seat_index']), tag)


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
    tags = [j[2] for j in all_jobs]
    totals = defaultdict(Counter)
    with ProcessPoolExecutor() as ex:
        for tag, res in zip(tags, ex.map(scan, all_jobs, chunksize=16)):
            totals[tag].update(res)
    print('{:6s} {:>8s} {:>12s} {:>12s} {:>14s} {:>16s} {:>16s}'.format(
        '', 'turns', 'attach/turn', 'offered%', 'take|offered', 'dark-in-hand%', 'dark-left-in-deck'))
    for t in ('elite', 'mid', 'low', 'v4'):
        c = totals[t]
        n = c['own_turns'] or 1
        print('{:6s} {:8d} {:11.1f}% {:11.1f}% {:13.1f}% {:15.1f}% {:17.2f}'.format(
            t, c['own_turns'], 100 * c['turns_attach_taken'] / n,
            100 * c['turns_attach_offered'] / n,
            100 * c['turns_attach_taken'] / max(1, c['turns_attach_offered']),
            100 * c['turns_hand_dark'] / n,
            c['deck_dark_sum'] / n))
    print()
    print('turn >= 5 only')
    for t in ('elite', 'mid', 'low', 'v4'):
        c = totals[t]
        n = c['late_turns'] or 1
        print('{:6s} {:8d} {:11.1f}% {:11.1f}% {:13.1f}% {:15.1f}% {:17.2f}'.format(
            t, c['late_turns'], 100 * c['late_attach_taken'] / n,
            100 * c['late_attach_offered'] / n,
            100 * c['late_attach_taken'] / max(1, c['late_attach_offered']),
            100 * c['late_hand_dark'] / n,
            c['late_deck_dark_sum'] / n))
