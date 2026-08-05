"""ctx21: when the Punk Up trigger itself is still below two Darkness and is on
the menu, do the pilots put the energy on it?"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v5'))
import ml_features as mf

GRIM = 648
DECK_HASH = '9714ab5c3996f6cc'


def scan(payload):
    path, seat, tag = payload
    try:
        rep = json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return Counter()
    steps = rep.get('steps') or []
    c = Counter()
    for i, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[i + 1]):
            continue
        rec = step[seat] or {}
        if rec.get('status') != 'ACTIVE':
            continue
        obs = rec.get('observation') or {}
        sel = obs.get('select') or {}
        if int(sel.get('context', -1)) != 21:
            continue
        eff = sel.get('effect') or {}
        if not isinstance(eff, dict) or int(eff.get('id', -1)) != GRIM:
            continue
        trigger_serial = eff.get('serial')
        cur = obs.get('current') or {}
        opts = sel.get('option') or []
        act = (steps[i + 1][seat] or {}).get('action')
        if not isinstance(act, list) or not act:
            continue
        rows = []
        for slot, o in enumerate(opts):
            card, _, _ = mf.resolve_option(cur, sel, o)
            if not card:
                continue
            rows.append((slot, int(card.get('serial', -2)),
                         mf._dark_energy_count(card), int(card.get('id', -1))))
        trig = next((r for r in rows if r[1] == trigger_serial), None)
        pick = next((r for r in rows if r[0] == act[0]), None)
        if trig is None or pick is None:
            continue
        if trig[2] < 2:
            c['trigger_hungry_offered'] += 1
            c['trigger_hungry_taken'] += int(pick[1] == trig[1])
        if pick[2] >= 4 and any(r[2] < pick[2] for r in rows):
            c['onto_e4plus_with_lower_alt'] += 1
        c['attaches'] += 1
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


if __name__ == '__main__':
    all_jobs = list(jobs()) + list(v4_jobs())
    tags = [j[2] for j in all_jobs]
    totals = defaultdict(Counter)
    with ProcessPoolExecutor() as ex:
        for tag, res in zip(tags, ex.map(scan, all_jobs, chunksize=16)):
            totals[tag].update(res)
    for t in ('top', 'mid', 'low', 'v4'):
        c = totals[t]
        d = c['trigger_hungry_offered'] or 1
        print('{:5s} attaches={:5d}  trigger-below-2-and-offered n={:5d} -> onto trigger {:5.1f}%   onto a body already at 4+ while a lower one was offered: {:4d} ({:.2f}%)'.format(
            t, c['attaches'], c['trigger_hungry_offered'],
            100 * c['trigger_hungry_taken'] / d,
            c['onto_e4plus_with_lower_alt'],
            100 * c['onto_e4plus_with_lower_alt'] / max(1, c['attaches'])))
