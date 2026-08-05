"""Every multi-pick select the rule policy still owns: how many do the pilots
actually take, and how many does v4 take?"""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v4'))
import ml_features as mf

DECK_HASH = '9714ab5c3996f6cc'
ROUTED = {0, 1, 3, 4, 5, 7, 8, 13, 15, 16, 21, 37, 38, 40, 41, 43}


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
        opts = sel.get('option') or []
        act = (steps[i + 1][seat] or {}).get('action')
        if not isinstance(act, list):
            continue
        ctx = int(sel.get('context', -1))
        lo = int(sel.get('minCount', 0) or 0)
        hi = int(sel.get('maxCount', 0) or 0)
        if hi <= 1 or lo == hi:
            continue  # single pick, or a forced count with no decision
        eff = nested_id(sel.get('effect'))
        took = len([a for a in act if isinstance(a, int) and 0 <= a < len(opts)])
        key = 'c{}_e{}_max{}'.format(ctx, eff, hi)
        c[key + '_n'] += 1
        c[key + '_sum'] += took
        c[key + '_max'] += int(took == hi)
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
    keys = sorted({k[:-2] for k in totals['top'] if k.endswith('_n')},
                  key=lambda k: -totals['top'][k + '_n'])
    print('{:24s} {:>28s} {:>28s}'.format('select', 'top (n, mean, took-max)', 'v4 (n, mean, took-max)'))
    for k in keys:
        row = '{:24s}'.format(k)
        for t in ('top', 'mid', 'low', 'v4'):
            c = totals[t]
            n = c[k + '_n']
            if not n:
                row += ' {:>28s}'.format('-')
                continue
            row += ' {:>28s}'.format('{}: {:5d} {:.2f} {:5.1f}%'.format(
                t, n, c[k + '_sum'] / n, 100 * c[k + '_max'] / n))
        print(row)
