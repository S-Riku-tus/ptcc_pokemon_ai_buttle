"""Pin the count rule: searched = min(max, d_trigger + hungry_other_bodies), floor 2."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

IMPI, MORG, GRIM = 646, 647, 648
events = json.loads(Path('.tmp/_v5_punk_events.json').read_text(encoding='utf-8'))


def tier(tag):
    if tag == 'v4':
        return 'v4'
    score = float(tag.split('|')[1])
    return 'elite' if score >= 1120 else ('mid' if score >= 1070 else 'low')


def feats(ev):
    trig_e = ev['trigger_e']
    board = list(ev['board'])
    # drop one copy of the trigger from the "other" list
    others = []
    dropped = False
    for b in board:
        if not dropped and b['id'] == GRIM and b['e'] == trig_e:
            dropped = True
            continue
        others.append(b)
    hungry = [b for b in others if b['e'] < 2]
    evolvable_hungry = []
    for b in hungry:
        if b['id'] == GRIM:
            evolvable_hungry.append(b)
        elif b['id'] == MORG and ev['hand_grim'] > 0:
            evolvable_hungry.append(b)
        elif b['id'] == IMPI and ((ev['hand_grim'] > 0 and ev['hand_candy'] > 0) or ev['hand_morg'] > 0):
            evolvable_hungry.append(b)
    return {
        'd_trigger': max(0, 2 - trig_e) if trig_e >= 0 else 2,
        'n_hungry': len(hungry),
        'n_evolvable': len(evolvable_hungry),
        'max': ev['max'],
        's': ev['searched'],
    }


rows = defaultdict(list)
for ev in events:
    rows[tier(ev['tag'])].append(feats(ev))


def evaluate(name, fn):
    print('rule:', name)
    for t in ('elite', 'mid', 'low', 'v4'):
        diffs = Counter()
        exact = over = under = 0
        pred_sum = 0
        for f in rows[t]:
            p = fn(f)
            pred_sum += p
            diffs[f['s'] - p] += 1
            exact += int(f['s'] == p)
            over += int(p > f['s'])
            under += int(p < f['s'])
        n = len(rows[t]) or 1
        print('  {:6s} n={:4d} exact {:5.1f}%  within1 {:5.1f}%  mean_pred {:.2f} vs actual {:.2f}'.format(
            t, n, 100 * exact / n,
            100 * (diffs[0] + diffs[1] + diffs[-1]) / n,
            pred_sum / n, mean(f['s'] for f in rows[t])))
    print()


evaluate('F: min(max, max(2, d_trigger + n_hungry))',
         lambda f: min(f['max'], max(2, f['d_trigger'] + f['n_hungry'])))
evaluate('F2: min(max, max(2, d_trigger + n_evolvable))',
         lambda f: min(f['max'], max(2, f['d_trigger'] + f['n_evolvable'])))
evaluate('G: min(max, max(2, d_trigger + 2*n_hungry))  [2 each]',
         lambda f: min(f['max'], max(2, f['d_trigger'] + 2 * f['n_hungry'])))
evaluate('H: always max (v4 today)', lambda f: f['max'])
evaluate('I: min(max, 2)', lambda f: min(f['max'], 2))
evaluate('J: min(max, max(2, d_trigger + n_hungry + 1))',
         lambda f: min(f['max'], max(2, f['d_trigger'] + f['n_hungry'] + 1)))

print('=== how often the elite exceed rule F, by n_hungry ===')
tab = defaultdict(Counter)
for f in rows['elite']:
    p = min(f['max'], max(2, f['d_trigger'] + f['n_hungry']))
    tab[(f['d_trigger'], f['n_hungry'])][f['s'] - p] += 1
for key in sorted(tab):
    c = tab[key]
    n = sum(c.values())
    if n < 20:
        continue
    print('  d_trigger={} n_hungry={}  n={:4d}  diff {}'.format(key[0], key[1], n, dict(sorted(c.items()))))
