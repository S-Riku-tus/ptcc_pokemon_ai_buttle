"""Does the fuel chain actually move in live play?

Reads a self-play trajectory JSONL and reports, per agent version and per own
turn: Darkness left in deck, a Darkness in hand, the attachment taken, Punk Up
search counts, and Adrena-Brain uses."""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'agents' / 'grimmsnarl' / 'grimmsnarl_ml_v5'))
import ml_features as mf

DARK, MUNK = 7, 112
IMPI, MORG, GRIM = 646, 647, 648
MARNIE = {IMPI, MORG, GRIM}
DECK_TOTAL_DARK = 10

path = Path(sys.argv[1] if len(sys.argv) > 1
            else 'data/runs/local_self_play/v5_vs_v4_traj/trajectories.jsonl')

# (version, game, seat) -> {turn -> row}
turns: dict = defaultdict(dict)
counts: dict = defaultdict(Counter)

with path.open(encoding='utf-8') as handle:
    for line in handle:
        rec = json.loads(line)
        version = rec['agent_version'].rsplit('/', 1)[-1]
        obs = rec.get('observation') or {}
        sel = obs.get('select') or {}
        cur = obs.get('current') or {}
        if not sel or not cur:
            continue
        players = cur.get('players') or []
        your = int(cur.get('yourIndex', 0))
        if your >= len(players):
            continue
        me = players[your]
        ctx = int(sel.get('context', -1))
        opts = sel.get('option') or []
        action = rec.get('selected_action') or []
        picks = [a for a in action if isinstance(a, int) and 0 <= a < len(opts)]
        key = (version, rec['game_id'], rec['seat'])

        effect = sel.get('effect')
        effect_id = int(effect.get('id', -1)) if isinstance(effect, dict) else -1
        if ctx == 22 and effect_id == GRIM:
            counts[version]['punk_events'] += 1
            counts[version]['punk_taken'] += len(picks)
            counts[version]['punk_offered'] += int(sel.get('maxCount', 0) or 0)
            counts[version]['punk_took_max'] += int(
                len(picks) == int(sel.get('maxCount', 0) or 0))
            counts[version]['punk_five'] += int(len(picks) >= 5)
        if ctx == 5 and effect_id == 1086:
            counts[version]['poffin_events'] += 1
            counts[version]['poffin_taken'] += len(picks)

        if ctx != 0 or not picks:
            continue
        turn = int(cur.get('turn', -1))
        row = turns[key].setdefault(
            turn, {'offered': False, 'took': False, 'hand_dark': 0,
                   'deck_dark': DECK_TOTAL_DARK, 'brains': 0})
        hand_dark = sum(1 for c in (me.get('hand') or [])
                        if int(c.get('id', -1)) == DARK)
        visible = hand_dark
        for c in mf._in_play(me):
            visible += mf._dark_energy_count(c)
        for c in (me.get('discard') or []):
            if int(c.get('id', -1)) == DARK:
                visible += 1
        row['hand_dark'] = max(row['hand_dark'], hand_dark)
        row['deck_dark'] = max(0, DECK_TOTAL_DARK - visible)
        kinds = [mf.action_type(cur, o, sel) for o in opts]
        if any(k == 'energy' for k in kinds):
            row['offered'] = True
        pick = picks[0]
        if kinds[pick] == 'energy':
            row['took'] = True
        if kinds[pick] == 'ability':
            card = mf.candidate_card(cur, opts[pick], sel) or {}
            if int(card.get('id', -1)) == MUNK:
                row['brains'] += 1

games = defaultdict(set)
for (version, game, seat), rows in turns.items():
    games[version].add((game, seat))
    c = counts[version]
    for row in rows.values():
        c['own_turns'] += 1
        c['offered'] += int(row['offered'])
        c['took'] += int(row['took'])
        c['hand_dark'] += int(row['hand_dark'] > 0)
        c['deck_dark_sum'] += row['deck_dark']
        c['brains'] += row['brains']
        if int(list(rows.keys())[0]) >= 0 and row is not None:
            pass
    for turn, row in rows.items():
        if turn >= 5:
            c['late_turns'] += 1
            c['late_deck_dark_sum'] += row['deck_dark']
            c['late_hand_dark'] += int(row['hand_dark'] > 0)
            c['late_took'] += int(row['took'])

print('{:24s} {:>7s} {:>7s} {:>10s} {:>12s} {:>13s} {:>11s} {:>11s}'.format(
    'agent', 'games', 'turns', 'attach/tn', 'dark-in-hand', 'dark-in-deck',
    'brains/gm', 'punk-taken'))
for version in sorted(counts):
    c = counts[version]
    n = c['own_turns'] or 1
    g = len(games[version]) or 1
    pe = c['punk_events'] or 1
    print('{:24s} {:7d} {:7d} {:9.1f}% {:11.1f}% {:13.2f} {:11.2f} {:11.2f}'.format(
        version, g, c['own_turns'], 100 * c['took'] / n,
        100 * c['hand_dark'] / n, c['deck_dark_sum'] / n,
        c['brains'] / g, c['punk_taken'] / pe))
print()
print('{:24s} {:>10s} {:>13s} {:>13s} {:>12s} {:>12s}'.format(
    'agent', 'punk n', 'took-max', 'five-searches', 'poffin n', 'poffin mean'))
for version in sorted(counts):
    c = counts[version]
    pe = c['punk_events'] or 1
    fe = c['poffin_events'] or 1
    print('{:24s} {:10d} {:12.1f}% {:13d} {:12d} {:12.2f}'.format(
        version, c['punk_events'], 100 * c['punk_took_max'] / pe,
        c['punk_five'], c['poffin_events'], c['poffin_taken'] / fe))
print()
print('turn >= 5 only')
for version in sorted(counts):
    c = counts[version]
    n = c['late_turns'] or 1
    print('{:24s} turns={:5d} attach/turn {:5.1f}%  dark-in-hand {:5.1f}%  dark-in-deck {:.2f}'.format(
        version, c['late_turns'], 100 * c['late_took'] / n,
        100 * c['late_hand_dark'] / n, c['late_deck_dark_sum'] / n))
