"""Summarise the deck archetype distribution of the current Kaggle top N.

Reads the per-submission bookkeeping written by
``scripts/collect_top100_submission_replays.py`` (``submissions/<id>/``),
extracts the 60-card list that each leaderboard team actually brought (the
target seat's opening deck from the replay header), maps card IDs to names
via ``vendor/cg/cards.json``, and reports how many teams play each deck list
and archetype.

The per-submission files are used rather than ``indexes/episodes.csv``
because that consolidated index is only written when every submission in the
run succeeds; a single Kaggle 429 drops the whole index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.core.replay_io import deck_hash  # noqa: E402
from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

POKEMON_CARD_TYPE = 0
MIN_DECK_SIZE = 40


def load_cards(path: Path) -> dict[int, dict[str, Any]]:
    cards = json.loads(path.read_text(encoding="utf-8"))
    return {int(card["cardId"]): card for card in cards}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def seat_for_submission(
    episode: dict[str, Any],
    submission_id: int,
) -> int | None:
    for seat in (0, 1):
        value = str(episode.get(f"agent_{seat}_submission_id", "")).strip()
        if value == str(submission_id):
            return seat
    return None


def deck_from_episode(
    root: Path,
    episode: dict[str, Any],
    submission_id: int,
) -> tuple[list[int], str]:
    """Return the target seat's opening 60-card list for one episode."""
    seat = seat_for_submission(episode, submission_id)
    if seat is None:
        return [], "seat_index_unknown"

    episode_id = episode.get("episode_id")
    replay_path = root / "replays" / f"episode_{episode_id}.json"
    if not replay_path.exists():
        return [], "replay_missing"

    try:
        header = extract_fast_header_from_file(replay_path)
    except Exception as exc:  # noqa: BLE001
        return [], f"header_error: {type(exc).__name__}: {exc}"

    decks = header.get("decks") or [[], []]
    deck = list(decks[seat]) if len(decks) > seat else []
    if len(deck) < MIN_DECK_SIZE:
        return [], f"deck_too_short: {len(deck)}"
    return deck, ""


def pokemon_lines(
    deck: list[int],
    cards: dict[int, dict[str, Any]],
) -> list[tuple[str, int, int]]:
    """Return (name, count, hp) for every Pokemon in the deck, best first."""
    rows: list[tuple[str, int, int]] = []
    for card_id, count in Counter(deck).items():
        card = cards.get(int(card_id))
        if card is None or card.get("cardType") != POKEMON_CARD_TYPE:
            continue
        name = str(card.get("name", card_id))
        rows.append((name, count, int(card.get("hp") or 0)))
    rows.sort(key=lambda item: (-item[2], -item[1], item[0]))
    return rows


def other_lines(
    deck: list[int],
    cards: dict[int, dict[str, Any]],
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for card_id, count in Counter(deck).items():
        card = cards.get(int(card_id))
        if card is None or card.get("cardType") == POKEMON_CARD_TYPE:
            continue
        rows.append((str(card.get("name", card_id)), count))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def archetype_label(
    deck: list[int],
    cards: dict[int, dict[str, Any]],
    depth: int = 2,
) -> str:
    """Name a deck after its highest-HP Pokemon (the likely main attacker)."""
    lines = pokemon_lines(deck, cards)
    if not lines:
        return "unknown"
    return " / ".join(name for name, _count, _hp in lines[:depth])


def deck_by_team_name(root: Path, episode_id: Any, name: str) -> list[int]:
    """Return the deck played by ``name`` in one replay, or []."""
    path = root / "replays" / f"episode_{episode_id}.json"
    if not path.exists():
        return []
    try:
        header = extract_fast_header_from_file(path)
    except Exception:  # noqa: BLE001
        return []
    decks = header.get("decks") or [[], []]
    for seat, team_name in enumerate(header.get("team_names") or []):
        if str(team_name) != name or seat >= len(decks):
            continue
        deck = list(decks[seat])
        if len(deck) >= MIN_DECK_SIZE:
            return deck
    return []


def decks_by_team_name(root: Path) -> dict[str, list[list[int]]]:
    """Map replay team name -> every distinct deck it played on disk.

    Kaggle's EpisodeService rate-limits hard, so ``episodes.json`` is often
    incomplete. The replay header carries ``TeamNames`` per seat, which
    recovers a team's deck without further API calls -- but a team running
    several submissions on the ladder at once shows up with several decks,
    so callers must treat multi-deck names as ambiguous.
    """
    found: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for path in sorted((root / "replays").glob("episode_*.json")):
        try:
            header = extract_fast_header_from_file(path)
        except Exception:  # noqa: BLE001
            continue
        decks = header.get("decks") or [[], []]
        for seat, team_name in enumerate(header.get("team_names") or []):
            if not team_name or seat >= len(decks):
                continue
            deck = list(decks[seat])
            if len(deck) >= MIN_DECK_SIZE:
                found[str(team_name)][deck_hash(deck)] = deck
    return {name: list(decks.values()) for name, decks in found.items()}


def collect_teams(
    root: Path,
    cards: dict[int, dict[str, Any]],
    label_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    teams: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    name_index = decks_by_team_name(root)

    for submission_dir in sorted((root / "submissions").iterdir()):
        meta_path = submission_dir / "submission.json"
        episodes_path = submission_dir / "episodes.json"
        if not meta_path.exists():
            continue

        meta = read_json(meta_path)
        submission_id = int(meta["submission_id"])
        team_name = str(meta.get("team_name", ""))
        rank_text = str(meta.get("leaderboard_rank", "")).strip()
        rank = int(rank_text) if rank_text.isdigit() else 0

        episodes: list[dict[str, Any]] = []
        if episodes_path.exists():
            episodes = list(read_json(episodes_path).get("episodes") or [])

        deck: list[int] = []
        reason = "no_episodes"
        source = "episode_seat"

        # 1. Authoritative: episodes.json carries the per-seat submission ID.
        for episode in episodes:
            deck, reason = deck_from_episode(root, episode, submission_id)
            if deck:
                break

        # 2. This submission's own replay index, seat picked by team name.
        if not deck:
            index_path = submission_dir / "replays" / "index.json"
            refs = read_json(index_path) if index_path.exists() else []
            for ref in refs:
                deck = deck_by_team_name(
                    root, ref.get("episode_id"), team_name
                )
                if deck:
                    reason = ""
                    source = "submission_replay_index"
                    break

        # 3. Any replay on disk -- only safe when the name played one deck.
        if not deck:
            candidates = name_index.get(team_name, [])
            if len(candidates) == 1:
                deck = candidates[0]
                reason = ""
                source = "team_name_match"
            elif len(candidates) > 1:
                reason = f"ambiguous_team_name: {len(candidates)} decks"

        if not deck:
            problems.append(
                {
                    "rank": str(rank),
                    "team_name": team_name,
                    "submission_id": str(submission_id),
                    "reason": reason,
                }
            )
            continue

        teams.append(
            {
                "rank": rank,
                "team_name": team_name,
                "submission_id": submission_id,
                "score": str(meta.get("public_score", "")),
                "deck": deck,
                "deck_hash": deck_hash(deck),
                "deck_source": source,
                "archetype": archetype_label(deck, cards, label_depth),
            }
        )

    teams.sort(key=lambda item: item["rank"])
    problems.sort(key=lambda item: int(item["rank"] or 0))
    return teams, problems


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data" / "kaggle_top50_meta",
        help="Collection root from collect_top100_submission_replays.py.",
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=ROOT / "vendor" / "cg" / "cards.json",
        help="Card metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory. Default: <root>/analysis.",
    )
    parser.add_argument(
        "--label-depth",
        type=int,
        default=2,
        help="How many top-HP Pokemon to use in the archetype label.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    cards = load_cards(args.cards)

    teams, problems = collect_teams(root, cards, args.label_depth)
    if not teams:
        raise SystemExit(f"No decks could be extracted under {root}")

    by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_deck_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for team in teams:
        by_archetype[team["archetype"]].append(team)
        by_deck_hash[team["deck_hash"]].append(team)

    output_dir = args.output_dir or (root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    team_fields = [
        "rank",
        "team_name",
        "submission_id",
        "score",
        "archetype",
        "deck_hash",
        "deck_source",
    ]
    write_csv(
        output_dir / "teams.csv",
        [{field: team[field] for field in team_fields} for team in teams],
        team_fields,
    )

    archetype_rows: list[dict[str, Any]] = []
    for name, group in sorted(
        by_archetype.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        archetype_rows.append(
            {
                "archetype": name,
                "teams": len(group),
                "share_pct": round(100.0 * len(group) / len(teams), 1),
                "best_rank": min(team["rank"] for team in group),
                "distinct_deck_lists": len(
                    {team["deck_hash"] for team in group}
                ),
                "ranks": " ".join(str(team["rank"]) for team in group),
            }
        )
    write_csv(
        output_dir / "archetypes.csv",
        archetype_rows,
        [
            "archetype",
            "teams",
            "share_pct",
            "best_rank",
            "distinct_deck_lists",
            "ranks",
        ],
    )

    deck_lists = []
    for key, group in sorted(
        by_deck_hash.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        sample = group[0]["deck"]
        deck_lists.append(
            {
                "deck_hash": key,
                "teams": len(group),
                "ranks": [team["rank"] for team in group],
                "team_names": [team["team_name"] for team in group],
                "archetype": group[0]["archetype"],
                "pokemon": [
                    {"name": name, "count": count, "hp": hp}
                    for name, count, hp in pokemon_lines(sample, cards)
                ],
                "other_cards": [
                    {"name": name, "count": count}
                    for name, count in other_lines(sample, cards)
                ],
            }
        )

    detail = {
        "root": str(root),
        "teams_analysed": len(teams),
        "teams_missing": len(problems),
        "distinct_deck_lists": len(by_deck_hash),
        "archetypes": archetype_rows,
        "deck_lists": deck_lists,
        "problems": problems,
    }
    (output_dir / "deck_distribution.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Teams analysed: {len(teams)} (missing: {len(problems)})")
    print(f"Distinct deck lists: {len(by_deck_hash)}")
    print()
    print(f"{'teams':>5} {'%':>6} {'best':>5}  archetype")
    for row in archetype_rows:
        print(
            f"{row['teams']:>5} {row['share_pct']:>6} "
            f"{row['best_rank']:>5}  {row['archetype']}"
        )
    if problems:
        print()
        print("Missing decks:")
        for problem in problems:
            print(
                f"  rank {problem['rank']:>3} "
                f"{problem['team_name']}: {problem['reason']}"
            )
    print()
    print(f"Report: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
