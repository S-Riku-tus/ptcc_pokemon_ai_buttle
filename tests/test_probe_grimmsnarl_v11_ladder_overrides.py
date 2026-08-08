from scripts.probe_grimmsnarl_v11_ladder_overrides import legal_replay_action


def test_legal_replay_action_preserves_multi_pick() -> None:
    assert legal_replay_action([0, 2, 4], 5) == [0, 2, 4]
    assert legal_replay_action([], 5) == []


def test_legal_replay_action_rejects_partial_or_invalid_selection() -> None:
    assert legal_replay_action([0, 5], 5) is None
    assert legal_replay_action([0, "2"], 5) is None
