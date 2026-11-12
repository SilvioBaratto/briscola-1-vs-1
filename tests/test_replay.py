"""Replaying a saved game with the visualization, without opening a browser."""

import pytest

pytest.importorskip("playwright")

from visualization.render_game import BriscolaGameSimulator  # noqa: E402


def test_replay_redraws_the_saved_game_to_the_same_score(make_game):
    """The simulator follows the recorded deck, opener and moves and ends on the saved result."""
    for index in (0, 1):
        record = make_game(index)
        simulator = BriscolaGameSimulator(replay=record)
        winner = simulator.simulate()
        assert winner == {"agent": "RL Agent", "llm": "LLM", "tie": "Tie"}[record.winner]
        assert simulator.env.state.player_score == record.agent_score
        assert simulator.env.state.opponent_score == record.llm_score
        assert simulator.frames[0].rl_player.is_leading is record.agent_opens
        assert simulator.frames[-1].phase == "game_over"


def test_replay_refuses_a_record_that_no_longer_adds_up(make_game):
    """If the rules drift from what was saved, the replay fails loudly rather than show a wrong game."""
    record = make_game(0)
    record.tricks[-1].agent_score += 1
    with pytest.raises(ValueError):
        BriscolaGameSimulator(replay=record).simulate()
