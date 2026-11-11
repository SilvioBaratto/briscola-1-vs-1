"""The 200-game evaluation: game logs, fixed decks, saving and resuming."""

import json

import pytest

from src.agents.llm_opponent import OllamaUnavailableError
from src.cards import Card, Deck
from src.game_record import AGENT, LLM
from tests.helpers import ScriptedOpponent
from training import evaluate
from training.evaluate import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_PAUSED,
    EvaluationConfig,
    game_path,
    run_evaluation,
)


def test_a_game_is_a_complete_consistent_log(make_game):
    """Twenty tricks, every card once, 120 points, and the trick winner leads the next trick."""
    record = make_game(4)
    assert len(record.tricks) == 20
    played = [Card.from_string(t.agent_card) for t in record.tricks] + [Card.from_string(t.llm_card) for t in record.tricks]
    assert len({c.to_string() for c in played}) == 40
    assert record.agent_score + record.llm_score == 120
    assert record.tricks[0].leader == (AGENT if record.agent_opens else LLM)
    for before, after in zip(record.tricks, record.tricks[1:]):
        assert after.leader == before.winner
    for trick in record.tricks:
        assert trick.agent_hand[trick.agent_index] == trick.agent_card
        assert trick.llm_hand[trick.llm_index] == trick.llm_card


def test_the_deck_and_the_opener_follow_the_game_number(make_game):
    """Even games are opened by the agent; game i uses seed + i; the briscola is the bottom card."""
    for index in (0, 1, 2, 3):
        record = make_game(index, seed=100)
        assert record.agent_opens is (index % 2 == 0)
        assert record.deck_seed == 100 + index
        assert record.deck_order == [c.to_string() for c in Deck.shuffled_cards(100 + index)]
        assert record.briscola == record.deck_order[0]


def test_the_same_game_is_replayed_identically(make_game):
    """Fixed decks and a deterministic agent make a game repeatable when the opponent is."""
    first = make_game(5)
    second = make_game(5)
    assert [t.llm_card for t in first.tricks] == [t.llm_card for t in second.tricks]
    assert first.to_dict()["tricks"][0]["agent_hand"] == second.to_dict()["tricks"][0]["agent_hand"]


def test_unusable_llm_answers_are_counted_per_game(make_game):
    """Moves the opponent reports as invalid end up in the record."""
    record = make_game(0, ScriptedOpponent(invalid_every=4))
    assert record.llm_invalid_moves == 5
    assert [t.llm_valid for t in record.tricks].count(False) == 5


def config(tmp_path, checkpoint, **overrides):
    """Build an evaluation config writing into a temporary results folder."""
    settings = dict(checkpoint=checkpoint, num_games=4, seed=7, results_dir=str(tmp_path / "results"))
    settings.update(overrides)
    return EvaluationConfig(**settings)


def test_interrupted_run_resumes_from_the_first_missing_game(tmp_path, checkpoint):
    """Ollama stops during game 2: games 0 and 1 stay on disk, the reason is written, and a rerun finishes only the rest."""
    results = tmp_path / "results"
    # 20 LLM moves per game: the 41st call is the first move of game 2.
    flaky = ScriptedOpponent(fail_on_call=(41, OllamaUnavailableError("ConnectError: refused")))
    assert run_evaluation(config(tmp_path, checkpoint), lambda timeout: flaky) == EXIT_INTERRUPTED

    assert sorted(p.name for p in (results / "partite").glob("*.json")) == ["partita_000.json", "partita_001.json"]
    status = json.loads((results / "stato.json").read_text(encoding="utf-8"))
    assert (status["games_done"], status["games_expected"], status["status"]) == (2, 4, "interrupted")
    assert "refused" in status["reason"]

    kept = {i: game_path(results, i).read_bytes() for i in (0, 1)}
    healthy = ScriptedOpponent()
    assert run_evaluation(config(tmp_path, checkpoint), lambda timeout: healthy) == EXIT_OK
    assert healthy.calls == 40
    assert all(game_path(results, i).read_bytes() == kept[i] for i in (0, 1))
    assert json.loads((results / "stato.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_a_deleted_game_is_the_only_one_replayed(tmp_path, checkpoint):
    """The run restarts from the first missing game, not from the last file."""
    results = tmp_path / "results"
    assert run_evaluation(config(tmp_path, checkpoint), lambda timeout: ScriptedOpponent()) == EXIT_OK
    game_path(results, 1).unlink()
    opponent = ScriptedOpponent()
    assert run_evaluation(config(tmp_path, checkpoint), lambda timeout: opponent) == EXIT_OK
    assert opponent.calls == 20
    assert game_path(results, 1).exists()


def test_a_half_written_game_is_played_again(tmp_path, checkpoint):
    """A truncated file does not count as a finished game."""
    results = tmp_path / "results"
    run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent())
    game_path(results, 1).write_text('{"format": "briscola-game/v', encoding="utf-8")
    opponent = ScriptedOpponent()
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: opponent) == EXIT_OK
    assert opponent.calls == 20


def test_time_limit_pauses_between_games(tmp_path, checkpoint):
    """A zero-minute budget plays nothing and says it paused, so the run can be continued."""
    code = run_evaluation(config(tmp_path, checkpoint, max_minutes=0), lambda timeout: ScriptedOpponent())
    assert code == EXIT_PAUSED
    status = json.loads((tmp_path / "results" / "stato.json").read_text(encoding="utf-8"))
    assert status["status"] == "paused" and status["games_done"] == 0


def test_results_of_another_experiment_are_never_mixed_in(tmp_path, checkpoint):
    """Resuming with a different seed is refused instead of blending two runs."""
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent()) == EXIT_OK
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2, seed=8), lambda timeout: ScriptedOpponent()) == EXIT_ERROR


def test_missing_checkpoint_is_an_error_not_a_crash(tmp_path):
    """A wrong checkpoint path ends the run with an error code."""
    assert run_evaluation(config(tmp_path, str(tmp_path / "nope.pt")), lambda timeout: ScriptedOpponent()) == EXIT_ERROR


def test_checkpoint_path_in_the_run_file_is_not_personal(tmp_path, checkpoint):
    """Only a repository-relative path or a bare file name is written to committed files."""
    run_evaluation(config(tmp_path, checkpoint, num_games=1), lambda timeout: ScriptedOpponent())
    run = json.loads((tmp_path / "results" / "run.json").read_text(encoding="utf-8"))
    assert run["checkpoint"] == "agent.pt"
    assert run["llm_model"] == "gemma4:31b-cloud"
    assert evaluate.PROTOCOL in run["protocol"]


def test_hands_shrink_as_the_deck_runs_out(make_game):
    """Sanity check on the log: hands shrink by one card per trick until the deck runs out."""
    record = make_game(2)
    assert len(record.tricks[0].agent_hand) == 3
    assert len(record.tricks[-1].agent_hand) == 1
    assert all(len(t.agent_hand) == len(t.llm_hand) for t in record.tricks)


def test_opponent_errors_other_than_ollama_down_do_not_hide_in_the_loop(tmp_path, checkpoint):
    """A programming error in the opponent is raised, not turned into a quiet stop."""
    broken = ScriptedOpponent(fail_on_call=(1, RuntimeError("bug")))
    with pytest.raises(RuntimeError):
        run_evaluation(config(tmp_path, checkpoint), lambda timeout: broken)


def test_rerunning_a_finished_evaluation_leaves_every_file_untouched(tmp_path, checkpoint, monkeypatch):
    """Nothing is rewritten when all games are already saved, not even the status timestamp."""
    clock = iter(f"2026-01-01T00:00:{second:02d}+00:00" for second in range(60))
    monkeypatch.setattr(evaluate, "_utc_now", lambda: next(clock))
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent()) == EXIT_OK
    results = tmp_path / "results"
    before = {path.name: path.read_bytes() for path in results.rglob("*.json")}
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent()) == EXIT_OK
    assert {path.name: path.read_bytes() for path in results.rglob("*.json")} == before


def test_games_without_a_run_file_are_not_adopted(tmp_path, checkpoint):
    """If run.json is lost, the saved games are refused: nothing proves which weights played them."""
    run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent())
    (tmp_path / "results" / "run.json").unlink()
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent()) == EXIT_ERROR


def test_the_same_weights_under_another_path_resume(tmp_path, checkpoint):
    """The run is identified by the hash of the weights, not by where the file lives."""
    import shutil

    run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent())
    copy = tmp_path / "renamed.pt"
    shutil.copy(checkpoint, copy)
    opponent = ScriptedOpponent()
    assert run_evaluation(config(tmp_path, str(copy), num_games=3), lambda timeout: opponent) == EXIT_OK
    assert opponent.calls == 20


def test_other_weights_are_refused(tmp_path, checkpoint, model):
    """A different checkpoint cannot continue a run, even under the same file name."""
    import torch

    run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent())
    torch.manual_seed(99)
    other = type(model)(obs_dim=model.shared[0].in_features, action_dim=3, hidden_dim=16)
    torch.save({"model_state_dict": other.state_dict()}, checkpoint)
    assert run_evaluation(config(tmp_path, checkpoint, num_games=3), lambda timeout: ScriptedOpponent()) == EXIT_ERROR


def test_a_foreign_json_file_stops_the_run_instead_of_being_overwritten(tmp_path, checkpoint):
    """Only a truncated write is replayed; valid JSON of another format is left for the owner to look at."""
    run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: ScriptedOpponent())
    path = game_path(tmp_path / "results", 1)
    path.write_text('{"format": "briscola-game/v2"}', encoding="utf-8")
    opponent = ScriptedOpponent()
    assert run_evaluation(config(tmp_path, checkpoint, num_games=2), lambda timeout: opponent) == EXIT_ERROR
    assert opponent.calls == 0
    assert path.read_text(encoding="utf-8") == '{"format": "briscola-game/v2"}'
