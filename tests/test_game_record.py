"""Saving and loading a game record."""

import json

import pytest

from src.game_record import GameRecord


def test_round_trip_keeps_every_field(tmp_path, make_game):
    """What is saved is what is loaded, including the derived totals."""
    record = make_game(0)
    path = tmp_path / "partita_000.json"
    record.save(path)
    loaded = GameRecord.load(path)
    assert loaded == record
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["winner"] == record.winner
    assert data["agent_score"] + data["llm_score"] == 120


def test_save_leaves_no_temporary_file(tmp_path, make_game):
    """The atomic write cleans up after itself."""
    make_game(0).save(tmp_path / "partite" / "partita_000.json")
    assert [p.name for p in (tmp_path / "partite").iterdir()] == ["partita_000.json"]


@pytest.mark.parametrize("content", ["", "{", "[]", '{"format": "other"}', '{"format": "briscola-game/v1"}'])
def test_broken_files_are_rejected_as_value_errors(tmp_path, content):
    """Any unreadable or foreign file raises ValueError, which resume treats as a missing game."""
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        GameRecord.load(path)
