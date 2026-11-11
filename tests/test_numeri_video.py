"""Statistics for the video: intervals, point-losing moves, report and the two games to show."""

import json
import statistics

import pytest

from src.game_record import AGENT, LLM, GameRecord, TrickLog
from tests.helpers import ScriptedOpponent
from training import numeri_video as nv
from training.numeri_video import Stats


def trick(number, leader, agent_hand, llm_hand, agent_card, llm_card, winner, points=0):
    """Build a trick whose indices follow from the hands."""
    return TrickLog(
        number=number,
        leader=leader,
        agent_hand=agent_hand,
        llm_hand=llm_hand,
        agent_card=agent_card,
        agent_index=agent_hand.index(agent_card),
        llm_card=llm_card,
        llm_index=llm_hand.index(llm_card),
        llm_valid=True,
        llm_seconds=0.5,
        winner=winner,
        points=points,
        agent_score=0,
        llm_score=0,
    )


def record_with(tricks, briscola="4 coins"):
    """Wrap tricks in a record; only what the statistics read matters."""
    return GameRecord(0, 42, True, [], briscola, "gemma4:31b-cloud", "checkpoints/final_model.pt", tricks)


def test_wilson_interval_matches_a_hand_computation():
    """75 wins out of 200 gives 31.1% to 44.4%, and the bounds stay inside 0 to 1 at the edges."""
    low, high = nv.wilson_interval(75, 200)
    assert (round(low, 3), round(high, 3)) == (0.311, 0.444)
    assert nv.wilson_interval(0, 50)[0] == 0.0
    assert nv.wilson_interval(50, 50)[1] == 1.0
    assert nv.wilson_interval(0, 0) == (0.0, 0.0)


def test_difference_interval_is_centred_near_the_difference_and_contains_zero_for_equal_samples():
    """Equal rates give a symmetric interval around zero; a big gap excludes zero."""
    low, high = nv.difference_interval(40, 100, 40, 100)
    assert low < 0 < high and abs(low + high) < 1e-9
    low, high = nv.difference_interval(80, 100, 20, 100)
    assert low > 0.4


def test_a_losing_card_with_points_is_counted_by_role_and_kind():
    """The agent's A of cups beats the LLM's K, and the LLM's own K led into a 3 is lost too."""
    game = record_with(
        [
            trick(1, AGENT, ["A cups", "2 clubs", "5 clubs"], ["K cups", "2 swords", "5 swords"], "A cups", "K cups", AGENT, 15),
            trick(2, LLM, ["3 swords", "2 clubs", "5 clubs"], ["K swords", "2 swords", "5 swords"], "3 swords", "K swords", AGENT, 14),
            trick(3, LLM, ["7 clubs", "6 clubs", "5 clubs"], ["K coins", "6 swords", "4 swords"], "7 clubs", "K coins", LLM, 4),
        ]
    )
    rows = {(r.role, r.kind): (r.count, r.points, r.played) for r in nv.lost_point_moves([game])}
    assert rows == {("risponde", "figura"): (1, 4, 1), ("apre", "figura"): (1, 4, 1)}


def test_a_gift_is_avoidable_only_if_a_cheaper_card_would_have_lost_as_well():
    """Answering a lead of A cups with K cups loses 4 points; the 2 of swords would have lost for free."""
    avoidable = record_with(
        [trick(1, AGENT, ["A cups", "2 clubs", "5 clubs"], ["K cups", "2 swords", "5 swords"], "A cups", "K cups", AGENT, 15)]
    )
    rows = nv.avoidable_gifts([avoidable])
    assert [(r.kind, r.count, r.points) for r in rows] == [("figura", 1, 4)]

    # The only cheaper card is a trump that would have won: giving the K away was not a gift.
    forced = record_with(
        [trick(1, AGENT, ["A cups", "2 clubs", "5 clubs"], ["K cups", "2 coins", "5 coins"], "A cups", "K cups", AGENT, 15)]
    )
    assert nv.avoidable_gifts([forced]) == []

    # Leading is never judged: the agent's reply is unknown.
    led = record_with(
        [trick(1, LLM, ["A cups", "2 clubs", "5 clubs"], ["K swords", "2 swords", "5 swords"], "A cups", "K swords", AGENT, 15)]
    )
    assert nv.avoidable_gifts([led]) == []


@pytest.fixture
def games(make_game):
    """Twenty scripted games, a mix of results, enough for the report."""
    return [make_game(i, ScriptedOpponent(policy=lambda hand, i=i: i % len(hand), invalid_every=7 if i == 3 else None)) for i in range(20)]


def stats_for(games, expected=20, status="complete", reason="all games saved"):
    """Wrap games in `Stats` the way `load_stats` does."""
    return Stats(games=games, expected=expected, status=status, reason=reason)


def test_report_has_at_least_five_numbers_each_with_its_command(games):
    """Every numbered section ends with a command that recomputes it."""
    report = nv.build_report(stats_for(games))
    commands = [line for line in report.splitlines() if "python -m training.numeri_video --mostra" in line]
    assert len(commands) >= 5
    for name in nv.NUMBERS:
        assert f"--mostra {name}`" in report


def test_every_command_in_the_report_prints_a_number(games, tmp_path, capsys):
    """The advertised commands work on a results folder and agree with the report text."""
    for game in games:
        game.save(tmp_path / "partite" / f"partita_{game.game_index:03d}.json")
    (tmp_path / "stato.json").write_text(
        json.dumps({"games_expected": 20, "games_done": 20, "status": "complete", "reason": "all games saved"}), encoding="utf-8"
    )
    for name in nv.NUMBERS:
        assert nv.main(["--results-dir", str(tmp_path), "--mostra", name]) == 0
        assert capsys.readouterr().out.strip()


def test_report_says_so_when_the_data_is_incomplete(games):
    """A run that stopped early is announced at the top, with the reason, and the counts are honest."""
    report = nv.build_report(stats_for(games, expected=200, status="interrupted", reason="Ollama did not answer at game 20: refused"))
    assert "ATTENZIONE" in report
    assert "20 partite su 200" in report
    assert "Ollama did not answer at game 20" in report


def test_showcase_games_are_picked_by_a_fixed_rule(games):
    """One won and one lost game, the same ones on every call, typical rather than extreme."""
    won, lost = nv.pick_showcase(games)
    assert won is None or won.winner == AGENT
    assert lost is None or lost.winner == LLM
    assert (won, lost) == nv.pick_showcase(list(reversed(games)))
    if lost is not None:
        margins = {g.game_index: abs(g.agent_score - g.llm_score) for g in games if g.winner == LLM}
        median = statistics.median(margins.values())
        expected = min(margins, key=lambda index: (abs(margins[index] - median), index))
        assert lost.game_index == expected


def test_write_outputs_creates_the_report_and_both_games(games, tmp_path):
    """Running the module on a results folder leaves the report and the replayable games next to the data."""
    for game in games:
        game.save(tmp_path / "partite" / f"partita_{game.game_index:03d}.json")
    stats = nv.write_outputs(tmp_path)
    assert (tmp_path / nv.REPORT_FILE).read_text(encoding="utf-8").startswith("# Numeri per il video")
    won, lost = nv.pick_showcase(stats.games)
    for game, name in ((won, nv.WON_FILE), (lost, nv.LOST_FILE)):
        target = tmp_path / nv.SHOWCASE_DIR / name
        assert target.exists() == (game is not None)
        if game is not None:
            assert GameRecord.load(target) == game


def test_report_survives_a_run_with_a_single_game(make_game):
    """An evaluation stopped after one game still gets a report, with the missing group stated."""
    report = nv.build_report(stats_for([make_game(0)], expected=200, status="interrupted", reason="refused"))
    assert "ATTENZIONE" in report
    assert "nessuna partita in questa serie" in report
    assert "non calcolabile" in report
    assert "nan" not in report.lower()


def test_report_states_what_it_cannot_know(games):
    """Without stato.json the report does not claim the data is complete, and the seed comes from the games."""
    report = nv.build_report(stats_for(games, expected=None, status="unknown", reason=""))
    assert "Dati completi" not in report
    assert "stato.json" in report
    assert "seme 42 + *i*" in report
