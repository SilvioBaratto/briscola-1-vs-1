"""The LLM opponent: prompt, model settings, retries and what counts as an invalid move."""

import json
import time
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from src.agents import llm_opponent as opp
from src.agents.llm_opponent import Action, LLMOpponent, OllamaUnavailableError

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "choose_card_prompts.json").read_text(encoding="utf-8"))
HAND = ["A🪙", "3⚔️", "K🏆"]


class FakeChat:
    """Replaces `ChatOllama`; hands out a structured runnable driven by `behaviour`."""

    created = []
    behaviour = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeChat.created.append(kwargs)

    def with_structured_output(self, schema, method=None):
        """Return a runnable that calls the test's behaviour once per attempt."""
        assert schema is Action
        assert method == "json_schema"
        return RunnableLambda(lambda _messages: FakeChat.behaviour())


@pytest.fixture
def fake_chat(monkeypatch):
    """Install `FakeChat` and return a setter for the per-attempt behaviour."""
    FakeChat.created = []
    monkeypatch.setattr(opp, "ChatOllama", FakeChat)

    def install(behaviour):
        """Use `behaviour(attempt)` for every model call; return the list of call times."""
        calls = []

        def wrapped():
            """Log the call time, then defer to the test's behaviour."""
            calls.append(time.monotonic())
            return behaviour(len(calls))

        FakeChat.behaviour = staticmethod(wrapped)
        return calls

    return install


def failing(error):
    """Return a behaviour that raises `error` on every attempt."""

    def behaviour(attempt):
        """Raise the error."""
        raise error

    return behaviour


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_prompt_is_the_one_sent_before(name):
    """The rendered prompt is identical, character for character, to the previous implementation's."""
    case = FIXTURES[name]
    assert opp.build_prompt(**case["inputs"]) == case["prompt"]


def test_base_url_defaults_and_drops_the_v1_suffix(monkeypatch):
    """ChatOllama gets the bare address, whichever form the environment variable has."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert opp.ollama_base_url() == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
    assert opp.ollama_base_url() == "http://host.docker.internal:11434"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    assert opp.ollama_base_url() == "http://ollama:11434"


def test_only_gemma_is_called_with_the_previous_sampling_settings(fake_chat, monkeypatch):
    """Model, temperature, token limit and address are the agreed ones."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    fake_chat(lambda attempt: Action(card_index=1))
    opponent = LLMOpponent(timeout_seconds=12.0)
    assert opponent.choose_action(HAND, "5⚔️", "", None) == 1
    assert FakeChat.created == [
        {
            "model": "gemma4:31b-cloud",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "num_predict": 500,
            "client_kwargs": {"timeout": 12.0},
        }
    ]
    assert opponent.last_move.valid and opponent.last_move.card_index == 1


def test_two_retries_with_growing_wait_then_success(fake_chat):
    """A move that fails twice is asked a third time, waiting 0.1 s and then 0.15 s."""

    def behaviour(attempt):
        """Fail the first two attempts, then answer."""
        if attempt < 3:
            raise ConnectionError("down")
        return Action(card_index=2)

    calls = fake_chat(behaviour)
    assert LLMOpponent().choose_action(HAND, "5⚔️", "", None) == 2
    assert len(calls) == 3
    first_wait, second_wait = calls[1] - calls[0], calls[2] - calls[1]
    assert 0.09 <= first_wait < 0.5
    assert second_wait > first_wait


def test_no_third_retry_and_strict_mode_reports_ollama_down(fake_chat):
    """After three failed attempts strict mode raises instead of inventing a move."""
    calls = fake_chat(failing(ConnectionError("down")))
    with pytest.raises(OllamaUnavailableError):
        LLMOpponent(strict=True).choose_action(HAND, "5⚔️", "", None)
    assert len(calls) == 3


def test_ollama_down_without_strict_mode_plays_random_and_counts_it(fake_chat):
    """The training behaviour is kept: a random card, recorded as an invalid move."""
    fake_chat(failing(ConnectionError("down")))
    opponent = LLMOpponent()
    index = opponent.choose_action(HAND, "5⚔️", "", None)
    assert 0 <= index < 3
    assert opponent.last_move.valid is False
    assert opponent.get_statistics()["random_fallback_calls"] == 1


def test_index_outside_the_hand_is_an_invalid_move_without_retry(fake_chat):
    """card_index 7 for a three-card hand is unusable, and asking again would not help."""
    calls = fake_chat(lambda attempt: Action(card_index=7))
    opponent = LLMOpponent(strict=True)
    index = opponent.choose_action(HAND, "5⚔️", "", None)
    assert 0 <= index < 3
    assert opponent.last_move.valid is False
    assert len(calls) == 1
    assert opponent.get_statistics()["successful_calls"] == 0


def test_unparseable_answer_is_an_invalid_move_not_an_outage(fake_chat):
    """Malformed output counts as an invalid move even in strict mode."""
    calls = fake_chat(failing(OutputParserException("not json")))
    opponent = LLMOpponent(strict=True)
    index = opponent.choose_action(HAND, "5⚔️", "", None)
    assert 0 <= index < 3
    assert opponent.last_move.valid is False
    assert len(calls) == 3


def test_negative_index_is_invalid(fake_chat):
    """Python's negative indexing must not make -1 look like a card."""
    fake_chat(lambda attempt: Action(card_index=-1))
    opponent = LLMOpponent()
    opponent.choose_action(["7🪙"], "5⚔️", "", None)
    assert opponent.last_move.valid is False
