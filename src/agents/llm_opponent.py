"""LLM-based opponent served by Ollama and called through LangChain."""

import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.briscola_env import BriscolaEnv
from src.cards import Card

logger = logging.getLogger(__name__)

# The only opponent model of the project: no other model is ever called.
MODEL_NAME = "gemma4:31b-cloud"
DEFAULT_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.1
MAX_TOKENS = 500

# Retries come after the first attempt, so one move costs at most 3 calls. The wait
# grows 100 ms, 150 ms, ... and is capped at 2 s, as in the retry policy used before.
MAX_RETRIES = 2
RETRY_INITIAL_DELAY = 0.1
RETRY_MULTIPLIER = 1.5
RETRY_MAX_DELAY = 2.0

# Neapolitan suits: Denari, Coppe, Spade, Bastoni
SUIT_SYMBOLS = {"coins": "🪙", "cups": "🏆", "swords": "⚔️", "clubs": "🪵"}


class Action(BaseModel):
    """Structured answer the LLM must give.

    Attributes:
        card_index: Zero-based position, in the hand shown to the LLM, of the card to play.
    """

    card_index: int


class OllamaUnavailableError(RuntimeError):
    """Raised in strict mode when Ollama cannot answer, so no move could be asked."""


@dataclass
class MoveInfo:
    """What happened on the last call to `LLMOpponent.choose_action`.

    Attributes:
        card_index: Index actually played (a random one when the answer was invalid).
        valid: False when the LLM answer could not be used.
        seconds: Wall-clock time of the whole move, retries included.
    """

    card_index: int
    valid: bool
    seconds: float


def ollama_base_url() -> str:
    """Return the Ollama address from ``OLLAMA_BASE_URL`` without a ``/v1`` suffix.

    ``ChatOllama`` talks to the native Ollama API, so the OpenAI-style ``/v1`` suffix
    that older configurations carry must not reach it.
    """
    url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def format_history_by_suit(cards: List[str]) -> str:
    """
    Format card history grouped by suit (multi-line).

    Input: ["A coins", "3 swords", "K coins", "2 clubs"]
    Output: "- 🪙: A,K\n- ⚔️: 3\n- 🪵: 2"
    """
    if not cards:
        return ""

    by_suit = defaultdict(list)
    for card in cards:
        parts = card.split(" ", 1)
        if len(parts) == 2:
            value, suit = parts
            by_suit[suit].append(value)

    # Format: - symbol: values (one per line)
    lines = []
    for suit in ["coins", "cups", "swords", "clubs"]:
        if suit in by_suit:
            symbol = SUIT_SYMBOLS[suit]
            lines.append(f"- {symbol}: {','.join(by_suit[suit])}")

    return "\n".join(lines)


def format_card_compact(card: str) -> str:
    """Convert 'A coins' to 'A🪙'."""
    parts = card.split(" ", 1)
    if len(parts) == 2:
        value, suit = parts
        return f"{value}{SUIT_SYMBOLS.get(suit, suit)}"
    return card


def build_prompt(
    hand: List[str],
    briscola: str,
    history: str,
    opponent_card: Optional[str] = None,
) -> str:
    """Render the card-choice prompt sent to the LLM as a single user message.

    Args:
        hand: Cards in hand, already in compact form (e.g. ``"A🪙"``).
        briscola: The trump card.
        history: Played cards grouped by suit, empty before the first trick.
        opponent_card: Card the other player already played this trick, if any.

    Returns:
        The prompt text, ending with the JSON schema the answer must follow.
    """
    lines = ["=== STATE ===", f"Trump: {briscola}"]
    if opponent_card:
        lines.append(f"Opp: {opponent_card}")
    if history:
        lines += ["Played:", history]
    lines.append("=== HAND ===")
    lines.append("".join(f"{i}:{card} " for i, card in enumerate(hand)))
    lines.append(f"Pick: {','.join(str(i) for i in range(len(hand)))}")
    lines += ["Answer in JSON using this schema:", "{", "  card_index: int,", "}"]
    return "\n".join(lines)


def _build_chain(timeout_seconds: float):
    """Build the LangChain runnable: ChatOllama, structured output, retries."""
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=ollama_base_url(),
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        client_kwargs={"timeout": timeout_seconds},
    )
    return llm.with_structured_output(Action, method="json_schema").with_retry(
        stop_after_attempt=MAX_RETRIES + 1,
        wait_exponential_jitter=True,
        exponential_jitter_params={
            "initial": RETRY_INITIAL_DELAY,
            "exp_base": RETRY_MULTIPLIER,
            "max": RETRY_MAX_DELAY,
            "jitter": 0.0,
        },
    )


class LLMOpponent:
    """
    LLM opponent that asks gemma4:31b-cloud, through LangChain, which card to play.

    Features:
    - ChatOllama call with a Pydantic structured answer and repeated attempts
    - Fallback to a random valid action when the answer is unusable; every fallback is
      an invalid move (`random_fallback_calls`, and ``valid=False`` in `last_move`)
    - Optional strict mode that raises instead of guessing when Ollama is down
    """

    def __init__(
        self,
        timeout_seconds: float = 60.0,
        strict: bool = False,
    ):
        """
        Args:
            timeout_seconds: Max time to wait for one LLM request
            strict: When True, an Ollama failure raises `OllamaUnavailableError`
                instead of falling back to a random card, so no number is invented
                from moves the LLM never made.
        """
        self.timeout_seconds = timeout_seconds
        self.strict = strict
        self._chain = _build_chain(timeout_seconds)

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.random_fallback_calls = 0
        self.last_move: Optional[MoveInfo] = None

    def choose_action(
        self,
        hand: List[str],
        briscola: str,
        history: str,
        opponent_card: Optional[str] = None
    ) -> int:
        """
        Choose which card to play from hand.

        Args:
            hand: List of card strings (e.g., ["A🪙", "3⚔️", "K🏆"])
            briscola: Trump card
            history: Formatted history string (e.g., "- 🪙: A,3\n- ⚔️: K")
            opponent_card: Opponent's card if they played first this trick

        Returns:
            Index of card to play (0 to len(hand)-1). A random valid index when the
            LLM answer is not usable; `last_move` then has ``valid=False``.

        Raises:
            OllamaUnavailableError: In strict mode, when Ollama does not answer.
        """
        self.total_calls += 1
        start = time.perf_counter()

        if not hand or len(hand) > 3:
            logger.error(f"Invalid hand size: {len(hand)}")
            return self._fallback(len(hand), start)

        try:
            action = self._chain.invoke(
                [HumanMessage(content=build_prompt(hand, briscola, history, opponent_card))]
            )
        except (OutputParserException, ValueError) as e:
            # The model answered, but not with a usable Action (malformed JSON, wrong type).
            logger.warning(f"Unusable LLM answer: {e}")
            return self._fallback(len(hand), start)
        except Exception as e:
            if self.strict:
                raise OllamaUnavailableError(f"{type(e).__name__}: {e}") from e
            logger.warning(f"LLM call failed: {e}")
            return self._fallback(len(hand), start)

        if not self._is_valid_action(action.card_index, len(hand)):
            logger.warning(f"Invalid card_index from LLM: {action.card_index} (hand size: {len(hand)})")
            return self._fallback(len(hand), start)

        self.successful_calls += 1
        self.last_move = MoveInfo(action.card_index, True, time.perf_counter() - start)
        return action.card_index

    def _fallback(self, hand_size: int, start: float) -> int:
        """Play a random card and record the move as invalid."""
        logger.warning("Falling back to random action")
        self.random_fallback_calls += 1
        index = self._random_action(hand_size)
        self.last_move = MoveInfo(index, False, time.perf_counter() - start)
        return index

    def _is_valid_action(self, card_index: int, hand_size: int) -> bool:
        """Check if action is valid."""
        return 0 <= card_index < hand_size

    def _random_action(self, hand_size: int) -> int:
        """Return random valid action."""
        return random.randint(0, hand_size - 1)

    def get_statistics(self) -> dict:
        """Get opponent statistics."""
        success_rate = (
            (self.successful_calls / self.total_calls * 100)
            if self.total_calls > 0 else 0
        )
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "random_fallback_calls": self.random_fallback_calls,
            "success_rate": success_rate,
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_calls = 0
        self.successful_calls = 0
        self.random_fallback_calls = 0
        self.last_move = None


# Convenience wrapper for use with environment
def create_llm_opponent_action_fn(opponent: LLMOpponent):
    """
    Create action function compatible with BriscolaEnv.

    Args:
        opponent: LLMOpponent instance

    Returns:
        Function that takes (hand, briscola, game_history, opponent_card) and returns int
    """
    # Only used for its card formatting helper; building it once keeps the global
    # random state, which seeds the decks, untouched between moves.
    env = BriscolaEnv()

    def action_fn(
        hand: List[Card],
        briscola: Card,
        game_history: List[Card],
        opponent_card: Optional[Card]
    ) -> int:
        """Ask the LLM for the index of the card to play, given engine-side cards."""
        # Compact hand format: ["A🪙", "3⚔️", "K🏆"]
        hand_str = [format_card_compact(env.card_to_string(c)) for c in hand]
        history_cards = [env.card_to_string(c) for c in game_history]
        history_formatted = format_history_by_suit(history_cards)
        opponent_str = format_card_compact(env.card_to_string(opponent_card)) if opponent_card else None
        # Full briscola card (e.g., "5⚔️") - this is the last card in deck
        briscola_str = format_card_compact(env.card_to_string(briscola))

        return opponent.choose_action(hand_str, briscola_str, history_formatted, opponent_str)

    return action_fn
