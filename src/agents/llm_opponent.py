"""LLM-based opponent using BAML with robust error handling."""

import logging
import os
from collections import defaultdict
from typing import List, Optional
import random

from baml_client import b as baml_client
from baml_client.types import Action
from baml_py import ClientRegistry

from src.cards import Card

# Configure BAML client at runtime to handle Docker networking
_ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_client_registry = ClientRegistry()
_client_registry.add_llm_client(
    name="Model",
    provider="openai-generic",
    options={
        "base_url": _ollama_url,
        "model": "briscola",
        "temperature": 0.1,
        "max_tokens": 500,
    },
)
_client_registry.set_primary("Model")

# Create configured BAML client
b = baml_client.with_options(client_registry=_client_registry)

logger = logging.getLogger(__name__)


# Neapolitan suits: Denari, Coppe, Spade, Bastoni
SUIT_SYMBOLS = {"coins": "🪙", "cups": "🏆", "swords": "⚔️", "clubs": "🪵"}


def format_history_by_suit(cards: List[str]) -> str:
    """
    Format card history grouped by suit (multi-line).

    Input: ["A coins", "3 swords", "K coins", "2 clubs"]
    Output: "- ♦: A,K\n- ♠: 3\n- ♣: 2"
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
    """Convert 'A coins' to 'A♦'."""
    parts = card.split(" ", 1)
    if len(parts) == 2:
        value, suit = parts
        return f"{value}{SUIT_SYMBOLS.get(suit, suit)}"
    return card


class LLMOpponent:
    """
    LLM opponent that uses BAML to select cards.

    Features:
    - BAML function call to LLM
    - Fallback to random valid action on failure
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
    ):
        """
        Args:
            timeout_seconds: Max time to wait for LLM response
        """
        self.timeout_seconds = timeout_seconds

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.random_fallback_calls = 0

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
            hand: List of card strings (e.g., ["A coins", "3 swords", "K cups"])
            briscola: Trump suit
            history: Formatted history string (e.g., "coins:A,3 | swords:K")
            opponent_card: Opponent's card if they played first this trick

        Returns:
            Index of card to play (0 to len(hand)-1)
        """
        self.total_calls += 1

        # Validate hand size
        if not hand or len(hand) > 3:
            logger.error(f"Invalid hand size: {len(hand)}")
            return self._random_action(len(hand))

        # Try BAML function
        try:
            action = b.ChooseCard(
                hand=hand,
                briscola=briscola,
                history=history,
                opponent_card=opponent_card
            )

            # Validate action
            if self._is_valid_action(action.card_index, len(hand)):
                self.successful_calls += 1
                return action.card_index
            else:
                logger.warning(f"Invalid card_index from LLM: {action.card_index} (hand size: {len(hand)})")

        except Exception as e:
            logger.debug(f"BAML call failed: {e}")

        # Fallback: random valid action
        logger.warning("Falling back to random action")
        self.random_fallback_calls += 1
        return self._random_action(len(hand))

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


# Convenience wrapper for use with environment
def create_llm_opponent_action_fn(opponent: LLMOpponent):
    """
    Create action function compatible with BriscolaEnv.

    Args:
        opponent: LLMOpponent instance

    Returns:
        Function that takes (hand, briscola, game_history, opponent_card) and returns int
    """
    def action_fn(
        hand: List[Card],
        briscola: Card,
        game_history: List[Card],
        opponent_card: Optional[Card]
    ) -> int:
        # Convert cards to strings
        from src.briscola_env import BriscolaEnv
        env = BriscolaEnv()  # Just for utility methods

        # Compact hand format: ["A🪙", "3⚔️", "K🏆"]
        hand_str = [format_card_compact(env.card_to_string(c)) for c in hand]
        history_cards = [env.card_to_string(c) for c in game_history]
        history_formatted = format_history_by_suit(history_cards)
        opponent_str = format_card_compact(env.card_to_string(opponent_card)) if opponent_card else None
        # Full briscola card (e.g., "5⚔️") - this is the last card in deck
        briscola_str = format_card_compact(env.card_to_string(briscola))

        return opponent.choose_action(hand_str, briscola_str, history_formatted, opponent_str)

    return action_fn
