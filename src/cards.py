"""Card definitions and deck management for Briscola."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List
import random


class CardValue(IntEnum):
    """Card values for Briscola (numeric values are for identification, not ranking)."""
    TWO = 2
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    JACK = 8
    HORSE = 9  # Queen in standard deck
    KING = 10
    THREE = 3
    ACE = 1

    @staticmethod
    def get_rank(value: "CardValue") -> int:
        """
        Get the rank for trick comparison (higher is better).
        Ranking from highest to lowest: Ace, Three, King, Horse, Jack, 7, 6, 5, 4, 2
        """
        rank_order = {
            CardValue.ACE: 10,
            CardValue.THREE: 9,
            CardValue.KING: 8,
            CardValue.HORSE: 7,
            CardValue.JACK: 6,
            CardValue.SEVEN: 5,
            CardValue.SIX: 4,
            CardValue.FIVE: 3,
            CardValue.FOUR: 2,
            CardValue.TWO: 1,
        }
        return rank_order[value]

    @staticmethod
    def get_points(value: "CardValue") -> int:
        """
        Get point value for scoring.
        Ace: 11, Three: 10, King: 4, Horse: 3, Jack: 2, others: 0
        """
        points_map = {
            CardValue.ACE: 11,
            CardValue.THREE: 10,
            CardValue.KING: 4,
            CardValue.HORSE: 3,
            CardValue.JACK: 2,
            CardValue.SEVEN: 0,
            CardValue.SIX: 0,
            CardValue.FIVE: 0,
            CardValue.FOUR: 0,
            CardValue.TWO: 0,
        }
        return points_map[value]


@dataclass
class Card:
    """Represents a single card in Briscola."""
    value: CardValue
    suit: str  # "coins", "cups", "swords", "clubs"

    def get_rank(self) -> int:
        """Get the rank of this card for trick comparison."""
        return CardValue.get_rank(self.value)

    def get_points(self) -> int:
        """Get the point value of this card."""
        return CardValue.get_points(self.value)

    def __lt__(self, other: "Card") -> bool:
        """Compare cards by rank (not numeric value)."""
        return self.get_rank() < other.get_rank()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.value == other.value and self.suit == other.suit

    def __repr__(self) -> str:
        return f"Card({self.value.name}, {self.suit})"

    def beats(self, other: "Card", briscola_suit: str, is_leading: bool = False) -> bool:
        """
        Determine if this card beats another card in Briscola.

        Args:
            other: The other card to compare against
            briscola_suit: The trump suit for this game
            is_leading: True if this card was played first

        Returns:
            True if this card wins the trick

        Rules:
        1. If both cards are same suit, higher rank wins
        2. If different suits and neither is briscola, leading card wins
        3. If one card is briscola (trump), it wins
        """
        this_is_briscola = self.suit == briscola_suit
        other_is_briscola = other.suit == briscola_suit

        # Rule 3: Trump beats non-trump
        if this_is_briscola and not other_is_briscola:
            return True
        if other_is_briscola and not this_is_briscola:
            return False

        # Rule 1: Same suit - higher rank wins
        if self.suit == other.suit:
            return self.get_rank() > other.get_rank()

        # Rule 2: Different suits, neither is trump - leading card wins
        return is_leading


class Deck:
    """A Sicilian/Neapolitan deck of 40 cards for Briscola."""

    SUITS = ["coins", "cups", "swords", "clubs"]

    def __init__(self):
        self.cards: List[Card] = []
        self.briscola_card: Card | None = None
        self.reset()

    def reset(self) -> None:
        """Reset deck with all 40 cards and shuffle."""
        self.cards = [
            Card(value, suit)
            for suit in self.SUITS
            for value in CardValue
        ]
        self.briscola_card = None
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def draw(self) -> Card | None:
        """Draw a card from the top of the deck."""
        if self.cards:
            return self.cards.pop()
        return None

    def reveal_briscola(self) -> Card | None:
        """
        Reveal the briscola (trump) card after dealing initial hands.
        This card stays visible and is drawn last.
        """
        if self.cards:
            self.briscola_card = self.cards[0]  # Bottom card of deck
            return self.briscola_card
        return None

    def get_briscola_suit(self) -> str | None:
        """Get the trump suit for this game."""
        return self.briscola_card.suit if self.briscola_card else None

    def __len__(self) -> int:
        return len(self.cards)
