"""Reproducible decks, forced openers and the card text codec."""

import random

import pytest

from src.briscola_env import BriscolaEnv
from src.cards import Card, Deck


def test_every_card_survives_a_text_round_trip():
    """to_string and from_string are inverses for the whole deck."""
    for card in Deck.ordered_cards():
        assert Card.from_string(card.to_string()) == card


@pytest.mark.parametrize("text", ["", "A", "Z coins", "A moons", "A  coins"])
def test_unknown_card_text_is_rejected(text):
    """Garbage never turns into a card."""
    with pytest.raises(ValueError):
        Card.from_string(text)


def test_same_seed_same_deck_and_global_random_untouched():
    """A seed fixes the deck without consuming the global random stream."""
    assert Deck.shuffled_cards(7) == Deck.shuffled_cards(7)
    assert Deck.shuffled_cards(7) != Deck.shuffled_cards(8)

    random.seed(123)
    expected = random.random()
    random.seed(123)
    Deck.shuffled_cards(7)
    assert random.random() == expected


def test_set_order_requires_a_full_deck():
    """A short or duplicated deck is refused."""
    deck = Deck()
    cards = Deck.ordered_cards()
    with pytest.raises(ValueError):
        deck.set_order(cards[:39])
    with pytest.raises(ValueError):
        deck.set_order(cards[:39] + [cards[0]])


def test_reset_deals_the_given_deck_and_opener():
    """Cards come off the end of the list; the first six are the two hands, the bottom is the briscola."""
    order = Deck.shuffled_cards(3)
    env = BriscolaEnv()
    env.reset(deck_order=order, player_is_leading=False)
    assert env.state is not None
    assert env.state.player_hand == [order[-1], order[-2], order[-3]]
    assert env.state.opponent_hand == [order[-4], order[-5], order[-6]]
    assert env.state.briscola_card == order[0]
    assert env.state.player_is_leading is False

    env.reset(deck_order=order, player_is_leading=True)
    assert env.state.player_is_leading is True


def test_reset_without_arguments_still_shuffles_and_picks_an_opener():
    """The default behaviour used by training is unchanged."""
    env = BriscolaEnv()
    env.reset()
    first = list(env.deck.cards)
    env.reset()
    assert first != env.deck.cards
