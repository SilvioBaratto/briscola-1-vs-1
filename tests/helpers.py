"""Test doubles shared by the test modules."""

from typing import Callable, List, Optional, Tuple

from src.agents.llm_opponent import MoveInfo


class ScriptedOpponent:
    """Stands in for `LLMOpponent`: applies a fixed policy and reports each move like the real one.

    Args:
        policy: Maps the hand to the index to play.
        invalid_every: Report every n-th move as unusable (None: never).
        fail_on_call: ``(n, error)``: raise ``error`` on the n-th call, counted from 1 (None: never).
    """

    def __init__(
        self,
        policy: Callable[[List[str]], int] = lambda hand: 0,
        invalid_every: Optional[int] = None,
        fail_on_call: Optional[Tuple[int, Exception]] = None,
    ):
        self.policy = policy
        self.invalid_every = invalid_every
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.last_move: Optional[MoveInfo] = None

    def choose_action(self, hand, briscola, history, opponent_card=None) -> int:
        """Return the scripted index and record the move."""
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call[0]:
            raise self.fail_on_call[1]
        valid = not (self.invalid_every and self.calls % self.invalid_every == 0)
        index = self.policy(hand)
        self.last_move = MoveInfo(index, valid, 0.01 * self.calls)
        return index
