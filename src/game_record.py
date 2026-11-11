"""Complete record of one evaluated game, stored as JSON.

The same file feeds the statistics (``training/numeri_video.py``) and the replay of
a game in ``visualization/render_game.py``, so it holds everything a replay needs:
the deck order, who opens, the hands before every trick and the card each side played.
"""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

FORMAT = "briscola-game/v1"
AGENT = "agent"
LLM = "llm"
TIE = "tie"


@dataclass
class TrickLog:
    """One trick as it was played.

    Attributes:
        number: Trick number, 1 to 20.
        leader: ``"agent"`` or ``"llm"``, who played first.
        agent_hand: Agent's cards before the trick, as ``"A coins"`` strings.
        llm_hand: LLM's cards before the trick.
        agent_card: Card the agent played.
        agent_index: Position of that card in ``agent_hand``.
        llm_card: Card the LLM played.
        llm_index: Position of that card in ``llm_hand``.
        llm_valid: False when the LLM answer was unusable and a random card was played.
        llm_seconds: Time the LLM took for this move.
        winner: ``"agent"`` or ``"llm"``.
        points: Points the winner took.
        agent_score: Agent's total after the trick.
        llm_score: LLM's total after the trick.
    """

    number: int
    leader: str
    agent_hand: List[str]
    llm_hand: List[str]
    agent_card: str
    agent_index: int
    llm_card: str
    llm_index: int
    llm_valid: bool
    llm_seconds: float
    winner: str
    points: int
    agent_score: int
    llm_score: int


@dataclass
class GameRecord:
    """A finished game between the trained agent and the LLM.

    Attributes:
        game_index: Position of the game in the run, from 0.
        deck_seed: Seed of the deck (see `Deck.shuffled_cards`).
        agent_opens: True when the agent plays the first trick.
        deck_order: The 40 cards in draw order (last element is drawn first).
        briscola: The trump card, which is drawn last.
        llm_model: Ollama model that played the LLM side.
        checkpoint: Agent checkpoint the game was played with.
        tricks: The 20 tricks in order.
    """

    game_index: int
    deck_seed: int
    agent_opens: bool
    deck_order: List[str]
    briscola: str
    llm_model: str
    checkpoint: str
    tricks: List[TrickLog] = field(default_factory=list)

    @property
    def agent_score(self) -> int:
        """Agent's final points."""
        return self.tricks[-1].agent_score if self.tricks else 0

    @property
    def llm_score(self) -> int:
        """LLM's final points."""
        return self.tricks[-1].llm_score if self.tricks else 0

    @property
    def winner(self) -> str:
        """``"agent"``, ``"llm"`` or ``"tie"`` (60-60)."""
        if self.agent_score > self.llm_score:
            return AGENT
        if self.llm_score > self.agent_score:
            return LLM
        return TIE

    @property
    def llm_invalid_moves(self) -> int:
        """How many LLM answers were unusable."""
        return sum(1 for trick in self.tricks if not trick.llm_valid)

    @property
    def llm_seconds_per_move(self) -> List[float]:
        """Seconds the LLM took for each of its moves, in trick order."""
        return [trick.llm_seconds for trick in self.tricks]

    def to_dict(self) -> dict:
        """Return the JSON-ready form, with the derived totals next to the raw data."""
        seconds = self.llm_seconds_per_move
        return {
            "format": FORMAT,
            "game_index": self.game_index,
            "deck_seed": self.deck_seed,
            "agent_opens": self.agent_opens,
            "winner": self.winner,
            "agent_score": self.agent_score,
            "llm_score": self.llm_score,
            "llm_invalid_moves": self.llm_invalid_moves,
            "llm_mean_seconds_per_move": round(sum(seconds) / len(seconds), 3) if seconds else 0.0,
            "llm_model": self.llm_model,
            "checkpoint": self.checkpoint,
            "briscola": self.briscola,
            "deck_order": self.deck_order,
            "tricks": [asdict(trick) for trick in self.tricks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameRecord":
        """Rebuild a record from `to_dict` output, ignoring the derived totals.

        Raises:
            ValueError: If the data is not in the ``briscola-game/v1`` format.
        """
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(f"Not a {FORMAT} record")
        try:
            return cls(
                game_index=data["game_index"],
                deck_seed=data["deck_seed"],
                agent_opens=data["agent_opens"],
                deck_order=list(data["deck_order"]),
                briscola=data["briscola"],
                llm_model=data["llm_model"],
                checkpoint=data["checkpoint"],
                tricks=[TrickLog(**trick) for trick in data["tricks"]],
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Malformed {FORMAT} record: {e}") from e

    def save(self, path: Path) -> None:
        """Write the record so that a crash never leaves a half-written file.

        The data goes to a temporary file in the same folder and is then renamed,
        which is atomic; an interrupted run therefore sees either the whole game or
        no game at all.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, separators=(", ", ": "))
                handle.write("\n")
            # mkstemp creates the file private to its owner; results are meant to be shared.
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "GameRecord":
        """Read a record written by `save`.

        Raises:
            ValueError: If the file is not valid JSON or not a game record.
            OSError: If the file cannot be read.
        """
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
