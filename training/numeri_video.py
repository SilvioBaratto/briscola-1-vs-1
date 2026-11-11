"""Compute the numbers for the video from the saved games.

``python -m training.numeri_video`` writes ``results/numeri-video.md`` and the two
games to show; ``python -m training.numeri_video --mostra NOME`` prints one number so
that every figure in the report can be checked on its own.

Everything is derived from ``results/partite/*.json`` and ``results/stato.json``,
never from memory, so the report is the same each time it is regenerated.
"""

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.agents.llm_opponent import TEMPERATURE
from src.cards import Card
from src.game_record import AGENT, LLM, TIE, GameRecord
from training.evaluate import GAMES_DIR, STATUS_FILE

REPORT_FILE = "numeri-video.md"
SHOWCASE_DIR = "da-mostrare"
WON_FILE = "partita-vinta.json"
LOST_FILE = "partita-persa.json"
Z95 = 1.96


def load_games(results_dir: Path) -> List[GameRecord]:
    """Load every saved game, ordered by game index.

    Raises:
        FileNotFoundError: If no game is saved yet.
    """
    paths = sorted((Path(results_dir) / GAMES_DIR).glob("partita_*.json"))
    if not paths:
        raise FileNotFoundError(f"No games in {Path(results_dir) / GAMES_DIR}")
    return [GameRecord.load(path) for path in paths]


def wilson_interval(successes: int, total: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a proportion (95% by default).

    Args:
        successes: Number of successes.
        total: Number of trials.
        z: Normal quantile of the confidence level.

    Returns:
        Lower and upper bound as fractions between 0 and 1.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def difference_interval(
    wins_a: int, total_a: int, wins_b: int, total_b: int, z: float = Z95
) -> Tuple[float, float]:
    """Newcombe interval for the difference of two proportions, ``a - b``.

    Built from the two Wilson intervals, so it behaves well with few games.

    Returns:
        Lower and upper bound of ``p_a - p_b`` as fractions.
    """
    p_a, p_b = wins_a / total_a, wins_b / total_b
    low_a, high_a = wilson_interval(wins_a, total_a, z)
    low_b, high_b = wilson_interval(wins_b, total_b, z)
    diff = p_a - p_b
    return (
        diff - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2),
        diff + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2),
    )


def card_class(card: Card, briscola_suit: str) -> str:
    """Name the kind of a card that carries points: carico, figura, with 'briscola' if trump."""
    kind = "carico" if card.get_points() >= 10 else "figura"
    return f"{kind} di briscola" if card.suit == briscola_suit else kind


@dataclass
class MoveRow:
    """How often a kind of LLM move gave points away.

    Attributes:
        role: ``"risponde"`` or ``"apre"``.
        kind: Card kind, from `card_class`.
        count: Times the LLM played it and lost the trick.
        points: Points the agent collected from those cards.
        played: Times the LLM played that kind of card in that role, won or lost.
    """

    role: str
    kind: str
    count: int
    points: int
    played: int


def lost_point_moves(games: List[GameRecord]) -> List[MoveRow]:
    """Group the LLM moves that lost a trick with a card worth points.

    A card with points that loses the trick goes to the agent, so this is every
    move in which the LLM handed points over, whether or not it could do better.

    Returns:
        Rows sorted from the most frequent to the least frequent.
    """
    lost: Dict[Tuple[str, str], List[int]] = {}
    played: Dict[Tuple[str, str], int] = {}
    for game in games:
        suit = Card.from_string(game.briscola).suit
        for trick in game.tricks:
            card = Card.from_string(trick.llm_card)
            if card.get_points() == 0:
                continue
            key = ("apre" if trick.leader == LLM else "risponde", card_class(card, suit))
            played[key] = played.get(key, 0) + 1
            if trick.winner == AGENT:
                entry = lost.setdefault(key, [0, 0])
                entry[0] += 1
                entry[1] += card.get_points()
    rows = [MoveRow(role, kind, c, p, played[(role, kind)]) for (role, kind), (c, p) in lost.items()]
    return sorted(rows, key=lambda row: (-row.count, -row.points, row.role, row.kind))


def avoidable_gifts(games: List[GameRecord]) -> List[MoveRow]:
    """Find the LLM answers to the agent's lead that cost points for nothing.

    A gift is avoidable when the LLM lost the trick with a card worth points while
    another card in its hand, worth fewer points, would have lost it just the same.
    Only answers are checked: when the LLM leads, the agent's reply is unknown.
    The ``points`` of a row are the points saved by playing the cheapest losing card.

    Returns:
        Rows sorted from the most frequent to the least frequent.
    """
    found: Dict[str, List[int]] = {}
    answered: Dict[str, int] = {}
    for game in games:
        suit = Card.from_string(game.briscola).suit
        for trick in game.tricks:
            if trick.leader != AGENT:
                continue
            lead = Card.from_string(trick.agent_card)
            played = Card.from_string(trick.llm_card)
            if played.get_points() == 0:
                continue
            kind = card_class(played, suit)
            answered[kind] = answered.get(kind, 0) + 1
            if trick.winner != AGENT:
                continue
            cheaper = [
                Card.from_string(text).get_points()
                for text in trick.llm_hand
                if Card.from_string(text).get_points() < played.get_points()
                and lead.beats(Card.from_string(text), suit, is_leading=True)
            ]
            if cheaper:
                entry = found.setdefault(kind, [0, 0])
                entry[0] += 1
                entry[1] += played.get_points() - min(cheaper)
    rows = [MoveRow("risponde", kind, c, p, answered[kind]) for kind, (c, p) in found.items()]
    return sorted(rows, key=lambda row: (-row.count, -row.points, row.kind))


@dataclass
class OpenerStats:
    """Results of the games in which one side played the first trick."""

    games: int
    wins: int
    losses: int
    ties: int
    mean_agent: float
    mean_llm: float

    @property
    def win_rate(self) -> float:
        """Share of games the agent won."""
        return self.wins / self.games


def opener_stats(games: List[GameRecord]) -> OpenerStats:
    """Summarise a group of games from the agent's point of view."""
    return OpenerStats(
        games=len(games),
        wins=sum(g.winner == AGENT for g in games),
        losses=sum(g.winner == LLM for g in games),
        ties=sum(g.winner == TIE for g in games),
        mean_agent=statistics.mean(g.agent_score for g in games),
        mean_llm=statistics.mean(g.llm_score for g in games),
    )


def pick_showcase(games: List[GameRecord]) -> Tuple[Optional[GameRecord], Optional[GameRecord]]:
    """Choose one game the agent won and one it lost, both typical rather than extreme.

    The pick is the game whose margin is closest to the median margin of its group,
    the lowest game index winning a tie, so the choice is the same on every run.

    Returns:
        ``(won, lost)``; either is None if the agent never had that result.
    """

    def typical(group: List[GameRecord]) -> Optional[GameRecord]:
        """Game of the group whose margin is closest to the median margin."""
        if not group:
            return None
        median = statistics.median(abs(g.agent_score - g.llm_score) for g in group)
        return min(group, key=lambda g: (abs(abs(g.agent_score - g.llm_score) - median), g.game_index))

    return typical([g for g in games if g.winner == AGENT]), typical([g for g in games if g.winner == LLM])


@dataclass
class Stats:
    """Every number of the report, computed once from the saved games."""

    games: List[GameRecord]
    expected: Optional[int]
    status: str
    reason: str

    @property
    def n(self) -> int:
        """Games available."""
        return len(self.games)

    def count(self, winner: str) -> int:
        """Games with a given winner (``agent``, ``llm`` or ``tie``)."""
        return sum(g.winner == winner for g in self.games)

    @property
    def win_rate(self) -> float:
        """Share of games the agent won; ties are not wins."""
        return self.count(AGENT) / self.n

    @property
    def win_interval(self) -> Tuple[float, float]:
        """95% Wilson interval of `win_rate`."""
        return wilson_interval(self.count(AGENT), self.n)

    @property
    def mean_agent(self) -> float:
        """Average points of the agent."""
        return statistics.mean(g.agent_score for g in self.games)

    @property
    def mean_llm(self) -> float:
        """Average points of the LLM."""
        return statistics.mean(g.llm_score for g in self.games)

    @property
    def agent_opens(self) -> OpenerStats:
        """Games the agent opened."""
        return opener_stats([g for g in self.games if g.agent_opens])

    @property
    def llm_opens(self) -> OpenerStats:
        """Games the LLM opened."""
        return opener_stats([g for g in self.games if not g.agent_opens])

    @property
    def opener_difference(self) -> Tuple[float, Tuple[float, float]]:
        """Win rate with the agent opening minus with the LLM opening, and its 95% interval."""
        a, b = self.agent_opens, self.llm_opens
        return a.win_rate - b.win_rate, difference_interval(a.wins, a.games, b.wins, b.games)

    @property
    def moves(self) -> int:
        """LLM moves in all games."""
        return sum(len(g.tricks) for g in self.games)

    @property
    def invalid_moves(self) -> int:
        """LLM answers that could not be used."""
        return sum(g.llm_invalid_moves for g in self.games)

    @property
    def seconds(self) -> List[float]:
        """Seconds the LLM took for each move."""
        return [s for g in self.games for s in g.llm_seconds_per_move]

    @property
    def slowest_game(self) -> GameRecord:
        """Game that holds the slowest LLM move."""
        return max(self.games, key=lambda g: max(g.llm_seconds_per_move))


def load_stats(results_dir: Path) -> Stats:
    """Read the games and the run status from a results folder."""
    games = load_games(results_dir)
    status_path = Path(results_dir) / STATUS_FILE
    expected, status, reason = None, "unknown", ""
    if status_path.exists():
        data = json.loads(status_path.read_text(encoding="utf-8"))
        expected, status, reason = data["games_expected"], data["status"], data["reason"]
    return Stats(games=games, expected=expected, status=status, reason=reason)


def it(value: float, digits: int = 1) -> str:
    """Format a number with the Italian decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    """Format a fraction as an Italian percentage."""
    return f"{it(value * 100, digits)}%"


def play_phrase(row: "MoveRow") -> str:
    """Say a move in Italian, e.g. ``rispondere con una figura``."""
    verb = "aprire" if row.role == "apre" else "rispondere"
    article = "un" if row.kind.startswith("carico") else "una"
    return f"{verb} con {article} {row.kind}"


def describe_result(s: Stats) -> str:
    """Wins, losses and ties of the agent."""
    return f"{s.count(AGENT)} vittorie, {s.count(LLM)} sconfitte, {s.count(TIE)} pareggi (60-60) su {s.n} partite"


def describe_win_rate(s: Stats) -> str:
    """Win percentage of the agent with its 95% interval."""
    low, high = s.win_interval
    return f"{pct(s.win_rate)} di vittorie, intervallo al 95%: da {pct(low)} a {pct(high)}"


def describe_points(s: Stats) -> str:
    """Average points of both sides and the margin."""
    margin = s.mean_agent - s.mean_llm
    return (
        f"agente {it(s.mean_agent)} punti, {s.games[0].llm_model} {it(s.mean_llm)} punti, "
        f"margine {it(margin)} per l'agente (su 120 punti in palio)"
    )


def describe_opener(s: Stats) -> str:
    """Win rate with each opener and the difference."""
    a, b = s.agent_opens, s.llm_opens
    diff, (low, high) = s.opener_difference
    return (
        f"apre l'agente: {a.wins} vittorie su {a.games} ({pct(a.win_rate)}), punti {it(a.mean_agent)}-{it(a.mean_llm)}; "
        f"apre il modello: {b.wins} vittorie su {b.games} ({pct(b.win_rate)}), punti {it(b.mean_agent)}-{it(b.mean_llm)}; "
        f"differenza {it(diff * 100)} punti percentuali, intervallo al 95%: da {it(low * 100)} a {it(high * 100)}"
    )


def describe_lost_points(s: Stats) -> str:
    """The LLM move that gives points away most often, and the one that costs most."""
    rows = lost_point_moves(s.games)
    if not rows:
        return "il modello non ha mai perso una mano con una carta che vale punti"
    top = rows[0]
    costly = max(rows, key=lambda r: r.points)
    return (
        f"più frequente: {play_phrase(top)} e perdere la mano, {top.count} volte ({top.points} punti ceduti); "
        f"più costosa: {play_phrase(costly)} e perdere la mano, {costly.count} volte ({costly.points} punti ceduti)"
    )


def describe_gifts(s: Stats) -> str:
    """Avoidable point gifts of the LLM when answering."""
    rows = avoidable_gifts(s.games)
    total = sum(r.count for r in rows)
    points = sum(r.points for r in rows)
    if not rows:
        return "0 regali evitabili"
    top = rows[0]
    return (
        f"{total} regali evitabili ({points} punti risparmiabili); "
        f"il più frequente: {play_phrase(top)}, {top.count} volte ({top.points} punti risparmiabili)"
    )


def describe_invalid(s: Stats) -> str:
    """Unusable LLM answers."""
    return f"{s.invalid_moves} mosse non valide su {s.moves} mosse del modello"


def describe_seconds(s: Stats) -> str:
    """Time the LLM took per move."""
    secs = sorted(s.seconds)
    p95 = secs[min(len(secs) - 1, math.ceil(0.95 * len(secs)) - 1)]
    return (
        f"media {it(statistics.mean(secs), 2)} s, mediana {it(statistics.median(secs), 2)} s, "
        f"95° percentile {it(p95, 2)} s, massimo {it(max(secs), 2)} s"
    )


NUMBERS: Dict[str, Callable[[Stats], str]] = {
    "risultato": describe_result,
    "percentuale-vittorie": describe_win_rate,
    "punti-medi": describe_points,
    "chi-apre": describe_opener,
    "mosse-che-perdono-punti": describe_lost_points,
    "regali-evitabili": describe_gifts,
    "mosse-non-valide": describe_invalid,
    "secondi-per-mossa": describe_seconds,
}


def command_for(name: str, results_dir: str) -> str:
    """Return the shell command that prints one number."""
    suffix = "" if results_dir == "results" else f" --results-dir {results_dir}"
    return f"python -m training.numeri_video --mostra {name}{suffix}"


def _table(rows: List[MoveRow], with_role: bool = True) -> List[str]:
    header = "| Ruolo | Carta | Volte che ha perso la mano | Punti ceduti | Volte giocata |" if with_role else (
        "| Carta | Volte | Punti risparmiabili | Volte giocata in risposta |"
    )
    lines = [header, "|---|---|---:|---:|---:|" if with_role else "|---|---:|---:|---:|"]
    for row in rows:
        if with_role:
            lines.append(f"| {row.role} | {row.kind} | {row.count} | {row.points} | {row.played} |")
        else:
            lines.append(f"| {row.kind} | {row.count} | {row.points} | {row.played} |")
    return lines


def _game_line(game: GameRecord, results_dir: str) -> str:
    opener = "apre l'agente" if game.agent_opens else "apre il modello"
    return (
        f"partita {game.game_index} ({opener}): agente {game.agent_score}, modello {game.llm_score}, "
        f"file `{results_dir}/{GAMES_DIR}/partita_{game.game_index:03d}.json`"
    )


def build_report(s: Stats, results_dir: str = "results") -> str:
    """Write the Italian report with every number, how to recompute it and what is surprising.

    Args:
        s: Statistics of the saved games.
        results_dir: Folder shown in the recompute commands.

    Returns:
        The Markdown text of ``numeri-video.md``.
    """
    model = s.games[0].llm_model
    checkpoint = s.games[0].checkpoint
    low, high = s.win_interval
    margin = s.mean_agent - s.mean_llm
    won, lost = pick_showcase(s.games)
    by_margin = sorted(s.games, key=lambda g: g.agent_score - g.llm_score)
    worst, best = by_margin[0], by_margin[-1]
    gifts = avoidable_gifts(s.games)
    moves = lost_point_moves(s.games)
    diff, (diff_low, diff_high) = s.opener_difference
    secs = s.seconds
    out: List[str] = []
    add = out.append

    add(f"# Numeri per il video: agente PPO contro {model}")
    add("")
    if s.expected is not None and s.n < s.expected:
        add(
            f"> **ATTENZIONE: i dati sono incompleti.** Ci sono {s.n} partite su {s.expected}: "
            f"il run si è fermato ({s.status}): {s.reason}. I numeri qui sotto valgono solo per queste {s.n} partite."
        )
    else:
        add(f"> Dati completi: {s.n} partite su {s.expected if s.expected is not None else s.n}, tutte giocate e salvate.")
    add("")
    add(
        f"L'agente è `{checkpoint}` (rete PPO già addestrata, mossa sempre la più probabile). "
        f"L'avversario è `{model}` servito da Ollama e chiamato solo con LangChain, temperatura {it(TEMPERATURE)}. "
        f"Il mazzo della partita *i* si mescola con il seme 42 + *i*, e l'agente apre le partite pari, il modello le dispari: "
        f"{s.agent_opens.games} partite aperte dall'agente e {s.llm_opens.games} dal modello."
    )
    add("")
    add(
        "Ogni numero ha accanto il comando che lo ricalcola dai file in "
        f"`{results_dir}/{GAMES_DIR}/`. Rilanciando la valutazione i mazzi e chi apre sono gli stessi, "
        "ma il modello è un servizio remoto e può rispondere in modo un po' diverso: le mosse, e quindi i punteggi, possono cambiare di poco."
    )
    add("")
    add("## In una frase")
    add("")
    if high < 0.5:
        headline = "perde contro il modello"
    elif low > 0.5:
        headline = "vince contro il modello"
    else:
        headline = "non ha un vincitore chiaro contro il modello"
    add(
        f"L'agente addestrato {headline}: {s.count(AGENT)} vittorie contro {s.count(LLM)} sconfitte "
        f"({pct(s.win_rate)} di vittorie, intervallo al 95% da {pct(low)} a {pct(high)})."
    )
    add("")
    add("## I numeri")
    add("")

    add("### 1. Chi ha vinto")
    add("")
    add(f"- **{s.count(AGENT)} vittorie, {s.count(LLM)} sconfitte e {s.count(TIE)} pareggi** dell'agente, su {s.n} partite. Un pareggio è 60 a 60.")
    add(f"- In parole povere: ogni 100 partite l'agente ne vince circa {round(s.win_rate * 100)} e ne perde circa {round(s.count(LLM) / s.n * 100)}.")
    add(f"- Si ricalcola con: `{command_for('risultato', results_dir)}`")
    add("")

    add("### 2. Percentuale di vittorie e margine di errore")
    add("")
    add(f"- **{pct(s.win_rate)} di vittorie** (vittorie / partite, i pareggi non contano come vittorie), con un **intervallo di confidenza al 95% da {pct(low)} a {pct(high)}**.")
    if high < 0.5:
        verdict_win = "il 50% (testa o croce) cade fuori, sopra la forbice: non è sfortuna, l'agente è davvero peggiore del modello"
    elif low > 0.5:
        verdict_win = "il 50% (testa o croce) cade fuori, sotto la forbice: l'agente è davvero migliore del modello"
    else:
        verdict_win = "il 50% (testa o croce) cade dentro la forbice: la serie non basta per dire chi è più forte"
    add(f"- In parole povere: con {s.n} partite il vero valore sta, con il 95% di fiducia, in quella forbice, e {verdict_win}.")
    add("- Metodo: intervallo di Wilson, `training/numeri_video.py:wilson_interval`.")
    add(f"- Si ricalcola con: `{command_for('percentuale-vittorie', results_dir)}`")
    add("")

    add("### 3. Punti medi e margine")
    add("")
    add(f"- **Agente {it(s.mean_agent)} punti, modello {it(s.mean_llm)} punti**, margine **{it(margin)}** per l'agente (su 120 punti in palio, per vincere ne servono 61).")
    ahead = "il modello prende circa {} punti più dell'agente" if margin < 0 else "l'agente prende circa {} punti più del modello"
    add(f"- In parole povere: in una partita media {ahead.format(it(abs(margin), 0))}.")
    add(f"- Si ricalcola con: `{command_for('punti-medi', results_dir)}`")
    add("")

    a, b = s.agent_opens, s.llm_opens
    add("### 4. Aprire cambia qualcosa?")
    add("")
    add(f"- **Apre l'agente:** {a.wins} vittorie, {a.losses} sconfitte e {a.ties} pareggi su {a.games} partite ({pct(a.win_rate)} di vittorie); punti medi {it(a.mean_agent)} a {it(a.mean_llm)}.")
    add(f"- **Apre il modello:** {b.wins} vittorie, {b.losses} sconfitte e {b.ties} pareggi su {b.games} partite ({pct(b.win_rate)} di vittorie); punti medi {it(b.mean_agent)} a {it(b.mean_llm)}.")
    add(f"- **Differenza:** {it(diff * 100)} punti percentuali per l'agente quando apre, intervallo al 95% da {it(diff_low * 100)} a {it(diff_high * 100)}.")
    verdict = (
        "lo zero cade dentro la forbice: con queste partite non si può dire che aprire sia un vantaggio né uno svantaggio"
        if diff_low <= 0 <= diff_high
        else "lo zero cade fuori dalla forbice: aprire cambia davvero il risultato"
    )
    add(f"- In parole povere: {verdict}.")
    add(f"- Si ricalcola con: `{command_for('chi-apre', results_dir)}`")
    add("")

    add("### 5. La mossa in cui il modello perde più punti")
    add("")
    add("- Carte con punti: *carico* = asso o tre (11 e 10 punti), *figura* = re, cavallo o fante (4, 3 e 2 punti).")
    if moves:
        top = moves[0]
        costly = max(moves, key=lambda r: r.points)
        add(f"- **La più frequente:** {play_phrase(top)} e perdere la mano: **{top.count} volte**, {top.points} punti regalati all'agente.")
        add(f"- **La più costosa:** {play_phrase(costly)} e perdere la mano: {costly.count} volte, **{costly.points} punti** regalati.")
    else:
        add("- Il modello non ha mai perso una mano con una carta che vale punti.")
    add("- In parole povere: ogni volta che il modello gioca una carta con punti e non vince la mano, quei punti vanno all'agente. Questa tabella conta tutte quelle volte, divise per chi apre e per tipo di carta.")
    add("")
    out.extend(_table(moves))
    add("")
    add(f"- Si ricalcola con: `{command_for('mosse-che-perdono-punti', results_dir)}`")
    add("")
    total_gifts = sum(r.count for r in gifts)
    gift_points = sum(r.points for r in gifts)
    add("#### Quanti di questi regali erano evitabili")
    add("")
    add(
        f"- Quando risponde all'apertura dell'agente, il modello butta una carta con punti e perde la mano pur avendo in mano una carta che vale meno e avrebbe perso lo stesso: "
        f"**{total_gifts} volte, {gift_points} punti** che poteva tenersi. Questa è un'euristica: non dice che sia sempre un errore, dice che il regalo non era obbligato."
    )
    if gifts:
        add(f"- Il caso più frequente: {play_phrase(gifts[0])}, {gifts[0].count} volte ({gifts[0].points} punti risparmiabili).")
        add("")
        out.extend(_table(gifts, with_role=False))
        add("")
    add(f"- Si ricalcola con: `{command_for('regali-evitabili', results_dir)}`")
    add("")

    add("### 6. Risposte non valide del modello")
    add("")
    add(f"- **{s.invalid_moves} mosse non valide su {s.moves}** mosse del modello (risposta senza un indice di carta utilizzabile).")
    add("- In parole povere: l'uscita strutturata di LangChain ha funzionato sempre; il modello non ha mai fatto una mossa impossibile." if s.invalid_moves == 0 else "- In parole povere: in quei casi il programma ha giocato una carta a caso al posto del modello e la mossa è contata come non valida.")
    add(f"- Si ricalcola con: `{command_for('mosse-non-valide', results_dir)}`")
    add("")

    add("### 7. Quanto ci mette a muovere")
    add("")
    add(f"- **{describe_seconds(s)}** per mossa del modello ({len(secs)} mosse).")
    add("- In parole povere: una partita da 20 mosse costa al modello meno di dieci secondi di attesa, ritentativi compresi." if statistics.mean(secs) * 20 < 10 else "- In parole povere: una partita da 20 mosse costa al modello diversi secondi di attesa.")
    add(f"- Si ricalcola con: `{command_for('secondi-per-mossa', results_dir)}`")
    add("")

    add("## Cosa è contro-intuitivo")
    add("")
    n_flag = 0
    if high < 0.5:
        n_flag += 1
        add(
            f"- **L'agente addestrato perde contro il modello**, e non per sfortuna: {pct(s.win_rate)} di vittorie, e anche il limite alto della forbice ({pct(high)}) resta sotto il 50%. "
            f"Una rete addestrata apposta per questo gioco perde con un modello generico. "
            f"La partita che lo mostra: {_game_line(lost, results_dir) if lost else _game_line(worst, results_dir)}. "
            f"Il caso peggiore è la {_game_line(worst, results_dir)}."
        )
    if best.agent_score - best.llm_score >= 100:
        add(
            f"- **Ma la fortuna pesa:** la miglior partita dell'agente è un {best.agent_score}-{best.llm_score}, la {_game_line(best, results_dir)}. "
            f"Con una sola partita si può dire tutto: servono le {s.n}."
        )
    if diff_low <= 0 <= diff_high:
        n_flag += 1
        add(
            f"- **Chi apre non conta, o conta troppo poco per vederlo:** {pct(a.win_rate)} di vittorie aprendo contro {pct(b.win_rate)} lasciando aprire il modello, "
            f"con una differenza ({it(diff * 100)} punti) più piccola dell'errore. "
            "È un numero medio su 100 + 100 partite: nessuna partita singola lo mostra, vale la tabella al punto 4."
        )
    if s.invalid_moves == 0:
        n_flag += 1
        slow = s.slowest_game
        add(
            f"- **Il modello non sbaglia mai il formato:** 0 mosse non valide su {s.moves}, e in media risponde in {it(statistics.mean(secs), 2)} secondi a mossa. "
            f"La mossa più lenta ({it(max(secs), 1)} s) è nella {_game_line(slow, results_dir)}."
        )
    if gifts and s.mean_llm > s.mean_agent and total_gifts > 0:
        n_flag += 1
        llm_wins = [g for g in s.games if g.winner == LLM]
        gift_by_game = {g.game_index: avoidable_gifts([g]) for g in llm_wins}
        top_game = max(
            llm_wins,
            key=lambda g: (sum(r.points for r in gift_by_game[g.game_index]), -g.game_index),
        )
        add(
            f"- **Il modello vince anche se regala punti:** {total_gifts} regali evitabili ({gift_points} punti) nel complesso. "
            f"La partita in cui ne ha regalati di più e ha vinto lo stesso: {_game_line(top_game, results_dir)}, "
            f"con {sum(r.points for r in gift_by_game[top_game.game_index])} punti risparmiabili."
        )
    if n_flag == 0:
        add("- Nessun numero va contro l'intuizione in questa serie.")
    add("")

    add("## Le due partite da mostrare")
    add("")
    add("Scelte con una regola fissa, perché siano tipiche e non estreme: tra le partite vinte (e tra quelle perse) dall'agente, quella con il margine più vicino alla mediana del suo gruppo.")
    add("")
    for label, game, name in (("Vinta dall'agente", won, WON_FILE), ("Persa dall'agente", lost, LOST_FILE)):
        if game is None:
            add(f"- {label}: nessuna in questa serie.")
        else:
            add(f"- {label}: {_game_line(game, results_dir)}; copia in `{results_dir}/{SHOWCASE_DIR}/{name}`.")
    add("")
    add("Per vederle:")
    add("")
    add("```bash")
    for name in (WON_FILE, LOST_FILE):
        add(
            f"python -m visualization.render_game --replay {results_dir}/{SHOWCASE_DIR}/{name} "
            f"--output {results_dir}/{SHOWCASE_DIR}/{name.replace('.json', '.gif')}"
        )
    add("```")
    add("")
    add("## Come rifare tutto")
    add("")
    add("```bash")
    add("python -m training.evaluate        # gioca le partite mancanti, salva ogni partita appena finita")
    add("python -m training.numeri_video    # riscrive questo file e le due partite da mostrare")
    add("```")
    add("")
    return "\n".join(out)


def write_outputs(results_dir: Path) -> Stats:
    """Write the report and copy the two games to show.

    Returns:
        The statistics the report was built from.
    """
    results_dir = Path(results_dir)
    stats = load_stats(results_dir)
    (results_dir / REPORT_FILE).write_text(build_report(stats, results_dir.as_posix()), encoding="utf-8")
    won, lost = pick_showcase(stats.games)
    for game, name in ((won, WON_FILE), (lost, LOST_FILE)):
        target = results_dir / SHOWCASE_DIR / name
        if game is None:
            target.unlink(missing_ok=True)
        else:
            game.save(target)
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name; ``sys.argv`` when omitted.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Numbers for the video, from the saved games")
    parser.add_argument("--results-dir", default="results", help="Folder with the saved games")
    parser.add_argument("--mostra", choices=sorted(NUMBERS), help="Print one number and exit")
    args = parser.parse_args(argv)

    if args.mostra:
        print(NUMBERS[args.mostra](load_stats(Path(args.results_dir))))
        return 0
    stats = write_outputs(Path(args.results_dir))
    print(f"Written {Path(args.results_dir) / REPORT_FILE} from {stats.n} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
