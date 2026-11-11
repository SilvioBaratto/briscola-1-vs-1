"""Evaluate the trained PPO agent against gemma4:31b-cloud, one saved game at a time.

Every finished game is written to ``<results-dir>/partite/partita_NNN.json`` right
away. Running the command again skips the games already on disk and continues from
the first missing one, so an interrupted run (Ollama down, a usage limit, Ctrl-C)
never costs more than the game in progress.

Deck ``i`` is shuffled with ``seed + i`` and the agent opens the even-numbered
games, the LLM the odd-numbered ones, so the same command replays the same deals.
"""

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.agents.llm_opponent import (
    MODEL_NAME,
    LLMOpponent,
    OllamaUnavailableError,
    create_llm_opponent_action_fn,
)
from src.briscola_env import BriscolaEnv
from src.cards import Deck
from src.game_record import AGENT, LLM, TIE, GameRecord, TrickLog
from src.models.actor_critic import ActorCritic
from src.models.checkpoint import checkpoint_sha256, load_actor_critic

logger = logging.getLogger(__name__)

PROTOCOL = "briscola-eval/v1"
DEFAULT_CHECKPOINT = "checkpoints/final_model.pt"
GAMES_DIR = "partite"
RUN_FILE = "run.json"
STATUS_FILE = "stato.json"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 2
EXIT_PAUSED = 3


def agent_opens(game_index: int) -> bool:
    """Tell who plays the first trick of a game: the agent in even games, the LLM in odd ones."""
    return game_index % 2 == 0


def game_path(results_dir: Path, game_index: int) -> Path:
    """Return where the record of a game is stored."""
    return Path(results_dir) / GAMES_DIR / f"partita_{game_index:03d}.json"


def load_finished_games(results_dir: Path, num_games: int, seed: int) -> dict:
    """Read the games already saved for this run.

    Args:
        results_dir: Folder that holds the ``partite`` folder.
        num_games: Number of games the run must reach.
        seed: Deck seed of game 0.

    Returns:
        Map from game index to its record, for the games that are complete.

    Raises:
        ValueError: If a saved game does not belong to this run (other deck seed or
            opener), which would mix results of different experiments.
    """
    finished = {}
    for index in range(num_games):
        path = game_path(results_dir, index)
        if not path.exists():
            continue
        try:
            record = GameRecord.load(path)
        except (ValueError, OSError):
            # Unreadable file: play the game again and overwrite it.
            logger.warning(f"Ignoring unreadable game file {path}")
            continue
        if record.game_index != index or record.deck_seed != seed + index or record.agent_opens != agent_opens(index):
            raise ValueError(f"{path} was played with other deck seeds or openers than this run")
        if len(record.tricks) != 20:
            continue
        finished[index] = record
    return finished


def _portable_path(path: str) -> str:
    """Return a path that is safe to commit: relative to the working folder, or just the file name."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def check_run_identity(results_dir: Path, identity: dict) -> None:
    """Write ``run.json`` the first time, and refuse to continue a run made with other settings.

    Raises:
        ValueError: If ``results_dir`` already holds a run with a different model,
            checkpoint or seed.
    """
    path = Path(results_dir) / RUN_FILE
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(
                f"{path} describes a different run ({saved}); use another --results-dir "
                "or move the old results away"
            )
        return
    _write_json(path, identity)


def write_status(results_dir: Path, num_games: int, games_done: int, status: str, reason: str) -> None:
    """Record how far the run got and why it stopped, next to the games."""
    _write_json(
        Path(results_dir) / STATUS_FILE,
        {
            "games_expected": num_games,
            "games_done": games_done,
            "status": status,
            "reason": reason,
            "updated_at": _utc_now(),
        },
    )


def agent_action(model: ActorCritic, env: BriscolaEnv, obs: np.ndarray) -> int:
    """Greedy action of the trained agent for an observation."""
    mask = torch.BoolTensor(env.get_action_mask()).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _, _ = model.get_action(
            torch.FloatTensor(obs).unsqueeze(0), deterministic=True, action_mask=mask
        )
    return int(action.item())


def play_game(
    env: BriscolaEnv,
    model: ActorCritic,
    opponent: LLMOpponent,
    game_index: int,
    seed: int,
    checkpoint: str,
) -> GameRecord:
    """Play one full game and log every trick.

    When the LLM opens, the agent answers after seeing its card, as it does in the
    app and in the visualization.

    Args:
        env: Environment, built with the history length of the agent.
        model: Trained network, in evaluation mode.
        opponent: LLM opponent; its `last_move` is read after each of its moves.
        game_index: Position of the game in the run; decides the opener.
        seed: Deck seed of game 0; this game's deck uses ``seed + game_index``.
        checkpoint: Portable checkpoint name stored in the record.

    Returns:
        The finished record.

    Raises:
        OllamaUnavailableError: If the opponent is strict and Ollama stops answering.
    """
    deck_seed = seed + game_index
    opens = agent_opens(game_index)
    # The random fallback for unusable answers draws from the global generator.
    random.seed(deck_seed)

    deck_order = Deck.shuffled_cards(deck_seed)
    obs = env.reset(deck_order=deck_order, player_is_leading=opens)
    assert env.state is not None
    record = GameRecord(
        game_index=game_index,
        deck_seed=deck_seed,
        agent_opens=opens,
        deck_order=[card.to_string() for card in deck_order],
        briscola=env.state.briscola_card.to_string(),
        llm_model=MODEL_NAME,
        checkpoint=checkpoint,
    )

    ask_llm = create_llm_opponent_action_fn(opponent)
    seen: dict = {}

    def llm_action(hand, briscola, game_history, opponent_card) -> int:
        """Ask the LLM and keep the hand it saw and how the move went."""
        seen["hand"] = [card.to_string() for card in hand]
        index = ask_llm(hand, briscola, game_history, opponent_card)
        seen["move"] = opponent.last_move
        return index

    done = False
    while not done:
        agent_hand = [card.to_string() for card in env.state.player_hand]
        leader = AGENT if env.state.player_is_leading else LLM
        if leader == AGENT:
            agent_index = agent_action(model, env, obs)
            obs, _, done, _ = env.step(agent_index, llm_action)
        else:
            env.opponent_lead(llm_action)
            agent_index = agent_action(model, env, env._get_observation())
            obs, _, done, _ = env.player_respond(agent_index)

        trick = env.state.trick_history[-1]
        move = seen["move"]
        record.tricks.append(
            TrickLog(
                number=len(record.tricks) + 1,
                leader=leader,
                agent_hand=agent_hand,
                llm_hand=seen["hand"],
                agent_card=agent_hand[agent_index],
                agent_index=agent_index,
                llm_card=seen["hand"][move.card_index],
                llm_index=move.card_index,
                llm_valid=move.valid,
                llm_seconds=round(move.seconds, 3),
                winner=AGENT if trick.winner_id == 0 else LLM,
                points=trick.points,
                agent_score=env.state.player_score,
                llm_score=env.state.opponent_score,
            )
        )
    return record


@dataclass
class EvaluationConfig:
    """Settings of one evaluation run.

    Attributes:
        checkpoint: Path of the agent checkpoint.
        num_games: Games the run must reach.
        seed: Deck seed of game 0.
        results_dir: Folder for the saved games and the run files.
        device: Torch device of the agent.
        timeout_seconds: Max wait for one LLM request.
        max_minutes: Stop starting new games after this many minutes (None: no limit).
        verbose: Print one line per finished game.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    num_games: int = 200
    seed: int = 42
    results_dir: str = "results"
    device: str = "cpu"
    timeout_seconds: float = 60.0
    max_minutes: Optional[float] = None
    verbose: bool = False


def run_evaluation(
    config: EvaluationConfig,
    opponent_factory: Callable[[float], LLMOpponent] = lambda timeout: LLMOpponent(
        timeout_seconds=timeout, strict=True
    ),
) -> int:
    """Play the missing games of the run, saving each one as soon as it ends.

    Args:
        config: Run settings.
        opponent_factory: Builds the LLM opponent from the request timeout in seconds;
            replaced in tests.

    Returns:
        ``EXIT_OK`` when all games are saved, ``EXIT_INTERRUPTED`` when Ollama stopped
        answering or the run was cancelled, ``EXIT_PAUSED`` when the time limit ended
        it, ``EXIT_ERROR`` when the run cannot start.
    """
    results_dir = Path(config.results_dir)
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return EXIT_ERROR

    identity = {
        "protocol": PROTOCOL,
        "llm_model": MODEL_NAME,
        "checkpoint": _portable_path(config.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(config.checkpoint),
        "seed": config.seed,
    }
    try:
        check_run_identity(results_dir, identity)
        finished = load_finished_games(results_dir, config.num_games, config.seed)
    except ValueError as e:
        logger.error(str(e))
        return EXIT_ERROR

    missing: List[int] = [i for i in range(config.num_games) if i not in finished]
    if not missing:
        write_status(results_dir, config.num_games, len(finished), "complete", "all games saved")
        logger.info(f"All {config.num_games} games are already saved in {results_dir}")
        return EXIT_OK
    logger.info(f"{len(finished)} games saved, {len(missing)} to play (first: {missing[0]})")

    model, history_length = load_actor_critic(config.checkpoint, config.device)
    env = BriscolaEnv(history_length=history_length)
    opponent = opponent_factory(config.timeout_seconds)

    started = time.monotonic()
    status, reason, code = "complete", "all games saved", EXIT_OK
    index = missing[0]
    progress = tqdm(missing, desc="Games", unit="game")
    try:
        for index in progress:
            if config.max_minutes is not None and (time.monotonic() - started) / 60 >= config.max_minutes:
                status, reason, code = "paused", f"time limit of {config.max_minutes} minutes reached", EXIT_PAUSED
                break
            record = play_game(env, model, opponent, index, config.seed, identity["checkpoint"])
            record.save(game_path(results_dir, index))
            finished[index] = record
            write_status(results_dir, config.num_games, len(finished), "running", "")
            if config.verbose:
                logger.info(
                    f"Game {index}: {record.winner} {record.agent_score}-{record.llm_score} "
                    f"(agent opens: {record.agent_opens}, invalid LLM moves: {record.llm_invalid_moves})"
                )
    except OllamaUnavailableError as e:
        status, code = "interrupted", EXIT_INTERRUPTED
        reason = f"Ollama did not answer at game {index}: {e}"
        logger.error(reason)
    except KeyboardInterrupt:
        status, code = "interrupted", EXIT_INTERRUPTED
        reason = f"cancelled by the user at game {index}"
        logger.error(reason)
    finally:
        progress.close()

    write_status(results_dir, config.num_games, len(finished), status, reason)
    logger.info(f"{len(finished)}/{config.num_games} games saved ({status}: {reason})")
    if finished:
        outcomes = [record.winner for record in finished.values()]
        logger.info(
            f"Agent: {outcomes.count(AGENT)} wins, {outcomes.count(LLM)} losses, "
            f"{outcomes.count(TIE)} ties; run `python -m training.numeri_video` for the numbers"
        )
    if code != EXIT_OK:
        logger.info("Run the same command again to continue from the first missing game")
    return code


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments without the program name; ``sys.argv`` when omitted.

    Returns:
        The process exit code (see `run_evaluation`).
    """
    parser = argparse.ArgumentParser(
        description=f"Play the trained PPO agent against {MODEL_NAME} and save every game"
    )
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--num-games", "--num-episodes", dest="num_games", type=int, default=200,
                        help="Number of games (the agent opens the even ones, the LLM the odd ones)")
    parser.add_argument("--seed", type=int, default=42, help="Deck seed of game 0; game i uses seed + i")
    parser.add_argument("--results-dir", type=str, default="results", help="Where games and status are saved")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for one LLM request")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop starting new games after this many minutes; rerun to continue")
    parser.add_argument("--verbose", action="store_true", help="Print one line per finished game")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # One INFO line per trick (reward shaping) and per HTTP request would bury the progress bar.
    for noisy in ("src.briscola_env", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return run_evaluation(
        EvaluationConfig(
            checkpoint=args.checkpoint,
            num_games=args.num_games,
            seed=args.seed,
            results_dir=args.results_dir,
            device=args.device,
            timeout_seconds=args.timeout,
            max_minutes=args.max_minutes,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
