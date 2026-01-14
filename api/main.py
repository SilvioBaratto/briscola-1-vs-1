"""FastAPI backend for Briscola card game with RL opponent."""

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging - must be before importing modules that use logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

from api.config import settings
from src.briscola_env import BriscolaEnv, BriscolaState
from src.cards import Card, CardValue, Deck
from src.models import ActorCritic, PPO


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class CardSchema(BaseModel):
    """Representation of a card for the API."""
    value: str
    suit: str
    points: int
    display_name: str

    @classmethod
    def from_card(cls, card: Card) -> "CardSchema":
        """Convert a Card object to CardSchema."""
        value_names = {
            CardValue.ACE: "A",
            CardValue.TWO: "2",
            CardValue.THREE: "3",
            CardValue.FOUR: "4",
            CardValue.FIVE: "5",
            CardValue.SIX: "6",
            CardValue.SEVEN: "7",
            CardValue.JACK: "J",
            CardValue.HORSE: "H",
            CardValue.KING: "K",
        }
        return cls(
            value=card.value.name,
            suit=card.suit,
            points=card.get_points(),
            display_name=f"{value_names[card.value]} {card.suit}",
        )


class HandCardSchema(BaseModel):
    """A card in the player's hand with its index."""
    index: int
    card: CardSchema


class TrickCardSchema(BaseModel):
    """A card played in the current trick."""
    player: str  # "player" or "opponent"
    card: CardSchema


class GameStateResponse(BaseModel):
    """Current game state response."""
    game_id: str
    player_hand: List[HandCardSchema]
    opponent_hand_size: int
    trump_card: CardSchema
    trump_suit: str
    player_score: int
    opponent_score: int
    cards_in_deck: int
    player_leads: bool
    current_trick: List[TrickCardSchema]
    game_over: bool
    winner: Optional[str] = None
    message: Optional[str] = None


class NewGameRequest(BaseModel):
    """Request to start a new game."""
    player_starts: Optional[bool] = None  # None = random


class NewGameResponse(BaseModel):
    """Response when starting a new game."""
    game_id: str
    state: GameStateResponse
    message: str


class PlayCardRequest(BaseModel):
    """Request to play a card."""
    card_index: int = Field(..., ge=0, le=2, description="Index of card in hand (0-2)")


class TrickResult(BaseModel):
    """Result of a completed trick."""
    player_card: CardSchema
    opponent_card: CardSchema
    winner: str  # "player" or "opponent"
    points_won: int


class PlayCardResponse(BaseModel):
    """Response after playing a card."""
    state: GameStateResponse
    trick_result: Optional[TrickResult] = None
    message: str


class GameHistoryEntry(BaseModel):
    """A single entry in the game history."""
    trick_number: int
    player_card: CardSchema
    opponent_card: CardSchema
    winner: str
    points: int
    player_score_after: int
    opponent_score_after: int


class GameHistoryResponse(BaseModel):
    """Full game history response."""
    game_id: str
    history: List[GameHistoryEntry]
    total_tricks: int
    final_player_score: Optional[int] = None
    final_opponent_score: Optional[int] = None
    game_over: bool
    winner: Optional[str] = None


class ModelLoadRequest(BaseModel):
    """Request to load a model checkpoint."""
    checkpoint_path: str


class ModelLoadResponse(BaseModel):
    """Response after loading a model."""
    success: bool
    message: str
    model_info: Optional[Dict[str, Any]] = None


class ModelStatusResponse(BaseModel):
    """Response with current model status."""
    loaded: bool
    checkpoint_path: Optional[str] = None
    obs_dim: int
    action_dim: int
    hidden_dim: int
    device: str
    buffer_size: int
    games_collected: int


class ModelSaveRequest(BaseModel):
    """Request to save the model."""
    checkpoint_path: str


class ModelSaveResponse(BaseModel):
    """Response after saving the model."""
    success: bool
    message: str
    checkpoint_path: str


class TrainRequest(BaseModel):
    """Request to train the model on collected experiences."""
    min_experiences: int = Field(
        default=64,
        ge=1,
        description="Minimum number of experiences required to train",
    )


class TrainResponse(BaseModel):
    """Response after training."""
    success: bool
    message: str
    metrics: Optional[Dict[str, float]] = None
    experiences_used: int


# -----------------------------------------------------------------------------
# Game Session Manager
# -----------------------------------------------------------------------------

class GameSession:
    """Represents a single game session."""

    def __init__(self, game_id: str, env: BriscolaEnv, player_leads: bool):
        self.game_id = game_id
        self.env = env
        self.player_leads_first = player_leads
        self.history: List[GameHistoryEntry] = []
        self.trick_number = 0
        self.created_at = datetime.utcnow()
        self.last_observation: Optional[np.ndarray] = None
        self.pending_trick: Optional[Dict] = None  # For collecting experiences


class GameManager:
    """Manages all active game sessions."""

    def __init__(self):
        self.games: Dict[str, GameSession] = {}
        self.max_games = settings.max_active_games

    def create_game(self, player_starts: Optional[bool] = None) -> GameSession:
        """Create a new game session."""
        # Clean up old games if too many
        if len(self.games) >= self.max_games:
            oldest_id = min(
                self.games.keys(),
                key=lambda k: self.games[k].created_at,
            )
            del self.games[oldest_id]

        game_id = str(uuid.uuid4())
        # In API: human is "player", RL agent is "opponent"
        # So rewards should be from opponent's (agent's) perspective for training
        env = BriscolaEnv(reward_perspective="opponent")
        obs = env.reset()

        # Override random start if specified
        if player_starts is not None:
            env.state.player_is_leading = player_starts

        session = GameSession(game_id, env, env.state.player_is_leading)
        session.last_observation = obs
        self.games[game_id] = session
        return session

    def get_game(self, game_id: str) -> Optional[GameSession]:
        """Get a game session by ID."""
        return self.games.get(game_id)

    def remove_game(self, game_id: str) -> None:
        """Remove a game session."""
        if game_id in self.games:
            del self.games[game_id]


# -----------------------------------------------------------------------------
# RL Model Manager
# -----------------------------------------------------------------------------

class RLModelManager:
    """Manages the RL model for the opponent."""

    # Use absolute path relative to project root
    DEFAULT_CHECKPOINT = str(Path(__file__).parent.parent / "checkpoints" / "best_model.pt")

    def __init__(self):
        self.device = settings.get_device()

        # Create environment to get dimensions
        temp_env = BriscolaEnv()
        self.obs_dim = temp_env.obs_dim
        self.action_dim = temp_env.action_dim

        # Initialize model and PPO (hidden_dim=128 to match saved checkpoint)
        self.hidden_dim = 128
        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )
        self.ppo = PPO(
            model=self.model,
            device=self.device,
        )

        self.checkpoint_path: Optional[str] = None
        self.games_collected = 0

        # Auto-load best model if it exists
        self._auto_load_checkpoint()

    def _auto_load_checkpoint(self) -> None:
        """Automatically load the best model checkpoint if it exists."""
        if os.path.exists(self.DEFAULT_CHECKPOINT):
            try:
                self.ppo.load(self.DEFAULT_CHECKPOINT)
                self.checkpoint_path = self.DEFAULT_CHECKPOINT
                logger.info(f"Loaded checkpoint: {self.DEFAULT_CHECKPOINT}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.info("Starting with untrained model")
        else:
            logger.warning(f"No checkpoint found at {self.DEFAULT_CHECKPOINT}")
            logger.info("Starting with untrained model")

    def auto_save_checkpoint(self) -> None:
        """Automatically save to the default checkpoint path."""
        try:
            os.makedirs(os.path.dirname(self.DEFAULT_CHECKPOINT) or ".", exist_ok=True)
            self.ppo.save(self.DEFAULT_CHECKPOINT)
            self.checkpoint_path = self.DEFAULT_CHECKPOINT
            logger.info(f"Auto-saved checkpoint: {self.DEFAULT_CHECKPOINT}")
        except Exception as e:
            logger.error(f"Failed to auto-save checkpoint: {e}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a model checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.ppo.load(path)
        self.checkpoint_path = path

        # Get model info
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save the current model."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.ppo.save(path)
        self.checkpoint_path = path

    def get_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Get action from the RL model."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, _, _ = self.model.get_action(
                obs_tensor,
                deterministic=deterministic,
                action_mask=mask_tensor,
            )

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def collect_experience(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        action_mask: np.ndarray,
    ) -> None:
        """Add experience to the training buffer."""
        self.ppo.buffer.add(
            obs=obs.flatten(),
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            action_mask=action_mask,
        )

    def train(self, min_experiences: int = 64) -> Optional[Dict[str, float]]:
        """Train on collected experiences."""
        buffer_size = len(self.ppo.buffer)
        if buffer_size < min_experiences:
            return None

        # Compute returns and advantages
        self.ppo.buffer.compute_returns_and_advantages(
            gamma=self.ppo.gamma,
            gae_lambda=self.ppo.gae_lambda,
        )

        # Run PPO update
        metrics = self.ppo.update()
        self.games_collected = 0  # Reset counter after training

        return metrics

    def get_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "loaded": self.checkpoint_path is not None,
            "checkpoint_path": self.checkpoint_path,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "device": self.device,
            "buffer_size": len(self.ppo.buffer),
            "games_collected": self.games_collected,
        }


# -----------------------------------------------------------------------------
# Global State
# -----------------------------------------------------------------------------

game_manager: Optional[GameManager] = None
model_manager: Optional[RLModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global game_manager, model_manager

    # Startup
    game_manager = GameManager()
    model_manager = RLModelManager()

    yield

    # Shutdown
    game_manager = None
    model_manager = None


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Briscola RL API",
    description="API for playing Briscola against an RL-trained opponent",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def build_game_state_response(session: GameSession) -> GameStateResponse:
    """Build a GameStateResponse from a game session."""
    state = session.env.state
    assert state is not None

    # Build player hand
    player_hand = [
        HandCardSchema(
            index=i,
            card=CardSchema.from_card(card),
        )
        for i, card in enumerate(state.player_hand)
    ]

    # Build current trick
    current_trick = []
    for player_id, card in state.current_trick:
        current_trick.append(
            TrickCardSchema(
                player="player" if player_id == 0 else "opponent",
                card=CardSchema.from_card(card),
            )
        )

    # Determine game over and winner
    game_over = (
        len(state.player_hand) == 0
        and len(state.opponent_hand) == 0
        and len(state.current_trick) == 0
    )

    winner = None
    if game_over:
        if state.player_score > state.opponent_score:
            winner = "player"
        elif state.opponent_score > state.player_score:
            winner = "opponent"
        else:
            winner = "tie"

    return GameStateResponse(
        game_id=session.game_id,
        player_hand=player_hand,
        opponent_hand_size=len(state.opponent_hand),
        trump_card=CardSchema.from_card(state.briscola_card),
        trump_suit=state.briscola_suit,
        player_score=state.player_score,
        opponent_score=state.opponent_score,
        cards_in_deck=len(state.deck),
        player_leads=state.player_is_leading,
        current_trick=current_trick,
        game_over=game_over,
        winner=winner,
    )


def get_opponent_action(
    session: GameSession,
    obs: np.ndarray,
    opponent_hand: List[Card],
) -> int:
    """Get the opponent's action using the RL model."""
    # Build action mask for opponent
    action_mask = np.zeros(3, dtype=bool)
    action_mask[: len(opponent_hand)] = True

    # Get action from RL model
    action, _, _ = model_manager.get_action(obs, action_mask, deterministic=False)

    # Ensure action is valid
    if action >= len(opponent_hand):
        action = 0  # Fallback to first card

    return action


# -----------------------------------------------------------------------------
# Game Endpoints
# -----------------------------------------------------------------------------

def get_opponent_action_fn(session: GameSession):
    """Create opponent action function for a session."""
    def opponent_action_fn(
        hand: List[Card],
        briscola: Card,
        game_history: List[Card],
        opponent_card: Optional[Card],
    ) -> int:
        """Get opponent action from RL model."""
        action_mask = np.zeros(3, dtype=bool)
        action_mask[: len(hand)] = True
        obs = session.env._get_observation()
        action, _, _ = model_manager.get_action(obs, action_mask, deterministic=False)
        if action >= len(hand):
            action = len(hand) - 1 if len(hand) > 0 else 0
        return action
    return opponent_action_fn


@app.post("/game/new", response_model=NewGameResponse, tags=["Game"])
async def create_new_game(request: NewGameRequest = None):
    """Start a new game of Briscola."""
    if request is None:
        request = NewGameRequest()

    session = game_manager.create_game(player_starts=request.player_starts)

    # If opponent leads first, have them play their card immediately
    if not session.env.state.player_is_leading:
        opponent_action_fn = get_opponent_action_fn(session)
        opponent_card = session.env.opponent_lead(opponent_action_fn)
        message = f"Game started. Opponent leads with {opponent_card}. Your turn to respond."
    else:
        message = "Game started. You lead first."

    state = build_game_state_response(session)

    return NewGameResponse(
        game_id=session.game_id,
        state=state,
        message=message,
    )


@app.get("/game/{game_id}/state", response_model=GameStateResponse, tags=["Game"])
async def get_game_state(game_id: str):
    """Get the current state of a game."""
    session = game_manager.get_game(game_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game not found: {game_id}",
        )

    return build_game_state_response(session)


@app.post("/game/{game_id}/play", response_model=PlayCardResponse, tags=["Game"])
async def play_card(game_id: str, request: PlayCardRequest):
    """
    Play a card from your hand.

    If you are leading, the opponent will respond.
    If opponent led, you are responding to their card.
    After the trick is resolved, if opponent leads next, their card is played immediately.
    """
    session = game_manager.get_game(game_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game not found: {game_id}",
        )

    state = session.env.state
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Game not initialized",
        )

    # Check if game is already over
    if len(state.player_hand) == 0 and len(state.opponent_hand) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Game is already over",
        )

    # Validate card index
    if request.card_index >= len(state.player_hand):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid card index: {request.card_index}. You only have {len(state.player_hand)} cards.",
        )

    # Store the current observation for experience collection
    current_obs = session.last_observation

    # Get the player's card
    player_card_played = state.player_hand[request.card_index]

    # Capture agent's hand BEFORE the trick (agent = opponent in the environment)
    agent_hand_before = list(state.opponent_hand)

    # Check if player is responding to opponent's lead or leading themselves
    is_responding = len(state.current_trick) == 1 and state.current_trick[0][0] == 1

    try:
        if is_responding:
            # Player is responding to opponent's lead
            opponent_card_played = state.current_trick[0][1]
            obs, reward, done, info = session.env.player_respond(request.card_index)
            # info now contains player_card and opponent_card directly
        else:
            # Player is leading - use the original step function
            opponent_action_fn = get_opponent_action_fn(session)
            obs, reward, done, info = session.env.step(
                action=request.card_index,
                opponent_action_fn=opponent_action_fn,
            )
            # Get opponent card from played_cards (player led, opponent responded)
            opponent_card_played = state.played_cards[-1]

        session.last_observation = obs
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing game step: {str(e)}",
        )

    # Build trick result
    session.trick_number += 1
    trick_winner = "player" if info["winner"] == 0 else "opponent"
    trick_points = info["trick_points"]

    # Log agent's hand and action (agent-focused logging)
    agent_hand_str = ", ".join(str(c) for c in agent_hand_before)
    logger.info(f"Agent hand: [{agent_hand_str}] -> played {opponent_card_played}")

    trick_result = TrickResult(
        player_card=CardSchema.from_card(player_card_played),
        opponent_card=CardSchema.from_card(opponent_card_played),
        winner=trick_winner,
        points_won=trick_points,
    )

    # Add to history
    session.history.append(
        GameHistoryEntry(
            trick_number=session.trick_number,
            player_card=CardSchema.from_card(player_card_played),
            opponent_card=CardSchema.from_card(opponent_card_played),
            winner=trick_winner,
            points=trick_points,
            player_score_after=info["player_score"],
            opponent_score_after=info["opponent_score"],
        )
    )

    # Collect experience for training (from player's perspective)
    action_mask = np.zeros(3, dtype=bool)
    action_mask[: len(state.player_hand) + 1] = True  # +1 because we just played

    # Get value estimate for the current state
    if current_obs is not None:
        _, log_prob, value = model_manager.get_action(
            current_obs,
            action_mask,
            deterministic=True,
        )

        model_manager.collect_experience(
            obs=current_obs,
            action=request.card_index,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            action_mask=action_mask,
        )

    if done:
        model_manager.games_collected += 1
        # Auto-save the model after each game to continuously improve
        model_manager.auto_save_checkpoint()

    # If opponent leads next and game isn't over, have them play their leading card
    if not done and not state.player_is_leading:
        opponent_action_fn = get_opponent_action_fn(session)
        next_opponent_card = session.env.opponent_lead(opponent_action_fn)
        logger.debug(f"Opponent leads next with: {next_opponent_card}")

    # Build response
    game_state = build_game_state_response(session)

    if done:
        if game_state.winner == "player":
            message = f"Game over! You won {info['player_score']} to {info['opponent_score']}!"
        elif game_state.winner == "opponent":
            message = f"Game over! Opponent won {info['opponent_score']} to {info['player_score']}."
        else:
            message = f"Game over! It's a tie at {info['player_score']} points each."
    else:
        if trick_winner == "player":
            message = f"You won the trick and gained {trick_points} points."
        else:
            message = f"Opponent won the trick and gained {trick_points} points."

        if game_state.player_leads:
            message += " You lead next."
        else:
            # Opponent already played their lead, show it
            if len(state.current_trick) == 1:
                opp_card = state.current_trick[0][1]
                message += f" Opponent leads with {opp_card}. Your turn to respond."

    return PlayCardResponse(
        state=game_state,
        trick_result=trick_result,
        message=message,
    )


@app.get("/game/{game_id}/history", response_model=GameHistoryResponse, tags=["Game"])
async def get_game_history(game_id: str):
    """Get the full history of a game."""
    session = game_manager.get_game(game_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game not found: {game_id}",
        )

    state = session.env.state
    game_over = (
        state is not None
        and len(state.player_hand) == 0
        and len(state.opponent_hand) == 0
    )

    winner = None
    final_player_score = None
    final_opponent_score = None

    if game_over and state is not None:
        final_player_score = state.player_score
        final_opponent_score = state.opponent_score
        if state.player_score > state.opponent_score:
            winner = "player"
        elif state.opponent_score > state.player_score:
            winner = "opponent"
        else:
            winner = "tie"

    return GameHistoryResponse(
        game_id=game_id,
        history=session.history,
        total_tricks=session.trick_number,
        final_player_score=final_player_score,
        final_opponent_score=final_opponent_score,
        game_over=game_over,
        winner=winner,
    )


@app.delete("/game/{game_id}", tags=["Game"])
async def delete_game(game_id: str):
    """Delete/end a game session."""
    session = game_manager.get_game(game_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game not found: {game_id}",
        )

    game_manager.remove_game(game_id)
    return {"message": f"Game {game_id} deleted successfully"}


@app.get("/games", tags=["Game"])
async def list_games():
    """List all active game sessions."""
    games_info = []
    for game_id, session in game_manager.games.items():
        state = session.env.state
        game_over = (
            state is not None
            and len(state.player_hand) == 0
            and len(state.opponent_hand) == 0
        )
        games_info.append({
            "game_id": game_id,
            "created_at": session.created_at.isoformat(),
            "trick_number": session.trick_number,
            "game_over": game_over,
            "player_score": state.player_score if state else 0,
            "opponent_score": state.opponent_score if state else 0,
        })
    return {"games": games_info, "total": len(games_info)}


# -----------------------------------------------------------------------------
# Model Endpoints
# -----------------------------------------------------------------------------

@app.post("/model/load", response_model=ModelLoadResponse, tags=["Model"])
async def load_model(request: ModelLoadRequest):
    """Load a trained model checkpoint."""
    try:
        model_info = model_manager.load_checkpoint(request.checkpoint_path)
        return ModelLoadResponse(
            success=True,
            message=f"Model loaded from {request.checkpoint_path}",
            model_info=model_info,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.get("/model/status", response_model=ModelStatusResponse, tags=["Model"])
async def get_model_status():
    """Get the current status of the RL model."""
    status_info = model_manager.get_status()
    return ModelStatusResponse(**status_info)


@app.get("/model/checkpoints", tags=["Model"])
async def list_checkpoints():
    """List available model checkpoints in the checkpoints directory."""
    checkpoint_dir = Path(settings.default_checkpoint_dir)
    checkpoints = []

    if checkpoint_dir.exists():
        for path in checkpoint_dir.rglob("*.pt"):
            stat = path.stat()
            checkpoints.append({
                "path": str(path),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        # Also check for .pth files
        for path in checkpoint_dir.rglob("*.pth"):
            stat = path.stat()
            checkpoints.append({
                "path": str(path),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x["modified_at"], reverse=True)

    return {
        "checkpoints": checkpoints,
        "total": len(checkpoints),
        "checkpoint_dir": str(checkpoint_dir),
    }


@app.post("/model/save", response_model=ModelSaveResponse, tags=["Model"])
async def save_model(request: ModelSaveRequest):
    """Save the current model to a checkpoint file."""
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized",
        )
    try:
        model_manager.save_checkpoint(request.checkpoint_path)
        return ModelSaveResponse(
            success=True,
            message=f"Model saved to {request.checkpoint_path}",
            checkpoint_path=request.checkpoint_path,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save model: {str(e)}",
        )


@app.post("/model/train", response_model=TrainResponse, tags=["Model"])
async def train_model(request: Optional[TrainRequest] = None):
    """
    Train the model on collected game experiences.

    This performs a batch PPO update using all experiences collected
    since the last training run.
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized",
        )

    if request is None:
        request = TrainRequest()

    buffer_size = len(model_manager.ppo.buffer)

    if buffer_size < request.min_experiences:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not enough experiences. Have {buffer_size}, need {request.min_experiences}.",
        )

    try:
        metrics = model_manager.train(min_experiences=request.min_experiences)

        if metrics is None:
            return TrainResponse(
                success=False,
                message="Training skipped - not enough experiences",
                metrics=None,
                experiences_used=0,
            )

        return TrainResponse(
            success=True,
            message=f"Training completed using {buffer_size} experiences",
            metrics=metrics,
            experiences_used=buffer_size,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_games": len(game_manager.games) if game_manager else 0,
        "model_loaded": model_manager.checkpoint_path is not None if model_manager else False,
    }


# -----------------------------------------------------------------------------
# Run with Uvicorn
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
