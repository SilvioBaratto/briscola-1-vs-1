"""Pydantic schemas for the Briscola API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import Card and CardValue for type conversion
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cards import Card, CardValue


# -----------------------------------------------------------------------------
# Card Schemas
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


# -----------------------------------------------------------------------------
# Game State Schemas
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Game History Schemas
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Model Schemas
# -----------------------------------------------------------------------------

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
