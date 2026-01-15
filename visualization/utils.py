"""Utility constants and functions for Briscola game visualization."""

import base64
import os
from functools import lru_cache
from src.cards import Card, CardValue

# Path to card images
CARDS_DIR = os.path.join(os.path.dirname(__file__), "cards")


@lru_cache(maxsize=50)
def _load_image_base64(image_path: str) -> str:
    """Load an image and return as base64 data URI (cached)."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

# Suit symbols (Neapolitan deck)
SUIT_SYMBOLS = {
    "coins": "🪙",
    "cups": "🏆",
    "swords": "⚔️",
    "clubs": "🪵"
}

# Map suit names to Italian image names
SUIT_IMAGE_NAMES = {
    "coins": "denara",
    "cups": "coppe",
    "swords": "spade",
    "clubs": "bastoni"
}

# Card value display (short)
VALUE_SHORT = {
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

# Card value display (full name)
VALUE_FULL = {
    CardValue.ACE: "Ace",
    CardValue.TWO: "Two",
    CardValue.THREE: "Three",
    CardValue.FOUR: "Four",
    CardValue.FIVE: "Five",
    CardValue.SIX: "Six",
    CardValue.SEVEN: "Seven",
    CardValue.JACK: "Jack",
    CardValue.HORSE: "Horse",
    CardValue.KING: "King",
}

# Suit CSS classes
SUIT_CSS = {
    "coins": "suit-coins",
    "cups": "suit-cups",
    "swords": "suit-swords",
    "clubs": "suit-clubs",
}


def format_card(card: Card) -> str:
    """Format card as 'Value Symbol' (e.g., 'A 🪙')."""
    return f"{VALUE_SHORT[card.value]} {SUIT_SYMBOLS[card.suit]}"


def get_card_image_path(card: Card) -> str:
    """Get the file path for a card's image."""
    suit_name = SUIT_IMAGE_NAMES[card.suit]
    # CardValue int maps directly to image number (1=Ace, 2=Two, ..., 10=King)
    number = int(card.value)
    return os.path.join(CARDS_DIR, f"{suit_name}{number}.png")


def get_card_back_image_path() -> str:
    """Get the file path for the card back image."""
    return os.path.join(CARDS_DIR, "card-back.png")


def get_card_image_data(card: Card) -> str:
    """Get base64 data URI for a card's image."""
    return _load_image_base64(get_card_image_path(card))


def get_card_back_image_data() -> str:
    """Get base64 data URI for the card back image."""
    return _load_image_base64(get_card_back_image_path())


def format_card_html(card: Card, face_up: bool = True) -> str:
    """Generate HTML for a card using PNG images (base64 embedded)."""
    if not face_up:
        img_data = get_card_back_image_data()
        return f'<div class="card face-down"><img src="{img_data}" alt="card back"></div>'

    img_data = get_card_image_data(card)
    suit_class = SUIT_CSS[card.suit]
    return f'<div class="card face-up {suit_class}"><img src="{img_data}" alt="{card}"></div>'


# CSS styles for Briscola HTML rendering
CSS = """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    width: 800px;
    height: 700px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
}

/* Player Areas */
.player-area {
    height: 160px;
    padding: 15px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
}

.opponent-area {
    background: linear-gradient(180deg, #2d1f3d 0%, #1a1a2e 100%);
    border-bottom: 2px solid #4a3a5a;
}

.agent-area {
    background: linear-gradient(0deg, #1f2d3d 0%, #1a1a2e 100%);
    border-top: 2px solid #3a4a5a;
}

.player-info {
    text-align: center;
    min-width: 100px;
}

.player-name {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 8px;
}

.player-name.llm {
    color: #9d4edd;
}

.player-name.rl {
    color: #00d4ff;
}

.score {
    font-size: 28px;
    font-weight: bold;
    color: #ffffff;
}

.score-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
}

.hand {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex: 1;
}

/* Cards */
.card {
    width: 70px;
    height: 100px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    font-weight: bold;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}

.card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 6px;
}

.card.face-up {
    background: transparent;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.card.face-down {
    background: transparent;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.card .suit {
    font-size: 20px;
    margin-top: 4px;
}

.card.played {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}

.card.highlight {
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
}

/* Suit colors */
.suit-coins {
    color: #d4a017;
    border: 2px solid #d4a017;
}

.suit-cups {
    color: #c0392b;
    border: 2px solid #c0392b;
}

.suit-swords {
    color: #2980b9;
    border: 2px solid #2980b9;
}

.suit-clubs {
    color: #27ae60;
    border: 2px solid #27ae60;
}

/* Table Center */
.table-center {
    flex: 1;
    background: radial-gradient(ellipse at center, #1e3a2f 0%, #152a22 100%);
    border: 3px solid #2d5a4a;
    display: flex;
    align-items: center;
    justify-content: space-around;
    position: relative;
    padding: 20px;
}

.briscola-area {
    text-align: center;
}

.briscola-label {
    font-size: 12px;
    color: #8ab89a;
    text-transform: uppercase;
    margin-bottom: 8px;
    letter-spacing: 1px;
}

.briscola-card {
    transform: rotate(-5deg);
}

.briscola-card .card {
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.4);
    border: 2px solid gold !important;
}

.trick-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}

.trick-cards {
    display: flex;
    gap: 30px;
    align-items: center;
}

.trick-card-wrapper {
    text-align: center;
}

.trick-card-label {
    font-size: 11px;
    color: #6a8a7a;
    margin-bottom: 5px;
    text-transform: uppercase;
}

.trick-card-wrapper .card {
    transform: scale(0.95);
}

.trick-card-wrapper.winner .card {
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.6);
    transform: scale(1.05);
}

.trick-empty {
    width: 70px;
    height: 100px;
    border: 2px dashed #3a5a4a;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4a6a5a;
    font-size: 12px;
}

.deck-area {
    text-align: center;
}

.deck-label {
    font-size: 12px;
    color: #8ab89a;
    text-transform: uppercase;
    margin-bottom: 8px;
    letter-spacing: 1px;
}

.deck-count {
    width: 70px;
    height: 100px;
    background: linear-gradient(145deg, #2a3a5a 0%, #1a2a3a 100%);
    border: 2px solid #3a4a6a;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #8a9aba;
}

.deck-number {
    font-size: 28px;
    font-weight: bold;
}

.deck-text {
    font-size: 10px;
    text-transform: uppercase;
}

/* Info Bar */
.info-bar {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.6);
    padding: 8px 20px;
    border-radius: 20px;
    color: #ffffff;
    font-size: 14px;
    display: flex;
    gap: 20px;
    align-items: center;
}

.trick-number {
    color: #8ab89a;
}

.current-turn {
    font-weight: bold;
}

.current-turn.llm {
    color: #9d4edd;
}

.current-turn.rl {
    color: #00d4ff;
}

/* Message Banner */
.message {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0, 0, 0, 0.85);
    padding: 15px 30px;
    border-radius: 10px;
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    border: 2px solid #4a5a6a;
    z-index: 100;
    max-width: 80%;
}

.message.winner {
    background: linear-gradient(135deg, #1a3a2a 0%, #0f2518 100%);
    border-color: #4caf50;
    color: #8bc34a;
}

.message.loser {
    background: linear-gradient(135deg, #3a1a1a 0%, #250f0f 100%);
    border-color: #f44336;
    color: #ef9a9a;
}

/* Leading indicator */
.leading-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: #ffc107;
    color: #000;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: bold;
    text-transform: uppercase;
}

/* Empty card slot */
.card-slot {
    width: 70px;
    height: 100px;
    border: 2px dashed #3a4a5a;
    border-radius: 8px;
    opacity: 0.3;
}
"""
