"""
Render Briscola game (RL Agent vs LLM) as GIF or MP4 using HTML + Playwright.
"""

import argparse
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from playwright.sync_api import sync_playwright, Playwright, Browser, Page

from src.briscola_env import BriscolaEnv
from src.cards import Card
from src.models.actor_critic import ActorCritic
from visualization.utils import (
    CSS, SUIT_CSS, VALUE_SHORT, SUIT_SYMBOLS,
    get_card_image_data, get_card_back_image_data
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PlayerState:
    """State of a player for rendering."""
    name: str
    hand: List[Card]
    score: int
    is_rl: bool
    is_leading: bool


@dataclass
class GameFrame:
    """A single frame of the game for rendering."""
    trick_number: int
    phase: str  # "start", "rl_plays", "llm_plays", "trick_end", "game_over"
    rl_player: PlayerState
    llm_player: PlayerState
    briscola_card: Card
    current_trick: List[Tuple[str, Card]]  # [(player_name, card), ...]
    deck_remaining: int
    message: str
    trick_winner: Optional[str] = None
    show_llm_hand: bool = False  # Whether to reveal LLM's cards


# =============================================================================
# HTML RENDERER
# =============================================================================

def render_card_html(card: Card, face_up: bool = True, extra_class: str = "") -> str:
    """Render a single card as HTML using PNG images (base64 embedded)."""
    if not face_up:
        img_data = get_card_back_image_data()
        return f'<div class="card face-down {extra_class}"><img src="{img_data}" alt="card back"></div>'

    img_data = get_card_image_data(card)
    suit_class = SUIT_CSS[card.suit]
    return f'<div class="card face-up {suit_class} {extra_class}"><img src="{img_data}" alt="{card}"></div>'


def render_hand_html(hand: List[Card], face_up: bool = True) -> str:
    """Render a player's hand as HTML."""
    cards_html = ""
    for card in hand:
        cards_html += render_card_html(card, face_up)
    # Add empty slots for missing cards (hand can have 0-3 cards)
    for _ in range(3 - len(hand)):
        cards_html += '<div class="card-slot"></div>'
    return cards_html


def render_html(frame: GameFrame) -> str:
    """Generate complete HTML for a frame."""
    # LLM opponent area (top)
    llm = frame.llm_player
    llm_leading = '<span class="leading-badge">Leads</span>' if llm.is_leading else ""
    llm_hand_html = render_hand_html(llm.hand, face_up=frame.show_llm_hand)

    # RL agent area (bottom)
    rl = frame.rl_player
    rl_leading = '<span class="leading-badge">Leads</span>' if rl.is_leading else ""
    rl_hand_html = render_hand_html(rl.hand, face_up=True)

    # Briscola card
    briscola_html = render_card_html(frame.briscola_card, face_up=True)

    # Current trick area
    trick_html = ""
    llm_card_html = '<div class="trick-empty">-</div>'
    rl_card_html = '<div class="trick-empty">-</div>'

    for player_name, card in frame.current_trick:
        card_html = render_card_html(card, face_up=True, extra_class="played")
        if player_name == "LLM":
            winner_class = "winner" if frame.trick_winner == "LLM" else ""
            llm_card_html = f'<div class="trick-card-wrapper {winner_class}">' \
                           f'<div class="trick-card-label">LLM</div>{card_html}</div>'
        else:
            winner_class = "winner" if frame.trick_winner == "RL Agent" else ""
            rl_card_html = f'<div class="trick-card-wrapper {winner_class}">' \
                          f'<div class="trick-card-label">RL</div>{card_html}</div>'

    # Format trick cards (LLM on top, RL on bottom for vertical layout)
    if frame.current_trick:
        trick_html = f'''
        <div class="trick-cards">
            {llm_card_html}
            {rl_card_html}
        </div>
        '''
    else:
        trick_html = '''
        <div class="trick-cards">
            <div class="trick-empty">-</div>
            <div class="trick-empty">-</div>
        </div>
        '''

    # Deck count
    deck_html = f'''
    <div class="deck-count">
        <div class="deck-number">{frame.deck_remaining}</div>
        <div class="deck-text">cards</div>
    </div>
    '''

    # Info bar
    turn_class = "llm" if frame.llm_player.is_leading else "rl"
    turn_name = "LLM" if frame.llm_player.is_leading else "RL"
    if frame.phase == "game_over":
        info_html = f'<span class="trick-number">Game Over</span>'
    else:
        info_html = f'''
        <span class="trick-number">Trick {frame.trick_number}/20</span>
        <span class="current-turn {turn_class}">{turn_name} leads</span>
        '''

    # Message
    message_class = ""
    if "wins" in frame.message.lower() and "rl" in frame.message.lower():
        message_class = "winner"
    elif "wins" in frame.message.lower() and "llm" in frame.message.lower():
        message_class = "loser"

    message_html = f'<div class="message {message_class}">{frame.message}</div>' if frame.message else ""

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>{CSS}</style>
</head>
<body>
    <!-- LLM Opponent (Top) -->
    <div class="player-area opponent-area">
        {llm_leading}
        <div class="player-info">
            <div class="player-name llm">LLM</div>
            <div class="score-label">Score</div>
            <div class="score">{llm.score}</div>
        </div>
        <div class="hand">
            {llm_hand_html}
        </div>
    </div>

    <!-- Table Center -->
    <div class="table-center">
        <div class="briscola-area">
            <div class="briscola-label">Trump</div>
            <div class="briscola-card">
                {briscola_html}
            </div>
        </div>

        <div class="trick-area">
            {trick_html}
        </div>

        <div class="deck-area">
            <div class="deck-label">Deck</div>
            {deck_html}
        </div>

        <div class="info-bar">
            {info_html}
        </div>
    </div>

    <!-- RL Agent (Bottom) -->
    <div class="player-area agent-area">
        {rl_leading}
        <div class="player-info">
            <div class="player-name rl">RL Agent</div>
            <div class="score-label">Score</div>
            <div class="score">{rl.score}</div>
        </div>
        <div class="hand">
            {rl_hand_html}
        </div>
    </div>

    {message_html}
</body>
</html>'''

    return html


class HTMLRenderer:
    """Renders HTML frames to PNG using Playwright."""

    def __init__(self) -> None:
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    def start(self) -> None:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.page = self.browser.new_page(viewport={'width': 800, 'height': 700})

    def render_frame(self, frame: GameFrame, path: str) -> None:
        html = render_html(frame)
        if self.page is not None:
            self.page.set_content(html)
            self.page.screenshot(path=path)

    def stop(self) -> None:
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


# =============================================================================
# GAME SIMULATION
# =============================================================================

class BriscolaGameSimulator:
    """Simulates a Briscola game between RL agent and LLM opponent."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_llm: bool = True,
        show_llm_hand: bool = True,
    ):
        self.device = device
        self.use_llm = use_llm
        self.show_llm_hand = show_llm_hand
        self.frames: List[GameFrame] = []

        # Load RL model - get dimensions from checkpoint
        self.model: Optional[ActorCritic] = None
        history_length = 0  # Default for compatibility with older checkpoints

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            # Extract dimensions from saved weights
            shared_weight = checkpoint['model_state_dict']['shared.0.weight']
            actor_out = checkpoint['model_state_dict']['actor.2.weight']
            obs_dim = shared_weight.shape[1]
            hidden_dim = shared_weight.shape[0]
            action_dim = actor_out.shape[0]

            # Determine history_length from obs_dim
            # obs_dim = 205 + 82 * history_length
            if obs_dim == 205:
                history_length = 0
            elif obs_dim == 287:
                history_length = 1
            elif obs_dim == 369:
                history_length = 2
            elif obs_dim == 533:
                history_length = 4
            else:
                history_length = 0  # Default fallback

            self.model = ActorCritic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded RL model: {model_path} (obs={obs_dim}, hidden={hidden_dim})")

        # Initialize environment with matching history_length
        self.env = BriscolaEnv(history_length=history_length)

        if self.model is None:
            # Create default model for random play
            self.model = ActorCritic(
                obs_dim=self.env.obs_dim,
                action_dim=self.env.action_dim,
            )
            print("Warning: No model loaded, using random RL agent")

        # Initialize LLM opponent
        self.llm_opponent = None
        if use_llm:
            try:
                from src.agents.llm_opponent import LLMOpponent, create_llm_opponent_action_fn
                self.llm_opponent = LLMOpponent()
                self.llm_action_fn = create_llm_opponent_action_fn(self.llm_opponent)
                print("LLM opponent initialized")
            except Exception as e:
                print(f"Warning: Could not initialize LLM opponent: {e}")
                self.llm_opponent = None

    def _get_rl_action(self, obs: np.ndarray) -> int:
        """Get action from RL model."""
        assert self.model is not None
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        mask = self.env.get_action_mask()
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _, _, _ = self.model.get_action(
                obs_tensor, deterministic=True, action_mask=mask_tensor
            )
        return int(action.item())

    def _get_llm_action(
        self,
        hand: List[Card],
        briscola: Card,
        game_history: List[Card],
        opponent_card: Optional[Card]
    ) -> int:
        """Get action from LLM opponent (or random fallback)."""
        if self.llm_opponent is not None:
            return self.llm_action_fn(hand, briscola, game_history, opponent_card)
        # Random fallback
        return random.randint(0, len(hand) - 1)

    def _create_frame(
        self,
        phase: str,
        message: str,
        current_trick: Optional[List[Tuple[str, Card]]] = None,
        trick_winner: Optional[str] = None,
    ) -> GameFrame:
        """Create a game frame from current state."""
        assert self.env.state is not None

        rl_player = PlayerState(
            name="RL Agent",
            hand=list(self.env.state.player_hand),
            score=self.env.state.player_score,
            is_rl=True,
            is_leading=self.env.state.player_is_leading,
        )

        llm_player = PlayerState(
            name="LLM",
            hand=list(self.env.state.opponent_hand),
            score=self.env.state.opponent_score,
            is_rl=False,
            is_leading=not self.env.state.player_is_leading,
        )

        return GameFrame(
            trick_number=self.env.state.trick_number + 1,
            phase=phase,
            rl_player=rl_player,
            llm_player=llm_player,
            briscola_card=self.env.state.briscola_card,
            current_trick=current_trick or [],
            deck_remaining=len(self.env.deck),
            message=message,
            trick_winner=trick_winner,
            show_llm_hand=self.show_llm_hand,
        )

    def _add_frames(self, frame: GameFrame, count: int = 1) -> None:
        """Add frame(s) to the list."""
        for _ in range(count):
            self.frames.append(frame)

    def simulate(self) -> str:
        """
        Simulate a complete game and generate frames.

        Returns:
            Winner name ("RL Agent" or "LLM")
        """
        self.frames = []

        # Reset environment
        obs = self.env.reset()
        assert self.env.state is not None

        # Initial frame
        frame = self._create_frame("start", "Game starting...")
        self._add_frames(frame, 5)

        # Play tricks
        while True:
            current_trick: List[Tuple[str, Card]] = []
            winner: str = "RL Agent"  # Default, will be set based on trick result

            if self.env.state.player_is_leading:
                # RL leads
                # Frame: RL's turn
                frame = self._create_frame("rl_plays", "RL Agent's turn...")
                self._add_frames(frame, 3)

                # RL plays
                action = self._get_rl_action(obs)
                rl_card = self.env.state.player_hand[action]
                current_trick.append(("RL Agent", rl_card))

                # Frame: RL played
                frame = self._create_frame(
                    "rl_plays",
                    f"RL plays {VALUE_SHORT[rl_card.value]} {SUIT_SYMBOLS[rl_card.suit]}",
                    current_trick=current_trick,
                )
                self._add_frames(frame, 4)

                # Execute step (LLM responds inside env.step)
                obs, _, done, info = self.env.step(action, self._get_llm_action)

                # Get LLM's response card from the last trick
                if self.env.state.trick_history:
                    last_trick = self.env.state.trick_history[-1]
                    llm_card = last_trick.responder_card
                    current_trick.append(("LLM", llm_card))

                    # Determine winner
                    winner = "RL Agent" if last_trick.winner_id == 0 else "LLM"

            else:
                # LLM leads
                # Frame: LLM's turn
                frame = self._create_frame("llm_plays", "LLM's turn...")
                self._add_frames(frame, 3)

                # LLM plays (via opponent_lead)
                llm_card = self.env.opponent_lead(self._get_llm_action)
                current_trick.append(("LLM", llm_card))

                # Frame: LLM played
                frame = self._create_frame(
                    "llm_plays",
                    f"LLM plays {VALUE_SHORT[llm_card.value]} {SUIT_SYMBOLS[llm_card.suit]}",
                    current_trick=current_trick,
                )
                self._add_frames(frame, 4)

                # Frame: RL's turn to respond
                frame = self._create_frame(
                    "rl_plays",
                    "RL Agent responds...",
                    current_trick=current_trick,
                )
                self._add_frames(frame, 3)

                # RL responds
                action = self._get_rl_action(self.env._get_observation())
                rl_card = self.env.state.player_hand[action]
                current_trick.append(("RL Agent", rl_card))

                # Execute step
                obs, _, done, info = self.env.player_respond(action)

                # Determine winner
                if self.env.state.trick_history:
                    last_trick = self.env.state.trick_history[-1]
                    winner = "RL Agent" if last_trick.winner_id == 0 else "LLM"

            # Frame: Trick result
            trick_points = info.get("trick_points", 0)
            frame = self._create_frame(
                "trick_end",
                f"{winner} wins trick! (+{trick_points} pts)",
                current_trick=current_trick,
                trick_winner=winner,
            )
            self._add_frames(frame, 6)

            if done:
                break

            # Frame: New trick starting
            frame = self._create_frame("start", f"Trick {self.env.state.trick_number + 1}")
            self._add_frames(frame, 3)

        # Game over frame
        rl_score = self.env.state.player_score
        llm_score = self.env.state.opponent_score

        if rl_score > llm_score:
            winner = "RL Agent"
            message = f"RL Agent wins! {rl_score} - {llm_score}"
        elif llm_score > rl_score:
            winner = "LLM"
            message = f"LLM wins! {llm_score} - {rl_score}"
        else:
            winner = "Tie"
            message = f"It's a tie! {rl_score} - {llm_score}"

        frame = self._create_frame("game_over", message)
        self._add_frames(frame, 15)

        return winner


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def make_video(frames_dir: str, output: str, fps: int = 5) -> bool:
    """Create MP4 video from frames."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def make_gif(frames_dir: str, output: str, fps: int = 5) -> bool:
    """Create high-quality GIF using ffmpeg with palette generation."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found")
        return False

    palette_filter = f"fps={fps},scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%04d.png",
        "-vf", palette_filter,
        "-loop", "0",
        output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Render Briscola game visualization")
    parser.add_argument("--model", default="checkpoints/best_model.pt",
                       help="Path to RL model checkpoint")
    parser.add_argument("--output", default="briscola_game.gif",
                       help="Output file (GIF or MP4)")
    parser.add_argument("--fps", type=int, default=5,
                       help="Frames per second")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-llm", action="store_true",
                       help="Use random opponent instead of LLM")
    parser.add_argument("--hide-llm-hand", action="store_true",
                       help="Hide LLM's cards (show face-down)")
    parser.add_argument("--keep-frames", action="store_true",
                       help="Keep PNG frames after generating video")
    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Using LLM: {not args.no_llm}")

    # Prepare frames directory
    frames_dir = "visualization/frames"
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frames_dir, f))

    # Simulate game
    print("Simulating game...")
    simulator = BriscolaGameSimulator(
        model_path=args.model,
        use_llm=not args.no_llm,
        show_llm_hand=not args.hide_llm_hand,
    )
    winner = simulator.simulate()
    print(f"Winner: {winner}, Frames: {len(simulator.frames)}")

    # Render frames
    print("Rendering frames with HTML...")
    renderer = HTMLRenderer()
    renderer.start()

    for i, frame in enumerate(simulator.frames):
        renderer.render_frame(frame, f"{frames_dir}/frame_{i:04d}.png")
        if (i + 1) % 50 == 0:
            print(f"  Rendered {i + 1}/{len(simulator.frames)}")

    renderer.stop()
    print(f"  Rendered {len(simulator.frames)} frames")

    # Generate output
    is_gif = args.output.lower().endswith(".gif")
    format_name = "GIF" if is_gif else "video"
    print(f"Generating {format_name}...")

    if is_gif:
        success = make_gif(frames_dir, args.output, args.fps)
    else:
        success = make_video(frames_dir, args.output, args.fps)

    if success:
        print(f"Saved: {args.output}")
        if not args.keep_frames:
            for f in os.listdir(frames_dir):
                if f.endswith(".png"):
                    os.remove(os.path.join(frames_dir, f))
    else:
        print(f"{format_name} generation failed, frames saved in: {frames_dir}")


if __name__ == "__main__":
    main()
