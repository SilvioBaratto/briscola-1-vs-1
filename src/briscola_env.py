"""Briscola game environment for RL training with LLM opponent."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.cards import Card, CardValue, Deck

logger = logging.getLogger(__name__)


@dataclass
class TrickRecord:
    """Record of a completed trick."""
    leader_card: Card      # Card played by the leader
    responder_card: Card   # Card played by the responder
    leader_id: int         # 0 = player, 1 = opponent
    winner_id: int         # 0 = player, 1 = opponent
    points: int            # Points captured in this trick


@dataclass
class BriscolaState:
    """Current game state."""
    player_hand: List[Card]
    opponent_hand: List[Card]
    played_cards: List[Card]
    player_score: int
    opponent_score: int
    briscola_card: Card  # The actual trump card (drawn last)
    briscola_suit: str   # Trump suit for quick access
    deck: Deck
    current_trick: List[Tuple[int, Card]]  # (player_id, card)
    player_is_leading: bool
    trick_number: int = 0  # Current trick number (1-20)
    trick_history: List[TrickRecord] = field(default_factory=list)  # History of completed tricks


class BriscolaEnv:
    """
    Briscola environment for 1-vs-1 training.

    Observation space:
        - One-hot encoded hand (3 cards * 40 possible cards)
        - Briscola suit encoding (4 values)
        - Played cards encoding (40 possible cards)
        - Current trick state (if opponent played first)
        - Score differential

    Action space:
        - Index of card to play from hand (0, 1, or 2)

    Reward structure:
        - Final: margin-based (+1 to +2 for win, -1 to -2 for loss)
        - Intermediate: scaled by card value lost/won
          - Losing Ace/Three: heavy penalty (-0.4 to -0.5)
          - Losing King/Horse/Jack: moderate penalty (-0.15 to -0.2)
          - Losing zero-point cards: minimal (-0.02)
        - Card waste penalty: ratio-based, scales with how bad the "trade" is
          - Ace/Three (weight >= 0.9) wasted on junk: penalty = -ratio * 0.5
            - Three → Seven: -(0.9-0.05)*0.5 = -0.43 (terrible!)
          - Trump wasted on non-trump junk: penalty = -ratio * 0.4
          - King/Horse (weight >= 0.25) wasted on junk: penalty = -ratio * 0.2
          - Low-value cards capturing anything: NO penalty
          - Any card capturing Ace/Three: NO penalty (good trade!)
        - Missed capture penalty: when responding and opponent plays Ace/Three (non-trump)
          - If we had a small briscola (2,4,5,6,7 of trump) but didn't use it: -0.54 to -0.6
          - Never give away free points when you can capture cheaply!
    """

    SUITS = ["coins", "cups", "swords", "clubs"]
    VALUES = list(CardValue)

    # Card value weights for reward calculation (higher = more important to keep)
    CARD_VALUE_WEIGHTS = {
        CardValue.ACE: 1.0,    # 11 points - most valuable (Carico)
        CardValue.THREE: 0.9,  # 10 points - second most valuable (Carico)
        CardValue.KING: 0.35,  # 4 points (Figura)
        CardValue.HORSE: 0.25, # 3 points (Figura)
        CardValue.JACK: 0.18,  # 2 points (Figura)
        CardValue.SEVEN: 0.05, # 0 points - highest non-point (Liscio Alto)
        CardValue.SIX: 0.04,   # 0 points (Liscio)
        CardValue.FIVE: 0.03,  # 0 points (Liscio)
        CardValue.FOUR: 0.02,  # 0 points (Liscio)
        CardValue.TWO: 0.01,   # 0 points - lowest (Liscio Basso)
    }

    # Briscola (trump) modifiers - trump cards are worth more strategically
    BRISCOLA_MODIFIERS = {
        CardValue.ACE: 1.5,    # Asso Briscola: ×1.5
        CardValue.THREE: 1.4,  # Tre Briscola: ×1.4
        CardValue.KING: 1.3,   # Re Briscola: ×1.3
        CardValue.HORSE: 1.3,  # Cavallo Briscola: ×1.3
        CardValue.JACK: 1.3,   # Fante Briscola: ×1.3
        CardValue.SEVEN: 1.2,  # Lisci Briscola: ×1.2
        CardValue.SIX: 1.2,
        CardValue.FIVE: 1.2,
        CardValue.FOUR: 1.2,
        CardValue.TWO: 1.2,
    }

    # Card categories for easier checks
    CARICHI = {CardValue.ACE, CardValue.THREE}  # High-value cards (10-11 pts)
    FIGURE = {CardValue.KING, CardValue.HORSE, CardValue.JACK}  # Face cards
    LISCI_ALTI = {CardValue.SEVEN}  # High non-point cards
    LISCI_BASSI = {CardValue.TWO, CardValue.FOUR, CardValue.FIVE, CardValue.SIX}  # Low cards

    def __init__(self, reward_perspective: str = "player", history_length: int = 4):
        """
        Initialize Briscola environment.

        Args:
            reward_perspective: Who receives the rewards - "player" or "opponent".
                - "player": Rewards calculated from player's perspective (default)
                - "opponent": Rewards calculated from opponent's perspective
                  Use this when the RL agent is the opponent (e.g., in API where
                  human is player and RL agent is opponent)
            history_length: Number of past tricks to include in observation.
                - Default: 4 (last 4 tricks visible to the agent)
                - Each trick adds 82 dims: leader card (40) + responder card (40) +
                  leader_id (1) + winner_id (1)
        """
        self.deck = Deck()
        self.state: Optional[BriscolaState] = None
        self.reward_perspective = reward_perspective
        self.history_length = history_length

        # Observation dimensions
        self.hand_dim = 3 * 40  # 3 cards, 40 possible card types
        self.briscola_dim = 4   # 4 suits
        self.played_dim = 40    # Track which cards have been played
        self.trick_dim = 40     # Current opponent card (if any)
        self.score_dim = 1      # Score differential

        # Trick history dimensions (per trick):
        # - Leader card: 40 (one-hot)
        # - Responder card: 40 (one-hot)
        # - Leader ID: 1 (0 = player, 1 = opponent)
        # - Winner ID: 1 (0 = player, 1 = opponent)
        self.trick_record_dim = 40 + 40 + 1 + 1  # 82 dims per trick
        self.history_dim = self.history_length * self.trick_record_dim

        self.obs_dim = (
            self.hand_dim +
            self.briscola_dim +
            self.played_dim +
            self.trick_dim +
            self.score_dim +
            self.history_dim
        )

        self.action_dim = 3  # Max 3 cards in hand

    def _apply_perspective(self, reward: float) -> float:
        """
        Apply reward perspective transformation.

        If reward_perspective is "opponent", negate the reward since
        the reward was calculated from player's perspective.
        """
        if self.reward_perspective == "opponent":
            return -reward
        return reward

    def _get_effective_weight(self, card: Card) -> float:
        """
        Get the effective strategic weight of a card, applying briscola modifier if trump.
        """
        assert self.state is not None
        base_weight = self.CARD_VALUE_WEIGHTS[card.value]
        if card.suit == self.state.briscola_suit:
            return base_weight * self.BRISCOLA_MODIFIERS[card.value]
        return base_weight

    def _get_game_phase(self) -> str:
        """
        Determine current game phase based on trick number and deck size.

        Returns: "early" (tricks 1-6), "mid" (tricks 7-14), or "end" (tricks 15-20)
        """
        assert self.state is not None
        trick_num = self.state.trick_number
        deck_size = len(self.deck)

        if trick_num <= 6 and deck_size > 20:
            return "early"
        elif trick_num <= 14 and deck_size >= 8:
            return "mid"
        else:
            return "end"

    def _get_phase_modifier(self) -> float:
        """
        Get reward modifier based on game phase.

        Early game: standard (1.0)
        Mid game: standard (1.0)
        End game: amplified (1.5)
        """
        phase = self._get_game_phase()
        if phase == "end":
            return 1.5
        return 1.0

    def _is_critical_moment(self) -> bool:
        """
        Check if this is a critical moment (close score in late game).
        """
        assert self.state is not None
        score_diff = abs(self.state.player_score - self.state.opponent_score)
        # Close game in end phase
        return self._get_game_phase() == "end" and score_diff <= 10

    def reset(self) -> np.ndarray:
        """Reset environment for new game."""
        self.deck.reset()

        # Deal initial hands
        player_hand: List[Card] = []
        for _ in range(3):
            card = self.deck.draw()
            assert card is not None, "Deck should have cards for initial deal"
            player_hand.append(card)

        opponent_hand: List[Card] = []
        for _ in range(3):
            card = self.deck.draw()
            assert card is not None, "Deck should have cards for initial deal"
            opponent_hand.append(card)

        # Reveal briscola
        briscola_card = self.deck.reveal_briscola()
        assert briscola_card is not None, "Deck should have briscola card"
        briscola_suit = briscola_card.suit

        # Random starting player
        player_is_leading = np.random.rand() > 0.5

        self.state = BriscolaState(
            player_hand=player_hand,
            opponent_hand=opponent_hand,
            played_cards=[],
            player_score=0,
            opponent_score=0,
            briscola_card=briscola_card,
            briscola_suit=briscola_suit,
            deck=self.deck,
            current_trick=[],
            player_is_leading=player_is_leading,
        )

        return self._get_observation()

    def _card_to_index(self, card: Card) -> int:
        """Convert card to unique index (0-39)."""
        suit_idx = self.SUITS.index(card.suit)
        value_idx = self.VALUES.index(card.value)
        return suit_idx * 10 + value_idx

    def _index_to_card(self, idx: int) -> Card:
        """Convert index back to card."""
        suit_idx = idx // 10
        value_idx = idx % 10
        return Card(self.VALUES[value_idx], self.SUITS[suit_idx])

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        assert self.state is not None, "State not initialized, call reset() first"
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # Encode hand (one-hot for each of 3 slots)
        for i in range(3):
            if i < len(self.state.player_hand):
                card_idx = self._card_to_index(self.state.player_hand[i])
                obs[idx + card_idx] = 1.0
            idx += 40

        # Encode briscola suit
        briscola_idx = self.SUITS.index(self.state.briscola_suit)
        obs[idx + briscola_idx] = 1.0
        idx += 4

        # Encode played cards
        for card in self.state.played_cards:
            card_idx = self._card_to_index(card)
            obs[idx + card_idx] = 1.0
        idx += 40

        # Encode current trick (opponent's card if they played first)
        if len(self.state.current_trick) == 1 and self.state.current_trick[0][0] == 1:
            card_idx = self._card_to_index(self.state.current_trick[0][1])
            obs[idx + card_idx] = 1.0
        idx += 40

        # Encode score differential (normalized to [-1, 1])
        score_diff = (self.state.player_score - self.state.opponent_score) / 120.0
        obs[idx] = score_diff
        idx += 1

        # Encode trick history (last N tricks)
        # Get the most recent tricks (up to history_length)
        recent_tricks = self.state.trick_history[-self.history_length:]

        for i in range(self.history_length):
            if i < len(recent_tricks):
                trick = recent_tricks[i]
                # Leader card (one-hot, 40 dims)
                leader_card_idx = self._card_to_index(trick.leader_card)
                obs[idx + leader_card_idx] = 1.0
                idx += 40

                # Responder card (one-hot, 40 dims)
                responder_card_idx = self._card_to_index(trick.responder_card)
                obs[idx + responder_card_idx] = 1.0
                idx += 40

                # Leader ID (1 dim: 0 = player, 1 = opponent)
                obs[idx] = float(trick.leader_id)
                idx += 1

                # Winner ID (1 dim: 0 = player, 1 = opponent)
                obs[idx] = float(trick.winner_id)
                idx += 1
            else:
                # Empty slots for early game (all zeros)
                idx += self.trick_record_dim

        return obs

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions (True = valid)."""
        assert self.state is not None, "State not initialized, call reset() first"
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[:len(self.state.player_hand)] = True
        return mask

    def _calculate_trick_reward(
        self,
        winner: int,
        player_card: Card,
        opponent_card: Card,
        trick_points: int,
        player_hand_before: Optional[List[Card]] = None,
        player_was_responding: bool = False,
        player_was_leading: bool = False
    ) -> float:
        """
        Calculate intermediate reward based on trick outcome with comprehensive reward system.

        Features:
        - Briscola modifier for trump cards
        - Efficiency bonus for capturing with lower value cards
        - Granular waste penalties (high/medium/low briscola)
        - Missed capture penalties for carichi AND figure
        - Positional rewards (leading vs responding)
        - Game phase modifiers (early/mid/end)
        - Critical moment amplification
        """
        assert self.state is not None
        briscola_suit = self.state.briscola_suit

        # Base weights for reward calculations
        player_base_weight = self.CARD_VALUE_WEIGHTS[player_card.value]
        opponent_base_weight = self.CARD_VALUE_WEIGHTS[opponent_card.value]

        player_is_trump = player_card.suit == briscola_suit
        opponent_is_trump = opponent_card.suit == briscola_suit

        # Get phase and modifiers
        phase = self._get_game_phase()
        critical_modifier = 2.0 if self._is_critical_moment() else 1.0

        # Initialize reward components
        total_reward = 0.0
        reward_breakdown = {}

        # ==================== MISSED CAPTURE PENALTIES ====================
        # Note: These are penalties for the PLAYER (human in API context)
        # We don't log these since we only care about agent mistakes
        missed_capture_penalty = 0.0
        if winner == 1 and player_was_responding and player_hand_before is not None:
            if not opponent_is_trump:  # Can only capture non-trump with trump
                # Check for small briscola (2-7 of trump)
                small_briscola_values = {
                    CardValue.TWO, CardValue.FOUR, CardValue.FIVE,
                    CardValue.SIX, CardValue.SEVEN
                }
                # Very small briscola (2-4) for capturing figure
                very_small_briscola_values = {
                    CardValue.TWO, CardValue.FOUR
                }

                had_small_briscola = any(
                    card.suit == briscola_suit and card.value in small_briscola_values
                    for card in player_hand_before
                )
                had_very_small_briscola = any(
                    card.suit == briscola_suit and card.value in very_small_briscola_values
                    for card in player_hand_before
                )

                # Missed capture of Carichi (A/3) with small briscola
                if opponent_card.value in self.CARICHI and had_small_briscola:
                    missed_capture_penalty = -opponent_base_weight * 0.6

                # Missed capture of Figure (K/C/J) with very small briscola (2-4)
                elif opponent_card.value in self.FIGURE and had_very_small_briscola:
                    missed_capture_penalty = -opponent_base_weight * 0.3

        # ==================== PLAYER WINS ====================
        if winner == 0:
            # Base reward
            base_reward = 0.05
            reward_breakdown["base"] = base_reward

            # Capture bonus (scaled by opponent card value)
            capture_bonus = opponent_base_weight * 0.4
            reward_breakdown["capture"] = capture_bonus

            # Points bonus
            points_bonus = trick_points / 120.0
            reward_breakdown["points"] = points_bonus

            # EFFICIENCY BONUS: Capturing with lower value card
            efficiency_bonus = 0.0
            efficiency_ratio = opponent_base_weight - player_base_weight
            if efficiency_ratio > 0:
                efficiency_bonus = efficiency_ratio * 0.3
                reward_breakdown["efficiency"] = efficiency_bonus

            # WASTE PENALTIES (for player - not logged, we only care about agent)
            card_waste_penalty = 0.0
            trade_ratio = player_base_weight - opponent_base_weight

            if trade_ratio > 0 and opponent_base_weight <= 0.05:  # Opponent played liscio
                if player_is_trump:
                    # Trump waste - granular by card type
                    if player_card.value in self.CARICHI:
                        card_waste_penalty = -trade_ratio * 0.6
                    elif player_card.value in self.FIGURE:
                        card_waste_penalty = -trade_ratio * 0.4
                    else:
                        card_waste_penalty = -trade_ratio * 0.1
                else:
                    # Non-trump waste
                    if player_card.value in self.CARICHI:
                        card_waste_penalty = -trade_ratio * 0.5
                    elif player_card.value in self.FIGURE:
                        card_waste_penalty = -trade_ratio * 0.2

            reward_breakdown["waste"] = card_waste_penalty

            # POSITIONAL REWARDS (when leading) - not logged for player
            positional_reward = 0.0
            if player_was_leading:
                if player_card.value in self.LISCI_BASSI or player_card.value in self.LISCI_ALTI:
                    if not player_is_trump:
                        positional_reward = 0.02  # Good opening with liscio
                elif player_card.value in self.CARICHI:
                    positional_reward = -0.15  # Bad to expose carichi
                elif player_is_trump:
                    if player_card.value in self.CARICHI:
                        positional_reward = -0.20  # Very bad to lead with high briscola
                    else:
                        positional_reward = -0.05  # Moderate risk with low briscola

            reward_breakdown["positional"] = positional_reward

            # Apply phase modifier (amplify in end game)
            if phase == "early" and player_is_trump:
                # Extra penalty for wasting briscola early
                card_waste_penalty *= 1.3
                reward_breakdown["waste"] = card_waste_penalty

            # Calculate total
            total_reward = (base_reward + capture_bonus + points_bonus +
                           efficiency_bonus + card_waste_penalty + positional_reward)

            # Apply critical moment modifier
            total_reward *= critical_modifier

        # ==================== OPPONENT (AGENT) WINS ====================
        else:
            # Base penalty (for player losing)
            base_penalty = -0.05
            reward_breakdown["base"] = base_penalty

            # Loss penalty (scaled by card value lost)
            loss_penalty = -player_base_weight * 0.5
            reward_breakdown["loss"] = loss_penalty

            # Points penalty
            points_penalty = -trick_points / 120.0
            reward_breakdown["points"] = points_penalty

            # AGENT WASTE PENALTY (logged as agent mistake)
            # This is "opponent_waste_bonus" from player perspective = agent made a mistake
            agent_waste_penalty = 0.0
            trade_ratio = opponent_base_weight - player_base_weight

            if trade_ratio > 0 and player_base_weight <= 0.05:  # Human played liscio
                if opponent_is_trump:
                    # Agent wasted trump on human's junk
                    agent_waste_penalty = trade_ratio * 0.7
                    logger.info(f"AGENT PENALTY spreco_briscola: wasted {opponent_card} "
                               f"to beat worthless {player_card} [phase={phase}]")
                elif opponent_card.value in self.CARICHI:
                    # Agent wasted carico
                    agent_waste_penalty = trade_ratio * 0.8
                    logger.info(f"AGENT PENALTY spreco_carico: wasted {opponent_card} "
                               f"to beat worthless {player_card} [phase={phase}]")
                elif opponent_card.value in self.FIGURE:
                    # Agent wasted figura
                    agent_waste_penalty = trade_ratio * 0.4
                    logger.info(f"AGENT PENALTY spreco_figura: used {opponent_card} "
                               f"to beat worthless {player_card} [phase={phase}]")

            reward_breakdown["agent_waste"] = agent_waste_penalty

            # POSITIONAL REWARDS (when responding - letting liscio pass)
            positional_reward = 0.0
            if player_was_responding:
                # Small bonus for not wasting cards on opponent's liscio
                if opponent_base_weight <= 0.05 and player_base_weight <= 0.05:
                    positional_reward = 0.01  # Saved our resources
                    reward_breakdown["positional"] = positional_reward

            # Calculate total (agent_waste_penalty is positive here = bonus for player)
            total_reward = (base_penalty + loss_penalty + points_penalty +
                           agent_waste_penalty + missed_capture_penalty + positional_reward)

            # Apply critical moment modifier
            total_reward *= critical_modifier

            # Log agent win (good for agent)
            agent_reward = -total_reward  # Flip for agent perspective
            if agent_waste_penalty == 0:
                logger.info(f"AGENT WIN: +{agent_reward:.3f} | {opponent_card} beat {player_card} "
                           f"(captured {trick_points}pts) [phase={phase}]")

        return total_reward

    def _calculate_final_reward(self) -> float:
        """
        Calculate margin-based final reward with special bonuses.

        Special bonuses (from REWARDS.md):
        - Cappotto (120-0): +3.0 / -3.0
        - Vittoria ≥91 punti: +2.5
        - Vittoria ≥80 punti: +2.0
        - Sconfitta ≤29 punti: -2.5

        Base rewards:
        - Win: +1.0 + margin_bonus (up to +1.0)
        - Loss: -1.0 - margin_penalty (up to -1.0)
        - Tie (60-60): 0.0
        """
        assert self.state is not None
        player_score = self.state.player_score
        opponent_score = self.state.opponent_score
        margin = player_score - opponent_score

        # Log from AGENT perspective (agent = opponent)
        if player_score > opponent_score:
            # Human wins = Agent loses
            if player_score == 120 and opponent_score == 0:
                total_reward = 3.0
                logger.info(f"AGENT GAME LOSS (cappotto): -3.0 | Agent: {opponent_score} - Human: {player_score}")
            elif player_score >= 91:
                total_reward = 2.5
                logger.info(f"AGENT GAME LOSS (dominant): -2.5 | Agent: {opponent_score} - Human: {player_score}")
            elif player_score >= 80:
                total_reward = 2.0
                logger.info(f"AGENT GAME LOSS (solid): -2.0 | Agent: {opponent_score} - Human: {player_score}")
            else:
                margin_bonus = min(abs(margin) / 60.0, 1.0)
                total_reward = 1.0 + margin_bonus
                logger.info(f"AGENT GAME LOSS: -{total_reward:.3f} | Agent: {opponent_score} - Human: {player_score}")
            return total_reward

        elif player_score < opponent_score:
            # Human loses = Agent wins
            if opponent_score == 120 and player_score == 0:
                total_reward = -3.0
                logger.info(f"AGENT GAME WIN (cappotto): +3.0 | Agent: {opponent_score} - Human: {player_score}")
            elif opponent_score >= 91:
                # Agent dominant win
                total_reward = -2.5
                logger.info(f"AGENT GAME WIN (dominant): +2.5 | Agent: {opponent_score} - Human: {player_score}")
            elif opponent_score >= 80:
                # Agent solid win
                total_reward = -2.0
                logger.info(f"AGENT GAME WIN (solid): +2.0 | Agent: {opponent_score} - Human: {player_score}")
            elif player_score <= 29:
                total_reward = -2.5
                logger.info(f"AGENT GAME WIN (crushed human): +2.5 | Agent: {opponent_score} - Human: {player_score}")
            else:
                margin_penalty = min(abs(margin) / 60.0, 1.0)
                total_reward = -1.0 - margin_penalty
                logger.info(f"AGENT GAME WIN: +{abs(total_reward):.3f} | Agent: {opponent_score} - Human: {player_score}")
            return total_reward

        else:
            # Tie (60-60)
            logger.info(f"AGENT GAME TIE: 0.0 | Agent: {opponent_score} - Human: {player_score}")
            return 0.0

    def step(
        self,
        action: int,
        opponent_action_fn: Callable[..., int]
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step of the game.

        Args:
            action: Index of card to play from player's hand
            opponent_action_fn: Function that takes (hand, briscola, game_history, opp_card)
                                and returns card index for opponent

        Returns:
            observation, reward, done, info
        """
        assert self.state is not None, "State not initialized, call reset() first"
        assert 0 <= action < len(self.state.player_hand), f"Invalid action: {action}"

        # Track player's position (leading or responding) for reward calculations
        player_was_leading = self.state.player_is_leading
        player_was_responding = not player_was_leading
        player_hand_before = list(self.state.player_hand) if player_was_responding else None

        # Increment trick number
        self.state.trick_number += 1

        # Execute trick based on who's leading
        if self.state.player_is_leading:
            # Player leads
            player_card = self.state.player_hand.pop(action)
            self.state.current_trick = [(0, player_card)]

            # Opponent responds
            opponent_action = opponent_action_fn(
                hand=self.state.opponent_hand,
                briscola=self.state.briscola_card,
                game_history=self.state.played_cards,
                opponent_card=player_card
            )
            opponent_card = self.state.opponent_hand.pop(opponent_action)
            self.state.current_trick.append((1, opponent_card))

            # Determine winner
            if player_card.beats(opponent_card, self.state.briscola_suit, is_leading=True):
                winner = 0  # Player wins
                self.state.player_is_leading = True
            else:
                winner = 1  # Opponent wins
                self.state.player_is_leading = False
        else:
            # Opponent leads
            opponent_action = opponent_action_fn(
                hand=self.state.opponent_hand,
                briscola=self.state.briscola_card,
                game_history=self.state.played_cards,
                opponent_card=None
            )
            opponent_card = self.state.opponent_hand.pop(opponent_action)
            self.state.current_trick = [(1, opponent_card)]

            # Player responds
            player_card = self.state.player_hand.pop(action)
            self.state.current_trick.append((0, player_card))

            # Determine winner
            if opponent_card.beats(player_card, self.state.briscola_suit, is_leading=True):
                winner = 1  # Opponent wins
                self.state.player_is_leading = False
            else:
                winner = 0  # Player wins
                self.state.player_is_leading = True

        # Calculate trick points
        trick_points = sum(card.get_points() for _, card in self.state.current_trick)

        if winner == 0:
            self.state.player_score += trick_points
        else:
            self.state.opponent_score += trick_points

        # Record trick in history before clearing
        if player_was_leading:
            # Player led: player_card is leader, opponent_card is responder
            trick_record = TrickRecord(
                leader_card=player_card,
                responder_card=opponent_card,
                leader_id=0,  # Player led
                winner_id=winner,
                points=trick_points
            )
        else:
            # Opponent led: opponent_card is leader, player_card is responder
            trick_record = TrickRecord(
                leader_card=opponent_card,
                responder_card=player_card,
                leader_id=1,  # Opponent led
                winner_id=winner,
                points=trick_points
            )
        self.state.trick_history.append(trick_record)

        # Add cards to played pile
        self.state.played_cards.extend([card for _, card in self.state.current_trick])
        self.state.current_trick = []

        # Draw new cards (winner draws first)
        if len(self.deck) > 0:
            if winner == 0:
                card = self.deck.draw()
                if card:
                    self.state.player_hand.append(card)
                if len(self.deck) > 0:
                    card = self.deck.draw()
                    if card:
                        self.state.opponent_hand.append(card)
            else:
                card = self.deck.draw()
                if card:
                    self.state.opponent_hand.append(card)
                if len(self.deck) > 0:
                    card = self.deck.draw()
                    if card:
                        self.state.player_hand.append(card)

        # Check if game is over
        done = len(self.state.player_hand) == 0 and len(self.state.opponent_hand) == 0

        # Calculate reward using new card-value-weighted system
        if done:
            reward = self._calculate_final_reward()
        else:
            reward = self._calculate_trick_reward(
                winner, player_card, opponent_card, trick_points,
                player_hand_before=player_hand_before,
                player_was_responding=player_was_responding,
                player_was_leading=player_was_leading
            )

        info = {
            "player_score": self.state.player_score,
            "opponent_score": self.state.opponent_score,
            "trick_points": trick_points,
            "winner": winner,
            "trick_number": self.state.trick_number,
        }

        return self._get_observation(), self._apply_perspective(reward), done, info

    def card_to_string(self, card: Card) -> str:
        """Convert card to compact string format."""
        value_map = {
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
        return f"{value_map[card.value]} {card.suit}"

    def get_state_for_llm(self, hand: List[Card], opponent_card: Optional[Card] = None) -> Dict:
        """Get state formatted for LLM prompt."""
        assert self.state is not None, "State not initialized, call reset() first"
        return {
            "hand": [self.card_to_string(c) for c in hand],
            "briscola": self.state.briscola_suit,
            "played": [self.card_to_string(c) for c in self.state.played_cards],
            "opponent_card": self.card_to_string(opponent_card) if opponent_card else None,
        }

    def opponent_lead(self, opponent_action_fn: Callable[..., int]) -> Card:
        """
        Have the opponent play their leading card.
        Call this when opponent leads (player_is_leading=False) before player responds.

        Returns:
            The card the opponent played
        """
        assert self.state is not None, "State not initialized, call reset() first"
        assert not self.state.player_is_leading, "Opponent can only lead when player_is_leading=False"
        assert len(self.state.current_trick) == 0, "Trick already in progress"

        # Get opponent's action
        opponent_action = opponent_action_fn(
            hand=self.state.opponent_hand,
            briscola=self.state.briscola_card,
            game_history=self.state.played_cards,
            opponent_card=None  # No card played yet
        )

        # Validate and play the card
        if opponent_action >= len(self.state.opponent_hand):
            opponent_action = len(self.state.opponent_hand) - 1 if len(self.state.opponent_hand) > 0 else 0

        opponent_card = self.state.opponent_hand.pop(opponent_action)
        self.state.current_trick = [(1, opponent_card)]

        return opponent_card

    def player_respond(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Player responds to opponent's lead.
        Call this after opponent_lead() when player_is_leading=False.

        Returns:
            observation, reward, done, info
        """
        assert self.state is not None, "State not initialized, call reset() first"
        assert len(self.state.current_trick) == 1, "Opponent must lead first"
        assert self.state.current_trick[0][0] == 1, "Opponent's card must be in current_trick"
        assert 0 <= action < len(self.state.player_hand), f"Invalid action: {action}"

        # Increment trick number
        self.state.trick_number += 1

        # Save hand before playing (for missed capture penalty calculation)
        player_hand_before = list(self.state.player_hand)

        # Get opponent's leading card
        opponent_card = self.state.current_trick[0][1]

        # Player responds
        player_card = self.state.player_hand.pop(action)
        self.state.current_trick.append((0, player_card))

        # Determine winner (opponent led, so opponent's card has is_leading=True)
        if opponent_card.beats(player_card, self.state.briscola_suit, is_leading=True):
            winner = 1  # Opponent wins
            self.state.player_is_leading = False
        else:
            winner = 0  # Player wins
            self.state.player_is_leading = True

        # Calculate trick points
        trick_points = sum(card.get_points() for _, card in self.state.current_trick)

        if winner == 0:
            self.state.player_score += trick_points
        else:
            self.state.opponent_score += trick_points

        # Record trick in history before clearing
        # In player_respond, opponent always led
        trick_record = TrickRecord(
            leader_card=opponent_card,
            responder_card=player_card,
            leader_id=1,  # Opponent led
            winner_id=winner,
            points=trick_points
        )
        self.state.trick_history.append(trick_record)

        # Add cards to played pile
        self.state.played_cards.extend([card for _, card in self.state.current_trick])
        self.state.current_trick = []

        # Draw new cards (winner draws first)
        if len(self.deck) > 0:
            if winner == 0:
                card = self.deck.draw()
                if card:
                    self.state.player_hand.append(card)
                if len(self.deck) > 0:
                    card = self.deck.draw()
                    if card:
                        self.state.opponent_hand.append(card)
            else:
                card = self.deck.draw()
                if card:
                    self.state.opponent_hand.append(card)
                if len(self.deck) > 0:
                    card = self.deck.draw()
                    if card:
                        self.state.player_hand.append(card)

        # Check if game is over
        done = len(self.state.player_hand) == 0 and len(self.state.opponent_hand) == 0

        # Calculate reward using new card-value-weighted system
        if done:
            reward = self._calculate_final_reward()
        else:
            reward = self._calculate_trick_reward(
                winner, player_card, opponent_card, trick_points,
                player_hand_before=player_hand_before,
                player_was_responding=True,  # player_respond is always responding
                player_was_leading=False  # player_respond means player is NOT leading
            )

        info = {
            "player_score": self.state.player_score,
            "opponent_score": self.state.opponent_score,
            "trick_points": trick_points,
            "winner": winner,
            "player_card": player_card,
            "opponent_card": opponent_card,
            "trick_number": self.state.trick_number,
        }

        return self._get_observation(), self._apply_perspective(reward), done, info
