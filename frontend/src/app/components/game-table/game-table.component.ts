import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import { CardComponent } from '../card/card.component';
import type { GameState, CardSchema } from '../../models/game.models';
import type { LastTrick } from '../../app';

@Component({
  selector: 'app-game-table',
  imports: [CardComponent],
  templateUrl: './game-table.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GameTableComponent {
  readonly gameState = input<GameState | null>(null);
  readonly lastTrick = input<LastTrick | null>(null);

  readonly opponentCardArray = (): number[] => {
    const size = this.gameState()?.opponent_hand_size || 0;
    return Array.from({ length: size }, (_, i) => i);
  };

  readonly opponentPlayedCard = computed((): CardSchema | null => {
    // Prioritize lastTrick when showing completed trick result
    const last = this.lastTrick();
    if (last?.opponentCard) {
      return last.opponentCard;
    }
    // Otherwise check current_trick (opponent's lead waiting for response)
    const trick = this.gameState()?.current_trick;
    if (trick && trick.length > 0) {
      const opponentTrick = trick.find(tc => tc.player === 'opponent');
      if (opponentTrick) return opponentTrick.card;
    }
    return null;
  });

  readonly playerPlayedCard = computed((): CardSchema | null => {
    // Prioritize lastTrick when showing completed trick result
    const last = this.lastTrick();
    if (last?.playerCard) {
      return last.playerCard;
    }
    // Otherwise check current_trick
    const trick = this.gameState()?.current_trick;
    if (trick && trick.length > 0) {
      const playerTrick = trick.find(tc => tc.player === 'player');
      if (playerTrick) return playerTrick.card;
    }
    return null;
  });

  readonly trickWinner = computed((): string | null => {
    return this.lastTrick()?.winner || null;
  });

  readonly isShowingTrickResult = computed((): boolean => {
    return this.lastTrick() !== null;
  });
}
