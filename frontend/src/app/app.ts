import { Component, signal, inject, effect, ChangeDetectionStrategy } from '@angular/core';
import { GameService } from './services/game.service';
import { GameTableComponent } from './components/game-table/game-table.component';
import { HandComponent } from './components/hand/hand.component';
import { TrainingPanelComponent } from './components/training-panel/training-panel.component';
import { GameOverModalComponent } from './components/game-over-modal/game-over-modal.component';
import type { CardSchema } from './models/game.models';

export interface LastTrick {
  playerCard: CardSchema;
  opponentCard: CardSchema;
  winner: string;
}

@Component({
  selector: 'app-root',
  imports: [
    GameTableComponent,
    HandComponent,
    TrainingPanelComponent,
    GameOverModalComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class App {
  private readonly gameService = inject(GameService);

  readonly selectedCardIndex = signal<number | null>(null);
  readonly showGameOver = signal(false);
  readonly trickMessage = signal<string | null>(null);
  readonly lastTrick = signal<LastTrick | null>(null);

  readonly gameState = this.gameService.gameState;
  readonly isLoading = this.gameService.isLoading;
  readonly error = this.gameService.error;
  readonly modelStatus = this.gameService.modelStatus;

  constructor() {
    // Auto-refresh model status periodically
    setInterval(() => {
      this.gameService.getModelStatus();
    }, 5000);

    // Initial model status fetch
    this.gameService.getModelStatus();

    // Watch for game over
    effect(() => {
      const state = this.gameState();
      if (state?.game_over) {
        setTimeout(() => {
          this.showGameOver.set(true);
        }, 1500);
      }
    });
  }

  async startNewGame(): Promise<void> {
    this.showGameOver.set(false);
    this.selectedCardIndex.set(null);
    this.trickMessage.set(null);
    this.lastTrick.set(null);
    await this.gameService.startNewGame();
  }

  async onCardSelected(cardIndex: number): Promise<void> {
    const state = this.gameState();
    if (!state || state.game_over) return;

    this.selectedCardIndex.set(cardIndex);

    try {
      const response = await this.gameService.playCard(cardIndex);
      this.selectedCardIndex.set(null);

      // Show trick result with cards displayed
      if (response.trick_result) {
        const { winner, points, player_card, opponent_card } = response.trick_result;

        // Debug logging
        console.log('Trick Result:', {
          player_card: player_card,
          opponent_card: opponent_card,
          winner: winner,
          points: points,
        });

        // Display the played cards on the board
        this.lastTrick.set({
          playerCard: player_card,
          opponentCard: opponent_card,
          winner: winner,
        });

        const winnerText = winner === 'player' ? 'You' : 'Opponent';
        this.trickMessage.set(`${winnerText} won the trick! (+${points} points)`);

        // Clear after delay to show next turn (4 seconds to see the result)
        setTimeout(() => {
          this.lastTrick.set(null);
          this.trickMessage.set(null);
        }, 4000);
      }
    } catch (err) {
      console.error('Failed to play card:', err);
      this.selectedCardIndex.set(null);
    }
  }

  async onTrainModel(): Promise<void> {
    try {
      const response = await this.gameService.trainModel();
      console.log('Training complete:', response);
    } catch (err) {
      console.error('Training failed:', err);
    }
  }

  async onSaveModel(): Promise<void> {
    try {
      await this.gameService.saveModel();
      console.log('Model saved successfully');
    } catch (err) {
      console.error('Save failed:', err);
    }
  }

  onCloseGameOver(): void {
    this.showGameOver.set(false);
  }

  async onPlayAgain(): Promise<void> {
    await this.startNewGame();
  }
}
