import { Injectable, signal, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import type {
  GameState,
  PlayCardResponse,
  ModelStatus,
  TrainingResponse,
} from '../models/game.models';

@Injectable({
  providedIn: 'root',
})
export class GameService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://localhost:8000';

  readonly currentGameId = signal<string | null>(null);
  readonly gameState = signal<GameState | null>(null);
  readonly isLoading = signal(false);
  readonly error = signal<string | null>(null);
  readonly modelStatus = signal<ModelStatus | null>(null);

  async startNewGame(): Promise<void> {
    this.isLoading.set(true);
    this.error.set(null);

    try {
      const response = await firstValueFrom(
        this.http.post<{ game_id: string; state: GameState }>(`${this.apiUrl}/game/new`, {})
      );
      this.currentGameId.set(response.game_id);
      this.gameState.set(response.state);
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to start new game');
      throw err;
    } finally {
      this.isLoading.set(false);
    }
  }

  async playCard(cardIndex: number): Promise<PlayCardResponse> {
    const gameId = this.currentGameId();
    if (!gameId) {
      throw new Error('No active game');
    }

    this.isLoading.set(true);
    this.error.set(null);

    try {
      const response = await firstValueFrom(
        this.http.post<PlayCardResponse>(`${this.apiUrl}/game/${gameId}/play`, {
          card_index: cardIndex,
        })
      );
      this.gameState.set(response.state);
      return response;
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to play card');
      throw err;
    } finally {
      this.isLoading.set(false);
    }
  }

  async getGameState(gameId: string): Promise<GameState> {
    try {
      const state = await firstValueFrom(
        this.http.get<GameState>(`${this.apiUrl}/game/${gameId}/state`)
      );
      this.gameState.set(state);
      return state;
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to get game state');
      throw err;
    }
  }

  async getModelStatus(): Promise<void> {
    try {
      const status = await firstValueFrom(
        this.http.get<ModelStatus>(`${this.apiUrl}/model/status`)
      );
      this.modelStatus.set(status);
    } catch (err) {
      console.error('Failed to get model status:', err);
    }
  }

  async trainModel(): Promise<TrainingResponse> {
    this.isLoading.set(true);
    this.error.set(null);

    try {
      const response = await firstValueFrom(
        this.http.post<TrainingResponse>(`${this.apiUrl}/model/train`, {})
      );
      await this.getModelStatus();
      return response;
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to train model');
      throw err;
    } finally {
      this.isLoading.set(false);
    }
  }

  async saveModel(checkpointPath?: string): Promise<void> {
    this.isLoading.set(true);
    this.error.set(null);

    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const path = checkpointPath || `checkpoints/briscola_model_${timestamp}.pt`;

      await firstValueFrom(
        this.http.post(`${this.apiUrl}/model/save`, { checkpoint_path: path })
      );
      await this.getModelStatus();
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to save model');
      throw err;
    } finally {
      this.isLoading.set(false);
    }
  }

  async loadModel(checkpointPath: string): Promise<void> {
    this.isLoading.set(true);
    this.error.set(null);

    try {
      await firstValueFrom(
        this.http.post(`${this.apiUrl}/model/load`, { checkpoint_path: checkpointPath })
      );
      await this.getModelStatus();
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to load model');
      throw err;
    } finally {
      this.isLoading.set(false);
    }
  }
}
