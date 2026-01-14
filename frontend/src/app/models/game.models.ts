export interface CardSchema {
  value: string;
  suit: string;
  points: number;
  display_name: string;
}

export interface HandCard {
  index: number;
  card: CardSchema;
}

export interface TrickCard {
  player: string;
  card: CardSchema;
}

export interface GameState {
  game_id: string;
  player_hand: HandCard[];
  opponent_hand_size: number;
  trump_card: CardSchema;
  trump_suit: string;
  player_score: number;
  opponent_score: number;
  cards_in_deck: number;
  player_leads: boolean;
  current_trick: TrickCard[];
  game_over: boolean;
  winner: string | null;
  message: string | null;
}

export interface PlayCardResponse {
  state: GameState;
  trick_result: {
    winner: string;
    points: number;
    player_card: CardSchema;
    opponent_card: CardSchema;
  } | null;
}

export interface ModelStatus {
  loaded: boolean;
  checkpoint_path: string | null;
  obs_dim: number;
  action_dim: number;
  hidden_dim: number;
  device: string;
  buffer_size: number;
  games_collected: number;
}

export interface TrainingResponse {
  success: boolean;
  message: string;
  metrics: Record<string, number> | null;
  experiences_used: number;
}
