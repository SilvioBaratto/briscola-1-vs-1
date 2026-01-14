// Minimal types for Briscola game - optimized for Ollama speed

// Card representation (compact format)
class CardInfo {
  // e.g., "A coins", "3 swords", "7 cups"
  card string @description("Card in format: value suit")
}

// Game state for LLM opponent (minimal)
class GameState {
  // Hand cards as simple strings (e.g., ["A coins", "3 swords", "K cups"])
  hand string[] @description("Your cards")

  // Trump suit
  briscola string @description("Trump suit")

  // Cards played so far (to track what's left)
  played string[] @description("All played cards")

  // Opponent's last card (if any)
  opponent_card string? @description("Card opponent just played (if any)")
}

// Action output - just the index
class Action {
  // Index of card to play from hand (0, 1, or 2)
  card_index int 
}

// Fallback: just return the index as int
class SimpleAction {
  card_index int
}
