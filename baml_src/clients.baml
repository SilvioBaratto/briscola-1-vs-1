// Retry policy for robustness
retry_policy BriscolaRetry {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 100
    multiplier 1.5
    max_delay_ms 2000
  }
}

// Main Ollama client for Briscola
// Uses OLLAMA_BASE_URL env var in Docker, defaults to localhost
client<llm> Model {
  provider "openai-generic"
  retry_policy BriscolaRetry
  options {
    base_url env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1"
    model briscola
    // Optimize for speed
    temperature 0.3
    max_tokens 100
  }
}

// Fast fallback client with even shorter responses
client<llm> FastModel {
  provider "openai-generic"
  retry_policy BriscolaRetry
  options {
    base_url env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1"
    model briscola
    temperature 0.1
    max_tokens 50
  }
}