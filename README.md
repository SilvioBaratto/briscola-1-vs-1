# Briscola RL: Deep Reinforcement Learning for Italian Card Games

A reinforcement learning system that trains AI agents to play **Briscola**, the classic Italian trick-taking card game. The agent learns optimal strategies through self-play against an LLM-powered opponent using Proximal Policy Optimization (PPO).

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Theoretical Background](#theoretical-background)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
  - [Actor-Critic Architecture](#actor-critic-architecture)
  - [Generalized Advantage Estimation (GAE)](#generalized-advantage-estimation-gae)
- [Reward System Design](#reward-system-design)
- [Game Rules](#game-rules)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

## Overview

This project implements a complete pipeline for training RL agents to master Briscola:

1. **Custom Game Engine**: Full implementation of Briscola rules with Italian card mechanics
2. **PPO Training**: From-scratch implementation of PPO with action masking for invalid moves
3. **LLM Opponent**: BAML-powered opponent using Ollama for curriculum learning
4. **Web Interface**: Angular frontend with FastAPI backend for human vs AI gameplay

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │   Briscola   │     │    PPO       │     │     LLM      │       │
│   │ Environment  │◄───►│   Agent      │◄───►│   Opponent   │       │
│   │              │     │              │     │   (Ollama)   │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│          │                    │                    │                │
│          ▼                    ▼                    ▼                │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │  Observation │     │ Actor-Critic │     │    BAML      │       │
│   │   Encoding   │     │   Network    │     │  Functions   │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Web Application                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │   Angular    │     │   FastAPI    │     │  RL Agent    │       │
│   │   Frontend   │◄───►│   Backend    │◄───►│  (Trained)   │       │
│   │              │     │              │     │              │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Theoretical Background

### Proximal Policy Optimization (PPO)

PPO is a policy gradient method that addresses the challenge of training stability in reinforcement learning. The key innovation is the **clipped surrogate objective** that prevents destructively large policy updates.

#### The Objective Function

```
L^CLIP(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio
- `A_t` is the advantage estimate
- `ε` is the clipping parameter (default: 0.2)

#### Why PPO Works for Briscola

1. **Stable Learning**: Card games have high variance in outcomes; PPO's clipping prevents catastrophic forgetting
2. **Sample Efficiency**: Each game provides multiple training samples (one per trick)
3. **Action Masking**: PPO handles invalid actions gracefully through logit masking

### Actor-Critic Architecture

The network uses a shared feature extractor with separate policy (actor) and value (critic) heads:

```
                    ┌─────────────────┐
                    │  Observation    │
                    │   (245 dims)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Shared MLP     │
                    │  (2 layers)     │
                    │  ReLU activation│
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │   Actor Head    │           │   Critic Head   │
     │  (Policy π)     │           │   (Value V)     │
     │                 │           │                 │
     │  Output: 3      │           │  Output: 1      │
     │  (card indices) │           │  (state value)  │
     └─────────────────┘           └─────────────────┘
```

#### Observation Space (245 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Hand encoding | 120 | 3 slots × 40 cards (one-hot) |
| Trump suit | 4 | One-hot encoded briscola suit |
| Played cards | 40 | Binary mask of cards seen |
| Current trick | 40 | Opponent's card if responding |
| Score differential | 1 | Normalized to [-1, 1] |

### Generalized Advantage Estimation (GAE)

GAE provides a balance between bias and variance in advantage estimation:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

Where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error.

| λ Value | Characteristics |
|---------|-----------------|
| λ = 0 | High bias, low variance (TD(0)) |
| λ = 1 | Low bias, high variance (Monte Carlo) |
| λ = 0.95 | Optimal balance for most tasks |

## Reward System Design

The reward system is carefully designed to teach strategic Briscola play:

### Final Game Rewards

| Outcome | Reward | Description |
|---------|--------|-------------|
| Cappotto (120-0) | ±3.0 | Perfect game bonus/penalty |
| Dominant win (≥91 pts) | ±2.5 | Strong victory |
| Solid win (≥80 pts) | ±2.0 | Clear victory |
| Normal win/loss | ±1.0 to ±2.0 | Margin-based scaling |
| Tie (60-60) | 0.0 | Draw |

### Intermediate Trick Rewards

The system provides dense feedback after each trick:

#### Positive Signals
- **Capture bonus**: Scaled by opponent card value (Ace/Three worth more)
- **Efficiency bonus**: Capturing high cards with low cards
- **Positional rewards**: Leading with low cards (lisci) is encouraged

#### Negative Signals (Penalties)
- **Card waste**: Using high cards (carichi) to capture worthless cards
- **Trump waste**: Wasting briscola on non-valuable tricks
- **Missed capture**: Not using small trump to capture opponent's Ace/Three

### Card Value Weights

```python
CARD_VALUE_WEIGHTS = {
    ACE:   1.0,   # 11 points - Carico
    THREE: 0.9,   # 10 points - Carico
    KING:  0.35,  # 4 points  - Figura
    HORSE: 0.25,  # 3 points  - Figura
    JACK:  0.18,  # 2 points  - Figura
    SEVEN: 0.05,  # 0 points  - Liscio Alto
    SIX:   0.04,  # 0 points  - Liscio
    FIVE:  0.03,  # 0 points  - Liscio
    FOUR:  0.02,  # 0 points  - Liscio
    TWO:   0.01,  # 0 points  - Liscio Basso
}
```

### Game Phase Modifiers

Rewards are amplified in critical moments:
- **End game** (tricks 15-20): 1.5× modifier
- **Critical moment** (close score in end game): 2.0× modifier

## Game Rules

Briscola is played with a 40-card Italian deck:

### Card Rankings (Highest to Lowest)
```
Ace (11 pts) > Three (10 pts) > King (4 pts) > Horse (3 pts) > Jack (2 pts) > 7-2 (0 pts)
```

### Gameplay
1. Each player receives 3 cards; one card is revealed as **briscola** (trump)
2. Players take turns playing one card per trick
3. **No obligation to follow suit** - play any card
4. Trick winner draws first from deck, then opponent
5. Trump (briscola) beats any non-trump card
6. Same suit: higher rank wins; different suits (no trump): first card wins
7. Game ends when all 40 cards are played; **61+ points wins**

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- Ollama (for LLM opponent)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/briscola-1-vs-1.git
cd briscola-1-vs-1

# Install Python dependencies
pip install -e .

# Setup Ollama model (optional, for LLM opponent)
ollama pull mistral:7b
ollama create briscola -f Modelfile

# Generate BAML client
baml-cli generate
```

## Usage

### Training

```bash
# Start Ollama server (in separate terminal)
ollama serve

# Run training
briscola train

# Or with custom parameters
briscola train --num-updates 1000 --lr 3e-4 --hidden-dim 128
```

Training metrics displayed:
- **WR**: Win rate against LLM opponent
- **R**: Mean cumulative reward per episode
- **PL**: Policy loss (negative = policy improving)

### Web Application

```bash
# Using Docker
docker compose up

# Access the game
# Frontend: http://localhost
# API docs: http://localhost:8000/docs
```

### CLI Options

```bash
briscola train --help

Options:
  --num-updates        Number of PPO updates (default: 1000)
  --episodes-per-update Episodes per update (default: 10)
  --eval-interval      Evaluation frequency (default: 10)
  --lr                 Learning rate (default: 3e-4)
  --gamma              Discount factor (default: 0.99)
  --clip-epsilon       PPO clip parameter (default: 0.2)
  --hidden-dim         Network hidden size (default: 128)
  --device             cpu or cuda (default: cpu)
```

## Project Structure

```
briscola-1-vs-1/
├── src/
│   ├── cards.py              # Card and deck definitions
│   ├── briscola_env.py       # RL environment with reward system
│   ├── models/
│   │   ├── actor_critic.py   # Neural network architecture
│   │   ├── ppo.py            # PPO algorithm implementation
│   │   ├── gae.py            # Advantage estimation
│   │   └── replay_buffer.py  # Experience storage
│   └── agents/
│       └── llm_opponent.py   # BAML-powered LLM opponent
├── training/
│   └── train_vs_llm.py       # Training loop
├── briscola_rl/
│   └── cli.py                # Command-line interface
├── api/
│   └── main.py               # FastAPI backend
├── frontend/                 # Angular web application
├── baml_src/                 # BAML function definitions
├── checkpoints/              # Saved model weights
└── Modelfile                 # Ollama model configuration
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| RL Framework | PyTorch 2.3+ |
| Environment | Custom (PettingZoo-compatible) |
| LLM Integration | BAML + Ollama |
| Backend | FastAPI |
| Frontend | Angular 19 |
| Containerization | Docker Compose |

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - Schulman et al., 2015
- [BAML Documentation](https://docs.boundaryml.com/)

## License

MIT License - See [LICENSE](LICENSE) for details.
