"""Evaluate trained PPO agent against LLM opponent."""

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from src.briscola_env import BriscolaEnv
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPO
from src.agents.llm_opponent import LLMOpponent, create_llm_opponent_action_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_agent(
    env: BriscolaEnv,
    ppo: PPO,
    opponent_action_fn: Callable,
    num_episodes: int = 100,
    verbose: bool = False,
):
    """
    Evaluate agent performance.

    Args:
        env: Game environment
        ppo: Trained PPO agent
        opponent_action_fn: Function for opponent actions
        num_episodes: Number of episodes to evaluate
        verbose: If True, print detailed episode information

    Returns:
        Dictionary with evaluation metrics
    """
    ppo.model.eval()

    wins = 0
    losses = 0
    ties = 0
    episode_rewards = []
    player_scores = []
    opponent_scores = []
    episode_lengths = []

    with torch.no_grad():
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            obs = env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)

            episode_reward = 0
            steps = 0

            for _ in range(40):  # Max steps
                action_mask = env.get_action_mask()
                action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(ppo.device)

                obs_input = obs_tensor.unsqueeze(1)  # Add seq dim
                action, _, _, _, _ = ppo.model.get_action(
                    obs_input,
                    deterministic=True,
                    action_mask=action_mask_tensor.unsqueeze(1)
                )

                action_int: int = int(action.item())
                next_obs, reward, done, info = env.step(action_int, opponent_action_fn)

                episode_reward += reward
                steps += 1

                if done:
                    player_score = info["player_score"]
                    opponent_score = info["opponent_score"]

                    if player_score > opponent_score:
                        wins += 1
                        result = "WIN"
                    elif player_score < opponent_score:
                        losses += 1
                        result = "LOSS"
                    else:
                        ties += 1
                        result = "TIE"

                    player_scores.append(player_score)
                    opponent_scores.append(opponent_score)

                    if verbose:
                        logger.info(
                            f"Episode {episode+1}: {result} | "
                            f"Score: {player_score}-{opponent_score} | "
                            f"Reward: {episode_reward:.3f} | "
                            f"Steps: {steps}"
                        )

                    break

                obs = next_obs
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

    ppo.model.train()

    return {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / num_episodes,
        "loss_rate": losses / num_episodes,
        "tie_rate": ties / num_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_player_score": np.mean(player_scores),
        "std_player_score": np.std(player_scores),
        "mean_opponent_score": np.mean(opponent_scores),
        "std_opponent_score": np.std(opponent_scores),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.mean(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed episode info")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Create environment
    env = BriscolaEnv()

    # Create model
    model = ActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
    )

    # Create PPO (just for model loading)
    ppo = PPO(model=model, device=args.device)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ppo.load(str(checkpoint_path))

    # Create LLM opponent
    llm_opponent = LLMOpponent()
    opponent_action_fn = create_llm_opponent_action_fn(llm_opponent)

    # Evaluate
    logger.info(f"Evaluating for {args.num_episodes} episodes...")
    results = evaluate_agent(
        env=env,
        ppo=ppo,
        opponent_action_fn=opponent_action_fn,
        num_episodes=args.num_episodes,
        verbose=args.verbose,
    )

    # Print results
    logger.info("\n" + "="*60)
    logger.info("Evaluation Results")
    logger.info("="*60)
    logger.info(f"Episodes: {results['num_episodes']}")
    logger.info(f"Wins: {results['wins']} ({results['win_rate']:.1%})")
    logger.info(f"Losses: {results['losses']} ({results['loss_rate']:.1%})")
    logger.info(f"Ties: {results['ties']} ({results['tie_rate']:.1%})")
    logger.info(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    logger.info(f"Mean Player Score: {results['mean_player_score']:.1f} ± {results['std_player_score']:.1f}")
    logger.info(f"Mean Opponent Score: {results['mean_opponent_score']:.1f} ± {results['std_opponent_score']:.1f}")
    logger.info(f"Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")
    logger.info("="*60)

    # LLM opponent stats
    llm_stats = llm_opponent.get_statistics()
    logger.info("\nLLM Opponent Statistics")
    logger.info("="*60)
    logger.info(f"Total Calls: {llm_stats['total_calls']}")
    logger.info(f"Successful Calls: {llm_stats['successful_calls']} ({llm_stats['success_rate']:.1f}%)")
    logger.info(f"Fallback Calls: {llm_stats['fallback_calls']}")
    logger.info(f"Random Fallback Calls: {llm_stats['random_fallback_calls']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
