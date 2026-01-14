"""Train PPO agent to play Briscola against LLM opponent."""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.briscola_env import BriscolaEnv
from src.models.actor_critic import ActorCritic
from src.models.ppo import PPO
from src.agents.llm_opponent import LLMOpponent, create_llm_opponent_action_fn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BriscolaTrainer:
    """PPO trainer for Briscola against LLM opponent."""

    def __init__(
        self,
        env: BriscolaEnv,
        ppo: PPO,
        llm_opponent: LLMOpponent,
        episodes_per_update: int = 10,
        max_steps_per_episode: int = 40,  # Max tricks in Briscola
        save_dir: str = "checkpoints",
        log_interval: int = 10,
    ):
        self.env = env
        self.ppo = ppo
        self.llm_opponent = llm_opponent
        self.opponent_action_fn = create_llm_opponent_action_fn(llm_opponent)

        self.episodes_per_update = episodes_per_update
        self.max_steps_per_episode = max_steps_per_episode
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.win_rates: List[float] = []

    def collect_rollouts(self, num_episodes: int) -> Dict[str, float]:
        """
        Collect rollouts for PPO update.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            Dictionary of episode statistics
        """
        episode_rewards = []
        episode_lengths = []
        wins = 0

        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0

            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.ppo.device)

            for _ in range(self.max_steps_per_episode):
                # Get action mask
                action_mask = self.env.get_action_mask()
                action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.ppo.device)

                # Sample action from policy
                with torch.no_grad():
                    obs_input = obs_tensor.unsqueeze(1)  # Add sequence dimension
                    action, log_prob, value, _, _ = self.ppo.model.get_action(
                        obs_input,
                        action_mask=action_mask_tensor.unsqueeze(1)
                    )

                action_int: int = int(action.item())

                # Execute step
                info = {}  # Initialize info to avoid unbound variable
                try:
                    next_obs, reward, done, info = self.env.step(
                        action_int,
                        self.opponent_action_fn
                    )
                except Exception as e:
                    logger.error(f"Step execution failed: {e}")
                    # End episode with penalty
                    reward = -1.0
                    done = True
                    next_obs = obs
                    info = {"player_score": 0, "opponent_score": 0}

                # Store transition
                self.ppo.collect_rollout(
                    obs=obs_tensor,
                    action=action_int,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    action_mask=action_mask_tensor,
                )

                episode_reward += reward
                episode_length += 1

                if done:
                    # Check if won
                    if info.get("player_score", 0) > info.get("opponent_score", 0):
                        wins += 1
                    break

                obs = next_obs
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.ppo.device)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "win_rate": float(wins / num_episodes),
        }

    def train(
        self,
        num_updates: int = 100,
        eval_interval: int = 10,
        eval_episodes: int = 20,
    ):
        """
        Main training loop.

        Args:
            num_updates: Number of PPO updates to perform
            eval_interval: Evaluate every N updates
            eval_episodes: Number of episodes for evaluation
        """
        logger.info(f"Starting training for {num_updates} updates")
        logger.info(f"Environment: obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}")
        logger.info(f"PPO config: lr={self.ppo.optimizer.param_groups[0]['lr']}, "
                   f"gamma={self.ppo.gamma}, clip_eps={self.ppo.clip_epsilon}")

        best_win_rate = 0.0
        start_time = time.time()

        for update in tqdm(range(num_updates), desc="Training"):
            # Collect rollouts
            update_start = time.time()
            rollout_stats = self.collect_rollouts(self.episodes_per_update)
            rollout_time = time.time() - update_start

            # Perform PPO update
            update_start = time.time()
            train_stats = self.ppo.update()
            update_time = time.time() - update_start

            # Log statistics
            self.episode_rewards.append(rollout_stats["mean_reward"])
            self.episode_lengths.append(int(rollout_stats["mean_length"]))
            self.win_rates.append(rollout_stats["win_rate"])

            if update % self.log_interval == 0:
                logger.info(
                    f"Update {update}/{num_updates} | "
                    f"Reward: {rollout_stats['mean_reward']:.3f} ± {rollout_stats['std_reward']:.3f} | "
                    f"Win Rate: {rollout_stats['win_rate']:.1%} | "
                    f"Len: {rollout_stats['mean_length']:.1f} | "
                    f"Policy Loss: {train_stats['policy_loss']:.4f} | "
                    f"Value Loss: {train_stats['value_loss']:.4f} | "
                    f"Entropy: {train_stats['entropy']:.4f} | "
                    f"Rollout: {rollout_time:.2f}s | Update: {update_time:.2f}s"
                )

            # Evaluation
            if update % eval_interval == 0:
                eval_stats = self.evaluate(eval_episodes)
                logger.info(
                    f"Evaluation | "
                    f"Win Rate: {eval_stats['win_rate']:.1%} | "
                    f"Mean Reward: {eval_stats['mean_reward']:.3f} | "
                    f"Mean Score: {eval_stats['mean_score']:.1f}"
                )

                # Save best model
                if eval_stats["win_rate"] > best_win_rate:
                    best_win_rate = eval_stats["win_rate"]
                    self.save_checkpoint(f"best_model_wr{best_win_rate:.2f}.pt")
                    logger.info(f"New best model saved! Win rate: {best_win_rate:.1%}")

            # Periodic checkpoint
            if update % 50 == 0 and update > 0:
                self.save_checkpoint(f"checkpoint_update{update}.pt")

            # Log LLM opponent statistics
            if update % (self.log_interval * 5) == 0:
                llm_stats = self.llm_opponent.get_statistics()
                logger.info(
                    f"LLM Opponent Stats | "
                    f"Success Rate: {llm_stats['success_rate']:.1f}% | "
                    f"Random Fallbacks: {llm_stats['random_fallback_calls']}"
                )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} min)")
        logger.info(f"Best win rate achieved: {best_win_rate:.1%}")

        # Final save
        self.save_checkpoint("final_model.pt")

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation statistics
        """
        rewards = []
        scores = []
        wins = 0

        # Set model to eval mode
        self.ppo.model.eval()

        with torch.no_grad():
            for _ in range(num_episodes):
                obs = self.env.reset()
                episode_reward = 0

                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.ppo.device)

                for _ in range(self.max_steps_per_episode):
                    action_mask = self.env.get_action_mask()
                    action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.ppo.device)

                    obs_input = obs_tensor.unsqueeze(1)
                    action, _, _, _, _ = self.ppo.model.get_action(
                        obs_input,
                        deterministic=True,  # Use greedy action for evaluation
                        action_mask=action_mask_tensor.unsqueeze(1)
                    )

                    action_int: int = int(action.item())
                    next_obs, reward, done, info = self.env.step(
                        action_int,
                        self.opponent_action_fn
                    )

                    episode_reward += reward

                    if done:
                        if info["player_score"] > info["opponent_score"]:
                            wins += 1
                        scores.append(info["player_score"])
                        break

                    obs = next_obs
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.ppo.device)

                rewards.append(episode_reward)

        # Set model back to train mode
        self.ppo.model.train()

        return {
            "win_rate": float(wins / num_episodes),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.save_dir / filename
        self.ppo.save(str(path))
        logger.info(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent vs LLM opponent")
    parser.add_argument("--num-updates", type=int, default=1000, help="Number of PPO updates")
    parser.add_argument("--episodes-per-update", type=int, default=10, help="Episodes per update")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of eval episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--llm-timeout", type=float, default=5.0, help="LLM timeout (seconds)")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = BriscolaEnv()
    logger.info(f"Environment created: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    # Create PPO agent
    model = ActorCritic(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
    )
    ppo = PPO(
        model=model,
        lr=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        device=args.device,
    )
    logger.info(f"PPO agent created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create LLM opponent
    llm_opponent = LLMOpponent(
        timeout_seconds=args.llm_timeout,
    )
    logger.info("LLM opponent initialized")

    # Create trainer
    trainer = BriscolaTrainer(
        env=env,
        ppo=ppo,
        llm_opponent=llm_opponent,
        episodes_per_update=args.episodes_per_update,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
    )

    # Train
    try:
        trainer.train(
            num_updates=args.num_updates,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("interrupted_model.pt")

    # Final statistics
    logger.info("\n" + "="*50)
    logger.info("Training Summary")
    logger.info("="*50)
    llm_stats = llm_opponent.get_statistics()
    logger.info(f"Total episodes: {len(trainer.episode_rewards)}")
    logger.info(f"Final win rate (last 20): {np.mean(trainer.win_rates[-20:]):.1%}")
    logger.info(f"LLM success rate: {llm_stats['success_rate']:.1f}%")
    logger.info(f"LLM total calls: {llm_stats['total_calls']}")
    logger.info(f"LLM random fallbacks: {llm_stats['random_fallback_calls']}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
