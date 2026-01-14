#!/usr/bin/env python3
"""
Briscola Reinforcement Learning - CLI Application

A modern CLI for training and evaluating RL agents on the Briscola card game.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Card display symbols
SUIT_SYMBOLS = {"coins": "🪙", "cups": "🏆", "swords": "⚔️", "clubs": "🪵"}


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🃏 BRISCOLA RL 🃏                          ║
║         Reinforcement Learning for Italian Card Game         ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def format_card(card_str: str) -> str:
    """Format card string with suit symbol."""
    parts = card_str.split(" ", 1)
    if len(parts) == 2:
        value, suit = parts
        symbol = SUIT_SYMBOLS.get(suit, suit)
        return f"{value}{symbol}"
    return card_str


def create_stats_table(stats: dict, title: str = "Statistics") -> Table:
    """Create a formatted table for statistics."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        if isinstance(value, float):
            if "rate" in key.lower():
                table.add_row(key.replace("_", " ").title(), f"{value:.1%}")
            else:
                table.add_row(key.replace("_", " ").title(), f"{value:.3f}")
        else:
            table.add_row(key.replace("_", " ").title(), str(value))

    return table


# =============================================================================
# TRAIN COMMAND
# =============================================================================

def cmd_train(args):
    """Train a PPO agent against LLM opponent."""
    from src.briscola_env import BriscolaEnv
    from src.models.actor_critic import ActorCritic
    from src.models.ppo import PPO
    from src.agents.llm_opponent import LLMOpponent, create_llm_opponent_action_fn

    print_banner()
    console.print("\n[bold green]Starting Training Session[/bold green]\n")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Configuration display
    config_table = Table(title="Training Configuration", show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Updates", str(args.num_updates))
    config_table.add_row("Episodes/Update", str(args.episodes_per_update))
    config_table.add_row("Learning Rate", str(args.lr))
    config_table.add_row("Gamma", str(args.gamma))
    config_table.add_row("Clip Epsilon", str(args.clip_epsilon))
    config_table.add_row("Hidden Dim", str(args.hidden_dim))
    config_table.add_row("Device", args.device)
    config_table.add_row("Save Directory", args.save_dir)

    console.print(config_table)
    console.print()

    # Create environment
    with console.status("[bold green]Initializing environment..."):
        env = BriscolaEnv()
        console.print(f"✓ Environment: obs_dim={env.obs_dim}, action_dim={env.action_dim}")

    # Create PPO agent
    with console.status("[bold green]Creating PPO agent..."):
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
        param_count = sum(p.numel() for p in model.parameters())
        console.print(f"✓ PPO agent created with {param_count:,} parameters")

    # Create LLM opponent
    with console.status("[bold green]Initializing LLM opponent..."):
        llm_opponent = LLMOpponent(timeout_seconds=args.llm_timeout)
        opponent_action_fn = create_llm_opponent_action_fn(llm_opponent)
        console.print("✓ LLM opponent initialized")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Starting training loop...[/bold]\n")

    # Training loop with progress
    best_win_rate = 0.0
    episode_rewards = []
    win_rates = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[stats]}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[green]Training...",
            total=args.num_updates,
            stats=""
        )

        for update in range(args.num_updates):
            # Collect rollouts
            rollout_stats = _collect_rollouts(
                env, ppo, opponent_action_fn,
                num_episodes=args.episodes_per_update,
                max_steps=40
            )

            # PPO update
            train_stats = ppo.update()

            # Track stats
            episode_rewards.append(rollout_stats["mean_reward"])
            win_rates.append(rollout_stats["win_rate"])

            # Update progress
            stats_str = (
                f"WR: {rollout_stats['win_rate']:.1%} | "
                f"R: {rollout_stats['mean_reward']:.2f} | "
                f"PL: {train_stats['policy_loss']:.3f}"
            )
            progress.update(task, advance=1, stats=stats_str)

            # Save best model
            if rollout_stats["win_rate"] > best_win_rate:
                best_win_rate = rollout_stats["win_rate"]
                ppo.save(str(save_dir / f"best_model_wr{best_win_rate:.2f}.pt"))

            # Periodic checkpoint
            if (update + 1) % 50 == 0:
                ppo.save(str(save_dir / f"checkpoint_update{update+1}.pt"))

    # Final save
    ppo.save(str(save_dir / "final_model.pt"))

    # Print summary
    console.print("\n")
    summary_table = Table(title="Training Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Updates", str(args.num_updates))
    summary_table.add_row("Best Win Rate", f"{best_win_rate:.1%}")
    summary_table.add_row("Final Win Rate (last 20)", f"{np.mean(win_rates[-20:]):.1%}")
    summary_table.add_row("Final Avg Reward", f"{np.mean(episode_rewards[-20:]):.3f}")

    llm_stats = llm_opponent.get_statistics()
    summary_table.add_row("LLM Success Rate", f"{llm_stats['success_rate']:.1f}%")
    summary_table.add_row("LLM Total Calls", str(llm_stats['total_calls']))
    summary_table.add_row("LLM Random Fallbacks", str(llm_stats['random_fallback_calls']))

    console.print(summary_table)
    console.print(f"\n[green]✓ Model saved to {save_dir}/final_model.pt[/green]")


def _collect_rollouts(env, ppo, opponent_action_fn, num_episodes: int, max_steps: int) -> dict:
    """Collect rollouts for PPO update."""
    episode_rewards = []
    wins = 0

    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)

        for _ in range(max_steps):
            action_mask = env.get_action_mask()
            action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(ppo.device)

            with torch.no_grad():
                obs_input = obs_tensor.unsqueeze(1)
                action, log_prob, value, _, _ = ppo.model.get_action(
                    obs_input,
                    action_mask=action_mask_tensor.unsqueeze(1)
                )

            action_int = int(action.item())

            try:
                next_obs, reward, done, info = env.step(action_int, opponent_action_fn)
            except Exception:
                reward = -1.0
                done = True
                next_obs = obs
                info = {"player_score": 0, "opponent_score": 0}

            ppo.collect_rollout(
                obs=obs_tensor,
                action=action_int,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
                action_mask=action_mask_tensor,
            )

            episode_reward += reward

            if done:
                if info.get("player_score", 0) > info.get("opponent_score", 0):
                    wins += 1
                break

            obs = next_obs
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)

        episode_rewards.append(episode_reward)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "win_rate": float(wins / num_episodes),
    }


# =============================================================================
# EVALUATE COMMAND
# =============================================================================

def cmd_evaluate(args):
    """Evaluate a trained model against LLM opponent."""
    from src.briscola_env import BriscolaEnv
    from src.models.actor_critic import ActorCritic
    from src.models.ppo import PPO
    from src.agents.llm_opponent import LLMOpponent, create_llm_opponent_action_fn

    print_banner()
    console.print("\n[bold green]Starting Evaluation[/bold green]\n")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[red]Error: Checkpoint not found: {checkpoint_path}[/red]")
        sys.exit(1)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create environment
    with console.status("[bold green]Initializing environment..."):
        env = BriscolaEnv()

    # Create and load model
    with console.status("[bold green]Loading model..."):
        model = ActorCritic(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            hidden_dim=args.hidden_dim,
        )
        ppo = PPO(model=model, device=args.device)
        ppo.load(str(checkpoint_path))
        console.print(f"✓ Loaded checkpoint: {checkpoint_path}")

    # Create LLM opponent
    with console.status("[bold green]Initializing LLM opponent..."):
        llm_opponent = LLMOpponent()
        opponent_action_fn = create_llm_opponent_action_fn(llm_opponent)

    console.print(f"\n[bold]Evaluating for {args.num_episodes} episodes...[/bold]\n")

    # Evaluation
    ppo.model.eval()
    wins, losses, ties = 0, 0, 0
    player_scores, opponent_scores = [], []
    episode_rewards = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating...", total=args.num_episodes)

        with torch.no_grad():
            for episode in range(args.num_episodes):
                obs = env.reset()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)
                episode_reward = 0

                for _ in range(40):
                    action_mask = env.get_action_mask()
                    action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(ppo.device)

                    obs_input = obs_tensor.unsqueeze(1)
                    action, _, _, _, _ = ppo.model.get_action(
                        obs_input,
                        deterministic=True,
                        action_mask=action_mask_tensor.unsqueeze(1)
                    )

                    action_int = int(action.item())
                    next_obs, reward, done, info = env.step(action_int, opponent_action_fn)
                    episode_reward += reward

                    if done:
                        ps, os_ = info["player_score"], info["opponent_score"]
                        player_scores.append(ps)
                        opponent_scores.append(os_)

                        if ps > os_:
                            wins += 1
                        elif ps < os_:
                            losses += 1
                        else:
                            ties += 1

                        if args.verbose:
                            result = "WIN" if ps > os_ else ("LOSS" if ps < os_ else "TIE")
                            console.print(
                                f"  Episode {episode+1}: [{'green' if ps > os_ else 'red'}]{result}[/] "
                                f"({ps}-{os_})"
                            )
                        break

                    obs = next_obs
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)

                episode_rewards.append(episode_reward)
                progress.update(task, advance=1)

    # Results
    console.print("\n")
    results_table = Table(title="Evaluation Results", show_header=True)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    total = args.num_episodes
    results_table.add_row("Episodes", str(total))
    results_table.add_row("Wins", f"{wins} ({wins/total:.1%})")
    results_table.add_row("Losses", f"{losses} ({losses/total:.1%})")
    results_table.add_row("Ties", f"{ties} ({ties/total:.1%})")
    results_table.add_row("Mean Player Score", f"{np.mean(player_scores):.1f} ± {np.std(player_scores):.1f}")
    results_table.add_row("Mean Opponent Score", f"{np.mean(opponent_scores):.1f} ± {np.std(opponent_scores):.1f}")
    results_table.add_row("Mean Reward", f"{np.mean(episode_rewards):.3f}")

    console.print(results_table)

    # LLM stats
    llm_stats = llm_opponent.get_statistics()
    llm_table = Table(title="LLM Opponent Statistics", show_header=True)
    llm_table.add_column("Metric", style="cyan")
    llm_table.add_column("Value", style="yellow")

    llm_table.add_row("Total Calls", str(llm_stats['total_calls']))
    llm_table.add_row("Successful", str(llm_stats['successful_calls']))
    llm_table.add_row("Success Rate", f"{llm_stats['success_rate']:.1f}%")
    llm_table.add_row("Random Fallbacks", str(llm_stats['random_fallback_calls']))

    console.print(llm_table)


# =============================================================================
# PLAY COMMAND
# =============================================================================

def cmd_play(args):
    """Play interactively against the RL agent."""
    from src.briscola_env import BriscolaEnv
    from src.models.actor_critic import ActorCritic
    from src.models.ppo import PPO

    print_banner()
    console.print("\n[bold green]Interactive Play Mode[/bold green]\n")

    # Load model if provided
    ppo = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            console.print(f"[red]Error: Checkpoint not found: {checkpoint_path}[/red]")
            sys.exit(1)

        env_temp = BriscolaEnv()
        model = ActorCritic(
            obs_dim=env_temp.obs_dim,
            action_dim=env_temp.action_dim,
            hidden_dim=args.hidden_dim,
        )
        ppo = PPO(model=model, device=args.device)
        ppo.load(str(checkpoint_path))
        ppo.model.eval()
        console.print(f"✓ Loaded RL agent: {checkpoint_path}")
    else:
        console.print("[yellow]No checkpoint provided, RL agent will play randomly.[/yellow]")

    # Create environment
    env = BriscolaEnv()

    # Game loop
    player_wins, agent_wins, ties = 0, 0, 0
    game_num = 0

    while True:
        game_num += 1
        console.print(f"\n[bold cyan]═══ Game {game_num} ═══[/bold cyan]\n")

        obs = env.reset()
        assert env.state is not None

        briscola_str = format_card(env.card_to_string(env.state.briscola_card))
        console.print(f"[bold]Trump (Briscola): {briscola_str}[/bold]\n")

        while True:
            assert env.state is not None

            # Display current state
            player_hand = env.state.player_hand
            agent_hand = env.state.opponent_hand

            console.print(f"[dim]Cards in deck: {len(env.deck)}[/dim]")
            console.print(f"[dim]Score: You {env.state.player_score} - {env.state.opponent_score} Agent[/dim]")

            # Show hands
            console.print("\n[bold]Your hand:[/bold]")
            for i, card in enumerate(player_hand):
                card_str = format_card(env.card_to_string(card))
                console.print(f"  [{i}] {card_str}")

            # Determine who leads
            if env.state.player_is_leading:
                # Player leads
                console.print("\n[green]You lead this trick.[/green]")

                # Get player action
                while True:
                    try:
                        choice = console.input("[bold]Choose card (0-{}): [/bold]".format(len(player_hand)-1))
                        action = int(choice)
                        if 0 <= action < len(player_hand):
                            break
                        console.print("[red]Invalid choice.[/red]")
                    except ValueError:
                        console.print("[red]Please enter a number.[/red]")

                # Player plays
                player_card = player_hand[action]
                console.print(f"\nYou played: {format_card(env.card_to_string(player_card))}")

                # Agent responds
                if ppo:
                    agent_action = _get_agent_action(env, ppo, agent_hand)
                else:
                    agent_action = random.randint(0, len(agent_hand) - 1)

                agent_card = agent_hand[agent_action]
                console.print(f"Agent played: {format_card(env.card_to_string(agent_card))}")

                # Define human opponent function that returns the player's chosen action
                def human_opponent_fn(**_kwargs):
                    return agent_action

                next_obs, reward, done, info = env.step(action, human_opponent_fn)
            else:
                # Agent leads
                console.print("\n[yellow]Agent leads this trick.[/yellow]")

                if ppo:
                    agent_action = _get_agent_action(env, ppo, agent_hand)
                else:
                    agent_action = random.randint(0, len(agent_hand) - 1)

                agent_card = agent_hand[agent_action]
                console.print(f"Agent played: {format_card(env.card_to_string(agent_card))}")

                # Get player response
                console.print("\n[bold]Your hand:[/bold]")
                for i, card in enumerate(player_hand):
                    card_str = format_card(env.card_to_string(card))
                    console.print(f"  [{i}] {card_str}")

                while True:
                    try:
                        choice = console.input("[bold]Choose card (0-{}): [/bold]".format(len(player_hand)-1))
                        action = int(choice)
                        if 0 <= action < len(player_hand):
                            break
                        console.print("[red]Invalid choice.[/red]")
                    except ValueError:
                        console.print("[red]Please enter a number.[/red]")

                player_card = player_hand[action]
                console.print(f"\nYou played: {format_card(env.card_to_string(player_card))}")

                def human_opponent_fn(**_kwargs):
                    return agent_action

                next_obs, reward, done, info = env.step(action, human_opponent_fn)

            # Show trick result
            winner = "You" if info["winner"] == 0 else "Agent"
            points = info["trick_points"]
            console.print(f"\n[bold]{winner} won the trick! (+{points} points)[/bold]")

            if done:
                ps, os_ = info["player_score"], info["opponent_score"]
                console.print("\n" + "="*40)
                if ps > os_:
                    console.print(f"[bold green]YOU WIN! {ps}-{os_}[/bold green]")
                    player_wins += 1
                elif ps < os_:
                    console.print(f"[bold red]YOU LOSE! {ps}-{os_}[/bold red]")
                    agent_wins += 1
                else:
                    console.print(f"[bold yellow]TIE! {ps}-{os_}[/bold yellow]")
                    ties += 1

                console.print(f"\n[dim]Overall: You {player_wins} - {agent_wins} Agent (Ties: {ties})[/dim]")
                break

            obs = next_obs

        # Play again?
        again = console.input("\n[bold]Play again? (y/n): [/bold]").lower()
        if again != 'y':
            break

    console.print("\n[bold]Thanks for playing![/bold]\n")


def _get_agent_action(env, ppo, hand) -> int:
    """Get action from RL agent."""
    obs = env._get_observation()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(ppo.device)

    # Create action mask for agent's hand size
    action_mask = np.zeros(env.action_dim, dtype=bool)
    action_mask[:len(hand)] = True
    action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(ppo.device)

    with torch.no_grad():
        obs_input = obs_tensor.unsqueeze(1)
        action, _, _, _, _ = ppo.model.get_action(
            obs_input,
            deterministic=True,
            action_mask=action_mask_tensor.unsqueeze(1)
        )

    return int(action.item())


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Briscola RL - Train and evaluate RL agents for Briscola",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  briscola train --num-updates 100 --device cuda
  briscola evaluate checkpoints/best_model.pt --num-episodes 50
  briscola play --checkpoint checkpoints/best_model.pt
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a PPO agent")
    train_parser.add_argument("--num-updates", type=int, default=1000, help="Number of PPO updates")
    train_parser.add_argument("--episodes-per-update", type=int, default=10, help="Episodes per update")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    train_parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    train_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    train_parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    train_parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--llm-timeout", type=float, default=5.0, help="LLM timeout (seconds)")
    train_parser.set_defaults(func=cmd_train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    eval_parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes")
    eval_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    eval_parser.add_argument("--device", type=str, default="cpu", help="Device")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Print episode details")
    eval_parser.set_defaults(func=cmd_evaluate)

    # Play command
    play_parser = subparsers.add_parser("play", help="Play interactively against RL agent")
    play_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    play_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    play_parser.add_argument("--device", type=str, default="cpu", help="Device")
    play_parser.set_defaults(func=cmd_play)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
