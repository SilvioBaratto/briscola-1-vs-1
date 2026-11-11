"""Load a trained PPO agent from a checkpoint file."""

import hashlib
from pathlib import Path
from typing import Tuple

import torch

from src.briscola_env import BriscolaEnv
from src.models.actor_critic import ActorCritic


def load_actor_critic(path: str, device: str = "cpu") -> Tuple[ActorCritic, int]:
    """Rebuild the network stored in a checkpoint, in evaluation mode.

    The layer sizes are read from the saved weights, so checkpoints trained with a
    different trick-history length load without extra flags.

    Args:
        path: Checkpoint written by `PPO.save`.
        device: Torch device to place the network on.

    Returns:
        The network and the `history_length` its environment must use.

    Raises:
        ValueError: If the observation size does not match any history length.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint["model_state_dict"]
    hidden_dim, obs_dim = state["shared.0.weight"].shape
    action_dim = state["actor.2.weight"].shape[0]

    base_dim = BriscolaEnv(history_length=0).obs_dim
    trick_dim = BriscolaEnv(history_length=1).obs_dim - base_dim
    history_length, remainder = divmod(obs_dim - base_dim, trick_dim)
    if remainder or history_length < 0:
        raise ValueError(f"Observation size {obs_dim} in {path} matches no history length")

    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, history_length


def checkpoint_sha256(path: str) -> str:
    """Return the SHA-256 of a checkpoint file, used to tie results to the exact weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
