"""Shared fixtures: a tiny random agent and a way to play scripted games."""

from typing import Optional

import pytest
import torch

from src.briscola_env import BriscolaEnv
from src.models.actor_critic import ActorCritic
from tests.helpers import ScriptedOpponent
from training.evaluate import play_game

OBS_DIM = BriscolaEnv(history_length=4).obs_dim


@pytest.fixture
def model() -> ActorCritic:
    """Small untrained network with the observation size of the shipped checkpoints."""
    torch.manual_seed(0)
    net = ActorCritic(obs_dim=OBS_DIM, action_dim=3, hidden_dim=16)
    net.eval()
    return net


@pytest.fixture
def checkpoint(tmp_path, model) -> str:
    """Path of a checkpoint file holding `model`."""
    path = tmp_path / "agent.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    return str(path)


@pytest.fixture
def make_game(model):
    """Return a function that plays one scripted game and gives its record."""

    def _make(index: int = 0, opponent: Optional[ScriptedOpponent] = None, seed: int = 42):
        """Play game `index` against `opponent` (a first-card player by default)."""
        env = BriscolaEnv(history_length=4)
        return play_game(env, model, opponent or ScriptedOpponent(), index, seed, "checkpoints/agent.pt")

    return _make
