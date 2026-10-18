"""Load an rsl-rl checkpoint for the mjlab viewer."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class CheckpointPolicy:
  """Viewer policy that runs a trained actor on the actor observation group."""

  def __init__(self, model: torch.nn.Module) -> None:
    self._model = model

  def reset(self) -> None:
    """No recurrent state to clear."""

  def __call__(self, obs: dict | TensorDict) -> torch.Tensor:
    """Return the deterministic action for the current observations."""
    return self._model(_as_tensordict(obs))


def _as_tensordict(obs: dict | TensorDict) -> TensorDict:
  """Wrap a plain observation dict. The actor reads the ``actor`` group."""
  if isinstance(obs, TensorDict):
    return obs
  actor = obs["actor"]
  return TensorDict(obs, batch_size=[actor.shape[0]])
