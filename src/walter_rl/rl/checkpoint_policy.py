"""Load an rsl-rl checkpoint for the mjlab viewer."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv

from walter_rl.rl.ppo import STATE_MACHINE_ACTOR_HIDDEN, STATE_MACHINE_CRITIC_HIDDEN
from walter_rl.rl.state_machine_env import _MAX_TANK_HIP_DEG
from walter_rl.rl.teacher_env import POLICY_ACTION_DIM


class CheckpointPolicy:
  """Viewer policy that runs a trained actor on the actor observation group.

  ``actor_hidden`` and ``critic_hidden`` are the widths a checkpoint from
  ``state_machine_ppo_runner_cfg`` was trained with.
  """

  actor_hidden = STATE_MACHINE_ACTOR_HIDDEN
  critic_hidden = STATE_MACHINE_CRITIC_HIDDEN

  def __init__(
    self,
    model: torch.nn.Module,
    env: ManagerBasedRlEnv | None = None,
  ) -> None:
    self._model = model
    self._env = env

  def reset(self) -> None:
    """No recurrent state to clear."""

  def control_errors(self, env_idx: int) -> tuple[tuple[str, float, float], ...]:
    """Policy outputs for the Viser motors overlay."""
    action = _policy_action(self._env)
    if action is not None and action.shape[-1] == POLICY_ACTION_DIM:
      return _teacher_control_errors(self._env, env_idx, action)
    return (
      ("tumble decision", _tumble_decision(self._env, env_idx), 1.0),
      ("tank hip (deg)", _tank_hip_deg(self._env, env_idx), _MAX_TANK_HIP_DEG),
    )

  def __call__(self, obs: dict | TensorDict) -> torch.Tensor:
    """Return the deterministic action for the current observations."""
    return self._model(_as_tensordict(obs))


def _tumble_decision(env: ManagerBasedRlEnv | None, env_idx: int) -> float:
  """Stored tumble toggle for one environment, or 0 when the env has none."""
  action = _policy_action(env)
  if action is None:
    return 0.0
  return float(action[env_idx, 0].detach().cpu())


def _tank_hip_deg(env: ManagerBasedRlEnv | None, env_idx: int) -> float:
  """Applied tank hip offset in degrees, or 0 when the env has none."""
  action = _policy_action(env)
  if action is None or action.shape[-1] < 2:
    return 0.0
  return float(action[env_idx, 1].detach().cpu()) * _MAX_TANK_HIP_DEG


def _teacher_control_errors(
  env: ManagerBasedRlEnv | None,
  env_idx: int,
  action: torch.Tensor,
) -> tuple[tuple[str, float, float], ...]:
  """Drive state and whether knee spin mode is active."""
  drive = float(action[env_idx, 12].detach().cpu())
  applied = getattr(env.unwrapped if env is not None else None, "_tumble_applied", None)
  spin = 0.0
  if isinstance(applied, torch.Tensor):
    spin = float(applied[env_idx].detach().cpu())
  return (
    ("drive state", drive, 1.0),
    ("knee spin mode", spin, 1.0),
  )


def _policy_action(env: ManagerBasedRlEnv | None) -> torch.Tensor | None:
  """Stored policy action, or None when this env does not keep one."""
  if env is None:
    return None
  action = getattr(env.unwrapped, "policy_action", None)
  if not isinstance(action, torch.Tensor):
    return None
  return action


def _as_tensordict(obs: dict | TensorDict) -> TensorDict:
  """Wrap a plain observation dict. The actor reads the ``actor`` group."""
  if isinstance(obs, TensorDict):
    return obs
  actor = obs["actor"]
  return TensorDict(obs, batch_size=[actor.shape[0]])
