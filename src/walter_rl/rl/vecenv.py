"""rsl-rl vector-env adapter for ``ManagerBasedRlEnv``."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


class RslRlVecEnvWrapper(VecEnv):
  """Present an mjlab environment as the vector env rsl-rl 5.4 expects."""

  def __init__(self, env: ManagerBasedRlEnv, clip_actions: float | None = None) -> None:
    self.env = env
    self.clip_actions = clip_actions
    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    # rsl-rl does not reset before the first rollout.
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    """Environment config, including reward and command settings."""
    return self.unwrapped.cfg

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    """The mjlab environment under this adapter."""
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    """Steps taken in the current episode, one entry per environment."""
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.unwrapped.episode_length_buf = value

  def get_observations(self) -> TensorDict:
    """Return actor and critic groups for the current state."""
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(obs_dict, batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    """Reset every environment and return the post-reset observations."""
    obs_dict, extras = self.env.reset()
    return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    """Step the environment and pack timeouts for infinite-horizon PPO."""
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, reward, terminated, truncated, extras = self.env.step(actions)
    reward = reward.reshape(self.num_envs)
    dones = (terminated | truncated).reshape(self.num_envs).to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated.reshape(self.num_envs)
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      reward,
      dones,
      extras,
    )

  def close(self) -> None:
    """Close the wrapped environment."""
    self.env.close()
