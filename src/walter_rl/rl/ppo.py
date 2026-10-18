"""PPO runner settings for rsl-rl 5.4."""

from __future__ import annotations

from walter_rl.rl.wandb_logger import WANDB_PROJECT

_ACTOR_HIDDEN = (512, 256, 128)
_CRITIC_HIDDEN = (512, 256, 128)


def walter_ppo_runner_cfg(
  *,
  max_iterations: int,
  experiment_name: str = "walter_fs_velocity",
  save_interval: int = 50,
) -> dict:
  """Build the dict ``OnPolicyRunner`` reads for the WaLTER velocity task.

  Actor and critic are separate MLPs. Metrics and checkpoints go to the
  Weights & Biases project ``walter-rl``. ``rnd_cfg`` is present and empty
  because the rsl-rl 5.4 runner indexes that key every iteration.
  """
  return {
    "seed": 42,
    "num_steps_per_env": 64,
    "max_iterations": max_iterations,
    "save_interval": save_interval,
    "experiment_name": experiment_name,
    "run_name": "",
    "logger": {
      "class_name": "walter_rl.rl.wandb_logger:WalterWandbLogWriter",
      "project_name": WANDB_PROJECT,
    },
    "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
    "clip_actions": None,
    "upload_model": True,
    "check_for_nan": True,
    "actor": _mlp_cfg(stochastic=True),
    "critic": _mlp_cfg(stochastic=False),
    "algorithm": {
      "class_name": "PPO",
      "num_learning_epochs": 5,
      "num_mini_batches": 4,
      "learning_rate": 1.0e-3,
      "schedule": "adaptive",
      "gamma": 0.99,
      "lam": 0.95,
      "entropy_coef": 0.005,
      "desired_kl": 0.01,
      "max_grad_norm": 1.0,
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": True,
      "clip_param": 0.2,
      "normalize_advantage_per_mini_batch": False,
      "optimizer": "adam",
      "share_cnn_encoders": False,
      "rnd_cfg": None,
      "symmetry_cfg": None,
    },
  }


def _mlp_cfg(*, stochastic: bool) -> dict:
  """One MLPModel entry. The critic has no action distribution."""
  hidden = _ACTOR_HIDDEN if stochastic else _CRITIC_HIDDEN
  cfg = {
    "class_name": "MLPModel",
    "hidden_dims": hidden,
    "activation": "elu",
    "obs_normalization": True,
  }
  if stochastic:
    cfg["distribution_cfg"] = {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  return cfg
