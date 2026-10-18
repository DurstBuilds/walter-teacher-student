"""Check that the PD action-matching reward prefers the teacher's action.

The match term must sit near its maximum when the applied action is the
teacher's, drop when the action is zeros, and stay below the combined
forward and yaw tracking weights.
"""

from __future__ import annotations

import torch

from walter_rl.rl.teacher_env import TeacherStudentEnv
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import (
  FORWARD_TRACK_WEIGHT,
  TRACK_TEACHER_WEIGHT,
  YAW_TRACK_WEIGHT,
  walter_fs_teacher_env_cfg,
)

_STEPS = 8
_FORWARD_COMMAND = 0.8
_YAW_COMMAND = 0.5
_MATCHED_KERNEL_MIN = 0.95


def main() -> None:
  """Roll out teacher actions and zeros, then fail if the reward is wrong."""
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = _check_cfg()
  _assert_weight_budget(cfg)
  env = TeacherStudentEnv(cfg, device=device)
  try:
    matched = _rollout(env, follow_teacher=True)
    mismatched = _rollout(env, follow_teacher=False)
  finally:
    env.close()
  _print_rates("teacher action", matched)
  _print_rates("zero action", mismatched)
  _assert_teacher_scores_higher(matched, mismatched)
  print("Teacher tracking reward passed.")


def _check_cfg():
  """Flat teacher task with a fixed nonzero forward and yaw command."""
  cfg = walter_fs_teacher_env_cfg(play=False)
  cfg.scene.num_envs = 2
  cfg.seed = 0
  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.0
  twist.zero_ang_vel_prob = 0.0
  twist.lin_vel_x_power = 1.0
  twist.resampling_time_range = (1.0e6, 1.0e6)
  twist.ranges.lin_vel_x = (_FORWARD_COMMAND, _FORWARD_COMMAND)
  twist.ranges.ang_vel_z = (_YAW_COMMAND, _YAW_COMMAND)
  return cfg


def _assert_weight_budget(cfg) -> None:
  """The match weight cannot outrank forward tracking plus yaw tracking."""
  match_weight = cfg.rewards["track_teacher"].weight
  task_weight = (
    cfg.rewards["track_linear_velocity"].weight
    + cfg.rewards["track_angular_velocity"].weight
  )
  if match_weight >= task_weight:
    raise SystemExit(
      f"track_teacher weight {match_weight} is not below the tracking "
      f"weights {task_weight}."
    )
  if match_weight != TRACK_TEACHER_WEIGHT:
    raise SystemExit(f"Expected teacher weight {TRACK_TEACHER_WEIGHT}.")
  if cfg.rewards["track_linear_velocity"].weight != FORWARD_TRACK_WEIGHT:
    raise SystemExit(f"Expected forward weight {FORWARD_TRACK_WEIGHT}.")
  if cfg.rewards["track_angular_velocity"].weight != YAW_TRACK_WEIGHT:
    raise SystemExit(f"Expected yaw weight {YAW_TRACK_WEIGHT}.")


def _rollout(env: TeacherStudentEnv, *, follow_teacher: bool) -> dict[str, float]:
  """Average unscaled reward rates and the action gap over a short rollout."""
  env.follow_teacher = follow_teacher
  env.reset()
  totals: dict[str, float] = {}
  mse_total = 0.0
  zeros = torch.zeros(
    env.num_envs,
    env.action_manager.total_action_dim,
    device=env.device,
  )
  for _ in range(_STEPS):
    _pin_command(env)
    env.step(zeros)
    for name, value in _mean_reward_rates(env).items():
      totals[name] = totals.get(name, 0.0) + value
    mse_total += _mean_mse(env)
  rates = {name: total / _STEPS for name, total in totals.items()}
  rates["teacher_action_mse"] = mse_total / _STEPS
  rates["track_teacher_kernel"] = rates["track_teacher"] / TRACK_TEACHER_WEIGHT
  return rates


def _pin_command(env: TeacherStudentEnv) -> None:
  """Hold a forward speed and a yaw rate so the teacher command is not zero."""
  command = env.command_manager.get_term("twist")
  command.vel_command_b[:, 0] = _FORWARD_COMMAND
  command.vel_command_b[:, 1] = 0.0
  command.vel_command_b[:, 2] = _YAW_COMMAND
  command.is_standing_env[:] = False


def _mean_reward_rates(env: TeacherStudentEnv) -> dict[str, float]:
  """Unscaled reward rate, averaged across environments."""
  totals: dict[str, float] = {}
  for env_index in range(env.num_envs):
    for name, values in env.reward_manager.get_active_iterable_terms(env_index):
      totals[name] = totals.get(name, 0.0) + values[0]
  return {name: total / env.num_envs for name, total in totals.items()}


def _mean_mse(env: TeacherStudentEnv) -> float:
  """Mean squared action error on this step."""
  total = 0.0
  for env_index in range(env.num_envs):
    terms = env.metrics_manager.get_active_iterable_terms(env_index)
    total += next(value[0] for name, value in terms if name == "teacher_action_mse")
  return total / env.num_envs


def _print_rates(label: str, rates: dict[str, float]) -> None:
  """Print one rollout's reward rates."""
  print(f"[{label}]")
  for name, value in rates.items():
    print(f"  {name}: {value:.6f}")


def _assert_teacher_scores_higher(
  matched: dict[str, float],
  mismatched: dict[str, float],
) -> None:
  """Teacher actions must earn a near-max kernel and beat a zero action."""
  matched_kernel = matched["track_teacher_kernel"]
  mismatched_kernel = mismatched["track_teacher_kernel"]
  if matched_kernel < _MATCHED_KERNEL_MIN:
    raise SystemExit(
      f"Teacher action kernel {matched_kernel:.4f} is below {_MATCHED_KERNEL_MIN}."
    )
  if mismatched_kernel >= matched_kernel - 0.05:
    raise SystemExit(
      "Zero action did not score clearly below the teacher action: "
      f"{mismatched_kernel:.4f} vs {matched_kernel:.4f}."
    )
  if mismatched["teacher_action_mse"] <= matched["teacher_action_mse"]:
    raise SystemExit("Zero action was not farther from the teacher than the copy.")


if __name__ == "__main__":
  main()
