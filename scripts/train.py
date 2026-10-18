"""Train a WaLTER FS velocity policy with PPO.

Examples:
  python scripts/train.py --num-envs 8 --max-iterations 2
  python scripts/train.py --teacher --num-envs 8 --max-iterations 2
  python scripts/train.py --teacher --task rough-test --max-iterations 500
  python scripts/train.py --resume logs/rsl_rl/.../model_500.pt --max-iterations 500
  python scripts/train.py --resume research/logs/training/.../model_500.pt

New runs write under ``research/logs/training``. Pass ``--log-root`` to put
them somewhere else. Checkpoints already in ``logs/rsl_rl/`` stay where they
are and still load through ``--resume`` with an explicit path.

``--resume`` loads the weights, optimizer, and iteration counter.
``--max-iterations`` is how many additional updates to run. The new
checkpoints go in a fresh timestamp directory under the log root.

``--task`` matches playback: ``flat``, ``rough``, ``rough-test``, and
``stairs``. The default is ``flat``. Metrics and checkpoints are uploaded
to the Weights & Biases project ``walter-rl``. Log in once with
``wandb login`` before the first run.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv

from walter_rl.rl.ppo import walter_ppo_runner_cfg
from walter_rl.rl.teacher_env import TeacherStudentEnv
from walter_rl.rl.vecenv import RslRlVecEnvWrapper
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import (
  RlSceneName,
  _TRAIN_ANG_VEL_Z,
  _TRAIN_LIN_VEL_X,
  walter_fs_rl_env_cfg,
  walter_fs_teacher_env_cfg,
)

_DEFAULT_LOG_ROOT = Path("research/logs/training")
_TASKS = ("flat", "rough", "rough-test", "stairs")


def main() -> None:
  """Build the selected scene and run PPO."""
  args = _parse_args()
  resume = _resume_checkpoint(args.resume)
  device = _resolve_device(args.device)
  task: RlSceneName = args.task
  if args.teacher:
    cfg = walter_fs_teacher_env_cfg(play=False, task=task)
    env_cls = TeacherStudentEnv
  else:
    cfg = walter_fs_rl_env_cfg(play=False, task=task)
    env_cls = ManagerBasedRlEnv
  cfg.scene.num_envs = args.num_envs
  cfg.seed = 42
  _apply_command_ranges(cfg, args)
  _apply_teacher_only_reward(cfg, args.teacher_only_reward)
  env = env_cls(cfg, device=device)
  _learn(
    env,
    args.max_iterations,
    _experiment_name(args.teacher, task),
    device,
    resume,
    args.log_root,
  )


def _learn(
  env: ManagerBasedRlEnv,
  max_iterations: int,
  experiment_name: str,
  device: str,
  resume: Path | None,
  log_root: Path,
) -> Path:
  """Run PPO and return the directory that holds the checkpoints."""
  agent_cfg = walter_ppo_runner_cfg(
    max_iterations=max_iterations,
    experiment_name=experiment_name,
  )
  log_dir = log_root / experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir.mkdir(parents=True, exist_ok=True)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
  runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=str(log_dir), device=device)
  _load_resume(runner, resume)
  print(f"[INFO] Training on {device}. Logs: {log_dir}")
  runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
  env.close()
  return log_dir


def _resume_checkpoint(path: str | None) -> Path | None:
  """Return a checkpoint file, or None when training starts fresh."""
  if path is None:
    return None
  checkpoint = Path(path)
  if not checkpoint.is_file():
    raise SystemExit(f"Checkpoint not found: {checkpoint}")
  return checkpoint


def _load_resume(runner: OnPolicyRunner, resume: Path | None) -> None:
  """Restore weights, optimizer, and the iteration counter."""
  if resume is None:
    return
  runner.load(str(resume))
  iteration = runner.current_learning_iteration
  print(f"[INFO] Resumed from {resume} at iteration {iteration}.")


def _apply_command_ranges(cfg, args: argparse.Namespace) -> None:
  """Replace the forward and yaw bounds. Sampling bias is unchanged."""
  if args.lin_vel_x_min > args.lin_vel_x_max:
    raise SystemExit("Forward speed minimum must not exceed the maximum.")
  if args.ang_vel_z_min > args.ang_vel_z_max:
    raise SystemExit("Yaw rate minimum must not exceed the maximum.")
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (args.lin_vel_x_min, args.lin_vel_x_max)
  twist.ranges.ang_vel_z = (args.ang_vel_z_min, args.ang_vel_z_max)


def _apply_teacher_only_reward(cfg, enabled: bool) -> None:
  """Zero every reward except PD action matching."""
  if not enabled:
    return
  if "track_teacher" not in cfg.rewards:
    raise SystemExit("--teacher-only-reward requires --teacher.")
  for name, term in cfg.rewards.items():
    if name != "track_teacher":
      term.weight = 0.0


def _experiment_name(teacher: bool, task: str) -> str:
  """Return the log folder. Flat keeps the original name."""
  base = "walter_fs_teacher" if teacher else "walter_fs_velocity"
  if task == "flat":
    return base
  return f"{base}_{task}"


def _resolve_device(requested: str | None) -> str:
  """Use CUDA when it is available and the caller did not force CPU."""
  if requested is not None:
    return requested
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--max-iterations", type=int, default=2)
  parser.add_argument(
    "--task",
    choices=_TASKS,
    default="flat",
    help="Scene. Same choices as playback: flat, rough, rough-test, stairs.",
  )
  parser.add_argument(
    "--teacher",
    action="store_true",
    help="Add the full-body PD action-matching reward.",
  )
  parser.add_argument(
    "--teacher-only-reward",
    action="store_true",
    help="Zero every reward except PD action matching. Requires --teacher.",
  )
  parser.add_argument("--lin-vel-x-min", type=float, default=_TRAIN_LIN_VEL_X[0])
  parser.add_argument("--lin-vel-x-max", type=float, default=_TRAIN_LIN_VEL_X[1])
  parser.add_argument("--ang-vel-z-min", type=float, default=_TRAIN_ANG_VEL_Z[0])
  parser.add_argument("--ang-vel-z-max", type=float, default=_TRAIN_ANG_VEL_Z[1])
  parser.add_argument(
    "--device",
    default=None,
    help="Torch device, for example cuda:0 or cpu. Default follows the GPU.",
  )
  parser.add_argument(
    "--log-root",
    type=Path,
    default=_DEFAULT_LOG_ROOT,
    help=(
      "Directory for new PPO logs. Defaults to research/logs/training. "
      "Existing logs/rsl_rl checkpoints are left in place."
    ),
  )
  parser.add_argument(
    "--resume",
    default=None,
    help=(
      "Checkpoint to continue from, including an older logs/rsl_rl file. "
      "--max-iterations counts additional updates. Logs go to a new "
      "timestamp directory under --log-root."
    ),
  )
  return parser.parse_args()


if __name__ == "__main__":
  main()
