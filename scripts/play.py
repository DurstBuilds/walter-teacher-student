"""Play a classic PD policy or a trained velocity checkpoint.

Examples:
  python scripts/play.py --policy pd --task flat
  python scripts/play.py --policy checkpoint --task rough-test \\
    --checkpoint logs/rsl_rl/.../model_600.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer

from walter_rl.assets.payloads import available_payloads
from walter_rl.control.full_body_pd.env import TaskName, make_pd_env
from walter_rl.control.full_body_pd.policy import FullBodyPdPolicy
from walter_rl.rl.checkpoint_policy import CheckpointPolicy
from walter_rl.rl.ppo import walter_ppo_runner_cfg
from walter_rl.rl.vecenv import RslRlVecEnvWrapper
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import walter_fs_rl_env_cfg
from walter_rl.viewer import WalterViserPlayViewer

_TASKS = ("flat", "rough", "rough-test", "stairs")


def main() -> None:
  """Open the viewer on either the PD controller or a checkpoint."""
  args = _parse_args()
  if args.policy == "pd":
    env, policy = _pd_policy(args)
  else:
    env, policy = _checkpoint_policy(args)
  viewer = WalterViserPlayViewer if args.viewer == "viser" else NativeMujocoViewer
  try:
    viewer(env, policy).run()
  finally:
    env.close()


def _pd_policy(args: argparse.Namespace) -> tuple[ManagerBasedRlEnv, FullBodyPdPolicy]:
  """Build a classic full-body PD play scene."""
  task: TaskName = args.task
  env = make_pd_env(
    task,
    play=True,
    num_envs=args.num_envs,
    payload=args.payload,
  )
  return env, FullBodyPdPolicy(env)


def _checkpoint_policy(
  args: argparse.Namespace,
) -> tuple[ManagerBasedRlEnv, CheckpointPolicy]:
  """Load an actor onto the scene selected by ``--task``."""
  if args.checkpoint is None:
    raise SystemExit("--policy checkpoint requires --checkpoint PATH")
  checkpoint = Path(args.checkpoint)
  if not checkpoint.is_file():
    raise SystemExit(f"Checkpoint not found: {checkpoint}")
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = walter_fs_rl_env_cfg(play=True, task=args.task)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  agent_cfg = walter_ppo_runner_cfg(max_iterations=1, experiment_name="walter_fs_play")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
  runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=None, device=device)
  runner.load(str(checkpoint))
  model = runner.get_inference_policy(device=device)
  return env, CheckpointPolicy(model)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", choices=("pd", "checkpoint"), default="pd")
  parser.add_argument("--checkpoint", type=str, default=None)
  parser.add_argument(
    "--task",
    choices=_TASKS,
    default="flat",
    help="Scene for both PD and checkpoint playback.",
  )
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--viewer", choices=("viser", "native"), default="viser")
  parser.add_argument(
    "--payload",
    choices=available_payloads(),
    default=None,
    help="Rigid mass welded to the top of the robot. Used by the PD policy.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  main()
