"""Record one rough-test cell for the PD teacher or a student checkpoint.

Examples:
  python scripts/eval_cells.py --controller pd --cell flat --difficulty medium
  python scripts/eval_cells.py --controller checkpoint --cell pyramid-stairs \\
    --difficulty hard --checkpoint logs/rsl_rl/.../model_600.pt
  python scripts/eval_cells.py --controller checkpoint --cell pyramid-stairs \\
    --difficulty hard --checkpoint research/logs/training/.../model_500.pt \\
    --ablate-scan

The clip is one environment on one sub-terrain. ``medium`` is difficulty 2/3,
the third of the four rough-test rows (0, 1/3, 2/3, 1). ``hard`` is the last
row. PD playback sets ``TeacherStudentEnv.follow_teacher``, so ``step`` applies
the full-body PD action. ``--ablate-scan`` replaces the actor and critic
``height_scan`` term with zeros before the policy reads it. The csv still
records the real forward-scan clearance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer.viewer_config import ViewerConfig

from walter_rl.rl.checkpoint_policy import CheckpointPolicy
from walter_rl.rl.ppo import walter_ppo_runner_cfg
from walter_rl.rl.teacher_env import TeacherStudentEnv
from walter_rl.rl.vecenv import RslRlVecEnvWrapper
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import (
  _TERRAIN_SCAN_OFFSET,
  _TERRAIN_SCAN_RESOLUTION,
  _TERRAIN_SCAN_SIZE,
  walter_fs_teacher_env_cfg,
)
from walter_rl.tasks.velocity.mdp.observations import finite_height_scan
from walter_rl.tasks.velocity.mdp.rewards import root_yaw

_FORWARD_COMMAND = 1.0
_CLIP_SECONDS = 8.0
# Long enough that an eval clip never resamples, and unequal so uniform_ is legal.
_RESAMPLE_S = (1.0e6, 1.0e6 + 1.0)
# Four rough-test rows use difficulties 0, 1/3, 2/3, and 1.
_DIFFICULTY = {"medium": 2.0 / 3.0, "hard": 1.0}
_CELL_SUBTERRAIN = {
  "flat": "flat",
  "pyramid-stairs": "pyramid_stairs",
  "inverted-stairs": "pyramid_stairs_inv",
  "box-grid": "random_grid",
  "slope": "hf_pyramid_slope",
}
_CELLS = tuple(_CELL_SUBTERRAIN)
_DIFFICULTIES = tuple(_DIFFICULTY)


def main() -> None:
  """Record one cell and write ``clip.mp4`` plus ``frames.csv``."""
  args = _parse_args()
  checkpoint = _checkpoint_path(args)
  device = _resolve_device(args.device)
  env, policy = _open_env(_build_cfg(args), device, checkpoint)
  try:
    steps = _step_count(args, env.step_dt)
    rows, frames = _rollout(
      env,
      policy,
      steps,
      ablate=args.ablate_scan,
      cell=args.cell,
      difficulty=args.difficulty,
    )
    out = _output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "frames.csv", rows)
    _write_video(out / "clip.mp4", frames, fps=round(1.0 / env.step_dt))
  finally:
    env.close()
  print(f"[INFO] Wrote {out / 'clip.mp4'} and {out / 'frames.csv'}")


def _build_cfg(args: argparse.Namespace) -> ManagerBasedRlEnvCfg:
  """Play-mode rough-test scene pinned to one cell and a fixed command."""
  cfg = walter_fs_teacher_env_cfg(play=True, task="rough-test")
  _pin_cell(cfg, args.cell, args.difficulty, args.seed)
  _pin_spawn(cfg)
  _pin_command_cfg(cfg)
  _keep_clip_running(cfg)
  _track_torso(cfg)
  if args.ablate_scan:
    _ablate_height_scan(cfg)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  return cfg


def _pin_cell(
  cfg: ManagerBasedRlEnvCfg,
  cell: str,
  difficulty: str,
  seed: int,
) -> None:
  """Keep one rough-test sub-terrain at one difficulty.

  A single row whose difficulty range is a point matches that rough-test
  row without spawning the random play-mode grid.
  """
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise SystemExit("rough-test did not build a terrain generator.")
  generator = terrain.terrain_generator
  name = _CELL_SUBTERRAIN[cell]
  if name not in generator.sub_terrains:
    known = ", ".join(generator.sub_terrains)
    raise SystemExit(f"Sub-terrain {name} is not in rough-test ({known}).")
  level = _DIFFICULTY[difficulty]
  chosen = replace(generator.sub_terrains[name], proportion=1.0)
  generator.sub_terrains = {name: chosen}
  generator.curriculum = True
  generator.num_rows = 1
  generator.num_cols = 1
  generator.difficulty_range = (level, level)
  generator.seed = seed
  terrain.max_init_terrain_level = 0
  cfg.events.pop("randomize_terrain", None)


def _pin_spawn(cfg: ManagerBasedRlEnvCfg) -> None:
  """Spawn at the patch origin, facing +x, with no pose noise."""
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.01, 0.01),
    "yaw": (0.0, 0.0),
  }


def _pin_command_cfg(cfg: ManagerBasedRlEnvCfg) -> None:
  """Stop command resampling from replacing the fixed forward speed."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.resampling_time_range = _RESAMPLE_S


def _command_fixed_forward(env: TeacherStudentEnv) -> None:
  """Write 1 m/s forward and zero yaw into the live command."""
  twist = env.command_manager.get_term("twist")
  twist.vel_command_b[:] = 0.0
  twist.vel_command_b[:, 0] = _FORWARD_COMMAND
  twist.vel_command_w[:] = twist.vel_command_b
  twist.is_standing_env[:] = False
  twist.is_heading_env[:] = False
  twist.is_world_env[:] = False
  twist.is_forward_env[:] = False


def _keep_clip_running(cfg: ManagerBasedRlEnvCfg) -> None:
  """Leave a fall in the clip instead of resetting the episode."""
  cfg.terminations.clear()


def _track_torso(cfg: ManagerBasedRlEnvCfg) -> None:
  """Follow the torso so the robot stays in frame as it drives."""
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
  cfg.viewer.entity_name = "robot"
  cfg.viewer.body_name = "torso"


def _ablate_height_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  """Zero ``height_scan`` in every group the networks read."""
  for group_name in ("actor", "critic"):
    term = cfg.observations[group_name].terms.get("height_scan")
    if term is None:
      raise SystemExit(f"{group_name} has no height_scan term to ablate.")
    term.func = _zeroed_height_scan


def _zeroed_height_scan(env: TeacherStudentEnv, sensor_name: str) -> torch.Tensor:
  """Match the scan shape and return zeros for the policy input."""
  heights = finite_height_scan(env, sensor_name=sensor_name)
  return torch.zeros_like(heights)


def _open_env(
  cfg: ManagerBasedRlEnvCfg,
  device: str,
  checkpoint: Path | None,
) -> tuple[TeacherStudentEnv, CheckpointPolicy | None]:
  """Build the teacher env and, for a checkpoint, the actor used by play."""
  env = TeacherStudentEnv(cfg, device=device, render_mode="rgb_array")
  if checkpoint is None:
    env.follow_teacher = True
    return env, None
  env.follow_teacher = False
  return env, _load_policy(env, checkpoint, device)


def _load_policy(
  env: TeacherStudentEnv,
  checkpoint: Path,
  device: str,
) -> CheckpointPolicy:
  """Load an actor the same way ``scripts/play.py`` loads one."""
  agent_cfg = walter_ppo_runner_cfg(
    max_iterations=1,
    experiment_name="walter_fs_eval",
  )
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
  runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=None, device=device)
  runner.load(str(checkpoint))
  model = runner.get_inference_policy(device=device)
  return CheckpointPolicy(model)


def _rollout(
  env: TeacherStudentEnv,
  policy: CheckpointPolicy | None,
  steps: int,
  *,
  ablate: bool,
  cell: str,
  difficulty: str,
) -> tuple[list[dict[str, float | int | str]], list[np.ndarray]]:
  """Step one environment and keep a csv row plus a frame each step."""
  obs, _ = env.reset()
  obs = _refresh_observations(env)
  if ablate:
    _assert_actor_scan_zero(env, obs)
  names = _action_names(env)
  knee_ids = _knee_ids(env)
  mask = _forward_ray_mask(env.device)
  prev_xy = env.scene["robot"].data.root_link_pos_w[0, :2].detach().clone()
  traveled = 0.0
  rows: list[dict[str, float | int | str]] = []
  frames: list[np.ndarray] = []
  for step in range(1, steps + 1):
    student = _policy_action(env, policy, obs)
    obs, _, _, _, _ = env.step(student)
    if ablate:
      _assert_actor_scan_zero(env, obs)
    metrics, prev_xy, traveled = _body_metrics(
      env, knee_ids, prev_xy, traveled, mask
    )
    rows.append(
      _frame_row(env, step, names, student, policy, metrics, cell, difficulty, ablate)
    )
    frames.append(_copy_frame(env.render()))
  return rows, frames


def _refresh_observations(env: TeacherStudentEnv) -> dict:
  """Pin the command, then rebuild observations so the policy sees it."""
  _command_fixed_forward(env)
  return env.observation_manager.compute()


def _policy_action(
  env: TeacherStudentEnv,
  policy: CheckpointPolicy | None,
  obs: dict,
) -> torch.Tensor:
  """Return the student action. PD playback ignores it inside ``step``."""
  if policy is None:
    dim = env.action_manager.total_action_dim
    return torch.zeros(env.num_envs, dim, device=env.device)
  return policy(obs).detach().clone()


def _body_metrics(
  env: TeacherStudentEnv,
  knee_ids: torch.Tensor,
  prev_xy: torch.Tensor,
  traveled: float,
  mask: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, float]:
  """Post-step speed, knee rate, uprightness, path length, and forward scan.

  Uprightness is 1 when the torso up-axis opposes gravity. Forward-scan
  height is the mean torso clearance of rays with positive body-frame x.
  """
  robot = env.scene["robot"]
  xy = robot.data.root_link_pos_w[0, :2]
  traveled += float(torch.linalg.norm(xy - prev_xy))
  metrics = {
    "forward_speed": float(robot.data.root_link_lin_vel_b[0, 0]),
    "along_spawn_speed": _along_spawn_speed(env, robot),
    "yaw_from_spawn": _yaw_from_spawn(env, robot),
    "knee_rate": float(robot.data.joint_vel[0, knee_ids].abs().mean()),
    "uprightness": float(-robot.data.projected_gravity_b[0, 2]),
    "distance_traveled": traveled,
    "forward_scan_height": _forward_scan_height(env, mask),
  }
  return metrics, xy.detach().clone(), traveled


def _frame_row(
  env: TeacherStudentEnv,
  step: int,
  names: list[str],
  student: torch.Tensor,
  policy: CheckpointPolicy | None,
  metrics: dict[str, float],
  cell: str,
  difficulty: str,
  ablate: bool,
) -> dict[str, float | int | str]:
  """One csv row. PD mode logs the applied PD action in the student columns."""
  pd_action = env.teacher_action
  logged_student = pd_action if policy is None else student
  row: dict[str, float | int | str] = {
    "step": step,
    "time_s": step * env.step_dt,
    "cell": cell,
    "difficulty": difficulty,
    "scan_ablated": int(ablate),
  }
  row.update(metrics)
  row["pd_tumble_gate"] = float(env.teacher.tumbling[0])
  _put_action(row, "student", names, logged_student)
  _put_action(row, "pd", names, pd_action)
  return row


def _put_action(
  row: dict[str, float | int | str],
  prefix: str,
  names: list[str],
  action: torch.Tensor,
) -> None:
  """Write one column per action target for environment 0."""
  values = action[0].detach().cpu().tolist()
  for name, value in zip(names, values, strict=True):
    row[f"{prefix}_{name}"] = float(value)


def _along_spawn_speed(env: TeacherStudentEnv, robot) -> float:
  """World speed along the yaw latched at the start of the episode."""
  yaw = getattr(env, "spawn_yaw_w", None)
  if yaw is None:
    return float("nan")
  vel = robot.data.root_link_lin_vel_w[0, :2]
  heading = float(yaw[0])
  return float(vel[0] * np.cos(heading) + vel[1] * np.sin(heading))


def _yaw_from_spawn(env: TeacherStudentEnv, robot) -> float:
  """Upright yaw error from the spawn heading, in radians."""
  yaw = getattr(env, "spawn_yaw_w", None)
  if yaw is None:
    return float("nan")
  current = float(root_yaw(robot.data.root_link_quat_w[:1])[0])
  error = current - float(yaw[0])
  return float((error + np.pi) % (2.0 * np.pi) - np.pi)


def _forward_scan_height(env: TeacherStudentEnv, mask: torch.Tensor) -> float:
  """Mean real scan clearance ahead of the torso, independent of ablation."""
  heights = finite_height_scan(env, sensor_name="terrain_scan")[0]
  if heights.numel() != mask.numel():
    raise SystemExit(
      f"Height scan has {heights.numel()} rays; forward mask has {mask.numel()}."
    )
  return float(heights[mask].mean())


def _forward_ray_mask(device: torch.device | str) -> torch.Tensor:
  """True for scan rays whose body-frame x is in front of the torso."""
  size_x, size_y = _TERRAIN_SCAN_SIZE
  res = _TERRAIN_SCAN_RESOLUTION
  offset_x, _offset_y = _TERRAIN_SCAN_OFFSET
  x = torch.arange(-size_x / 2, size_x / 2 + res * 0.5, res, device=device)
  y = torch.arange(-size_y / 2, size_y / 2 + res * 0.5, res, device=device)
  grid_x, _grid_y = torch.meshgrid(x + offset_x, y, indexing="xy")
  return grid_x.reshape(-1) > 0.0


def _knee_ids(env: TeacherStudentEnv) -> torch.Tensor:
  """Joint indices for every thigh-shin knee."""
  ids, _names = env.scene["robot"].find_joints(r".*_thigh_shin_joint")
  if not ids:
    raise SystemExit("No knee joints found.")
  return torch.tensor(ids, device=env.device, dtype=torch.long)


def _action_names(env: TeacherStudentEnv) -> list[str]:
  """Target name of each column in the action vector."""
  names: list[str] = []
  for term_name in env.action_manager.active_terms:
    term = env.action_manager.get_term(term_name)
    targets = getattr(term, "target_names", None)
    if not targets:
      targets = [f"{term_name}_{index}" for index in range(term.action_dim)]
    names.extend(targets)
  if len(names) != env.action_manager.total_action_dim:
    raise SystemExit("Action names do not match the action dimension.")
  return names


def _assert_actor_scan_zero(env: TeacherStudentEnv, obs: dict) -> None:
  """Fail if the actor still receives a non-zero height scan."""
  actor = obs["actor"]
  if not isinstance(actor, torch.Tensor):
    raise SystemExit("Actor observations are not concatenated.")
  scan = actor[:, _term_slice(env, "height_scan")]
  if float(scan.abs().max()) != 0.0:
    raise SystemExit("Scan ablation left a non-zero actor height_scan.")


def _term_slice(env: TeacherStudentEnv, term_name: str) -> slice:
  """Column slice of one concatenated actor term."""
  names = env.observation_manager.active_terms["actor"]
  dims = env.observation_manager.group_obs_term_dim["actor"]
  start = 0
  for name, dim in zip(names, dims, strict=True):
    width = 1
    for size in dim:
      width *= int(size)
    if name == term_name:
      return slice(start, start + width)
    start += width
  raise SystemExit("Actor observation has no height_scan term.")


def _copy_frame(frame: np.ndarray | None) -> np.ndarray:
  """Copy one RGB frame so the renderer can reuse its buffer."""
  if frame is None:
    raise SystemExit("Offscreen renderer returned no frame.")
  image = np.asarray(frame)
  if image.dtype != np.uint8:
    image = np.clip(image, 0, 255).astype(np.uint8)
  return np.array(image, copy=True)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
  """Write per-frame metrics. The first row fixes the column order."""
  if not rows:
    raise SystemExit("No frames to write.")
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
  """Write the offscreen frames to an mp4."""
  if not frames:
    raise SystemExit("No frames to write.")
  media.write_video(str(path), frames, fps=fps)


def _checkpoint_path(args: argparse.Namespace) -> Path | None:
  """Return the checkpoint file, or None for the PD controller."""
  if args.controller == "pd":
    return None
  if args.checkpoint is None:
    raise SystemExit("--controller checkpoint requires --checkpoint PATH")
  checkpoint = Path(args.checkpoint)
  if not checkpoint.is_file():
    raise SystemExit(f"Checkpoint not found: {checkpoint}")
  return checkpoint


def _step_count(args: argparse.Namespace, step_dt: float) -> int:
  """Use ``--steps`` when set, otherwise cover ``--seconds``."""
  if args.steps is not None:
    if args.steps < 1:
      raise SystemExit("--steps must be at least 1.")
    return args.steps
  if args.seconds <= 0.0:
    raise SystemExit("--seconds must be positive.")
  return max(1, round(args.seconds / step_dt))


def _output_dir(args: argparse.Namespace) -> Path:
  """Default to ``research/logs/analysis/<run_id>/``."""
  if args.output is not None:
    return Path(args.output)
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  name = f"{stamp}_{args.controller}_{args.cell}_{args.difficulty}"
  if args.ablate_scan:
    name += "_ablate-scan"
  return Path("research/logs/analysis") / name


def _resolve_device(requested: str | None) -> str:
  """Use CUDA when it is available and the caller did not force CPU."""
  if requested is not None:
    return requested
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--controller", choices=("pd", "checkpoint"), default="pd")
  parser.add_argument("--checkpoint", type=str, default=None)
  parser.add_argument("--cell", choices=_CELLS, required=True)
  parser.add_argument("--difficulty", choices=_DIFFICULTIES, required=True)
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Directory for clip.mp4 and frames.csv. Default is a timestamped run.",
  )
  parser.add_argument(
    "--ablate-scan",
    action="store_true",
    help="Zero the policy height_scan input. The csv still records the real scan.",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Environment steps to record. Overrides --seconds.",
  )
  parser.add_argument(
    "--seconds",
    type=float,
    default=_CLIP_SECONDS,
    help="Clip length when --steps is omitted.",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--device",
    default=None,
    help="Torch device, for example cuda:0 or cpu. Default follows the GPU.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  main()
