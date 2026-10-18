"""Scenes, payloads, and saved checkpoints for the panel."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from walter_rl.assets.payloads import available_payloads
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import (
  _TRAIN_ANG_VEL_Z,
  _TRAIN_LIN_VEL_X,
)

_RUN_FORMAT = "%Y-%m-%d_%H-%M-%S"
_MODEL_NAME = re.compile(r"^model_(\d+)\.pt$")
_PAYLOAD_DETAILS = {
  "rider": "75 kg seated human on the torso.",
}


def catalog() -> dict:
  """Return the choices the page renders."""
  return {
    "scenes": _scenes(),
    "policies": _policies(),
    "payloads": _payload_choices(),
    "viewers": _viewers(),
    "devices": _devices(),
    "defaults": _defaults(),
  }


def _scenes() -> list[dict]:
  """Ground the robot can be played or trained on."""
  return [
    {"id": "flat", "label": "Flat", "detail": "Level ground."},
    {"id": "rough", "label": "Rough", "detail": "Uneven terrain."},
    {
      "id": "rough-test",
      "label": "Rough test",
      "detail": "A short rough grid with taller stairs.",
    },
    {"id": "stairs", "label": "Stairs", "detail": "Stair climbs."},
  ]


def _policies() -> list[dict]:
  """Controllers the play tab can launch."""
  return [
    {
      "id": "pd",
      "label": "Classical PD",
      "detail": "Balance, speed, and skid steering in code.",
    },
    {
      "id": "smooth-tumble",
      "label": "Smooth tumble",
      "detail": "The same controller with a smoother tumble recovery.",
    },
    {
      "id": "checkpoint",
      "label": "Trained checkpoint",
      "detail": "A saved PPO policy.",
    },
  ]


def _viewers() -> list[dict]:
  """Where playback draws the robot."""
  return [
    {"id": "viser", "label": "Browser"},
    {"id": "native", "label": "Desktop window"},
  ]


def _devices() -> list[dict]:
  """Processors a training run can use."""
  return [
    {"id": "auto", "label": "Automatic"},
    {"id": "cuda:0", "label": "GPU"},
    {"id": "cpu", "label": "CPU"},
  ]


def _defaults() -> dict:
  """Values the page fills in before the user changes anything."""
  return {
    "scene": "flat",
    "policy": "pd",
    "viewer": "viser",
    "device": "auto",
    "play_num_envs": 1,
    "train_num_envs": 1024,
    "train_iterations": 1000,
    "lin_vel_x_min": _TRAIN_LIN_VEL_X[0],
    "lin_vel_x_max": _TRAIN_LIN_VEL_X[1],
    "ang_vel_z_min": _TRAIN_ANG_VEL_Z[0],
    "ang_vel_z_max": _TRAIN_ANG_VEL_Z[1],
  }


def list_runs(root: Path) -> list[dict]:
  """Group checkpoints by training run, newest run first."""
  found = [item for item in (_collect_run(root, path) for path in _run_dirs(root)) if item]
  found.sort(key=lambda item: item[0], reverse=True)
  return [item[1] for item in found]


def _payload_choices() -> list[dict]:
  """None, then each registered rigid mass."""
  choices = [{"id": "", "label": "None", "detail": "Robot only."}]
  for name in available_payloads():
    label = name[:1].upper() + name[1:]
    detail = _PAYLOAD_DETAILS.get(name, "Rigid mass on the torso.")
    choices.append({"id": name, "label": label, "detail": detail})
  return choices


def _run_dirs(root: Path) -> list[Path]:
  """Return timestamp directories under logs/rsl_rl."""
  logs = root / "logs" / "rsl_rl"
  if not logs.is_dir():
    return []
  return [path for path in logs.glob("*/*") if path.is_dir()]


def _collect_run(root: Path, path: Path) -> tuple[str, dict] | None:
  """One run, or None when it has no model files."""
  checkpoints = _checkpoints(root, path)
  if not checkpoints:
    return None
  run_label, sort_key = _run_label(path)
  return sort_key, {
    "experiment": _experiment_label(path.parent.name),
    "run": run_label,
    "checkpoints": checkpoints,
  }


def _checkpoints(root: Path, run_dir: Path) -> list[dict]:
  """Model files in this run, highest iteration first."""
  found: list[tuple[int, Path]] = []
  for path in run_dir.glob("model_*.pt"):
    match = _MODEL_NAME.match(path.name)
    if match is None or not path.is_file():
      continue
    found.append((int(match.group(1)), path))
  found.sort(key=lambda item: item[0], reverse=True)
  return [
    {"iteration": iteration, "path": path.resolve().relative_to(root.resolve()).as_posix()}
    for iteration, path in found
  ]


def _experiment_label(folder: str) -> str:
  """Turn a log folder name into a short run title."""
  if folder == "walter_fs_velocity":
    return "Velocity, flat"
  if folder == "walter_fs_teacher":
    return "Teacher, flat"
  if folder.startswith("walter_fs_teacher_"):
    return f"Teacher, {folder.removeprefix('walter_fs_teacher_')}"
  if folder.startswith("walter_fs_velocity_"):
    return f"Velocity, {folder.removeprefix('walter_fs_velocity_')}"
  return folder


def _run_label(path: Path) -> tuple[str, str]:
  """Return a display time and a sort key for one run directory."""
  try:
    moment = datetime.strptime(path.name, _RUN_FORMAT)
  except ValueError:
    stamp = datetime.fromtimestamp(path.stat().st_mtime)
    text = stamp.strftime("%Y-%m-%d %H:%M")
    return text, stamp.strftime("%Y-%m-%d %H:%M:%S")
  return moment.strftime("%Y-%m-%d %H:%M"), moment.strftime("%Y-%m-%d %H:%M:%S")
