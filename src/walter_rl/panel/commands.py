"""Build the play and train commands the panel is allowed to run."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from walter_rl.assets.payloads import available_payloads

_TASKS = ("flat", "rough", "rough-test", "stairs")
_POLICIES = ("pd", "smooth-tumble", "checkpoint")
_VIEWERS = ("viser", "native")
_DEVICES = ("auto", "cuda:0", "cpu")
_MODES = ("new", "resume")
_MAX_ENVS = 16384
_MAX_ITERS = 100_000
_SUMMARY_LIMIT = 240


class PanelError(Exception):
  """A request the panel should reject before starting a process."""


def play_command(root: Path, body: dict) -> list[str]:
  """Return the argv for one playback session."""
  policy = _choice(body, "policy", _POLICIES, "Choose a policy.")
  task = _choice(body, "task", _TASKS, "Choose a scene.")
  viewer = _choice(body, "viewer", _VIEWERS, "Choose a viewer.")
  robots = _count(body, "num_envs", _MAX_ENVS, "Number of robots")
  if policy == "smooth-tumble":
    return _tumble_command(root, body, task, viewer, robots)
  return _play_script_command(root, body, policy, task, viewer, robots)


def train_command(root: Path, body: dict) -> list[str]:
  """Return the argv for a new or continued training run."""
  mode = _choice(body, "mode", _MODES, "Choose a new run or a checkpoint.")
  task = _choice(body, "task", _TASKS, "Choose a scene.")
  robots = _count(body, "num_envs", _MAX_ENVS, "Number of robots")
  updates = _count(body, "max_iterations", _MAX_ITERS, "Updates")
  device = _choice(body, "device", _DEVICES, "Choose a processor.")
  command = _train_base(root, task, robots, updates)
  _append_teacher(command, body)
  _append_command_ranges(command, body)
  if device != "auto":
    command.extend(["--device", device])
  if mode == "resume":
    command.extend(["--resume", str(checkpoint_file(root, body.get("checkpoint")))])
  return command


def summary_text(body: dict) -> str:
  """Keep the sentence the page showed above the button."""
  text = body.get("summary", "")
  if not isinstance(text, str):
    return ""
  return " ".join(text.split())[:_SUMMARY_LIMIT]


def checkpoint_file(root: Path, raw: object) -> Path:
  """Resolve a checkpoint under logs/rsl_rl."""
  if not isinstance(raw, str) or not raw.strip():
    raise PanelError("Choose a checkpoint.")
  logs = (root / "logs" / "rsl_rl").resolve()
  path = Path(raw)
  if not path.is_absolute():
    path = root / path
  path = path.resolve()
  if path.suffix != ".pt" or not path.is_file():
    raise PanelError("Checkpoint not found.")
  try:
    path.relative_to(logs)
  except ValueError:
    raise PanelError("Checkpoint must be under logs/rsl_rl.") from None
  return path


def _play_script_command(
  root: Path,
  body: dict,
  policy: str,
  task: str,
  viewer: str,
  robots: int,
) -> list[str]:
  """argv for the PD policy or a trained checkpoint."""
  command = [
    sys.executable,
    str(root / "scripts" / "play.py"),
    "--policy",
    "checkpoint" if policy == "checkpoint" else "pd",
    "--task",
    task,
    "--num-envs",
    str(robots),
    "--viewer",
    viewer,
  ]
  if policy == "checkpoint":
    checkpoint = checkpoint_file(root, body.get("checkpoint"))
    command.extend(["--checkpoint", str(checkpoint)])
    return command
  _append_payload(command, body)
  return command


def _tumble_command(
  root: Path,
  body: dict,
  task: str,
  viewer: str,
  robots: int,
) -> list[str]:
  """argv for the smooth-tumble controller."""
  command = [
    sys.executable,
    str(root / "scripts" / "play_smooth_tumble.py"),
    "--task",
    task,
    "--num-envs",
    str(robots),
    "--viewer",
    viewer,
  ]
  _append_payload(command, body)
  return command


def _train_base(root: Path, task: str, robots: int, updates: int) -> list[str]:
  """argv shared by a new run and a continued run."""
  return [
    sys.executable,
    str(root / "scripts" / "train.py"),
    "--task",
    task,
    "--num-envs",
    str(robots),
    "--max-iterations",
    str(updates),
  ]


def _append_teacher(command: list[str], body: dict) -> None:
  """Add the teacher flag, and the teacher-only reward flag, when selected."""
  teacher = body.get("teacher", False)
  if not isinstance(teacher, bool):
    raise PanelError("Teacher must be on or off.")
  only_teacher = body.get("teacher_only_reward", False)
  if not isinstance(only_teacher, bool):
    raise PanelError("Teacher-only reward must be on or off.")
  if only_teacher and not teacher:
    raise PanelError("Reward only teacher tracking needs the PD teacher.")
  if teacher:
    command.append("--teacher")
  if only_teacher:
    command.append("--teacher-only-reward")


def _append_command_ranges(command: list[str], body: dict) -> None:
  """Pass the forward and yaw ranges. Bias and zero-yaw rate stay as they are."""
  forward_min = _real(body, "lin_vel_x_min", "Forward speed minimum")
  forward_max = _real(body, "lin_vel_x_max", "Forward speed maximum")
  yaw_min = _real(body, "ang_vel_z_min", "Yaw rate minimum")
  yaw_max = _real(body, "ang_vel_z_max", "Yaw rate maximum")
  if forward_min > forward_max:
    raise PanelError("Forward speed minimum must not exceed the maximum.")
  if yaw_min > yaw_max:
    raise PanelError("Yaw rate minimum must not exceed the maximum.")
  command.extend(
    [
      "--lin-vel-x-min",
      _number(forward_min),
      "--lin-vel-x-max",
      _number(forward_max),
      "--ang-vel-z-min",
      _number(yaw_min),
      "--ang-vel-z-max",
      _number(yaw_max),
    ]
  )


def _append_payload(command: list[str], body: dict) -> None:
  """Add a welded mass when one is selected."""
  payload = body.get("payload") or None
  if payload is None:
    return
  if not isinstance(payload, str) or payload not in available_payloads():
    raise PanelError("Choose a payload from the list.")
  command.extend(["--payload", payload])


def _choice(body: dict, key: str, allowed: tuple[str, ...], message: str) -> str:
  """Return one allow-listed string field."""
  value = body.get(key)
  if value not in allowed:
    raise PanelError(message)
  return str(value)


_RANGE_LIMIT = 20.0


def _real(body: dict, key: str, label: str) -> float:
  """Return a finite command bound inside ±20."""
  value = body.get(key)
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise PanelError(f"{label} must be a number.")
  if not math.isfinite(value):
    raise PanelError(f"{label} must be a finite number.")
  if value < -_RANGE_LIMIT or value > _RANGE_LIMIT:
    raise PanelError(f"{label} must be between -{_RANGE_LIMIT:g} and {_RANGE_LIMIT:g}.")
  return float(value)


def _number(value: float) -> str:
  """Format a bound without trailing zeros."""
  return format(value, "g")


def _count(body: dict, key: str, limit: int, label: str) -> int:
  """Return a whole number in 1..limit."""
  value = body.get(key)
  if isinstance(value, bool) or not isinstance(value, int):
    raise PanelError(f"{label} must be a whole number.")
  if value < 1 or value > limit:
    raise PanelError(f"{label} must be between 1 and {limit}.")
  return value
