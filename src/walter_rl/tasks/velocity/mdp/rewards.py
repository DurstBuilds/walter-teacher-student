"""WaLTER velocity reward terms."""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from walter_rl.control.full_body_pd.gains import TumbleGains
from walter_rl.tasks.velocity.mdp.observations import motor_torques

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# Longitudinal sample of the torso scan. Matches ``_TERRAIN_SCAN_RESOLUTION``.
_SCAN_SPACING_M = 0.1
# Forward edge is 1.6 m (size 2.4 m, offset 0.4 m): bins at 0.1, 0.2, ..., 1.6.
_FORWARD_BIN_COUNT = 16
_MATCH_STEP_FULL_M = 0.04
_MATCH_STEP_OFF_M = 0.08
_PD_MIN_FORWARD_CMD = TumbleGains().min_forward_cmd
_PD_TUMBLE_RATE = TumbleGains().rate
# Heading hold applies only while the torso is still upright. A pitch tumble
# past this value makes quaternion yaw meaningless, and that tumble is allowed.
_UPRIGHT_FOR_HEADING = 0.7


def track_teacher_action(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Reward copying the teacher action recorded from the pre-step state.

  The kernel is in ``[0, 1]``. On a flat scan that action is the reactive
  PD. On a step it is the same PD action with tumble forced on. Knee
  position error is the shortest angle. While that target is a tumble, the
  knee columns compare rate in rad/s instead.
  """
  error = _teacher_squared_error(env)
  return torch.exp(-error / std**2)


def teacher_action_mse(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean squared gap between the student action and the PD teacher."""
  return _teacher_squared_error(env)


def _teacher_squared_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean squared match error, including the env's knee comparison."""
  student = env.action_manager.action
  teacher = _teacher_action(env)
  delta_fn = getattr(env, "teacher_match_delta", None)
  delta = delta_fn(student, teacher) if callable(delta_fn) else student - teacher
  return torch.mean(torch.square(delta), dim=-1)


def _teacher_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  """PD action snapshotted before physics. Training envs set this attribute."""
  teacher = getattr(env, "teacher_action", None)
  if not isinstance(teacher, torch.Tensor):
    raise RuntimeError(
      "The teacher reward needs TeacherStudentEnv. It records the PD action "
      "before the physics step, which is the state the student acted on."
    )
  return teacher


def match_gate_from_step(step: torch.Tensor) -> torch.Tensor:
  """Map a forward height change to the teacher-match gate.

  ``step`` is the largest absolute jump between consecutive occupied
  bins. The gate is 1 at or below 0.04 m and 0 at or above 0.08 m, linear
  in between. A hard slope near 0.045 m per sample stays near full match.
  A stair face or box at or above 0.08 m, up or down, turns the match off.
  """
  span = _MATCH_STEP_OFF_M - _MATCH_STEP_FULL_M
  return ((_MATCH_STEP_OFF_M - step) / span).clamp(0.0, 1.0)


def forward_scan_step(
  x: torch.Tensor,
  z: torch.Tensor,
  valid: torch.Tensor,
) -> torch.Tensor:
  """Largest absolute jump between consecutive occupied forward bins.

  ``x`` and ``z`` are torso-yaw-frame coordinates in meters, shape ``[B, N]``.
  ``valid`` marks ray hits. Rays at ``x <= 0`` are behind the torso and are
  ignored. Empty bins are skipped, so a miss is not a step. A drop counts
  the same as a rise.
  """
  usable = valid & torch.isfinite(x) & torch.isfinite(z)
  # meshgrid(indexing="xy") flattens as iy * 25 + ix. x = -0.8 + 0.1 * ix,
  # so the torso (x <= 0) is ix <= 8 and bin b = round(x / 0.1) - 1 is ix - 9.
  bin_id = torch.round(x / _SCAN_SPACING_M).long() - 1
  forward = usable & (bin_id >= 0) & (bin_id < _FORWARD_BIN_COUNT)
  return _max_occupied_jump(_forward_bin_medians(z, bin_id, forward))


def forward_height_step(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """Largest absolute height change, in meters, in front of the torso.

  Reads ``terrain_scan`` ray hits. The ``height_scan`` observation is scaled
  and, on the actor, noised, so it is not the gate input.
  """
  x, z, valid = _yaw_frame_scan(env, sensor_name)
  return forward_scan_step(x, z, valid)


def forward_match_step(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Largest forward height jump for teacher match and curriculum gates.

  The legacy teacher scene uses one torso grid named ``terrain_scan``. The
  split-scan stack reads the head-mounted ``terrain_scan_front`` grid instead.
  """
  if "terrain_scan" in env.scene.sensors:
    return forward_height_step(env)
  sensor = env.scene["terrain_scan_front"]
  pattern = sensor.cfg.pattern
  half = pattern.size[0] / 2.0
  center = pattern.offset[0]
  return column_height_step(
    env,
    sensor.cfg.name,
    pattern.resolution,
    center - half,
    center + half,
  )


def gated_track_teacher_action(
  env: ManagerBasedRlEnv,
  std: float,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """Match the teacher action at full strength.

  The scan decision is already in the stored target: a flat scan keeps the
  reactive PD, and a step that turns the match gate off replaces knees and
  wheels with the tumble write. Multiplying by that gate would hide the
  tumble again. ``sensor_name`` stays in the signature so the reward config
  still loads; the scan is read when the target is recorded.
  """
  del sensor_name
  return track_teacher_action(env, std)


def limb_torque_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean squared hip and knee torque, each divided by its rated torque.

  Wheel torque is omitted. The value is near 1 when every limb motor sits
  at its continuous rating.
  """
  asset: Entity = env.scene[asset_cfg.name]
  rated = motor_torques(env, asset_cfg)
  limb = _limb_actuator_mask(env, asset, rated)
  return torch.mean(torch.square(rated[:, limb]), dim=-1)


def _limb_actuator_mask(
  env: ManagerBasedRlEnv,
  asset: Entity,
  rated: torch.Tensor,
) -> torch.Tensor:
  """True for actuators whose names are not wheels."""
  mask = getattr(env, "_limb_torque_mask", None)
  if isinstance(mask, torch.Tensor) and mask.shape[0] == rated.shape[-1]:
    return mask
  mask = torch.tensor(
    ["wheel" not in name.lower() for name in asset.actuator_names],
    device=rated.device,
  )
  env._limb_torque_mask = mask
  return mask


def policy_action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared change in the policy action, not the inner PD command.

  Both columns are on a scale of about one: the tumble toggle, including
  the off-dwell sign, and the tank hip offset in units of 30 degrees.
  """
  action = getattr(env, "policy_action", None)
  previous = getattr(env, "prev_policy_action", None)
  if not isinstance(action, torch.Tensor) or not isinstance(previous, torch.Tensor):
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(torch.square(action - previous), dim=1)


def drive_state_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared change of the raw ``drive_state`` sample.

  The first step after reset costs nothing. A later all-zero action is a
  real sample, so the next change is still charged. The sample is the
  policy column, before the knee-speed clamp.
  """
  zeros = torch.zeros(env.num_envs, device=env.device)
  action, previous = _policy_action_pair(env)
  if action is None or previous is None:
    return zeros
  column, _, width = _drive_state_layout()
  if not _is_policy_width(action, width) or not _is_policy_width(previous, width):
    return zeros
  delta = torch.square(action[:, column] - previous[:, column])
  fresh = getattr(env, "_drive_rate_fresh", None)
  if not isinstance(fresh, torch.Tensor) or fresh.shape[0] != env.num_envs:
    return delta
  delta = torch.where(fresh, torch.zeros_like(delta), delta)
  fresh[:] = False
  return delta


def drive_state_outside_band(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared distance of the raw ``drive_state`` sample outside the drive band."""
  zeros = torch.zeros(env.num_envs, device=env.device)
  action = getattr(env, "policy_action", None)
  if not isinstance(action, torch.Tensor):
    return zeros
  column, threshold, width = _drive_state_layout()
  if not _is_policy_width(action, width):
    return zeros
  excess = torch.relu(action[:, column].abs() - threshold)
  return torch.square(excess)


def _policy_action_pair(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Current and previous policy actions, or None when either is missing."""
  action = getattr(env, "policy_action", None)
  previous = getattr(env, "prev_policy_action", None)
  if not isinstance(action, torch.Tensor) or not isinstance(previous, torch.Tensor):
    return None, None
  return action, previous


def _is_policy_width(action: torch.Tensor, width: int) -> bool:
  """True when ``action`` is a batch of policy vectors of ``width``."""
  return action.ndim == 2 and action.shape[-1] == width


def _drive_state_layout() -> tuple[int, float, int]:
  """Column, band edge, and policy width from the teacher env.

  Imported at call time. The teacher env imports this module while it is
  still loading, so a module-level import would cycle. The literals apply
  only when that import fails.
  """
  try:
    from walter_rl.rl.teacher_env import (
      DRIVE_STATE_COLUMN,
      DRIVE_STATE_THRESHOLD,
      POLICY_ACTION_DIM,
    )
  except ImportError:
    return 12, 0.5, 17
  return DRIVE_STATE_COLUMN, float(DRIVE_STATE_THRESHOLD), POLICY_ACTION_DIM


def predictive_knee_tumble(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "terrain_scan_front",
  spacing: float = 1.0 / 7.0,
  x_min: float = 0.25,
  x_max: float = 1.25,
  max_predictive_tumble_distance: float | None = None,
) -> torch.Tensor:
  """Reward forward knee rate when an upward step is still ahead.

  The command gate is 0 below the PD ``min_forward_cmd`` and otherwise the
  sign of the forward command. Knee rate is the mean of the positive rates
  divided by ``TumbleGains.rate``. The front grid is read in the head yaw
  frame, from ``x_min`` to the nearer of ``x_max`` and
  ``max_predictive_tumble_distance``. The scan gate opens only for a rise:
  shut at or below 0.04 m and fully open at or above 0.08 m. A drop,
  including downstairs, keeps it shut. A forward command and an open gate
  still pay when body-x speed along that command is at or below zero. A
  flat scan keeps the gate shut. The speed-lag integral is not used.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_x = command[:, 0]
  pref = torch.sign(cmd_x)
  has_command = cmd_x.abs() >= _PD_MIN_FORWARD_CMD
  pref = torch.where(has_command, pref, torch.zeros_like(pref))
  knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  aligned = torch.relu(knee_vel * pref.unsqueeze(-1)).mean(dim=-1)
  aligned = aligned / _PD_TUMBLE_RATE
  along = asset.data.root_link_lin_vel_b[:, 0]
  reach = _predictive_reach(x_max, max_predictive_tumble_distance)
  heights = _column_heights(env, sensor_name, spacing, x_min, reach)
  tumble_gate = 1.0 - match_gate_from_step(_max_occupied_rise(heights))
  progress = _stall_progress(along * pref, tumble_gate)
  return aligned * progress * tumble_gate


def _predictive_reach(
  x_max: float,
  max_predictive_tumble_distance: float | None,
) -> float:
  """Nearer of the scan end and the predictive-tumble cap."""
  if max_predictive_tumble_distance is None:
    return x_max
  return min(x_max, max_predictive_tumble_distance)


def column_height_step(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  spacing: float,
  x_min: float,
  x_max: float,
) -> torch.Tensor:
  """Largest height jump between consecutive occupied scan columns.

  Columns are ``round(x / spacing)`` in the sensor yaw frame, from
  ``x_min`` through ``x_max``. Empty columns are skipped, so a missed ray
  is not a step.
  """
  return _max_occupied_jump(_column_heights(env, sensor_name, spacing, x_min, x_max))


def downhill_tumble(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan_front",
  spacing: float = 1.0 / 7.0,
  x_min: float = 0.25,
  x_max: float = 1.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Downward pitch, in radians, while tumbling toward lower ground.

  Pitch is 0 upright and negative when the nose points down. The return is
  that downward angle while the tumble latch is on and the forward scan
  falls away. The scan gate is 0 at a 4 cm drop and 1 at 8 cm, the same
  band as a stair face. A rise, a level or nose-up torso, or an off latch
  returns 0, so a climb tumble is not charged. The reward weight supplies
  the sign.
  """
  asset: Entity = env.scene[asset_cfg.name]
  downward = torch.relu(-_torso_pitch(asset.data.projected_gravity_b))
  heights = _column_heights(env, sensor_name, spacing, x_min, x_max)
  return downward * _descent_gate(_max_occupied_drop(heights)) * _applied_tumble(env)


def _column_heights(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  spacing: float,
  x_min: float,
  x_max: float,
) -> torch.Tensor:
  """Median height of each forward column. Empty columns are NaN."""
  x, z, valid = _yaw_frame_scan(env, sensor_name)
  usable = valid & torch.isfinite(x) & torch.isfinite(z)
  first = int(round(x_min / spacing))
  count = int(round((x_max - x_min) / spacing)) + 1
  index = torch.round(x / spacing).long() - first
  occupied = usable & (index >= 0) & (index < count)
  columns = [
    _masked_median(z, occupied & (index == column)) for column in range(count)
  ]
  return torch.stack(columns, dim=-1)


def _torso_pitch(gravity_b: torch.Tensor) -> torch.Tensor:
  """Pitch from upright, in radians. Negative when the nose points down.

  ``projected_gravity_b.x`` is positive nose-down and ``.z`` is about -1
  when the torso is level. The angle is ``-atan2(gx, -gz)``.
  """
  return -torch.atan2(gravity_b[:, 0], -gravity_b[:, 2])


def _descent_gate(drop: torch.Tensor) -> torch.Tensor:
  """Open from 0 at a 4 cm step to 1 at an 8 cm step."""
  return _open_gate(drop)


def _open_gate(step: torch.Tensor) -> torch.Tensor:
  """Ramp from shut at 4 cm to fully open at 8 cm."""
  span = _MATCH_STEP_OFF_M - _MATCH_STEP_FULL_M
  return ((step - _MATCH_STEP_FULL_M) / span).clamp(0.0, 1.0)


# Nominal axle x in the back-torso frame, from the FS kinematic tree.
_REAR_WHEEL_X = -0.027 - 0.1137
_FRONT_WHEEL_X = 0.1754 + 0.338 + 0.1137
_FOOTPRINT_BEHIND_REAR_M = 0.25
_FOOTPRINT_PAST_FRONT_M = 0.75
_FOOTPRINT_AHEAD_M = 0.35
_FOOTPRINT_RESOLUTION_M = 0.05
_FOOTPRINT_WIDTH_M = 0.4
# Snapped so the 5 cm grid covers both margins with a sample on each end.
_FOOTPRINT_X_START = -0.40
_FOOTPRINT_X_END = 1.40
_CLIMB_SPEED_SCALE_MPS = 0.3
_PROGRESS_MIN_CMD_MPS = 0.1
_TORSO_TOUCH_N = 1.0


def footprint_scan_pattern() -> dict[str, tuple[float, float] | float]:
  """Grid arguments for the privileged footprint height scan.

  The field runs from 0.25 m behind the rear axles to 0.75 m past the front
  axles. The center offset is in the back-torso frame.
  """
  size_x = _FOOTPRINT_X_END - _FOOTPRINT_X_START
  offset_x = 0.5 * (_FOOTPRINT_X_START + _FOOTPRINT_X_END)
  return {
    "size": (size_x, _FOOTPRINT_WIDTH_M),
    "resolution": _FOOTPRINT_RESOLUTION_M,
    "offset": (offset_x, 0.0),
  }


def forward_progress(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Body-x speed as a fraction of the forward command, capped at that command.

  Commands below 0.1 m/s pay nothing, so a requested stop is not a target.
  Any forward speed beats a stall, unlike the exponential tracking kernel.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd = command[:, 0]
  speed = asset.data.root_link_lin_vel_b[:, 0]
  active = cmd >= _PROGRESS_MIN_CMD_MPS
  progress = speed.clamp(min=-cmd, max=cmd) / cmd.clamp(min=1e-6)
  return torch.where(active, progress, torch.zeros_like(progress))


def climb_tumble(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  sensor_name: str = "footprint_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Forward knee rate while a riser is ahead of or under the wheels.

  Knee rate is the command-aligned positive rate divided by the tumble
  rate. The scan gate stays open until the rear axles clear the last riser.
  Credit grows with forward speed and does not jump at a stall.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  aligned = _aligned_knee_rate(asset, command[:, 0], asset_cfg)
  gate = climb_gate(env, sensor_name)
  speed = asset.data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
  factor = 0.5 + 0.5 * (speed / _CLIMB_SPEED_SCALE_MPS).clamp(0.0, 1.0)
  moving = command[:, 0] >= _PROGRESS_MIN_CMD_MPS
  return torch.where(moving, aligned * gate * factor, torch.zeros_like(aligned))


def climb_gate(
  env: ManagerBasedRlEnv,
  sensor_name: str = "footprint_scan",
) -> torch.Tensor:
  """Open while a riser is within 0.35 m ahead or still under the robot.

  The gate is shut at a 4 cm rise and fully open at 8 cm. A 0.45 slope
  changes about 2.3 cm per 5 cm sample, so it stays shut.
  """
  under, ahead = _footprint_rises(env, sensor_name)
  return torch.maximum(_open_gate(under), _open_gate(ahead))


def unneeded_tumble(
  env: ManagerBasedRlEnv,
  sensor_name: str = "footprint_scan",
) -> torch.Tensor:
  """Applied tumble when no riser sits between the rear axles and the lookahead.

  The charge fades as the rise gate opens, so a real step is not penalized.
  Flats, slopes, waves, and declines stay at a full charge.
  """
  rise = _span_step(env, sensor_name, "excuse", rise=True)
  return _applied_tumble(env) * (1.0 - _open_gate(rise))


def footprint_downhill_tumble(
  env: ManagerBasedRlEnv,
  sensor_name: str = "footprint_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Nose-down pitch, in radians, while tumbling toward a drop under the body.

  The drop is read from the footprint scan, from the rear axles to 0.75 m
  past the front axles. A rise, a level torso, or an off latch returns 0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  downward = torch.relu(-_torso_pitch(asset.data.projected_gravity_b))
  drop = _span_step(env, sensor_name, "excuse", rise=False)
  return downward * _open_gate(drop) * _applied_tumble(env)


def torso_contact_flag(
  env: ManagerBasedRlEnv,
  sensor_name: str = "torso_ground_touch",
) -> torch.Tensor:
  """1 when the belly touched during this policy step, else 0."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    magnitude = torch.linalg.vector_norm(data.force_history, dim=-1)
    return (magnitude > _TORSO_TOUCH_N).any(dim=-1).any(dim=-1).float()
  assert data.found is not None
  return torch.any(data.found > 0, dim=-1).float()


def footprint_height_profile(
  env: ManagerBasedRlEnv,
  sensor_name: str = "footprint_scan",
) -> torch.Tensor:
  """Median ground height along the footprint, relative to the scan frame.

  Missed rays are 0. Shape is ``(num_envs, num_samples)``.
  """
  profile, _xs, frame_z = _footprint_profile(env, sensor_name)
  relative = profile - frame_z.unsqueeze(-1)
  return torch.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)


def _aligned_knee_rate(
  asset: Entity,
  cmd_x: torch.Tensor,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Positive knee rate along the command, divided by the tumble rate."""
  pref = torch.sign(cmd_x)
  has_command = cmd_x.abs() >= _PROGRESS_MIN_CMD_MPS
  pref = torch.where(has_command, pref, torch.zeros_like(pref))
  knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  aligned = torch.relu(knee_vel * pref.unsqueeze(-1)).mean(dim=-1)
  return aligned / _PD_TUMBLE_RATE


def _footprint_rises(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Largest rise under the wheelbase and within 0.35 m ahead of the front axles."""
  under = _span_step(env, sensor_name, "under", rise=True)
  ahead = _span_step(env, sensor_name, "ahead", rise=True)
  return under, ahead


def _span_step(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  span: str,
  *,
  rise: bool,
) -> torch.Tensor:
  """Largest consecutive rise or drop on one footprint span."""
  profile, _xs, _frame_z = _footprint_profile(env, sensor_name)
  start, end = _footprint_spans(env, sensor_name)[span]
  if end <= start:
    return torch.zeros(profile.shape[0], device=profile.device)
  window = profile[:, start : end + 1]
  delta = window[:, 1:] - window[:, :-1]
  signed = delta if rise else -delta
  finite = torch.isfinite(signed)
  usable = torch.where(finite, signed, torch.zeros_like(signed))
  return usable.clamp(min=0.0).amax(dim=-1)


def _footprint_profile(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Median hit height per longitudinal sample, the sample x, and frame z."""
  sensor = env.scene[sensor_name]
  xs, n_y = _footprint_axes(env, sensor_name)
  hits = sensor.data.hit_pos_w
  valid = sensor.data.distances >= 0
  n_x = xs.numel()
  z = hits[..., 2].view(-1, n_y, n_x)
  mask = valid.view(-1, n_y, n_x)
  frame_z = sensor.data.frame_pos_w[:, 0, 2]
  return _median_over_width(z, mask), xs, frame_z


def _footprint_axes(env: ManagerBasedRlEnv, sensor_name: str) -> tuple[torch.Tensor, int]:
  """Sample x coordinates and the lateral ray count for ``sensor_name``."""
  cached = getattr(env, "_footprint_axes", None)
  if isinstance(cached, tuple) and cached[0] == sensor_name:
    return cached[1], cached[2]
  pattern = env.scene[sensor_name].cfg.pattern
  xs = _grid_axis(pattern.size[0], pattern.resolution, pattern.offset[0], env.device)
  ys = _grid_axis(pattern.size[1], pattern.resolution, pattern.offset[1], env.device)
  env._footprint_axes = (sensor_name, xs, ys.numel())
  return xs, ys.numel()


def _grid_axis(size: float, resolution: float, offset: float, device: str) -> torch.Tensor:
  """Match ``GridPatternCfg.generate_rays`` along one axis."""
  axis = torch.arange(
    -size / 2,
    size / 2 + resolution * 0.5,
    resolution,
    device=device,
    dtype=torch.float32,
  )
  return axis + offset


def _median_over_width(z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
  """Median across the width axis. Empty samples are NaN."""
  filled = torch.where(valid, z, torch.full_like(z, float("inf")))
  ordered = filled.sort(dim=1).values
  count = valid.sum(dim=1)
  last = z.shape[1] - 1
  lower = ((count - 1) // 2).clamp(0, last)
  upper = (count // 2).clamp(0, last)
  batch = torch.arange(z.shape[0], device=z.device)[:, None]
  column = torch.arange(z.shape[2], device=z.device)[None, :]
  median = (ordered[batch, lower, column] + ordered[batch, upper, column]) * 0.5
  return torch.where(count > 0, median, torch.full_like(median, float("nan")))


def _footprint_spans(env: ManagerBasedRlEnv, sensor_name: str) -> dict[str, tuple[int, int]]:
  """Inclusive sample ranges measured from the nominal wheel axles."""
  cached = getattr(env, "_footprint_spans", None)
  if isinstance(cached, dict) and cached.get("sensor") == sensor_name:
    return cached
  xs, _n_y = _footprint_axes(env, sensor_name)
  spans = {
    "sensor": sensor_name,
    "under": _sample_span(xs, _REAR_WHEEL_X, _FRONT_WHEEL_X),
    "ahead": _sample_span(xs, _FRONT_WHEEL_X, _FRONT_WHEEL_X + _FOOTPRINT_AHEAD_M),
    "excuse": _sample_span(xs, _REAR_WHEEL_X, _FRONT_WHEEL_X + _FOOTPRINT_PAST_FRONT_M),
  }
  env._footprint_spans = spans
  return spans


def _sample_span(xs: torch.Tensor, start: float, end: float) -> tuple[int, int]:
  """First sample at or after ``start`` through the last at or before ``end``."""
  host = xs.detach().cpu()
  first = int(torch.searchsorted(host, torch.tensor(start)).item())
  last = int(torch.searchsorted(host, torch.tensor(end), right=True).item()) - 1
  last_index = xs.numel() - 1
  return max(0, min(first, last_index)), max(0, min(last, last_index))


def _applied_tumble(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 while the state-machine tumble latch is on, else 0."""
  applied = getattr(env, "_tumble_applied", None)
  if not isinstance(applied, torch.Tensor):
    return torch.zeros(env.num_envs, device=env.device)
  return applied.to(dtype=torch.float32).reshape(env.num_envs)


def _yaw_frame_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Hit x, z in the torso yaw frame, and the valid-hit mask."""
  sensor = env.scene[sensor_name]
  hit = sensor.data.hit_pos_w
  origin = sensor.data.frame_pos_w[:, 0]
  yaw = yaw_quat(sensor.data.frame_quat_w[:, 0])
  yaw = yaw[:, None, :].expand(hit.shape[0], hit.shape[1], 4)
  local = quat_apply_inverse(yaw, hit - origin[:, None, :])
  return local[..., 0], local[..., 2], sensor.data.distances >= 0


def _forward_bin_medians(
  z: torch.Tensor,
  bin_id: torch.Tensor,
  forward: torch.Tensor,
) -> torch.Tensor:
  """Median height of each forward bin. Empty bins are NaN."""
  columns = [
    _masked_median(z, forward & (bin_id == index))
    for index in range(_FORWARD_BIN_COUNT)
  ]
  return torch.stack(columns, dim=-1)


def _masked_median(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Per-row median of ``values`` on ``mask``. Empty rows are NaN."""
  count = mask.sum(dim=-1)
  filled = torch.where(mask, values, torch.full_like(values, float("inf")))
  ordered = filled.sort(dim=-1).values
  last = values.shape[-1] - 1
  lower = ((count - 1) // 2).clamp(0, last)
  upper = (count // 2).clamp(0, last)
  rows = torch.arange(values.shape[0], device=values.device)
  median = (ordered[rows, lower] + ordered[rows, upper]) * 0.5
  return torch.where(count > 0, median, torch.full_like(median, float("nan")))


def _stall_progress(along: torch.Tensor, tumble_gate: torch.Tensor) -> torch.Tensor:
  """Speed scale for ``predictive_knee_tumble``.

  Positive speed along the command is clamped at 1. Non-positive speed
  still counts as full progress when the scan gate is open, because a
  stall at a step used to zero the term and leave only the lag reward.
  A shut gate does not get that credit. The caller multiplies by the gate,
  so a flat scan still pays nothing.
  """
  progress = torch.relu(along).clamp(max=1.0)
  stalled_on_step = (along <= 0) & (tumble_gate > 0)
  return torch.where(stalled_on_step, torch.ones_like(progress), progress)


def _max_occupied_rise(heights: torch.Tensor) -> torch.Tensor:
  """Largest rise between consecutive finite columns, in meters.

  Columns run from near to far. A drop does not add. Empty columns are
  skipped, so a missed ray is not a rise.
  """
  rise = torch.zeros(heights.shape[0], device=heights.device, dtype=heights.dtype)
  previous = torch.zeros_like(rise)
  has_previous = torch.zeros(
    heights.shape[0], dtype=torch.bool, device=heights.device
  )
  for index in range(heights.shape[-1]):
    column = heights[:, index]
    is_occupied = torch.isfinite(column)
    filled = torch.where(is_occupied, column, previous)
    climb = filled - previous
    can_compare = has_previous & is_occupied
    rise = torch.where(can_compare, torch.maximum(rise, climb), rise)
    previous = filled
    has_previous = has_previous | is_occupied
  return rise


def _max_occupied_drop(heights: torch.Tensor) -> torch.Tensor:
  """Largest fall between consecutive finite columns, in meters.

  Columns run from near to far. A rise does not add. Empty columns are
  skipped, so a missed ray is not a drop.
  """
  drop = torch.zeros(heights.shape[0], device=heights.device, dtype=heights.dtype)
  previous = torch.zeros_like(drop)
  has_previous = torch.zeros(
    heights.shape[0], dtype=torch.bool, device=heights.device
  )
  for index in range(heights.shape[-1]):
    column = heights[:, index]
    is_occupied = torch.isfinite(column)
    filled = torch.where(is_occupied, column, previous)
    fall = previous - filled
    can_compare = has_previous & is_occupied
    drop = torch.where(can_compare, torch.maximum(drop, fall), drop)
    previous = filled
    has_previous = has_previous | is_occupied
  return drop


def _max_occupied_jump(heights: torch.Tensor) -> torch.Tensor:
  """Max absolute jump between consecutive finite columns."""
  step = torch.zeros(heights.shape[0], device=heights.device, dtype=heights.dtype)
  previous = torch.zeros_like(step)
  has_previous = torch.zeros(
    heights.shape[0], dtype=torch.bool, device=heights.device
  )
  for index in range(heights.shape[-1]):
    column = heights[:, index]
    is_occupied = torch.isfinite(column)
    filled = torch.where(is_occupied, column, previous)
    jump = (filled - previous).abs()
    can_compare = has_previous & is_occupied
    step = torch.where(can_compare, torch.maximum(step, jump), step)
    previous = filled
    has_previous = has_previous | is_occupied
  return step


def root_yaw(quat: torch.Tensor) -> torch.Tensor:
  """Yaw of a wxyz quaternion, in radians.

  This is the upright-body yaw. Once the torso pitches past vertical the
  same formula jumps by half a turn, so callers gate on uprightness.
  """
  qw, qx, qy, qz = quat.unbind(dim=-1)
  return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _latch_spawn_yaw(env: ManagerBasedRlEnv, asset: Entity) -> None:
  """Record spawn yaw once per episode. Later steps keep that value."""
  yaw = getattr(env, "spawn_yaw_w", None)
  valid = getattr(env, "_spawn_yaw_valid", None)
  if yaw is None or valid is None:
    yaw = torch.zeros(env.num_envs, device=env.device)
    valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.spawn_yaw_w = yaw
    env._spawn_yaw_valid = valid
  measured = root_yaw(asset.data.root_link_quat_w)
  env.spawn_yaw_w = torch.where(valid, yaw, measured)
  env._spawn_yaw_valid = torch.ones_like(valid)


def _invalidate_spawn_yaw(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
) -> None:
  """Forget the latch for environments whose episode just ended."""
  valid = getattr(env, "_spawn_yaw_valid", None)
  if valid is None:
    return
  if env_ids is None:
    valid[:] = False
    return
  valid[env_ids] = False


def _spawn_axes_speed(
  env: ManagerBasedRlEnv,
  asset: Entity,
) -> tuple[torch.Tensor, torch.Tensor]:
  """World horizontal speed along the spawn yaw, and across it."""
  _latch_spawn_yaw(env, asset)
  vel = asset.data.root_link_lin_vel_w[:, :2]
  yaw = env.spawn_yaw_w
  cos_yaw = torch.cos(yaw)
  sin_yaw = torch.sin(yaw)
  along = cos_yaw * vel[:, 0] + sin_yaw * vel[:, 1]
  cross = -sin_yaw * vel[:, 0] + cos_yaw * vel[:, 1]
  return along, cross


class TrackLatchedForward(ManagerTermBase):
  """Reward speed along the spawn yaw, not the body's current forward axis.

  Body-frame tracking still pays after the robot yaws along a stair face.
  The latch is the yaw at the first reward of the episode, so turning off
  the steps and driving away does not match the forward command.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._std = float(cfg.params["std"])
    self._command_name = str(cfg.params["command_name"])
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Drop the spawn-yaw latch after the curriculum has read it."""
    _invalidate_spawn_yaw(self._env, env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del std, command_name, asset_cfg
    asset: Entity = env.scene[self._asset_cfg.name]
    command = env.command_manager.get_command(self._command_name)
    assert command is not None, f"Command '{self._command_name}' not found."
    along, cross = _spawn_axes_speed(env, asset)
    error = (command[:, 0] - along).square() + (command[:, 1] - cross).square()
    return torch.exp(-error / self._std**2)


def hold_spawn_heading(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Keep the upright torso on the spawn yaw.

  Returns 1 while the torso is pitching through a tumble, because yaw is not
  meaningful there. An upright yaw off the spawn heading goes to 0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  _latch_spawn_yaw(env, asset)
  error = wrap_to_pi(root_yaw(asset.data.root_link_quat_w) - env.spawn_yaw_w)
  upright = -asset.data.projected_gravity_b[:, 2] >= _UPRIGHT_FOR_HEADING
  kernel = torch.exp(-error.square() / std**2)
  return torch.where(upright, kernel, torch.ones_like(kernel))


def spawn_heading_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute upright yaw error from the spawn heading, in radians."""
  asset: Entity = env.scene[asset_cfg.name]
  _latch_spawn_yaw(env, asset)
  error = wrap_to_pi(root_yaw(asset.data.root_link_quat_w) - env.spawn_yaw_w).abs()
  upright = -asset.data.projected_gravity_b[:, 2] >= _UPRIGHT_FOR_HEADING
  return torch.where(upright, error, torch.zeros_like(error))


def track_body_forward(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body-frame forward speed matching the commanded speed.

  Lateral and vertical velocity are left out. A climb or a turn moves the
  root off the forward axis without missing this term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])
  return torch.exp(-error / std**2)


def reverse_drive(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared backward body-x speed while the command points forward.

  A requested stop returns 0. Forward motion returns 0. The square keeps a
  small rock cheap and a fast reverse, the stair backup exploit, large.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  backward = torch.relu(-asset.data.root_link_lin_vel_b[:, 0])
  commanded_forward = command[:, 0] > 0.0
  return torch.where(
    commanded_forward, backward.square(), torch.zeros_like(backward)
  )


def stuck_x_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Narrow penalty around zero body-x speed while a forward command is on.

  The kernel is 1 at zero speed and falls off with width ``std``. A command
  at or below zero returns 0, so a requested stop is not punished.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  speed = asset.data.root_link_lin_vel_b[:, 0]
  kernel = torch.exp(-speed.square() / std**2)
  moving = command[:, 0] > 0.0
  return torch.where(moving, kernel, torch.zeros_like(kernel))


def track_yaw_rate(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body yaw rate matching the commanded yaw rate.

  Roll and pitch rate are left out, so a tumble does not miss this term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std**2)


def track_lin_vel_xy(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward XY velocity tracking without punishing vertical speed.

  Climbing stairs needs nonzero body ``vz``. The factory tracker squares that
  component into the error and fights the climb.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  return torch.exp(-xy_error / std**2)


def integrate_speed_lag(
  lag: torch.Tensor,
  cmd_x: torch.Tensor,
  speed_x: torch.Tensor,
  dt: float,
  gains: TumbleGains,
) -> torch.Tensor:
  """Missed forward distance, in meters.

  Same update as ``FullBodyPdController._integrate_speed_lag``. Add
  ``v_cmd_x - v_x`` only while the forward command exceeds
  ``min_forward_cmd`` and the shortfall exceeds ``min_speed_error``.
  Otherwise leak toward zero at ``leak``, and never exceed ``cap``.
  """
  error = cmd_x - speed_x
  counting = (cmd_x > gains.min_forward_cmd) & (error > gains.min_speed_error)
  leaked = lag * math.exp(-gains.leak * dt)
  return torch.where(counting, lag + error * dt, leaked).clamp(max=gains.cap)


class KneeStuckTumble(ManagerTermBase):
  """Reward forward knee speed in proportion to missed forward distance.

  The integral is the PD stall integral. Positive knee rate is a forward
  tumble: it drops the front wheel. The return multiplies that integral by
  each knee's forward rate and sums them, so knee spin pays only while stuck.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._speed_lag = torch.zeros(env.num_envs, device=env.device)
    self._gains = TumbleGains()
    self._command_name = str(cfg.params["command_name"])
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Drop missed distance on environments that just ended."""
    self._speed_lag[slice(None) if env_ids is None else env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del command_name, asset_cfg
    self._update_lag(env)
    knee_vel = _knee_velocity(env, self._asset_cfg)
    return self._speed_lag * torch.relu(knee_vel).sum(dim=-1)

  def _update_lag(self, env: ManagerBasedRlEnv) -> None:
    """Step the stall integral from the command active during this step."""
    twist = env.command_manager.get_term(self._command_name)
    asset: Entity = env.scene[self._asset_cfg.name]
    along, _cross = _spawn_axes_speed(env, asset)
    self._speed_lag = integrate_speed_lag(
      self._speed_lag,
      twist.vel_command_b[:, 0],
      along,
      env.step_dt,
      self._gains,
    )


def _knee_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Joint velocity of the configured knees."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def knee_tumble(
  env: ManagerBasedRlEnv,
  velocity_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = "terrain_scan",
  rough_eps: float = 0.04,
  min_forward_cmd: float = 0.2,
  min_terrain_level: int = 2,
  min_body_vx: float = 0.15,
) -> torch.Tensor:
  """Reward forward-oriented knee winding only when climbing is useful.

  Positive knee rotation (axis +Y) drops the front wheel and lifts the rear
  so the rear wheel can roll up and over a step. Forward climbing therefore
  uses positive ``qdot`` (and negative ``qdot`` when reversing).

  Gated by non-flat stair columns at ``min_terrain_level``+, a minimum
  |forward| command, and actual body-x speed so spin-in-place cannot farm
  the term.
  """
  del sensor_name, rough_eps, min_body_vx  # Kept for config compatibility; unused.
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_x = command[:, 0]
  pref = torch.sign(cmd_x)
  pref = torch.where(cmd_x.abs() < min_forward_cmd, torch.zeros_like(pref), pref)
  knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  aligned = torch.relu(knee_vel * pref.unsqueeze(-1)).mean(dim=-1) / velocity_scale
  # Scale by forward body speed so spin-in-place cannot farm the term.
  body_vx = asset.data.root_link_lin_vel_b[:, 0]
  progress = torch.relu(body_vx * pref).clamp(max=1.0)
  return aligned * progress * _is_hard_stairs(env, min_terrain_level).float()


def knee_waste(
  env: ManagerBasedRlEnv,
  velocity_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "terrain_scan",
  rough_eps: float = 0.04,
  min_forward_cmd: float = 0.2,
  min_terrain_level: int = 2,
  min_body_vx: float = 0.15,
) -> torch.Tensor:
  """Penalize knee spinning only on the flat terrain column.

  Deliberately narrow: easy stairs (level < ``min_terrain_level``) and idle
  commands on stair columns do **not** count as waste. Prefer missing a flat
  penalty over punishing a robot that is still climbing.
  """
  del sensor_name, rough_eps, command_name, min_forward_cmd, min_terrain_level
  del min_body_vx
  asset: Entity = env.scene[asset_cfg.name]
  spin = asset.data.joint_vel[:, asset_cfg.joint_ids].abs().mean(dim=-1) / velocity_scale
  return spin * _is_flat_column(env).float()


def knee_tumble_reverse(
  env: ManagerBasedRlEnv,
  velocity_scale: float,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = "terrain_scan",
  rough_eps: float = 0.04,
  min_forward_cmd: float = 0.2,
  min_terrain_level: int = 2,
  min_body_vx: float = 0.15,
) -> torch.Tensor:
  """Count wrong-way knee wind (front-lift) on hard stairs.

  Used as a penalty so legs cannot tumble with the front wheel up while
  climbing forward. Same body-vx gate as ``knee_tumble``.
  """
  del sensor_name, rough_eps, min_body_vx
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_x = command[:, 0]
  pref = torch.sign(cmd_x)
  pref = torch.where(cmd_x.abs() < min_forward_cmd, torch.zeros_like(pref), pref)
  knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  wrong = torch.relu(-knee_vel * pref.unsqueeze(-1)).mean(dim=-1) / velocity_scale
  body_vx = asset.data.root_link_lin_vel_b[:, 0]
  progress = torch.relu(body_vx * pref).clamp(max=1.0)
  return wrong * progress * _is_hard_stairs(env, min_terrain_level).float()


def _is_flat_column(env: ManagerBasedRlEnv) -> torch.Tensor:
  """True only for envs assigned to the ``flat`` sub-terrain column."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  if "flat" not in names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return terrain.terrain_types == names.index("flat")


def _is_hard_stairs(env: ManagerBasedRlEnv, min_terrain_level: int) -> torch.Tensor:
  """True on non-flat stair columns at or above ``min_terrain_level``.

  Terrain-scan height std overlaps flat and stairs (~0.06 m both), so the
  curriculum column index is the reliable gate.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  levels = terrain.terrain_levels
  return (~_is_flat_column(env)) & (levels >= min_terrain_level)


def drive_forward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward positive body-x speed whenever a forward command is active.

  Looser than exact velocity tracking so the policy keeps covering ground on
  inverted stairs instead of stalling while matching a hard setpoint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  forward = (command[:, 0] > 0.1).float()
  return torch.relu(asset.data.root_link_lin_vel_b[:, 0]) * forward


def climb_vz(
  env: ManagerBasedRlEnv,
  sensor_name: str = "terrain_scan",
  rise_eps: float = 0.04,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upward world velocity when the height scan sees uneven terrain.

  Encourages gaining elevation on stairs without rewarding hops on flat ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  up = torch.relu(asset.data.root_link_lin_vel_w[:, 2])
  rough = _terrain_height_std(env, sensor_name)
  return up * (rough > rise_eps).float()


def _terrain_height_std(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-env standard deviation of valid terrain-scan hit heights."""
  sensor = env.scene[sensor_name]
  hit_z = sensor.data.hit_pos_w[..., 2]
  valid = sensor.data.distances >= 0
  fill = torch.where(valid, hit_z, torch.zeros_like(hit_z))
  count = valid.sum(dim=-1).clamp(min=1).to(hit_z.dtype)
  mean = fill.sum(dim=-1) / count
  centered = torch.where(valid, hit_z - mean.unsqueeze(-1), torch.zeros_like(hit_z))
  var = (centered * centered).sum(dim=-1) / count
  return torch.sqrt(var.clamp(min=0.0))


def head_level(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared head-roll error from zero (use a negative reward weight).

  The head hinge is about the forward axis. Zero keeps the head level with
  the torso; any sideways flop increases this term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  roll = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(roll), dim=-1)


def wheel_wheel_collision(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Count wheel pairs that are touching.

  Each ``wheel_pair_*`` sensor watches one unordered pair. The count is the
  penalty signal; the reward weight supplies the sign.
  """
  total: torch.Tensor | None = None
  for name, sensor in env.scene.sensors.items():
    # MuJoCo also registers each contact channel as a builtin sensor whose
    # name starts with the same prefix. Only the contact sensors count pairs.
    if not name.startswith("wheel_pair_") or not isinstance(sensor, ContactSensor):
      continue
    found = sensor.data.found
    assert found is not None
    hits = (found > 0).any(dim=-1).float()
    total = hits if total is None else total + hits
  if total is None:
    return torch.zeros(env.num_envs, device=env.device)
  return total


def track_ride_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  height_range: tuple[float, float],
  sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward torso clearance against the commanded ride height.

  Command 0 is the lowest legal stance and command 1 is thighs vertical down.
  Rough terrain uses the mean downward scan under the torso. Flat ground uses
  the torso origin height, which sits on the ground plane.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  ride = command[:, 3].clamp(0.0, 1.0)
  low, high = height_range
  target = low + ride * (high - low)
  clearance = _torso_clearance(env, sensor_name, asset_cfg)
  return torch.exp(-torch.square(clearance - target) / std**2)


def _torso_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str | None,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return height of the torso origin above the ground under the body."""
  scanned = _scan_clearance(env, sensor_name)
  if scanned is not None:
    return scanned
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]


def _scan_clearance(
  env: ManagerBasedRlEnv, sensor_name: str | None
) -> torch.Tensor | None:
  """Mean hit clearance, ignoring rays that miss."""
  if sensor_name is None or sensor_name not in env.scene.sensors:
    return None
  sensor = env.scene[sensor_name]
  distances = sensor.data.distances
  frame_z = sensor.data.frame_pos_w[:, 0, 2]
  clearance = frame_z.unsqueeze(-1) - sensor.data.hit_pos_w[..., 2]
  valid = distances >= 0
  clearance = torch.where(valid, clearance, torch.zeros_like(clearance))
  count = valid.sum(dim=-1).clamp(min=1).to(clearance.dtype)
  mean = clearance.sum(dim=-1) / count
  return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
