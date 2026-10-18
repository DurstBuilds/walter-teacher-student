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
  PD. On a step it is the same PD action with tumble forced on.
  """
  student = env.action_manager.action
  error = torch.mean(torch.square(student - _teacher_action(env)), dim=-1)
  return torch.exp(-error / std**2)


def teacher_action_mse(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean squared gap between the student action and the PD teacher."""
  student = env.action_manager.action
  return torch.mean(torch.square(student - _teacher_action(env)), dim=-1)


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


def predictive_knee_tumble(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "terrain_scan",
) -> torch.Tensor:
  """Reward forward knee rate when a step is still ahead in the scan.

  The command gate is 0 below the PD ``min_forward_cmd`` and otherwise the
  sign of the forward command. Knee rate is the mean of the positive rates
  divided by ``TumbleGains.rate``. The scan gate is one minus the
  teacher-match gate. A forward command and an open gate still pay when
  measured speed along that command is at or below zero. A flat scan keeps
  the gate shut. The speed-lag integral is not used.
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
  along, _cross = _spawn_axes_speed(env, asset)
  tumble_gate = 1.0 - match_gate_from_step(forward_height_step(env, sensor_name))
  progress = _stall_progress(along * pref, tumble_gate)
  return aligned * progress * tumble_gate


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
