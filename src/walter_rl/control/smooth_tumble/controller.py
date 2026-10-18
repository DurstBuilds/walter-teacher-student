"""PD loops that fill the Walter action vector, with level hips while tumbling.

Hips, knees, and wheels use the same action terms as the velocity task.
``head_joint`` is fixed at 0 degrees and is not an action.
While the knees tumble or settle, the hips hold a flat-ground contact
height instead of zeroing torso pitch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.utils.lab_api.math import quat_apply

from walter_rl.assets.walter_fs.walter_constants import (
  HIP_ACTION_SCALE,
  HIP_ANGLE_LIMIT,
  WHEEL_ACTION_SCALE,
  continuous_effort_limit,
  rated_speed_limit,
)
from walter_rl.control.smooth_tumble.gains import SmoothTumbleGains, default_gains

# Thigh drop and shin half-length from walter_fs.xml. The knee sits 0.18 m
# below a vertical hip, and each wheel center is 0.108 m along the shin.
_THIGH_DROP_M = 0.18
_SHIN_HALF_M = 0.108
_LEVEL_HIP_NEWTON_STEPS = 4
_HIP_HEIGHT_SAMPLES = 33
_SLOPE_EPSILON = 1e-4

_FRONT_HIPS = ("head_left_thigh_joint", "head_right_thigh_joint")
_REAR_HIPS = ("torso_left_thigh_joint", "torso_right_thigh_joint")
_LEFT_KNEES = ("torso_left_thigh_shin_joint", "head_left_thigh_shin_joint")
_RIGHT_KNEES = ("torso_right_thigh_shin_joint", "head_right_thigh_shin_joint")
_LEFT_THIGHS = ("torso_left_thigh", "head_left_thigh")
_RIGHT_THIGHS = ("torso_right_thigh", "head_right_thigh")
_WAIST = "head_joint"
# A half turn swaps which wheel is forward, so it is not the same zero.
_KNEE_PERIOD_RAD = 2.0 * math.pi
# Knee groups are (rear, front). Positive rear rotation and negative front
# rotation each drop the wheel toward the body center.


@dataclass
class _JointGroup:
  """Entity joint indices and the matching columns of the action vector."""

  ids: torch.Tensor
  columns: torch.Tensor


class SmoothTumbleController:
  """Map torso pitch, roll, forward speed, and yaw rate onto Walter's actions.

  A stall in forward speed overrides the roll knee pose and spins every knee
  about +Y. The knee position servo is still the inner loop. While tumbling,
  every wheel holds ``tumble.wheel_speed``, and the hips cancel the knee's
  ride-height change so the body stays level on flat ground.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    gains: SmoothTumbleGains | None = None,
  ) -> None:
    self._env = env
    self._gains = gains if gains is not None else default_gains()
    columns = _action_columns(env)
    robot = env.scene["robot"]
    device = env.device
    self._front_hips = _group(robot, columns, _FRONT_HIPS, device)
    self._rear_hips = _group(robot, columns, _REAR_HIPS, device)
    self._left_knees = _group(robot, columns, _LEFT_KNEES, device)
    self._right_knees = _group(robot, columns, _RIGHT_KNEES, device)
    self._left_thighs = _body_ids(robot, _LEFT_THIGHS, device)
    self._right_thighs = _body_ids(robot, _RIGHT_THIGHS, device)
    self._left_wheels = _wheel_columns(robot, columns, "left", device)
    self._right_wheels = _wheel_columns(robot, columns, "right", device)
    self._waist_column = columns.get(_WAIST)
    self._knee_scale = _knee_position_scale(env)
    self._knee_ids = torch.cat((self._left_knees.ids, self._right_knees.ids))
    self._knee_columns = torch.cat(
      (self._left_knees.columns, self._right_knees.columns)
    )
    self._action_dim = env.action_manager.total_action_dim
    self._speed_lag = torch.zeros(env.num_envs, device=device)
    self._tumbling = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    self._knee_ref = torch.zeros(
      env.num_envs, self._knee_ids.numel(), device=device
    )
    self._knee_pose = torch.zeros_like(self._knee_ref)
    self._settle_turns = torch.zeros_like(self._knee_ref)
    self._settling = torch.zeros_like(self._knee_ref, dtype=torch.bool)
    self._inner_down = torch.tensor([1.0, -1.0], device=device)
    # (rear, front). Tank Mode negates only the rear knee zero.
    self._tank_rear_flip = torch.tensor([-1.0, 1.0], device=device)
    # Flip front hip negates only the front knee zero.
    self._front_knee_flip = torch.tensor([1.0, -1.0], device=device)
    self._knee_ctrl_ids = _knee_ctrl_ids(robot)
    self._knee_servo_is_soft = False
    self._rated_caps = _rated_caps(robot)
    self._wheel_actuator = _wheel_actuator(robot)
    self._torque_is_rated = False
    self._speed_is_rated = False

  @property
  def gains(self) -> SmoothTumbleGains:
    """Gains used by the loops."""
    return self._gains

  def reset(self) -> None:
    """Clear the stall integral. Yaw has no accumulated heading."""
    self._clear_tumble(slice(None))

  def act(self) -> torch.Tensor:
    """Return one action row per environment."""
    self._sync_knee_servo()
    self._sync_torque_cap()
    self._sync_speed_cap()
    self._forget_finished_episodes()
    actions = torch.zeros(
      self._env.num_envs, self._action_dim, device=self._env.device
    )
    robot = self._env.scene["robot"]
    self._apply_roll(actions, robot)
    self._apply_tumble(actions, robot)
    self._apply_pitch(actions, robot)
    self._apply_wheels(actions, robot)
    if self._waist_column is not None:
      actions[:, self._waist_column] = 0.0
    return actions

  def _sync_knee_servo(self) -> None:
    """Use the compliant knee servo while Tank Mode is on."""
    command = self._env.command_manager.get_term("twist")
    soft = bool(getattr(command, "tank_mode", False))
    if soft == self._knee_servo_is_soft:
      return
    roll = self._gains.roll
    kp = roll.tank_actuator_kp if soft else roll.actuator_kp
    kd = roll.tank_actuator_kd if soft else roll.actuator_kd
    _write_knee_servo(self._env, self._knee_ctrl_ids, kp, kd)
    self._knee_servo_is_soft = soft

  def _sync_torque_cap(self) -> None:
    """Hold each DC motor at rated torque while that play switch is on."""
    rated = _twist_flag(self._env, "cap_rated_torque", default=False)
    if rated == self._torque_is_rated:
      return
    for actuator, torque, _speed in self._rated_caps:
      _write_effort_limit(actuator, torque if rated else None)
    self._torque_is_rated = rated

  def _sync_speed_cap(self) -> None:
    """Hold each DC motor at rated speed while that play switch is on."""
    rated = _twist_flag(self._env, "cap_rated_speed", default=False)
    if rated == self._speed_is_rated:
      return
    for actuator, _torque, speed in self._rated_caps:
      limit = speed if rated else float(actuator.cfg.velocity_limit)
      _write_velocity_limit(actuator, limit)
    self._speed_is_rated = rated

  def _forget_finished_episodes(self) -> None:
    """Drop tumble state on environments that just reset.

    The mask write stays on device. Reading ``any()`` back to the host would
    sync the GPU on every control step.
    """
    fresh = self._env.episode_length_buf == 0
    self._clear_tumble(fresh)

  def _clear_tumble(self, env_ids) -> None:
    """Zero the stall integral and the knee spin reference."""
    self._speed_lag[env_ids] = 0.0
    self._tumbling[env_ids] = False
    self._settling[env_ids] = False
    self._settle_turns[env_ids] = 0.0
    self._knee_ref[env_ids] = 0.0

  def _apply_tumble(self, actions: torch.Tensor, robot) -> None:
    """Spin every knee at ``tumble.rate`` once missed distance is large."""
    if not _twist_flag(self._env, "tumbling_enabled", default=True):
      self._clear_tumble(slice(None))
      return
    self._integrate_speed_lag(robot)
    was_tumbling = self._tumbling.clone()
    self._tumbling = _tumble_gate(was_tumbling, self._speed_lag, self._gains.tumble)
    self._advance_knee_reference(robot, was_tumbling)
    self._settle_knees(robot, was_tumbling)
    self._write_tumble(actions)
    self._settling = self._settling & ~self._settle_arrived

  def _integrate_speed_lag(self, robot) -> None:
    """Add forward shortfall up to ``tumble.cap`` once the error is large enough.

    Leak when the command is low or the speed error is within ``min_speed_error``.
    """
    command = self._env.command_manager.get_term("twist")
    dt = self._env.step_dt
    cmd_x = command.vel_command_b[:, 0]
    error = cmd_x - robot.data.root_link_lin_vel_b[:, 0]
    gains = self._gains.tumble
    counting = (cmd_x > gains.min_forward_cmd) & (error > gains.min_speed_error)
    leaked = self._speed_lag * math.exp(-gains.leak * dt)
    integrated = torch.where(counting, self._speed_lag + error * dt, leaked)
    self._speed_lag = integrated.clamp(max=gains.cap)

  def _advance_knee_reference(self, robot, was_tumbling: torch.Tensor) -> None:
    """Start the reference at the current angle, then integrate +Y rate."""
    just_on = self._tumbling & ~was_tumbling
    if bool(just_on.any()):
      self._knee_ref[just_on] = robot.data.joint_pos[just_on][:, self._knee_ids]
    self._knee_ref[self._tumbling] += self._gains.tumble.rate * self._env.step_dt

  def _settle_knees(self, robot, was_tumbling: torch.Tensor) -> None:
    """Carry each knee to the roll pose on the latched turn, then stop."""
    just_off = was_tumbling & ~self._tumbling
    self._settling = self._settling | just_off.unsqueeze(-1)
    self._settling = self._settling & ~self._tumbling.unsqueeze(-1)
    self._settle_arrived = torch.zeros_like(self._settling)
    if not bool(self._settling.any()):
      return
    self._latch_settle_turns(just_off)
    self._step_settle(robot)

  def _latch_settle_turns(self, just_off: torch.Tensor) -> None:
    """Lock the pose turn from the spin reference when tumbling ends.

    The turn is not taken from the measured angle. A shin that overshoots
    the pose would otherwise be given the next revolution, and each new
    overshoot would open another one.
    """
    if not bool(just_off.any()):
      return
    turns = _forward_periods(self._knee_ref, self._knee_pose)
    self._settle_turns = torch.where(
      just_off.unsqueeze(-1), turns, self._settle_turns
    )

  def _step_settle(self, robot) -> None:
    """Advance the reference to the latched pose. Crossing it ends the settle."""
    measured = robot.data.joint_pos[:, self._knee_ids]
    goal = self._knee_pose + self._settle_turns * _KNEE_PERIOD_RAD
    held, arrived = _advance_settle_reference(
      self._knee_ref,
      measured,
      goal,
      self._gains.tumble.rate,
      self._env.step_dt,
    )
    self._settle_arrived = self._settling & arrived
    self._knee_ref = torch.where(self._settling, held, self._knee_ref)

  def _write_tumble(self, actions: torch.Tensor) -> None:
    """Overwrite knee targets while tumbling or finishing the forward settle."""
    mask = self._tumbling.unsqueeze(-1) | self._settling
    if not bool(mask.any()):
      return
    rows, cols = mask.nonzero(as_tuple=True)
    actions[rows, self._knee_columns[cols]] = (
      self._knee_ref[rows, cols] / self._knee_scale
    )

  def _apply_pitch(self, actions: torch.Tensor, robot) -> None:
    """Level the body with the hips.

    Off tumble, drop the low end by swinging that end's hips outward from
    vertical. While tumbling or settling, skip that pitch-error loop and
    hold the straight-leg contact height for the knee reference.
    """
    hip = self._env.command_manager.get_term("twist").hip_angle
    front_sign = _front_hip_sign(self._env)
    rear, front = self._pitch_error_hips(robot, hip, front_sign)
    active = self._tumbling | self._settling.any(dim=-1)
    if bool(active.any()):
      # Re-solve for the flipped sign. Negating a hip that already matches
      # the knee does not keep the contact height, because the shin term
      # is |sin(θ + φ)|.
      level_rear, level_front = _level_hips(self._knee_ref, front_sign)
      rear = torch.where(
        active,
        (level_rear + hip).clamp(-HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT),
        rear,
      )
      front = torch.where(
        active,
        (level_front + front_sign * hip).clamp(-HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT),
        front,
      )
    _write_same(actions, self._rear_hips.columns, rear / HIP_ACTION_SCALE)
    _write_same(actions, self._front_hips.columns, front / HIP_ACTION_SCALE)

  def _pitch_error_hips(
    self,
    robot,
    hip: torch.Tensor,
    front_sign: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rear and front hip angles that zero torso pitch error."""
    gains = self._gains.pitch
    gravity_x = robot.data.projected_gravity_b[:, 0]
    pitch_rate = robot.data.root_link_ang_vel_b[:, 1]
    if not _twist_flag(self._env, "orientation_enabled", default=True):
      gravity_x = torch.zeros_like(gravity_x)
      pitch_rate = torch.zeros_like(pitch_rate)
    correction = gains.sign * (gains.kp * gravity_x + gains.kd * pitch_rate)
    # Zero correction is a vertical thigh. A positive angle is toward -X,
    # which is outward only for the rear hips. The twist hip slider is the
    # same sign: positive sends the front hips forward and the rear hips back.
    # Flip front hip reverses that front pose, the way Tank Mode reverses
    # the rear knee relative to the front. The downhill splay stays put.
    rear_out = correction.clamp(min=0.0, max=HIP_ANGLE_LIMIT)
    front_out = (-correction).clamp(min=0.0, max=HIP_ANGLE_LIMIT)
    rear = (rear_out + hip).clamp(-HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT)
    front = (-front_out + front_sign * hip).clamp(-HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT)
    return rear, front

  def _apply_roll(self, actions: torch.Tensor, robot) -> None:
    """Extend the low side by pushing its inner wheels down."""
    gains = self._gains.roll
    gravity_y = robot.data.projected_gravity_b[:, 1]
    roll_rate = robot.data.root_link_ang_vel_b[:, 0]
    if not _twist_flag(self._env, "orientation_enabled", default=True):
      gravity_y = torch.zeros_like(gravity_y)
      roll_rate = torch.zeros_like(roll_rate)
    extend_left = gains.sign * (gains.kp * gravity_y - gains.kd * roll_rate)
    left_mag = extend_left.clamp(min=0.0)
    right_mag = (-extend_left).clamp(min=0.0)
    self._command_knees(
      actions, robot, self._left_knees, self._left_thighs, left_mag, slice(0, 2)
    )
    self._command_knees(
      actions, robot, self._right_knees, self._right_thighs, right_mag, slice(2, 4)
    )

  def _command_knees(self, actions, robot, group, thighs, magnitude, pose_cols) -> None:
    """Hold each inner wheel down, stopping when it is straight down."""
    position = robot.data.joint_pos[:, group.ids]
    pitch = _thigh_pitch(robot, thighs)
    # Groups are (rear, front). Positive lowers each inner wheel. Tank Mode
    # flips the rear zero, so the same command raises the rear wheels.
    knee = self._env.command_manager.get_term("twist").knee_angle
    direction = self._inner_down
    pose_sign = direction
    if _twist_flag(self._env, "tank_mode", default=False):
      pose_sign = direction * self._tank_rear_flip
    if _twist_flag(self._env, "flip_front_hip", default=False):
      pose_sign = pose_sign * self._front_knee_flip
    level = -pitch + pose_sign * knee.unsqueeze(-1)
    room = _room_before_vertical(pitch, level, direction, self._gains.roll.q_max)
    fold = torch.minimum(magnitude.unsqueeze(-1), room)
    self._knee_pose[:, pose_cols] = level + direction * fold
    target = _nearest_inner_fold(position, fold, direction, level)
    actions[:, group.columns] = target / self._knee_scale

  def _apply_wheels(self, actions: torch.Tensor, robot) -> None:
    """Forward speed plus a skid-steer yaw rate from ``ang_vel_z``.

    Tumbling replaces both with ``tumble.wheel_speed`` on every wheel.
    """
    command = self._env.command_manager.get_term("twist")
    velocity = self._gains.velocity
    v_cmd = command.vel_command_b[:, 0]
    speed_error = v_cmd - robot.data.root_link_lin_vel_b[:, 0]
    omega_max = self._wheel_omega_max()
    omega = v_cmd * (omega_max / velocity.full_speed_command)
    omega = omega + velocity.kp * speed_error
    yaw = self._yaw_rate(robot, command)
    diff = yaw * self._gains.yaw.track_half_width / velocity.wheel_radius
    left = (omega - diff).clamp(-omega_max, omega_max) / WHEEL_ACTION_SCALE
    right = (omega + diff).clamp(-omega_max, omega_max) / WHEEL_ACTION_SCALE
    held = _clip_wheel_action(self._gains.tumble.wheel_speed, omega_max)
    left = torch.where(self._tumbling, held, left)
    right = torch.where(self._tumbling, held, right)
    _write_same(actions, self._left_wheels, left)
    _write_same(actions, self._right_wheels, right)

  def _wheel_omega_max(self) -> float:
    """Highest wheel rate the motor will still drive, in rad/s."""
    motor_limit = float(self._wheel_actuator.velocity_limit_motor[0, 0])
    extra = self._gains.velocity.wheel_speed_limit
    if extra > 0.0:
      return min(motor_limit, extra)
    return motor_limit

  def control_errors(self, env_idx: int) -> tuple[tuple[str, float, float], ...]:
    """Errors the loops are closing, for one environment.

    Each item is ``(name, error, full_scale)``. Pitch and roll are the
    gravity components driven toward zero. Linear velocity and yaw rate are
    command minus measured. The integral is missed distance, in meters.
    ``full_scale`` sizes the viewer bar and does not clip the loop.
    """
    robot = self._env.scene["robot"]
    command = self._env.command_manager.get_term("twist")
    gravity = robot.data.projected_gravity_b[env_idx]
    if not _twist_flag(self._env, "orientation_enabled", default=True):
      gravity = torch.zeros_like(gravity)
    lin_error = (
      command.vel_command_b[env_idx, 0] - robot.data.root_link_lin_vel_b[env_idx, 0]
    )
    yaw_error = (
      command.vel_command_b[env_idx, 2] - robot.data.root_link_ang_vel_b[env_idx, 2]
    )
    # One host copy. Separate float() calls each sync the GPU.
    pitch, roll, lin_vel, missed, yaw_rate = (
      torch.stack(
        (
          gravity[0],
          gravity[1],
          lin_error,
          self._speed_lag[env_idx],
          yaw_error,
        )
      )
      .detach()
      .cpu()
      .tolist()
    )
    gains = self._gains
    return (
      ("pitch", pitch, 1.0),
      ("roll", roll, 1.0),
      ("lin vel (m/s)", lin_vel, gains.velocity.full_speed_command),
      ("lin vel integral (m)", missed, gains.tumble.cap),
      ("yaw rate (rad/s)", yaw_rate, gains.yaw.yaw_rate_limit),
    )

  def _yaw_rate(self, robot, command) -> torch.Tensor:
    """Track the twist yaw rate. The Viser slider writes that same state."""
    yaw_cmd = command.vel_command_b[:, 2]
    measured = robot.data.root_link_ang_vel_b[:, 2]
    gains = self._gains.yaw
    rate = gains.sign * (yaw_cmd + gains.kp * (yaw_cmd - measured))
    return rate.clamp(-gains.yaw_rate_limit, gains.yaw_rate_limit)


def _twist_flag(env: ManagerBasedRlEnv, name: str, default: bool) -> bool:
  """Read a play-viewer switch from the twist command."""
  command = env.command_manager.get_term("twist")
  return bool(getattr(command, name, default))


def _front_hip_sign(env: ManagerBasedRlEnv) -> float:
  """Sign of the front hip pose. Flip matches the rear's positive sign."""
  if _twist_flag(env, "flip_front_hip", default=False):
    return 1.0
  return -1.0


def _rated_caps(robot) -> list[tuple[object, float, float]]:
  """DC motors that can be clamped to rated torque and rated speed."""
  caps: list[tuple[object, float, float]] = []
  for actuator in robot.actuators:
    if getattr(actuator, "velocity_limit_motor", None) is None:
      continue
    torque = continuous_effort_limit(actuator.target_names)
    speed = rated_speed_limit(actuator.target_names)
    if torque is None or speed is None:
      continue
    caps.append((actuator, torque, speed))
  return caps


def _write_effort_limit(actuator, rated_limit: float | None) -> None:
  """Set the continuous cap, or restore the peak limit stored at startup."""
  if rated_limit is None:
    actuator.set_effort_limit(slice(None), actuator.default_force_limit)
    return
  count = actuator.force_limit.shape[0]
  device = actuator.force_limit.device
  actuator.set_effort_limit(
    slice(None),
    torch.full((count,), rated_limit, device=device),
  )


def _write_velocity_limit(actuator, limit: float) -> None:
  """Write the joint speed where the torque-speed curve reaches zero torque."""
  actuator.velocity_limit_motor[:] = limit


def _tumble_gate(was_on: torch.Tensor, lag: torch.Tensor, gains) -> torch.Tensor:
  """Start tumbling above ``threshold`` and stay on until under ``release``."""
  stay = was_on & (lag >= gains.release)
  start = ~was_on & (lag > gains.threshold)
  return stay | start


def _level_hips(
  knee_ref: torch.Tensor,
  front_sign: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Rear and front hip angles that cancel the tumble knee's height change.

  ``knee_ref`` columns are left (rear, front) then right (rear, front).
  Each end uses the mean of its two knees. Rear hips swing positive.
  Front hips use ``front_sign``, negative unless Flip front hip is on.
  """
  rear_knee = 0.5 * (knee_ref[:, 0] + knee_ref[:, 2])
  front_knee = 0.5 * (knee_ref[:, 1] + knee_ref[:, 3])
  rear = _level_hip(rear_knee, outward_sign=1.0)
  front = _level_hip(front_knee, outward_sign=front_sign)
  return rear, front


def _level_hip(knee: torch.Tensor, outward_sign: float) -> torch.Tensor:
  """Signed hip angle that holds contact height at the straight-leg drop.

  The start is the hip-stop angle whose height is already nearest the
  thigh drop. An uncoupled ``acos`` seed overshoots once the hip swings
  the lower wheel, so Newton is only used to polish that start, and only
  when the residual gets smaller.
  """
  theta = outward_sign * _closest_outward_hip(knee, outward_sign)
  for _ in range(_LEVEL_HIP_NEWTON_STEPS):
    theta = _newton_level_hip(theta, knee, outward_sign)
  return theta


def _closest_outward_hip(knee: torch.Tensor, outward_sign: float) -> torch.Tensor:
  """Outward magnitude in the hip stop with contact height nearest the drop."""
  samples = torch.linspace(
    0.0, HIP_ANGLE_LIMIT, _HIP_HEIGHT_SAMPLES, device=knee.device, dtype=knee.dtype
  )
  theta = outward_sign * samples
  height = _contact_height(theta.unsqueeze(0), knee.unsqueeze(-1))
  error = (height - _THIGH_DROP_M).abs()
  return samples[error.argmin(dim=-1)]


def _newton_level_hip(
  theta: torch.Tensor,
  knee: torch.Tensor,
  outward_sign: float,
) -> torch.Tensor:
  """One damped Newton step toward the straight-leg contact height.

  A full step that increases the height error is halved until it improves
  or the step vanishes. The result stays on the outward side of vertical.
  """
  residual = _contact_height(theta, knee) - _THIGH_DROP_M
  slope = _contact_height_slope(theta, knee)
  safe = slope.abs() > _SLOPE_EPSILON
  full_step = torch.where(safe, residual / slope, torch.zeros_like(residual))
  best = theta
  best_error = residual.abs()
  scale = torch.ones_like(theta)
  for _ in range(_LEVEL_HIP_NEWTON_STEPS):
    candidate = _clamp_outward(theta - scale * full_step, outward_sign)
    error = (_contact_height(candidate, knee) - _THIGH_DROP_M).abs()
    improved = error < best_error
    best = torch.where(improved, candidate, best)
    best_error = torch.where(improved, error, best_error)
    scale = scale * 0.5
  return best


def _clamp_outward(theta: torch.Tensor, outward_sign: float) -> torch.Tensor:
  """Keep ``theta`` between vertical and the outward hip stop."""
  outward = (outward_sign * theta).clamp(0.0, HIP_ANGLE_LIMIT)
  return outward_sign * outward


def _contact_height(theta: torch.Tensor, knee: torch.Tensor) -> torch.Tensor:
  """Hip height above the lower wheel center with the body held level.

  ``θ = 0`` hangs the thigh straight down. Positive ``θ`` is about +Y.
  """
  return _THIGH_DROP_M * torch.cos(theta) + _SHIN_HALF_M * torch.abs(
    torch.sin(theta + knee)
  )


def _contact_height_slope(theta: torch.Tensor, knee: torch.Tensor) -> torch.Tensor:
  """Derivative of ``_contact_height`` with respect to the hip angle.

  The absolute value is flat where the two wheels are the same height, so
  the shin term drops out there instead of picking a one-sided slope.
  """
  shin = theta + knee
  sine = torch.sin(shin)
  sign = torch.where(sine > 0.0, 1.0, torch.where(sine < 0.0, -1.0, 0.0))
  thigh = -_THIGH_DROP_M * torch.sin(theta)
  return thigh + _SHIN_HALF_M * torch.cos(shin) * sign


def _knee_ctrl_ids(robot) -> torch.Tensor:
  """MuJoCo actuator indices for the knee position servos."""
  ids = [
    actuator.global_ctrl_ids
    for actuator in robot.actuators
    if any("thigh_shin_joint" in name for name in actuator.target_names)
  ]
  if not ids:
    raise KeyError("No knee position actuators found.")
  return torch.cat(ids)


def _write_knee_servo(env, ctrl_ids: torch.Tensor, kp: float, kd: float) -> None:
  """Write knee position-servo kp and kd.

  DC motor servos keep kp and kd on the Python actuator. Builtin position
  servos keep them in MuJoCo gainprm and biasprm.
  """
  robot = env.scene["robot"]
  for actuator in robot.actuators:
    if not any("thigh_shin_joint" in name for name in actuator.target_names):
      continue
    if getattr(actuator, "stiffness", None) is not None:
      _write_python_knee_servo(actuator, kp, kd)
      return
  model = env.sim.model
  model.actuator_gainprm[:, ctrl_ids, 0] = kp
  model.actuator_biasprm[:, ctrl_ids, 1] = -kp
  model.actuator_biasprm[:, ctrl_ids, 2] = -kd


def _write_python_knee_servo(actuator, kp: float, kd: float) -> None:
  """Set knee kp and kd on an explicit PD or DC motor actuator."""
  count = actuator.stiffness.shape[0]
  device = actuator.stiffness.device
  actuator.set_gains(
    slice(None),
    kp=torch.full((count,), kp, device=device),
    kd=torch.full((count,), kd, device=device),
  )


def _knee_position_scale(env: ManagerBasedRlEnv) -> float:
  """Radians of knee angle per unit of the knee action."""
  term = env.action_manager.get_term("knee_pos")
  if not isinstance(term, JointPositionAction):
    raise TypeError(
      "Roll control holds an absolute knee angle. The PD env must use "
      "JointPositionAction for knee_pos."
    )
  if not isinstance(term.scale, float):
    raise TypeError("Knee action scale must be one float for every knee.")
  return term.scale


def _thigh_pitch(robot, body_ids: torch.Tensor) -> torch.Tensor:
  """Angle of each thigh +X axis below the horizontal.

  The shin is fixed in this frame at knee joint zero, so the angle is the
  tilt a hip rotation (and the torso pitch it rides on) gives the leg assembly.
  """
  quat = robot.data.body_link_quat_w[:, body_ids]
  forward = quat.new_zeros(*quat.shape[:-1], 3)
  forward[..., 0] = 1.0
  axis = quat_apply(quat, forward)
  horizontal = torch.linalg.norm(axis[..., :2], dim=-1).clamp_min(1e-6)
  return torch.atan2(-axis[..., 2], horizontal)


def _room_before_vertical(
  pitch: torch.Tensor,
  level: torch.Tensor,
  direction: torch.Tensor,
  q_max: float,
) -> torch.Tensor:
  """Inner-down radians left before the wheel is straight down.

  ``level`` is the knee zero: it cancels thigh pitch and adds the knee
  slider. Both already tilt the shin, so they count against ``q_max``.
  """
  dropped = direction * (pitch + level)
  return (q_max - dropped).clamp(min=0.0)


def _forward_periods(position: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
  """Smallest turn count ``n`` such that ``base + n*2π`` is not behind ``position``.

  Tumbling only increases knee angle. The nearest turn can lie behind the
  shin, and the servo would reverse to reach it. The count is latched when
  tumbling ends so a later overshoot cannot select ``n + 1``.
  """
  return torch.ceil((position - base) / _KNEE_PERIOD_RAD - 1e-4)


def _advance_settle_reference(
  reference: torch.Tensor,
  measured: torch.Tensor,
  goal: torch.Tensor,
  rate: float,
  dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Step the settle reference toward ``goal``.

  Returns the held reference and whether this knee has arrived. The command
  stays ahead of a joint that is still short of ``goal``. A measurement past
  ``goal`` holds the goal and arrives on this step.
  """
  passed = measured >= goal
  ahead = torch.maximum(reference, torch.minimum(measured, goal))
  stepped = ahead + rate * dt
  arrived = passed | (stepped >= goal)
  held = torch.where(passed, goal, torch.minimum(stepped, goal))
  return held, arrived


def _nearest_inner_fold(
  position: torch.Tensor,
  magnitude: torch.Tensor,
  direction: torch.Tensor,
  level: torch.Tensor,
) -> torch.Tensor:
  """Nearest angle that lowers the inner wheel by ``magnitude``.

  ``level`` is the knee angle that holds the commanded zero. ``direction``
  is +1 on a rear knee and -1 on a front knee. ``magnitude`` is per knee
  and already stops at vertical. The measured angle does not choose the wheel.
  """
  target_fold = level + direction * magnitude
  turns = torch.round((position - target_fold) / _KNEE_PERIOD_RAD)
  return target_fold + turns * _KNEE_PERIOD_RAD


def _body_ids(robot, names: tuple[str, ...], device) -> torch.Tensor:
  """Body indices for ``names``, in that order."""
  ids, matched = robot.find_bodies(names, preserve_order=True)
  if tuple(matched) != names:
    raise KeyError(f"Expected bodies {names}, found {matched}")
  return torch.tensor(ids, dtype=torch.long, device=device)


def _action_columns(env: ManagerBasedRlEnv) -> dict[str, int]:
  """Map each actuated joint name to its column in the flat action vector."""
  columns: dict[str, int] = {}
  offset = 0
  for name in env.action_manager.active_terms:
    term = env.action_manager.get_term(name)
    for local, joint_name in enumerate(term.target_names):
      columns[joint_name] = offset + local
    offset += term.action_dim
  return columns


def _group(robot, columns: dict[str, int], names: tuple[str, ...], device) -> _JointGroup:
  """Resolve joint state indices and action columns in ``names`` order."""
  ids, matched = robot.find_joints(names, preserve_order=True)
  if tuple(matched) != names:
    raise KeyError(f"Expected joints {names}, found {matched}")
  missing = [name for name in names if name not in columns]
  if missing:
    raise KeyError(f"Joints missing from the action vector: {missing}")
  return _JointGroup(
    ids=torch.tensor(ids, dtype=torch.long, device=device),
    columns=torch.tensor(
      [columns[name] for name in names], dtype=torch.long, device=device
    ),
  )


def _wheel_actuator(robot):
  """The DC motor group that drives the wheel joints."""
  for actuator in robot.actuators:
    if any("wheel_joint" in name for name in actuator.target_names):
      return actuator
  raise KeyError("No wheel actuators found.")


def _clip_wheel_action(speed: float, omega_max: float) -> float:
  """Scale a wheel rate in rad/s into action units, inside the motor limit."""
  if speed > omega_max:
    speed = omega_max
  elif speed < -omega_max:
    speed = -omega_max
  return speed / WHEEL_ACTION_SCALE


def _wheel_columns(robot, columns: dict[str, int], side: str, device) -> torch.Tensor:
  """Action columns for the four wheels on ``side`` (``left`` or ``right``)."""
  _ids, names = robot.find_joints(".*wheel_joint")
  picked = [columns[name] for name in names if f"_{side}_" in name]
  if len(picked) != 4:
    raise KeyError(f"Expected 4 {side} wheels, found {picked}")
  return torch.tensor(picked, dtype=torch.long, device=device)


def _write_same(actions: torch.Tensor, columns: torch.Tensor, values: torch.Tensor) -> None:
  """Write one value per environment into every listed action column."""
  actions[:, columns] = values.unsqueeze(-1)
