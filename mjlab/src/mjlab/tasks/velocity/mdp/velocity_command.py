from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class _LockedJoystickAxis:
  """Command axis held at zero when its trained range is unused."""

  def __init__(self) -> None:
    self.value = 0.0


# I/K step forward speed through back, stop, and forward. J/L do the same
# for yaw. The outer positions are half of the slider maximum.
# Viewer starting limits. Training ranges can be tighter than this.
_DEFAULT_AXIS_MAX = {"lin_vel_x": 2.0, "ang_vel_z": 4.0}
_AXIS_COMMAND_FRACTION = 0.5
_AXIS_HOTKEYS = (
  (0, "I", 1.0, "Forward"),
  (0, "K", -1.0, "Back"),
  (2, "J", 1.0, "Yaw left"),
  (2, "L", -1.0, "Yaw right"),
)


def _bind_axis_hotkeys(server, enabled, sliders: list) -> None:
  """Bind I/K/J/L to step forward speed and yaw through -1, 0, and 1."""
  for index, key, sign, label in _AXIS_HOTKEYS:
    slider = sliders[index]
    if isinstance(slider, _LockedJoystickAxis):
      continue
    _register_axis_hotkey(server, enabled, slider, key, sign, label)


def _register_axis_hotkey(server, enabled, slider, key: str, sign: float, label: str) -> None:
  """Register one press that enables the joystick and sets this axis."""
  command = server.gui.add_command(
    label,
    description=f"{key} steps this axis through negative, zero, and positive.",
    hotkey=key,
  )

  @command.on_trigger
  def _(_event, _slider=slider, _sign=sign) -> None:
    # Enable is what copies the sliders into the command. Without it, the
    # heading loop still owns yaw.
    enabled.value = True
    _slider.value = _step_axis(_slider, _sign)


def _step_axis(slider, direction: float) -> float:
  """Move one notch among negative, zero, and positive half-max."""
  level = _axis_level(slider)
  if float(slider.value) > level * 0.5:
    state = 1
  elif float(slider.value) < -level * 0.5:
    state = -1
  else:
    state = 0
  stepped = max(-1, min(1, state + int(direction)))
  return stepped * level


# Walter hips stop at ±70 deg. The slider shows degrees; the command is radians.
# Positive splays the front hips forward and the rear hips backward.
_HIP_ANGLE_LIMIT_DEG = 70.0
_HIP_ANGLE_DEFAULT_DEG = 30.0
# Knee zero offset, same scale and front/rear signs as the hip slider.
_KNEE_ANGLE_LIMIT_DEG = 70.0
_KNEE_ANGLE_DEFAULT_DEG = 20.0


def _axis_level(slider) -> float:
  """Half the slider max, snapped to its step. This is the '1' position."""
  step = float(slider.step)
  return round((_AXIS_COMMAND_FRACTION * float(slider.max)) / step) * step


def _sample_range(
  rand: torch.Tensor,
  bounds: tuple[float, float],
  power: float,
) -> torch.Tensor:
  """Draw from ``bounds``. ``power`` below 1 bunches samples toward the top.

  A power of 1 is uniform. A power of 0.5 takes the square root of a unit
  uniform draw, so the density rises linearly and the top of the range is
  twice as likely as the midpoint.
  """
  low, high = bounds
  if high == low or power == 1.0:
    return rand.uniform_(low, high)
  unit = rand.uniform_(0.0, 1.0).pow_(power)
  return low + (high - low) * unit


def _zero_yaw(
  command: torch.Tensor,
  env_ids: torch.Tensor,
  rand: torch.Tensor,
  probability: float,
) -> None:
  """Set yaw to 0 on a random fraction of ``env_ids``."""
  if probability <= 0.0:
    return
  zero = rand.uniform_(0.0, 1.0) <= probability
  command[env_ids[zero], 2] = 0.0


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    # Radians. Positive points the front hips forward and the rear hips back.
    self.hip_angle = torch.zeros(self.num_envs, device=self.device)
    # Radians. Positive lowers the front inner wheels. Tank Mode flips the
    # rear zero, so the same command raises the rear inner wheels.
    self.knee_angle = torch.zeros(self.num_envs, device=self.device)
    # Latched with the pose sliders. The knee servo reads this each action.
    self.tank_mode = False
    # Positive hip_angle points the front hips forward. Flip matches the rear.
    self.flip_front_hip = False
    # Play switches. Tumbling and orientation start on. Torque and speed stay
    # on the peak envelope until the rated caps are checked.
    self.tumbling_enabled = True
    self.orientation_enabled = True
    self.cap_rated_torque = False
    self.cap_rated_speed = False

    ride_range = cfg.ranges.ride_height
    if ride_range is None:
      self.ride_height: torch.Tensor | None = None
      self._command_buf: torch.Tensor | None = None
    else:
      # Start at the top of the range. Most resamples stay there.
      self.ride_height = torch.full(
        (self.num_envs,), ride_range[1], device=self.device
      )
      self._command_buf = torch.zeros(self.num_envs, 4, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._tank_mode: viser.GuiCheckboxHandle | None = None
    self._flip_front_hip: viser.GuiCheckboxHandle | None = None
    self._tumbling_enabled: viser.GuiCheckboxHandle | None = None
    self._orientation_enabled: viser.GuiCheckboxHandle | None = None
    self._cap_rated_torque: viser.GuiCheckboxHandle | None = None
    self._cap_rated_speed: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._ride_height_slider: viser.GuiSliderHandle | None = None
    self._hip_angle_slider: viser.GuiSliderHandle | None = None
    self._knee_angle_slider: viser.GuiSliderHandle | None = None
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    if self.ride_height is None or self._command_buf is None:
      return self.vel_command_b
    self._command_buf[:, :3] = self.vel_command_b
    self._command_buf[:, 3] = self.ride_height
    return self._command_buf

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = _sample_range(
      r, self.cfg.ranges.lin_vel_x, self.cfg.lin_vel_x_power
    )
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    _zero_yaw(self.vel_command_b, env_ids, r, self.cfg.zero_ang_vel_prob)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    self._resample_ride_height(env_ids, r)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    if self.cfg.init_velocity_prob > 0.0:
      assert isinstance(env_ids, torch.Tensor)
      r = torch.empty(len(env_ids), device=self.device)
      init_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob]
      if len(init_ids) > 0:
        # Start these envs already moving at the commanded planar velocity.
        # Safe pre-forward: the body-frame write reads orientation from qpos.
        vel_b = torch.zeros(len(init_ids), 6, device=self.device)
        vel_b[:, :2] = self.vel_command_b[init_ids, :2]
        vel_b[:, 5] = self.vel_command_b[init_ids, 2]
        self.robot.write_root_link_velocity_b_to_sim(vel_b, env_ids=init_ids)
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Pure function of the current state; refreshing all envs is safe.
    del env_ids
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[heading_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[heading_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  def _resample_ride_height(self, env_ids: torch.Tensor, rand: torch.Tensor) -> None:
    """Sample torso ride height. A fixed fraction is forced to the maximum."""
    if self.ride_height is None:
      return
    ride_range = self.cfg.ranges.ride_height
    assert ride_range is not None
    low, high = ride_range
    if high <= low:
      self.ride_height[env_ids] = high
      return
    self.ride_height[env_ids] = rand.uniform_(low, high)
    at_max = rand.uniform_(0.0, 1.0) <= self.cfg.rel_max_ride_height_envs
    self.ride_height[env_ids[at_max]] = high

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      tank_mode = server.gui.add_checkbox("Tank Mode", initial_value=False)
      tumbling = server.gui.add_checkbox(
        "Tumbling",
        initial_value=True,
        hint="Spin the knees after a forward stall.",
      )
      orientation = server.gui.add_checkbox(
        "Orientation",
        initial_value=True,
        hint="Level pitch and roll. Unchecked, those errors are zero.",
      )
      rated_torque = server.gui.add_checkbox(
        "Rated torque cap",
        initial_value=False,
        hint="Hold each DC motor at its continuous torque instead of peak.",
      )
      rated_speed = server.gui.add_checkbox(
        "Rated speed cap",
        initial_value=False,
        hint="Drop each DC motor's driving torque to zero at rated speed.",
      )

      for label, max_val in axes:
        # A locked axis (for example lateral speed fixed at 0) cannot host a
        # slider: Viser requires the initial value to sit inside [min, max],
        # and the max-range slider starts at 0.1.
        if max_val < 0.1:
          sliders.append(_LockedJoystickAxis())
          continue
        limit = max(max_val, _DEFAULT_AXIS_MAX.get(label, max_val))
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=limit,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-limit,
          max=limit,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      flip_front_hip = server.gui.add_checkbox(
        "Flip front hip",
        initial_value=False,
        hint=(
          "Reverse the front hip so it swings with the rear. "
          "Unchecked, positive hip_angle points the front hips forward."
        ),
      )
      hip_slider = server.gui.add_slider(
        "hip_angle",
        min=-_HIP_ANGLE_LIMIT_DEG,
        max=_HIP_ANGLE_LIMIT_DEG,
        step=1.0,
        initial_value=_HIP_ANGLE_DEFAULT_DEG,
        marks=(
          (-_HIP_ANGLE_LIMIT_DEG, "-70°"),
          (0.0, "0°"),
          (_HIP_ANGLE_LIMIT_DEG, "70°"),
        ),
      )
      knee_slider = server.gui.add_slider(
        "knee_angle",
        min=-_KNEE_ANGLE_LIMIT_DEG,
        max=_KNEE_ANGLE_LIMIT_DEG,
        step=1.0,
        initial_value=_KNEE_ANGLE_DEFAULT_DEG,
        marks=(
          (-_KNEE_ANGLE_LIMIT_DEG, "-70°"),
          (0.0, "0°"),
          (_KNEE_ANGLE_LIMIT_DEG, "70°"),
        ),
      )

      ride_slider = None
      if ranges.ride_height is not None:
        ride_low, ride_high = ranges.ride_height
        ride_slider = server.gui.add_slider(
          "ride_height",
          min=ride_low,
          max=ride_high,
          step=0.05,
          initial_value=ride_high,
        )

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0
        # Zero clears velocity. Ride height 0 is a crouch, so return to full height.
        if ride_slider is not None and ranges.ride_height is not None:
          ride_slider.value = ranges.ride_height[1]
        hip_slider.value = 0.0
        knee_slider.value = 0.0

      server.gui.add_markdown(
        "Keys: **I**/**K** step forward speed and **J**/**L** step yaw "
        "through negative, stop, and positive (half of that axis max).\n\n"
        "**Tank Mode** applies **hip_angle** and **knee_angle**. Unchecked, "
        "both commands are zero and the sliders keep their values.\n\n"
        "**Tumbling** spins the knees after a forward stall. Unchecked, "
        "the knees stay on the roll pose.\n\n"
        "**Orientation** levels pitch and roll. Unchecked, those errors "
        "are zero and the hips and knees hold the pose commands.\n\n"
        "**Rated torque cap** holds every DC motor at its continuous "
        "torque. Unchecked, the peak envelope is used.\n\n"
        "**Rated speed cap** makes driving torque fall to zero at rated "
        "speed. Unchecked, that corner is the no-load speed.\n\n"
        "**hip_angle** is degrees. Positive points the front hips forward "
        "and the rear hips backward. **Flip front hip** reverses the front, "
        "so positive swings it with the rear. A tumble solves the front hip "
        "again for that sign, and the knee still holds the body level.\n\n"
        "**knee_angle** is degrees and shifts the knee zero. Positive lowers "
        "the front inner wheels. With **Tank Mode** on, the rear zero moves "
        "the other way, so positive raises the rear inner wheels."
      )

    _bind_axis_hotkeys(server, enabled, sliders)

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._tank_mode = tank_mode
    self.tank_mode = bool(tank_mode.value)
    self._flip_front_hip = flip_front_hip
    self.flip_front_hip = bool(flip_front_hip.value)
    self._tumbling_enabled = tumbling
    self.tumbling_enabled = bool(tumbling.value)
    self._orientation_enabled = orientation
    self.orientation_enabled = bool(orientation.value)
    self._cap_rated_torque = rated_torque
    self.cap_rated_torque = bool(rated_torque.value)
    self._cap_rated_speed = rated_speed
    self.cap_rated_speed = bool(rated_speed.value)
    self._joystick_sliders = sliders
    self._ride_height_slider = ride_slider
    self._hip_angle_slider = hip_slider
    self._knee_angle_slider = knee_slider
    self._joystick_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super().compute(dt, env_ids)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value
      if self._ride_height_slider is not None and self.ride_height is not None:
        self.ride_height[idx] = self._ride_height_slider.value
    self._apply_hip_slider()
    self._apply_knee_slider()
    if self._tank_mode is not None:
      self.tank_mode = bool(self._tank_mode.value)
    if self._flip_front_hip is not None:
      self.flip_front_hip = bool(self._flip_front_hip.value)
    if self._tumbling_enabled is not None:
      self.tumbling_enabled = bool(self._tumbling_enabled.value)
    if self._orientation_enabled is not None:
      self.orientation_enabled = bool(self._orientation_enabled.value)
    if self._cap_rated_torque is not None:
      self.cap_rated_torque = bool(self._cap_rated_torque.value)
    if self._cap_rated_speed is not None:
      self.cap_rated_speed = bool(self._cap_rated_speed.value)

  def _apply_hip_slider(self) -> None:
    """Copy the hip pose slider while Tank Mode is on."""
    self._apply_pose_slider(self._hip_angle_slider, self.hip_angle)

  def _apply_knee_slider(self) -> None:
    """Copy the knee zero slider while Tank Mode is on."""
    self._apply_pose_slider(self._knee_angle_slider, self.knee_angle)

  def _apply_pose_slider(self, slider, target: torch.Tensor) -> None:
    """Write one pose slider, or zero it when Tank Mode is off."""
    if slider is None or self._joystick_get_env_idx is None:
      return
    idx = self._joystick_get_env_idx()
    if self._tank_mode is not None and bool(self._tank_mode.value):
      target[idx] = math.radians(float(slider.value))
      return
    target[idx] = 0.0

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  rel_max_ride_height_envs: float = 0.0
  """Fraction of resamples forced to the top of ``ranges.ride_height``.
  Ignored when ride height is disabled."""
  init_velocity_prob: float = 0.0
  """Probability that an env starts its episode already moving at its sampled
  planar command velocity. Applied on reset only."""
  lin_vel_x_power: float = 1.0
  """Exponent on a unit uniform draw before it is scaled into ``lin_vel_x``.
  1 is uniform. Below 1 the density rises toward the top of the range."""
  zero_ang_vel_prob: float = 0.0
  """Chance a resampled yaw rate is set to 0. Independent of standing, which
  still zeros the whole command."""

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None
    ride_height: tuple[float, float] | None = None
    """Optional ``[min, max]`` torso ride-height command in ``[0, 1]``.
    When set, ``command`` is ``[vx, vy, yaw, ride_height]``."""

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
