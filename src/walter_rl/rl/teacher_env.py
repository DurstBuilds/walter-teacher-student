"""Environment that scores the student against the full-body PD teacher."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvStepReturn
from mjlab.utils.lab_api.math import wrap_to_pi

from walter_rl.control.full_body_pd.controller import FullBodyPdController
from walter_rl.control.full_body_pd.gains import TumbleGains
from walter_rl.tasks.velocity.mdp.rewards import forward_match_step, match_gate_from_step

# Policy vector: hips, wheels, drive_state, knee positions.
POLICY_ACTION_DIM = 17
DRIVE_STATE_THRESHOLD = 0.5
DRIVE_STATE_COLUMN = 12
MAX_KNEE_SPIN_RAD_S = 5.0
_HIP_POLICY = slice(0, 4)
_WHEEL_POLICY = slice(4, 12)
_KNEE_POLICY = slice(13, 17)
# Plant vector stays hip_pos, knee_pos, wheel_vel.
_HIP_PLANT = slice(0, 4)
_KNEE_PLANT = slice(4, 8)
_WHEEL_PLANT = slice(8, 16)
_PLANT_ACTION_DIM = 16
_ACTION_TERM_ORDER = ("hip_pos", "knee_pos", "wheel_vel")
_HIP_NAMES = ("br_hip", "bl_hip", "fr_hip", "fl_hip")
_KNEE_NAMES = ("br_knee", "bl_knee", "fr_knee", "fl_knee")
_WHEEL_NAMES = (
  "br_axle1",
  "br_axle2",
  "bl_axle1",
  "bl_axle2",
  "fr_axle1",
  "fr_axle2",
  "fl_axle1",
  "fl_axle2",
)
_TERM_TARGET_NAMES = {
  "hip_pos": _HIP_NAMES,
  "knee_pos": _KNEE_NAMES,
  "wheel_vel": _WHEEL_NAMES,
}


class TeacherStudentEnv(ManagerBasedRlEnv):
  """Velocity env that records the PD action before each physics step.

  Rewards run after the step, so calling the teacher there would score the
  next state. ``teacher_action`` is taken from the same state the student
  saw. A flat forward scan stores the reactive PD action. A step that turns
  the match gate off stores that action with tumble forced on.

  The student outputs 17 numbers: four hips, eight wheels, ``drive_state``,
  and four knee positions. ``step`` decodes that into the 16-D plant command.
  Inside ±0.5 the knee columns are the position actions. Outside that, every
  knee integrates one shared reference at ``drive_state * TumbleGains.rate``,
  clamped to ±``MAX_KNEE_SPIN_RAD_S`` rad/s.
  ``follow_teacher`` steps the stored 16-D PD target and skips the decode,
  so playback does not write a previous student action.
  """

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
  ) -> None:
    super().__init__(cfg, device=device, render_mode=render_mode)
    self.follow_teacher = False
    self._teacher = FullBodyPdController(self)
    self._assert_plant_layout()
    action_dim = self.action_manager.total_action_dim
    self.teacher_action = torch.zeros(self.num_envs, action_dim, device=self.device)
    self.policy_action = torch.zeros(
      self.num_envs, POLICY_ACTION_DIM, device=self.device
    )
    self.prev_policy_action = torch.zeros(
      self.num_envs, POLICY_ACTION_DIM, device=self.device
    )
    self._drive_rate_fresh = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._tumble_applied = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._spin_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._knee_ids = self._knee_joint_ids()
    self._knee_ref = torch.zeros(
      self.num_envs, self._knee_ids.numel(), device=self.device
    )
    self._knee_scale = _knee_action_scale(self)
    self._knee_rate = TumbleGains().rate
    self.teacher_knee_velocity_match = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.teacher_knee_rate = torch.zeros(self.num_envs, device=self.device)
    self.student_knee_rate = torch.zeros(self.num_envs, device=self.device)

  @property
  def teacher(self) -> FullBodyPdController:
    """Reactive PD controller snapshotted at the start of ``step``."""
    return self._teacher

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    """Snapshot the teacher target, then step the applied action.

    PD playback applies the reactive 16-D action and skips the decode.
    Training stores the reactive action on a closed scan and the forced
    tumble on an open scan, and steps the decoded 16-D plant command.
    """
    reactive = self._teacher.act()
    tumbling = self._teacher.tumbling.detach().clone()
    if self.follow_teacher:
      self.teacher_action = reactive.detach()
      self._tumble_applied = tumbling
      self._arm_knee_match(tumbling, student_is_teacher=True)
      return super().step(self.teacher_action)
    force = self._scan_forces_tumble()
    forced = self._teacher.overlay_forced_tumble(reactive, force)
    self.teacher_action = forced.detach()
    self._arm_knee_match(tumbling | force)
    return super().step(self._decode_policy_action(action))

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear spin state so the next tumble reseeds from the measured angle."""
    super()._reset_idx(env_ids)
    rows = slice(None) if env_ids is None else env_ids
    self._clear_drive_state(rows)

  def _decode_policy_action(self, action: torch.Tensor) -> torch.Tensor:
    """Copy hips and wheels, and switch knees on ``drive_state``."""
    if action.ndim != 2 or action.shape != (self.num_envs, POLICY_ACTION_DIM):
      raise ValueError(
        f"Teacher policy action must have shape ({self.num_envs}, "
        f"{POLICY_ACTION_DIM}), got {tuple(action.shape)}."
      )
    command = action.to(device=self.device)
    # Rewards run after the step and read both samples. Playback skips this.
    self.prev_policy_action = self.policy_action.detach().clone()
    self.policy_action = command.detach().clone()
    plant = torch.zeros(
      self.num_envs, _PLANT_ACTION_DIM, device=self.device
    )
    plant[:, _HIP_PLANT] = command[:, _HIP_POLICY]
    plant[:, _WHEEL_PLANT] = command[:, _WHEEL_POLICY]
    spinning = self._write_knees(plant, command)
    self._spin_mask = spinning
    self._tumble_applied = spinning
    self._record_student_knee_rate(command[:, DRIVE_STATE_COLUMN], spinning)
    return plant

  def _write_knees(
    self, plant: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    """Write knee positions, or the shared integrated reference."""
    drive = command[:, DRIVE_STATE_COLUMN]
    spinning = drive.abs() > DRIVE_STATE_THRESHOLD
    holding = ~spinning
    plant[holding, _KNEE_PLANT] = command[holding, _KNEE_POLICY]
    if bool(spinning.any()):
      self._integrate_knee_spin(spinning, drive)
      plant[spinning, _KNEE_PLANT] = (
        self._knee_ref[spinning] / self._knee_scale
      )
    return spinning

  def _integrate_knee_spin(
    self, spinning: torch.Tensor, drive: torch.Tensor
  ) -> None:
    """Seed each knee on entry, then add the same signed rate step."""
    just_on = spinning & ~self._spin_mask
    if bool(just_on.any()):
      measured = self.scene["robot"].data.joint_pos[just_on][:, self._knee_ids]
      self._knee_ref[just_on] = measured
    knee_speed = (drive * self._knee_rate).clamp(
      -MAX_KNEE_SPIN_RAD_S, MAX_KNEE_SPIN_RAD_S
    )
    increment = knee_speed.unsqueeze(-1) * self.step_dt
    self._knee_ref[spinning] = self._knee_ref[spinning] + increment[spinning]

  def _clear_drive_state(self, env_ids: torch.Tensor | slice) -> None:
    """Zero the spin latch, knee reference, and both stored policy actions."""
    self._spin_mask[env_ids] = False
    self._knee_ref[env_ids] = 0.0
    self.policy_action[env_ids] = 0.0
    self.prev_policy_action[env_ids] = 0.0
    self._tumble_applied[env_ids] = False
    if isinstance(getattr(self, "teacher_knee_velocity_match", None), torch.Tensor):
      self.teacher_knee_velocity_match[env_ids] = False
      self.teacher_knee_rate[env_ids] = 0.0
      self.student_knee_rate[env_ids] = 0.0
    fresh = getattr(self, "_drive_rate_fresh", None)
    if isinstance(fresh, torch.Tensor):
      fresh[env_ids] = True

  def _assert_plant_layout(self) -> None:
    """Require the action-manager joint order the policy vector is built on."""
    found_terms = tuple(self.action_manager.active_terms)
    if found_terms != _ACTION_TERM_ORDER:
      raise RuntimeError(
        f"Action terms {found_terms} do not match {_ACTION_TERM_ORDER}."
      )
    if self.action_manager.total_action_dim != _PLANT_ACTION_DIM:
      raise RuntimeError(
        "Plant action dim "
        f"{self.action_manager.total_action_dim} is not {_PLANT_ACTION_DIM}."
      )
    for term_name, wanted in _TERM_TARGET_NAMES.items():
      found = tuple(self.action_manager.get_term(term_name).target_names)
      if found != wanted:
        raise RuntimeError(
          f"{term_name} targets {found} do not match {wanted}."
        )

  def _knee_joint_ids(self) -> torch.Tensor:
    """Joint indices in the same order as the knee action columns."""
    ids, matched = self.scene["robot"].find_joints(
      _KNEE_NAMES, preserve_order=True
    )
    if tuple(matched) != _KNEE_NAMES:
      raise RuntimeError(f"Expected knee joints {_KNEE_NAMES}, found {matched}.")
    return torch.tensor(ids, dtype=torch.long, device=self.device)

  def teacher_match_delta(
    self, student: torch.Tensor, teacher: torch.Tensor
  ) -> torch.Tensor:
    """Per-column match error for the teacher reward.

    Knee columns are the shortest angle on the circle. While the stored
    teacher target is a tumble, those columns are knee rate in rad/s
    instead of angle.
    """
    delta = student - teacher
    knees = wrap_to_pi(delta[:, _KNEE_PLANT])
    rate_error = (self.student_knee_rate - self.teacher_knee_rate).unsqueeze(-1)
    use_rate = self.teacher_knee_velocity_match.unsqueeze(-1)
    delta = delta.clone()
    delta[:, _KNEE_PLANT] = torch.where(use_rate, rate_error, knees)
    return delta

  def _arm_knee_match(
    self, tumbling: torch.Tensor, *, student_is_teacher: bool = False
  ) -> None:
    """Record which rows compare knee rate, and the teacher's rate."""
    self.teacher_knee_velocity_match = tumbling.detach().clone()
    rate = torch.zeros(self.num_envs, device=self.device)
    rate[tumbling] = self._knee_rate
    self.teacher_knee_rate = rate
    if student_is_teacher:
      self.student_knee_rate = rate.clone()

  def _record_student_knee_rate(
    self, drive: torch.Tensor, spinning: torch.Tensor
  ) -> None:
    """Shared knee rate commanded this step. Position mode is 0 rad/s."""
    speed = torch.zeros(self.num_envs, device=self.device)
    if bool(spinning.any()):
      knee_speed = (drive * self._knee_rate).clamp(
        -MAX_KNEE_SPIN_RAD_S, MAX_KNEE_SPIN_RAD_S
      )
      speed[spinning] = knee_speed[spinning]
    self.student_knee_rate = speed

  def _scan_forces_tumble(self) -> torch.Tensor:
    """True where the forward step has turned the teacher-match gate off.

    The gate is 0 at a step of 0.08 m or more. Flatter scans, including a
    hard slope still inside the match ramp, do not force a tumble.
    """
    match = match_gate_from_step(forward_match_step(self))
    return match <= 0.0


def _knee_action_scale(env: ManagerBasedRlEnv) -> float:
  """Radians of knee angle per unit of the knee action column."""
  scale = env.action_manager.get_term("knee_pos").scale
  if not isinstance(scale, float):
    raise TypeError("Knee action scale must be one float for every knee.")
  return scale
