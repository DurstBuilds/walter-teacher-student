"""Environment that scores the student against the full-body PD teacher."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvStepReturn

from walter_rl.control.full_body_pd.controller import FullBodyPdController
from walter_rl.tasks.velocity.mdp.rewards import (
  forward_height_step,
  match_gate_from_step,
)


class TeacherStudentEnv(ManagerBasedRlEnv):
  """Velocity env that records the PD action before each physics step.

  Rewards run after the step, so calling the teacher there would score the
  next state. ``teacher_action`` is taken from the same state the student
  saw. A flat forward scan stores the reactive PD action. A step that turns
  the match gate off stores that action with tumble forced on.
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
    action_dim = self.action_manager.total_action_dim
    self.teacher_action = torch.zeros(self.num_envs, action_dim, device=self.device)

  @property
  def teacher(self) -> FullBodyPdController:
    """Reactive PD controller snapshotted at the start of ``step``."""
    return self._teacher

  def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    """Snapshot the teacher target, then step the applied action.

    PD playback applies the reactive action. Training stores the reactive
    action on a closed scan and the forced tumble on an open scan, and
    steps the student action either way.
    """
    reactive = self._teacher.act()
    if self.follow_teacher:
      self.teacher_action = reactive.detach()
    else:
      forced = self._teacher.overlay_forced_tumble(
        reactive, self._scan_forces_tumble()
      )
      self.teacher_action = forced.detach()
    applied = self.teacher_action if self.follow_teacher else action
    return super().step(applied)

  def _scan_forces_tumble(self) -> torch.Tensor:
    """True where the forward step has turned the teacher-match gate off.

    The gate is 0 at a step of 0.08 m or more. Flatter scans, including a
    hard slope still inside the match ramp, do not force a tumble.
    """
    match = match_gate_from_step(forward_height_step(self))
    return match <= 0.0
