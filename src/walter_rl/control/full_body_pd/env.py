"""PD and RL Walter environments share the locked head joint.

``head_joint`` is fixed at 0 degrees. It is not an action.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import mujoco
import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from walter_rl.assets.payloads import attach_payload
from walter_rl.assets.walter_fs.walter_constants import get_spec, get_walter_fs_robot_cfg
from walter_rl.control.full_body_pd.gains import FullBodyPdGains, default_gains
from walter_rl.tasks.velocity.config.walter_fs.env_cfgs import (
  walter_fs_flat_env_cfg,
  walter_fs_rough_env_cfg,
  walter_fs_rough_test_env_cfg,
  walter_fs_stairs_env_cfg,
)

TaskName = Literal["flat", "rough", "rough-test", "stairs"]

_EASY_NOISE_M = (0.02, 0.04)
_CFG_FOR_TASK = {
  "flat": walter_fs_flat_env_cfg,
  "rough": walter_fs_rough_env_cfg,
  "rough-test": walter_fs_rough_test_env_cfg,
  "stairs": walter_fs_stairs_env_cfg,
}


def make_pd_env(
  task: TaskName = "flat",
  *,
  play: bool = False,
  num_envs: int = 1,
  device: str | None = None,
  gains: FullBodyPdGains | None = None,
  easy_rough: bool = False,
  lock_speed: float | None = None,
  lock_heading: float = 0.0,
  episode_length_s: float | None = None,
  payload: str | None = None,
) -> ManagerBasedRlEnv:
  """Build a Walter env with ``head_joint`` fixed at 0 degrees."""
  chosen = "rough" if easy_rough else task
  cfg = _CFG_FOR_TASK[chosen](play=play)
  _add_waist_lock(cfg, gains if gains is not None else default_gains())
  if payload is not None:
    _weld_payload(cfg, payload)
  cfg.scene.num_envs = num_envs
  if episode_length_s is not None:
    cfg.episode_length_s = episode_length_s
  if easy_rough:
    _apply_easy_rough(cfg)
  if not play:
    _quiet(cfg)
  if lock_speed is not None:
    _lock_command(cfg, lock_speed, lock_heading)
  _drop_rewards_without_sensors(cfg)
  if play:
    _drop_play_overhead(cfg)
  return ManagerBasedRlEnv(cfg, device=device or _default_device())


def _weld_payload(cfg: ManagerBasedRlEnvCfg, name: str) -> None:
  """Weld a registered payload onto the robot after the waist spec is built."""
  robot = cfg.scene.entities["robot"]
  base_spec = robot.spec_fn

  def spec_with_payload() -> mujoco.MjSpec:
    spec = base_spec()
    attach_payload(spec, name)
    return spec

  robot.spec_fn = spec_with_payload


def _default_device() -> str:
  """Use the GPU when it is available."""
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _add_waist_lock(cfg: ManagerBasedRlEnvCfg, gains: FullBodyPdGains) -> None:
  """Fix ``head_joint`` at 0 degrees without changing the shared robot config."""
  robot = get_walter_fs_robot_cfg()
  assert robot.articulation is not None
  # The articulation object is shared. Copy it before editing gains or actuators.
  robot.articulation = replace(robot.articulation)
  robot.spec_fn = _spec_with_fixed_head
  _stiffen_hips(robot, gains)
  _stiffen_knees(robot, gains)
  cfg.scene.entities["robot"] = robot
  _use_absolute_knee_action(cfg)
  cfg.actions.pop("waist", None)


def _stiffen_hips(robot, gains: FullBodyPdGains) -> None:
  """Hold the commanded hip angle against wheel torque.

  The training hip servo is too soft, so a zero target still folds to the
  stop. One diagonal then points inward.
  """
  assert robot.articulation is not None
  robot.articulation.actuators = tuple(
    _stiff_hip(cfg, gains) for cfg in robot.articulation.actuators
  )


def _stiff_hip(cfg, gains: FullBodyPdGains):
  """Raise the hip position servo. Knee and wheel actuators stay as they are."""
  if not isinstance(cfg, (BuiltinPositionActuatorCfg, DcMotorActuatorCfg)):
    return cfg
  if not any(_is_hip_pattern(name) for name in cfg.target_names_expr):
    return cfg
  return replace(
    cfg,
    stiffness=gains.pitch.actuator_kp,
    damping=gains.pitch.actuator_kd,
  )


def _is_hip_pattern(name: str) -> bool:
  """True for the thigh-hip pattern, not the knee."""
  return "thigh_joint" in name and "shin" not in name


def _stiffen_knees(robot, gains: FullBodyPdGains) -> None:
  """Give the knee servo enough gain to hold a folded leg assembly.

  The training stiffness is 20 N·m/rad. The PD knee command used to be a
  0.03 rad increment, which produced about 0.6 N·m and never lifted a wheel.
  """
  assert robot.articulation is not None
  robot.articulation.actuators = tuple(
    _stiff_knee(cfg, gains) for cfg in robot.articulation.actuators
  )


def _stiff_knee(cfg, gains: FullBodyPdGains):
  """Raise the knee position servo. Hips and wheels are left alone."""
  if not isinstance(cfg, (BuiltinPositionActuatorCfg, DcMotorActuatorCfg)):
    return cfg
  if not any("thigh_shin_joint" in name for name in cfg.target_names_expr):
    return cfg
  return replace(
    cfg,
    stiffness=gains.roll.actuator_kp,
    damping=gains.roll.actuator_kd,
  )


def _use_absolute_knee_action(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the training's relative knee step with an angle target.

  A relative step cannot hold a load: zero rate means zero torque. The action
  value is the knee angle in radians.
  """
  cfg.actions["knee_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(r".*_thigh_shin_joint",),
    scale=1.0,
    use_default_offset=True,
  )


# Equality time constant. Short enough that a large moment leaves the head
# within a hundredth of a degree of zero.
_HEAD_LOCK_SOLREF = (0.002, 1.0)


def _spec_with_fixed_head() -> mujoco.MjSpec:
  """Drop the waist spring and constrain ``head_joint`` to 0."""
  spec = get_spec()
  _lock_head_joint(spec)
  return spec


def _lock_head_joint(spec: mujoco.MjSpec) -> None:
  """Replace the ±25 deg waist hinge with a joint equality at q = 0."""
  for joint in spec.joints:
    if joint.name != "head_joint":
      continue
    joint.stiffness[:] = 0.0
    joint.damping[:] = 0.0
  equality = spec.add_equality(
    name="head_fixed",
    type=mujoco.mjtEq.mjEQ_JOINT,
    name1="head_joint",
  )
  # a0 + a1 q = 0, with a1 = 1, so the joint angle is 0.
  equality.data[:5] = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
  equality.solref[:] = np.array(_HEAD_LOCK_SOLREF)


def _apply_easy_rough(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the training mix with one 2–4 cm random-rough tile."""
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    sub_terrains={
      "random_rough": random_rough(
        proportion=1.0,
        noise_range=_EASY_NOISE_M,
        noise_step=0.02,
      ),
    },
    add_lights=True,
  )
  cfg.scene.terrain.max_init_terrain_level = 0


def _quiet(cfg: ManagerBasedRlEnvCfg) -> None:
  """Turn off pushes, observation noise, and the terrain curriculum."""
  for group in cfg.observations.values():
    group.enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}


def _drop_play_overhead(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop training signals the PD controller never reads.

  A play step was spending most of its time on rewards and the torso height
  scan. Neither changes the physics or the action, and that cost is what held
  one environment under realtime.
  """
  cfg.rewards = {}
  cfg.observations.pop("critic", None)
  for group in cfg.observations.values():
    group.terms.pop("height_scan", None)
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if not _is_play_only_sensor(sensor.name)
  )


def _is_play_only_sensor(name: str) -> bool:
  """Height scans and wheel-pair contacts exist for rewards and the critic."""
  return name in ("terrain_scan", "body_height_scan", "foot_height_scan") or (
    name.startswith("wheel_pair_")
  )


def _drop_rewards_without_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  """Skip rewards whose sensors the flat scene removes."""
  names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  if "terrain_scan" not in names:
    cfg.rewards.pop("climb_vz", None)


def _lock_command(cfg: ManagerBasedRlEnvCfg, speed: float, heading: float) -> None:
  """Hold forward speed and heading fixed for a rollout."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = True
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 1.0
  twist.rel_forward_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.resampling_time_range = (1.0e6, 1.0e6)
  twist.ranges.lin_vel_x = (speed, speed)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.ranges.heading = (heading, heading)
