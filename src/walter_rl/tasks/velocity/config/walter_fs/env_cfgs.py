"""Full-scale WaLTER velocity environment configurations."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointVelocityActionCfg,
  RelativeJointPositionActionCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import box_random_grid, flat, pyramid_stairs, pyramid_stairs_inv
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from walter_rl.assets.walter_fs.walter_constants import (
  HIP_ACTION_SCALE,
  HIP_ANGLE_LIMIT,
  KNEE_ACTION_SCALE,
  RIDE_HEIGHT_MAX,
  RIDE_HEIGHT_MIN,
  WHEEL_ACTION_SCALE,
  get_walter_fs_robot_cfg,
)
from walter_rl.tasks.velocity.mdp import (
  climb_vz,
  drive_forward,
  drive_state_outside_band,
  drive_state_rate_l2,
  finite_builtin_sensor,
  finite_height_scan,
  finite_joint_pos_rel,
  finite_joint_vel_rel,
  finite_projected_gravity,
  KneeStuckTumble,
  track_body_forward,
  reverse_drive,
  stuck_x_velocity,
  track_yaw_rate,
  gated_track_teacher_action,
  head_level,
  hold_spawn_heading,
  joint_pos_sincos,
  last_policy_action,
  teacher_policy_action,
  terrain_level,
  tumble_latch_timers,
  knee_tumble,
  knee_tumble_reverse,
  knee_waste,
  joint_torques,
  wheel_torques,
  non_finite_root,
  tumble_applied,
  policy_action_rate_l2,
  climb_tumble,
  footprint_downhill_tumble,
  footprint_height_profile,
  footprint_scan_pattern,
  forward_progress,
  predictive_knee_tumble,
  torso_contact_flag,
  unneeded_tumble,
  spawn_heading_error,
  teacher_action_mse,
  SkillTerrainCurriculum,
  accumulate_skill_progress,
  spawn_skill_pose,
  TraversalTerrainCurriculum,
  accumulate_traversal_progress,
  spawn_traversal_pose,
  terrain_levels_along_spawn,
  terrain_levels_hold_max,
  track_lin_vel_xy,
  wrap_terrain_levels,
  track_ride_height,
  limb_torque_l2,
  wheel_wheel_collision,
)

# Physics is 200 Hz (5 ms). 20 substeps is a 10 Hz policy step.
_POLICY_DECIMATION = 20
# Rough training has eight sub-terrain columns. The critic one-hot is padded to
# this width so checkpoints load on rough-test (seven columns) or flat play.
_STATE_MACHINE_TERRAIN_COLUMN_SLOTS = 8
_HIP_WAIST_JOINTS = (r".*_hip", "waist")
_KNEE_JOINTS = (r".*_knee",)
_LEGGED_REWARD_TERMS = (
  "pose",
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "air_time",
  "foot_clearance",
  "foot_swing_height",
  "foot_slip",
  "soft_landing",
)
_WHEEL_GEOMS = (
  "fr_wheel1_col",
  "fr_wheel2_col",
  "fl_wheel1_col",
  "fl_wheel2_col",
  "br_wheel1_col",
  "br_wheel2_col",
  "bl_wheel1_col",
  "bl_wheel2_col",
)
_CRITIC_FOOT_TERMS = (
  "foot_height",
  "foot_air_time",
  "foot_contact",
  "foot_contact_forces",
)

# Stair-heavy curriculum. Flat is only a soft landing pad between climbs.
_WALTER_FS_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=3,
  curriculum=True,
  sub_terrains={
    "flat": flat(proportion=0.15),
    "pyramid_stairs": pyramid_stairs(
      proportion=0.425, step_height_range=(0.06, 0.18)
    ),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.425, step_height_range=(0.06, 0.18)
    ),
  },
  add_lights=True,
)


def walter_fs_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the rough-terrain WaLTER velocity config, with an optional play mode."""
  cfg = make_velocity_env_cfg()
  _configure_sim(cfg)
  _configure_scene(cfg)
  _configure_actions(cfg)
  _configure_commands(cfg)
  _configure_observations(cfg)
  _configure_rewards(cfg)
  _configure_terminations(cfg)
  if play:
    _apply_play_mode(cfg)
  return cfg


def walter_fs_rough_test_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough terrain with every third difficulty row, including both ends.

  Training starts every environment in the drive pool on row 0. Leaving a
  tile does not end the episode. Climb rows spawn on an approach.
  """
  cfg = walter_fs_rough_env_cfg(play=play)
  _keep_every_third_terrain_row(cfg)
  _scale_stairs(cfg, 1.5)
  _swap_noise_for_random_grid(cfg)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  if not play:
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.max_init_terrain_level = 0
    _configure_rough_test_curriculum(cfg)
  return cfg


def walter_fs_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat-ground WaLTER velocity config used for early smoke tests."""
  cfg = walter_fs_rough_env_cfg(play=play)
  _switch_to_flat_terrain(cfg)
  return cfg


def walter_fs_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the stair-heavy WaLTER velocity config for climb finetuning."""
  cfg = make_velocity_env_cfg()
  _configure_sim(cfg)
  _configure_scene(cfg)
  _switch_to_stairs_terrain(cfg)
  _configure_actions(cfg)
  _configure_commands(cfg)
  # Bias stairs practice toward driving forward into climbs.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_forward_envs = 0.8
  # Hold at hardest row when cleared so curriculum means can max out.
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_hold_max,
    params={"command_name": "twist"},
  )
  _configure_observations(cfg)
  _configure_rewards(cfg)
  _configure_terminations(cfg)
  if play:
    _apply_play_mode(cfg)
  return cfg


# REWARD WEIGHTS
# Tune these. Every term is registered. A weight of 0 leaves it on and pays nothing.
FORWARD_TRACK_WEIGHT = 2.0  # Drive at the commanded speed along body x.
HEADING_HOLD_WEIGHT = 0.0  # Stay pointed the way it spawned, except while tumbling.
YAW_TRACK_WEIGHT = 1.0  # Match the commanded turn rate.
UPRIGHT_WEIGHT = 0.2  # Keep the torso upright.
ACTION_RATE_WEIGHT = -1  # Penalize changes in the tumble toggle and tank hip.
HEAD_LEVEL_WEIGHT = -0.01  # Penalize the head rolling off level.
WHEEL_COLLISION_WEIGHT = -1.0  # One count per touching wheel pair.
LIMB_TORQUE_PENALTY_WEIGHT = -0.005  # Penalize hip and knee torque relative to rated torque.
KNEE_STUCK_TUMBLE_WEIGHT = 0.0  # Pay for knee spin while falling behind the speed command.
TRACK_TEACHER_WEIGHT = 1  # Match the PD teacher; kernel in [0, 1], so peak is 0.5.
PREDICTIVE_KNEE_TUMBLE_WEIGHT = 2.0  # Pay for knee spin when an upward step is still ahead.
# Per radian of nose-down pitch while tumbling toward a drop. A 12 cm stair
# is about 22 deg, so this is about -2.3/s and outweighs the tumble bonus.
DOWNHILL_TUMBLE_WEIGHT = -6.0
STUCK_SPEED_WEIGHT = -1.0  # Penalize body-x speed near zero while commanded forward.
# Squared backward speed (m/s)^2 while commanded forward. At 1 m/s this is
# -5/s, more than the forward-tracking peak, so a stair backup does not pay.
REVERSE_DRIVE_WEIGHT = -5.0
DRIVE_STATE_RATE_WEIGHT = -0.05  # Squared change of the raw drive_state sample.
DRIVE_STATE_BAND_WEIGHT = -0.2  # Squared amount raw drive_state sits outside ±0.5.

# State-machine table at a 10 Hz policy step. The constants above stay on the
# teacher and the 50 Hz tasks. Action rate is smaller because the step is 5x longer.
SM_FORWARD_TRACK_WEIGHT = 1.0
SM_FORWARD_PROGRESS_WEIGHT = 2.0
SM_CLIMB_TUMBLE_WEIGHT = 1.5
SM_UNNEEDED_TUMBLE_WEIGHT = -1.0
SM_STUCK_SPEED_WEIGHT = -0.5
SM_ACTION_RATE_WEIGHT = -0.1
SM_TORSO_CONTACT_WEIGHT = -2.0
_ROBOT_MASS_KG = 101.46
_HARD_TORSO_CONTACT_N = 0.3 * _ROBOT_MASS_KG * 9.81

# Kernel widths for the terms above. These are not weights.
HEADING_HOLD_STD = 0.5
TRACK_TEACHER_STD = 1.0
_LIN_VEL_STD = math.sqrt(0.25)
_ANG_VEL_STD = math.sqrt(0.5)
_STUCK_SPEED_STD = 0.1


RlSceneName = Literal["flat", "rough", "rough-test", "stairs"]


def walter_fs_rl_env_cfg(
  play: bool = False,
  task: RlSceneName = "flat",
) -> ManagerBasedRlEnvCfg:
  """Forward task on ``task`` terrain, with the trained action layout.

  Yaw rate is zero on three quarters of commands. The rest are drawn from
  the yaw range.

  Hips, absolute knees, and wheels match ``FullBodyPdController``.
  ``waist`` is fixed at 0 degrees and is not an action.
  Actor and critic both include the torso terrain-height scan. Flat ground
  keeps that sensor too, so the observation width matches the rough scenes.
  """
  scenes = {
    "flat": walter_fs_flat_env_cfg,
    "rough": walter_fs_rough_env_cfg,
    "rough-test": walter_fs_rough_test_env_cfg,
    "stairs": walter_fs_stairs_env_cfg,
  }
  cfg = scenes[task](play=play)
  _configure_simple_commands(cfg)
  _configure_simple_rewards(cfg)
  _add_terrain_scan_observation(cfg)
  # Imported inside the function. The PD env module imports this one.
  from walter_rl.control.full_body_pd.env import _add_waist_lock
  from walter_rl.control.full_body_pd.gains import default_gains

  _add_waist_lock(cfg, default_gains())
  # Rough-test installs its own level curriculum in the scene config.
  if task != "rough-test":
    _track_spawn_heading(cfg)
    _pin_hard_inverted_stairs(cfg)
  return cfg


def _configure_traversal_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the frozen heading curriculum with fixed-column traversal."""
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=TraversalTerrainCurriculum,
  )
  cfg.events["traversal_progress"] = EventTermCfg(
    func=accumulate_traversal_progress,
    mode="step",
  )
  cfg.events["traversal_spawn"] = EventTermCfg(
    func=spawn_traversal_pose,
    mode="reset",
  )


def _configure_rough_test_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drive on low rows, climb from an approach on the tall rows."""
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=SkillTerrainCurriculum,
    params={"command_name": "twist"},
  )
  cfg.events["skill_progress"] = EventTermCfg(
    func=accumulate_skill_progress,
    mode="step",
  )
  cfg.events["skill_spawn"] = EventTermCfg(
    func=spawn_skill_pose,
    mode="reset",
  )


def _configure_state_machine_rough_scene(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> None:
  """Rough training mix and fixed-column traversal curriculum."""
  _configure_rough_training_terrain(cfg)
  if not play:
    _configure_traversal_curriculum(cfg)


def _track_spawn_heading(cfg: ManagerBasedRlEnvCfg) -> None:
  """Promote terrain only for travel along the spawn yaw."""
  term = cfg.curriculum.get("terrain_levels")
  if term is None:
    return
  term.func = terrain_levels_along_spawn


def _pin_hard_inverted_stairs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep 80% of inverted-stair environments on the hardest row."""
  term = cfg.curriculum.get("terrain_levels")
  if term is None:
    return
  term.func = wrap_terrain_levels(term.func)


def walter_fs_teacher_env_cfg(
  play: bool = False,
  task: RlSceneName = "flat",
) -> ManagerBasedRlEnvCfg:
  """Velocity task on ``task`` terrain, matching a scan-conditioned PD target.

  Actor and critic see the same terms as the state-machine policy: front and
  rear height grids, proprioception, the twist command, and the applied
  tumble bit over 0.5 s. Neither sees the privileged critic extras. The
  previous-action term is the 17-D student command. The env decodes that
  into the 16-D plant command.

  On a flat scan the stored teacher action is the reactive PD. When the
  match gate is off, that action is the same PD command with tumble forced
  on. ``track_teacher`` matches it at ``TRACK_TEACHER_WEIGHT`` either way.
  ``predictive_knee_tumble`` still pays for knee rate when an upward step is ahead.
  ``knee_stuck_tumble`` remains the lag recovery signal.   ``drive_state_rate_l2``
  and ``drive_state_outside_band`` charge the raw sample so a tumble can leave
  the wheeled band and stay there. ``reverse_drive`` charges backward body
  speed while the command is forward.
  """
  cfg = walter_fs_rl_env_cfg(play=play, task=task)
  if task == "rough":
    _configure_state_machine_rough_scene(cfg, play=play)
  _install_split_scans(cfg)
  _configure_teacher_observations(cfg)
  cfg.rewards["track_teacher"] = RewardTermCfg(
    func=gated_track_teacher_action,
    weight=TRACK_TEACHER_WEIGHT,
    params={"std": TRACK_TEACHER_STD, "sensor_name": "terrain_scan_front"},
  )
  cfg.rewards["predictive_knee_tumble"] = RewardTermCfg(
    func=predictive_knee_tumble,
    weight=PREDICTIVE_KNEE_TUMBLE_WEIGHT,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=_KNEE_JOINTS),
    },
  )
  cfg.rewards["drive_state_rate_l2"] = RewardTermCfg(
    func=drive_state_rate_l2,
    weight=DRIVE_STATE_RATE_WEIGHT,
  )
  cfg.rewards["drive_state_outside_band"] = RewardTermCfg(
    func=drive_state_outside_band,
    weight=DRIVE_STATE_BAND_WEIGHT,
  )
  cfg.rewards["reverse_drive"] = RewardTermCfg(
    func=reverse_drive,
    weight=REVERSE_DRIVE_WEIGHT,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.metrics["teacher_action_mse"] = MetricsTermCfg(func=teacher_action_mse)
  cfg.metrics["spawn_heading_error"] = MetricsTermCfg(
    func=spawn_heading_error,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  return cfg


def walter_fs_state_machine_env_cfg(
  play: bool = False,
  task: RlSceneName = "flat",
) -> ManagerBasedRlEnvCfg:
  """PD inner loop with a learned tumble toggle and tank hip offset.

  Actor and critic see the head and torso height grids, projected gravity,
  hip and knee torque, wheel torque, the twist command, and the applied tumble mode over
  the last 0.5 s. The policy action is two numbers: the tumble toggle and
  a hip offset from 0 to 30 degrees. Tank Mode stays on and the tank knee
  command stays at zero. PD still writes the 16 joint targets. The
  stall-integral tumble gate is not used.
  """
  cfg = walter_fs_rl_env_cfg(play=play, task=task)
  # Set before observations so tumble memory counts policy steps, not PD ticks.
  cfg.decimation = _POLICY_DECIMATION
  if task == "rough":
    _configure_state_machine_rough_scene(cfg, play=play)
  _configure_state_machine_contact(cfg)
  _install_footprint_scan(cfg)
  _configure_state_machine_observations(cfg)
  _configure_state_machine_rewards(cfg)
  return cfg


_AXLE_JOINTS = (r".*_axle\d",)
_HIP_JOINTS = (r".*_hip",)


def _configure_teacher_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """State-machine policy terms for both networks. No privileged critic."""
  cfg.observations["actor"].terms = _teacher_terms(cfg, noisy=True)
  cfg.observations["critic"].terms = _teacher_terms(cfg, noisy=False)


def _teacher_terms(
  cfg: ManagerBasedRlEnvCfg,
  *,
  noisy: bool,
) -> dict[str, ObservationTermCfg]:
  """State-machine policy terms, with the 17-D student action in place of the 2-D one."""
  terms = _state_machine_terms(cfg, noisy=noisy)
  ordered: dict[str, ObservationTermCfg] = {}
  for name, term in terms.items():
    if name == "last_policy_action":
      ordered["actions"] = ObservationTermCfg(func=teacher_policy_action)
      continue
    ordered[name] = term
  return ordered


def _configure_state_machine_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Height grids, proprioception, and a privileged critic.

  The front grid is attached to the front torso and the rear grid to the back torso.
  Each is 1 m by 1 m with an 8×8 grid, so each field has 64 rays.
  The near edge sits 0.1 m outside that body's box. Tumble memory is the
  applied mode, oldest to newest, over the same 0.5 s the latch stays on.
  The actor adds the gyro, hip angles, knee sin/cos, joint and wheel
  torques, and the last policy action.
  The critic sees those without noise, plus body velocity, joint rates, the
  footprint profile, the terrain row, and the latch timers.
  """
  _install_split_scans(cfg)
  cfg.observations["actor"].terms = _state_machine_terms(cfg, noisy=True)
  cfg.observations["critic"].terms = _state_machine_terms(cfg, noisy=False)
  cfg.observations["critic"].terms.update(_privileged_terms())


def _state_machine_terms(
  cfg: ManagerBasedRlEnvCfg,
  *,
  noisy: bool,
) -> dict[str, ObservationTermCfg]:
  """Terms shared by the actor and the critic."""
  return {
    "height_scan_front": _split_height_scan_term("terrain_scan_front", noisy=noisy),
    "height_scan_rear": _split_height_scan_term("terrain_scan_rear", noisy=noisy),
    "projected_gravity": _projected_gravity_term(noisy=noisy),
    "base_ang_vel": _gyro_term(noisy=noisy),
    "hip_pos": _hip_pos_term(noisy=noisy),
    "knee_sincos": ObservationTermCfg(
      func=joint_pos_sincos,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_KNEE_JOINTS)},
    ),
    "joint_torques": ObservationTermCfg(
      func=joint_torques,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "wheel_torques": ObservationTermCfg(
      func=wheel_torques,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "last_policy_action": ObservationTermCfg(func=last_policy_action),
    "tumble_applied": _tumble_memory_term(cfg),
  }


def _privileged_terms() -> dict[str, ObservationTermCfg]:
  """Sim-only terms. These are not on the actor."""
  return {
    "base_lin_vel": ObservationTermCfg(
      func=finite_builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "joint_vel": ObservationTermCfg(
      func=finite_joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=_KNEE_JOINTS + _AXLE_JOINTS
        ),
      },
    ),
    "footprint_scan": ObservationTermCfg(
      func=footprint_height_profile,
      params={"sensor_name": "footprint_scan"},
    ),
    "terrain_level": ObservationTermCfg(
      func=terrain_level,
      params={"column_slots": _STATE_MACHINE_TERRAIN_COLUMN_SLOTS},
    ),
    "tumble_latch": ObservationTermCfg(func=tumble_latch_timers),
  }


def _gyro_term(*, noisy: bool) -> ObservationTermCfg:
  """IMU angular velocity. The actor term gets the velocity-task gyro noise."""
  return ObservationTermCfg(
    func=finite_builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2) if noisy else None,
  )


def _hip_pos_term(*, noisy: bool) -> ObservationTermCfg:
  """Hip angle relative to the default pose. Actor encoders are biased."""
  return ObservationTermCfg(
    func=finite_joint_pos_rel,
    params={
      "biased": noisy,
      "asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_JOINTS),
    },
  )


# Outward face distance in each segment's body frame (+x forward).
_FRONT_FACE_X = 0.15
_BACK_FACE_X = 0.23
_SPLIT_SCAN_GAP = 0.1
_SPLIT_SCAN_SIZE = (1.0, 1.0)
# Eight samples along each axis on a 1 m field (64 rays total).
_SPLIT_SCAN_RESOLUTION = _SPLIT_SCAN_SIZE[0] / 7.0
_FRONT_SCAN_OFFSET_X = _FRONT_FACE_X + _SPLIT_SCAN_GAP + _SPLIT_SCAN_SIZE[0] / 2
_REAR_SCAN_OFFSET_X = -(_BACK_FACE_X + _SPLIT_SCAN_GAP + _SPLIT_SCAN_SIZE[0] / 2)


def _install_split_scans(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the torso height grid with the front and rear fields."""
  sensors = [
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != "terrain_scan"
  ]
  names = {sensor.name for sensor in sensors}
  if "terrain_scan_front" not in names:
    sensors.append(
      _split_scan_sensor("terrain_scan_front", "head", _FRONT_SCAN_OFFSET_X)
    )
  if "terrain_scan_rear" not in names:
    sensors.append(
      _split_scan_sensor("terrain_scan_rear", "torso", _REAR_SCAN_OFFSET_X)
    )
  cfg.scene.sensors = tuple(sensors)


def _split_scan_sensor(name: str, body: str, offset_x: float) -> RayCastSensorCfg:
  """One yaw-aligned 1 m × 1 m height grid with 64 rays."""
  return RayCastSensorCfg(
    name=name,
    frame=ObjRef(type="body", name=body, entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(
      size=_SPLIT_SCAN_SIZE,
      resolution=_SPLIT_SCAN_RESOLUTION,
      offset=(offset_x, 0.0),
    ),
    max_distance=_TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
    debug_vis_enabled=False,
  )


_TUMBLE_MEMORY_S = 0.5


def _tumble_memory_term(cfg: ManagerBasedRlEnvCfg) -> ObservationTermCfg:
  """Applied tumble bit for the last ``_TUMBLE_MEMORY_S``, oldest first."""
  step_dt = cfg.sim.mujoco.timestep * cfg.decimation
  return ObservationTermCfg(
    func=tumble_applied,
    history_length=max(1, math.ceil(_TUMBLE_MEMORY_S / step_dt)),
  )


def _projected_gravity_term(*, noisy: bool) -> ObservationTermCfg:
  """Torso gravity direction. The actor term gets the velocity-task IMU noise."""
  return ObservationTermCfg(
    func=finite_projected_gravity,
    params={"asset_cfg": SceneEntityCfg("robot")},
    noise=Unoise(n_min=-0.05, n_max=0.05) if noisy else None,
  )


def _split_height_scan_term(sensor_name: str, *, noisy: bool) -> ObservationTermCfg:
  """Height of one split scan, scaled by the ray max distance."""
  return ObservationTermCfg(
    func=finite_height_scan,
    params={"sensor_name": sensor_name},
    noise=Unoise(n_min=-0.1, n_max=0.1) if noisy else None,
    scale=1.0 / _TERRAIN_SCAN_MAX_DISTANCE,
  )


def _configure_state_machine_contact(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remember every substep of belly force and terminate only on a hard hit."""
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "torso_ground_touch":
      sensor.history_length = _POLICY_DECIMATION
  term = cfg.terminations.get("torso_contact")
  if term is not None:
    term.params["force_threshold"] = _HARD_TORSO_CONTACT_N


def _install_footprint_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the privileged height grid used by the climb and downhill gates."""
  sensors = [
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != "footprint_scan"
  ]
  pattern = footprint_scan_pattern()
  sensors.append(
    RayCastSensorCfg(
      name="footprint_scan",
      frame=ObjRef(type="body", name="torso", entity="robot"),
      ray_alignment="yaw",
      pattern=GridPatternCfg(
        size=pattern["size"],
        resolution=pattern["resolution"],
        offset=pattern["offset"],
      ),
      max_distance=_TERRAIN_SCAN_MAX_DISTANCE,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=False,
    )
  )
  cfg.scene.sensors = tuple(sensors)


def _configure_state_machine_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Register every reward-weight term. Zero weights stay in the table."""
  knees = SceneEntityCfg("robot", joint_names=_KNEE_JOINTS)
  cfg.rewards["track_linear_velocity"].weight = SM_FORWARD_TRACK_WEIGHT
  cfg.rewards["stuck_x_velocity"].weight = SM_STUCK_SPEED_WEIGHT
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=policy_action_rate_l2,
    weight=SM_ACTION_RATE_WEIGHT,
  )
  cfg.rewards["forward_progress"] = RewardTermCfg(
    func=forward_progress,
    weight=SM_FORWARD_PROGRESS_WEIGHT,
    params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["climb_tumble"] = RewardTermCfg(
    func=climb_tumble,
    weight=SM_CLIMB_TUMBLE_WEIGHT,
    params={
      "command_name": "twist",
      "sensor_name": "footprint_scan",
      "asset_cfg": knees,
    },
  )
  cfg.rewards["unneeded_tumble"] = RewardTermCfg(
    func=unneeded_tumble,
    weight=SM_UNNEEDED_TUMBLE_WEIGHT,
    params={"sensor_name": "footprint_scan"},
  )
  cfg.rewards["torso_contact"] = RewardTermCfg(
    func=torso_contact_flag,
    weight=SM_TORSO_CONTACT_WEIGHT,
    params={"sensor_name": "torso_ground_touch"},
  )
  cfg.rewards["knee_stuck_tumble"].weight = KNEE_STUCK_TUMBLE_WEIGHT
  cfg.rewards["track_teacher"] = RewardTermCfg(
    func=gated_track_teacher_action,
    weight=TRACK_TEACHER_WEIGHT,
    params={"std": TRACK_TEACHER_STD, "sensor_name": "terrain_scan"},
  )
  cfg.rewards["limb_torque"] = RewardTermCfg(
    func=limb_torque_l2,
    weight=LIMB_TORQUE_PENALTY_WEIGHT,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["downhill_tumble"] = RewardTermCfg(
    func=footprint_downhill_tumble,
    weight=DOWNHILL_TUMBLE_WEIGHT,
    params={
      "sensor_name": "footprint_scan",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )


_TERRAIN_SCAN_SIZE = (2.4, 1.2)
_TERRAIN_SCAN_RESOLUTION = 0.1
_TERRAIN_SCAN_OFFSET = (0.4, 0.0)
_TERRAIN_SCAN_MAX_DISTANCE = 5.0


def _add_terrain_scan_observation(cfg: ManagerBasedRlEnvCfg) -> None:
  """Put the torso terrain-height grid in the actor and the critic.

  Flat scenes drop the scan because the PD task never reads it. The policy
  needs the same grid on every scene, placed just before the knee encoding.
  """
  _ensure_terrain_scan_sensor(cfg)
  for group_name in ("actor", "critic"):
    _install_height_scan(cfg, group_name)


def _ensure_terrain_scan_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Restore the torso scan when a flat scene has removed it."""
  sensors = list(cfg.scene.sensors or ())
  if any(sensor.name == "terrain_scan" for sensor in sensors):
    return
  sensors.append(_terrain_scan_sensor())
  cfg.scene.sensors = tuple(sensors)


def _install_height_scan(cfg: ManagerBasedRlEnvCfg, group_name: str) -> None:
  """Insert ``height_scan`` immediately before ``knee_sincos``."""
  terms = cfg.observations[group_name].terms
  existing = terms.pop("height_scan", None)
  knee = terms.pop("knee_sincos", None)
  terms["height_scan"] = existing if existing is not None else _height_scan_term(
    noisy=group_name == "actor"
  )
  if knee is not None:
    terms["knee_sincos"] = knee


def _height_scan_term(*, noisy: bool) -> ObservationTermCfg:
  """Terrain height above each ray, scaled by the scan's max distance."""
  return ObservationTermCfg(
    func=finite_height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.1, n_max=0.1) if noisy else None,
    scale=1.0 / _TERRAIN_SCAN_MAX_DISTANCE,
  )


def _terrain_scan_sensor() -> RayCastSensorCfg:
  """Torso grid looking 1.6 m ahead and 0.8 m behind."""
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="torso", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(
      size=_TERRAIN_SCAN_SIZE,
      resolution=_TERRAIN_SCAN_RESOLUTION,
      offset=_TERRAIN_SCAN_OFFSET,
    ),
    max_distance=_TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
    debug_vis_enabled=False,
  )


# Cube root of a unit uniform draw. Its mean is 3/4, so forward speed
# averages 0.25 + 0.75 = 1 m/s.
_TRAIN_LIN_VEL_X = (0.25, 1.25)
_TRAIN_LIN_VEL_X_POWER = 1.0 / 3.0
# Uniform yaw rate, rad/s. Three quarters of resamples are forced to zero.
_TRAIN_ANG_VEL_Z = (-1.0, 1.0)
_TRAIN_ZERO_YAW_PROB = 0.75
_TRAIN_STANDING_PROB = 0.05


def _configure_simple_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  """Sample body-frame forward speed, and a yaw rate on one quarter of draws.

  Forward speed is drawn from 0.25–1.25 m/s with a cube-root bias, so the
  average draw is 1 m/s. Yaw rate is uniform on ±1 rad/s, then set to zero
  on 75% of resamples. Five percent of resamples are a full stop, which
  also zeros yaw.
  """
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.rel_standing_envs = _TRAIN_STANDING_PROB
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_max_ride_height_envs = 0.0
  twist.lin_vel_x_power = _TRAIN_LIN_VEL_X_POWER
  twist.zero_ang_vel_prob = _TRAIN_ZERO_YAW_PROB
  twist.ranges.lin_vel_x = _TRAIN_LIN_VEL_X
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = _TRAIN_ANG_VEL_Z
  twist.ranges.heading = None
  twist.ranges.ride_height = None
  cfg.curriculum.pop("command_vel", None)


def _configure_simple_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Track body forward speed and yaw rate, and pay for a tumble while stuck."""
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=track_body_forward,
      weight=FORWARD_TRACK_WEIGHT,
      params={
        "command_name": "twist",
        "std": _LIN_VEL_STD,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "hold_spawn_heading": RewardTermCfg(
      func=hold_spawn_heading,
      weight=HEADING_HOLD_WEIGHT,
      params={"std": HEADING_HOLD_STD, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=track_yaw_rate,
      weight=YAW_TRACK_WEIGHT,
      params={"command_name": "twist", "std": _ANG_VEL_STD},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=UPRIGHT_WEIGHT,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
      },
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=ACTION_RATE_WEIGHT,
    ),
    "head_level": RewardTermCfg(
      func=head_level,
      weight=HEAD_LEVEL_WEIGHT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist",))},
    ),
    "wheel_wheel_collision": RewardTermCfg(
      func=wheel_wheel_collision,
      weight=WHEEL_COLLISION_WEIGHT,
    ),
    "knee_stuck_tumble": RewardTermCfg(
      func=KneeStuckTumble,
      weight=KNEE_STUCK_TUMBLE_WEIGHT,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", joint_names=_KNEE_JOINTS),
      },
    ),
    "stuck_x_velocity": RewardTermCfg(
      func=stuck_x_velocity,
      weight=STUCK_SPEED_WEIGHT,
      params={
        "command_name": "twist",
        "std": _STUCK_SPEED_STD,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
  }


def _configure_sim(cfg: ManagerBasedRlEnvCfg) -> None:
  """Give the wheel contacts a bit more solver budget than the factory default."""
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 256


def _configure_scene(cfg: ManagerBasedRlEnvCfg) -> None:
  """Attach the FS robot, terrain scan, and a torso/head ground-contact sensor."""
  cfg.scene.entities = {"robot": get_walter_fs_robot_cfg()}
  cfg.scene.extent = 3.0
  _point_terrain_scan_at_torso(cfg)
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != "foot_height_scan"
  ) + (_torso_ground_sensor(), _body_height_scan(), *_wheel_pair_sensors())

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = True
    # Start on the easier rows. Curriculum still promotes robots that drive far.
    cfg.scene.terrain.max_init_terrain_level = 2
    # Copy before editing so the shared mjlab preset dict stays intact.
    # Rise/run of 1 is a 45 deg face and tips this body. Cap near 24 deg.
    sub_terrains = dict(generator.sub_terrains)
    sub_terrains["hf_pyramid_slope"] = replace(
      sub_terrains["hf_pyramid_slope"], slope_range=(0.0, 0.45)
    )
    sub_terrains["hf_pyramid_slope_inv"] = replace(
      sub_terrains["hf_pyramid_slope_inv"], slope_range=(0.0, 0.45)
    )
    sub_terrains["wave_terrain"] = replace(
      sub_terrains["wave_terrain"], amplitude_range=(0.0, 0.12)
    )
    sub_terrains["pyramid_stairs"] = replace(
      sub_terrains["pyramid_stairs"], step_height_range=(0.0, 0.12)
    )
    sub_terrains["pyramid_stairs_inv"] = replace(
      sub_terrains["pyramid_stairs_inv"], step_height_range=(0.0, 0.12)
    )
    generator.sub_terrains = sub_terrains

  cfg.viewer.body_name = "torso"
  cfg.viewer.distance = 2.8
  cfg.viewer.elevation = -10.0

  # The factory foot-friction event has empty geom names and randomizes every
  # friction axis, which would wipe the low wheel rolling friction.
  cfg.events.pop("foot_friction", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)


def _point_terrain_scan_at_torso(cfg: ManagerBasedRlEnvCfg) -> None:
  for sensor in cfg.scene.sensors or ():
    if sensor.name != "terrain_scan":
      continue
    assert isinstance(sensor, RayCastSensorCfg)
    assert isinstance(sensor.frame, ObjRef)
    sensor.frame.name = "torso"
    # Rear edge stays at -0.8 m. The extra length and a forward shift look
    # out to 1.6 m in front of the torso.
    sensor.pattern = GridPatternCfg(
      size=_TERRAIN_SCAN_SIZE,
      resolution=_TERRAIN_SCAN_RESOLUTION,
      offset=_TERRAIN_SCAN_OFFSET,
    )


def _body_height_scan() -> RayCastSensorCfg:
  """Small downward grid under the torso, used only by the ride-height reward."""
  return RayCastSensorCfg(
    name="body_height_scan",
    frame=ObjRef(type="body", name="torso", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.3, 0.2), resolution=0.1),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )


def _wheel_pair_sensors() -> tuple[ContactSensorCfg, ...]:
  """One contact sensor for each unordered pair of wheel geoms."""
  sensors = []
  for index, (left, right) in enumerate(combinations(_WHEEL_GEOMS, 2)):
    sensors.append(
      ContactSensorCfg(
        name=f"wheel_pair_{index}",
        primary=ContactMatch(mode="geom", pattern=left, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=right, entity="robot"),
        fields=("found",),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="error",
      )
    )
  return tuple(sensors)


def _torso_ground_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="torso_ground_touch",
    primary=ContactMatch(
      mode="geom",
      pattern=(r"torso_bump_.*", r"head_bump_.*"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def _configure_actions(cfg: ManagerBasedRlEnvCfg) -> None:
  """Hips hold a stance, knees wind, and wheels take a velocity command."""
  cfg.actions = {
    "hip_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(r".*_hip",),
      scale=HIP_ACTION_SCALE,
      use_default_offset=True,
      clip={r".*_hip": (-HIP_ANGLE_LIMIT, HIP_ANGLE_LIMIT)},
    ),
    "knee_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(r".*_knee",),
      scale=KNEE_ACTION_SCALE,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=(r".*_axle[12]",),
      scale=WHEEL_ACTION_SCALE,
      use_default_offset=False,
    ),
  }


def _configure_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  """Track forward speed, yaw, and ride height. Lateral speed stays at zero."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 0.02
  twist.rel_heading_envs = 0.2
  twist.rel_forward_envs = 0.55
  twist.rel_max_ride_height_envs = 0.65
  twist.heading_command = True
  twist.heading_control_stiffness = 0.5
  twist.ranges.lin_vel_x = (0.3, 1.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (-1.0, 1.0)
  twist.ranges.heading = (-math.pi, math.pi)
  twist.ranges.ride_height = (0.0, 1.0)

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (0.3, 1.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-1.0, 1.0),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-0.8, 1.5),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-1.2, 1.2),
    },
  ]


def _configure_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Hide unbounded wheel and knee positions. Knees are reported as sin/cos."""
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    biased = group_name == "actor"
    terms["joint_pos"] = ObservationTermCfg(
      func=finite_joint_pos_rel,
      params={
        "biased": biased,
        "asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_WAIST_JOINTS),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01) if biased else None,
    )
    terms["knee_sincos"] = ObservationTermCfg(
      func=joint_pos_sincos,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=_KNEE_JOINTS)},
    )
    terms["joint_vel"].func = finite_joint_vel_rel
    terms["base_lin_vel"].func = finite_builtin_sensor
    terms["base_ang_vel"].func = finite_builtin_sensor
    terms["projected_gravity"].func = finite_projected_gravity
    if "height_scan" in terms:
      terms["height_scan"].func = finite_height_scan
  for term_name in _CRITIC_FOOT_TERMS:
    cfg.observations["critic"].terms.pop(term_name, None)


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep velocity tracking and uprightness. Drop legged-gait terms."""
  for term_name in _LEGGED_REWARD_TERMS:
    cfg.rewards.pop(term_name, None)
  # XY-only tracking: factory term also squares vz, which fights stair climbs.
  lin_std = cfg.rewards["track_linear_velocity"].params["std"]
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_lin_vel_xy,
    weight=2.0,
    params={"command_name": "twist", "std": lin_std},
  )
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["upright"].weight = 0.1
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["action_rate_l2"].weight = -0.02
  cfg.rewards["track_ride_height"] = RewardTermCfg(
    func=track_ride_height,
    weight=0.2,
    params={
      "command_name": "twist",
      "std": 0.05,
      "sensor_name": "body_height_scan",
      "height_range": (RIDE_HEIGHT_MIN, RIDE_HEIGHT_MAX),
    },
  )
  cfg.rewards["head_level"] = RewardTermCfg(
    func=head_level,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("waist",)),
    },
  )
  cfg.rewards["wheel_wheel_collision"] = RewardTermCfg(
    func=wheel_wheel_collision,
    weight=WHEEL_COLLISION_WEIGHT,
  )
  # Oriented, terrain+command gated tumble. Prefer front-down (+qdot when
  # driving forward). Penalize reverse tumble; waste only on the flat column.
  _knee_params = {
    "velocity_scale": 6.0,
    "command_name": "twist",
    "sensor_name": "terrain_scan",
    "rough_eps": 0.04,
    "min_forward_cmd": 0.2,
    "min_terrain_level": 2,
    "min_body_vx": 0.2,
    "asset_cfg": SceneEntityCfg("robot", joint_names=_KNEE_JOINTS),
  }
  cfg.rewards["knee_tumble"] = RewardTermCfg(
    func=knee_tumble,
    weight=3.0,
    params=dict(_knee_params),
  )
  cfg.rewards["knee_tumble_reverse"] = RewardTermCfg(
    func=knee_tumble_reverse,
    weight=-2.0,
    params=dict(_knee_params),
  )
  cfg.rewards["knee_waste"] = RewardTermCfg(
    func=knee_waste,
    weight=-6.0,
    params=dict(_knee_params),
  )
  cfg.rewards["climb_vz"] = RewardTermCfg(
    func=climb_vz,
    weight=1.5,
    params={
      "sensor_name": "terrain_scan",
      "rise_eps": 0.04,
    },
  )
  cfg.rewards["drive_forward"] = RewardTermCfg(
    func=drive_forward,
    weight=1.0,
    params={"command_name": "twist"},
  )


def _configure_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
  # World-frame tilt. Steep slopes legitimately pitch the body, so this only
  # catches a near-flip. Belly bumps cover a face-plant on the rough task.
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(100.0)
  cfg.terminations["torso_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "torso_ground_touch", "force_threshold": 20.0},
  )
  cfg.terminations["non_finite"] = TerminationTermCfg(
    func=non_finite_root,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )


# Rough training stairs. Row 0 is 6 cm and row 9 is 22 cm. The generator
# interpolates the rows between. rough-test tops out at 0.18 m.
_ROUGH_TRAIN_STAIR_HEIGHT = (0.06, 0.22)
# One column per type. Proportions only set how many robots spawn there.
_ROUGH_TRAIN_PROPORTIONS = {
  "pyramid_stairs_inv": 0.30,
  "random_grid": 0.20,
  "pyramid_stairs": 0.15,
  "hf_pyramid_slope_inv": 0.125,
  "hf_pyramid_slope": 0.075,
  "flat": 0.05,
  "random_rough": 0.05,
  "wave_terrain": 0.05,
}
# Four fifths of the preset cube-field height is the hard row. Easier rows
# keep the same linear ramp, so they land at 0, 1/3, and 2/3 of that cap.
_CUBE_FIELD_HEIGHT_SCALE = 4.0 / 5.0
# The generator turns this into a 1.6 m flat apron. The head wheels sit 0.64 m
# ahead of the root and the tail 0.32 m behind, so the stock 0.4 m apron
# cannot hold an approach without standing in the cubes or the next tile.
_CUBE_APPROACH_BORDER_M = 1.5


def _configure_rough_training_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Raise rough stairs past rough-test and add a cube column.

  Stair height is 6 cm on row 0 and 22 cm on row 9. The shared rough scene
  stays at 0.12 m stairs. rough-test multiplies those by 1.5, so writing
  0.22 m there would push the benchmark to 0.33 m.
  """
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  sub_terrains = dict(generator.sub_terrains)
  for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    sub_terrains[name] = replace(
      sub_terrains[name], step_height_range=_ROUGH_TRAIN_STAIR_HEIGHT
    )
  sub_terrains["random_rough"] = replace(
    sub_terrains["random_rough"], scale_with_difficulty=True
  )
  grid = box_random_grid(proportion=_ROUGH_TRAIN_PROPORTIONS["random_grid"])
  low, high = grid.grid_height_range
  sub_terrains["random_grid"] = replace(
    grid,
    platform_width=2.0,
    border_width=1.0,
    grid_height_range=(low, low + (high - low) * _CUBE_FIELD_HEIGHT_SCALE),
  )
  ordered = {}
  for name, proportion in _ROUGH_TRAIN_PROPORTIONS.items():
    ordered[name] = replace(sub_terrains[name], proportion=proportion)
  generator.sub_terrains = ordered


def _swap_noise_for_random_grid(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the uniform-noise column with a difficulty-scaled cube field.

  Cell height interpolates ``grid_height_range`` by row difficulty. The hard
  row stops at four fifths of the preset height, and the easier rows scale
  by the same factor.
  """
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  replaced = {}
  for name, sub in generator.sub_terrains.items():
    if name != "random_rough":
      replaced[name] = sub
      continue
    grid = box_random_grid(proportion=sub.proportion)
    low, high = grid.grid_height_range
    replaced["random_grid"] = replace(
      grid,
      border_width=_CUBE_APPROACH_BORDER_M,
      grid_height_range=(low, low + (high - low) * _CUBE_FIELD_HEIGHT_SCALE),
    )
  generator.sub_terrains = replaced


def _scale_stairs(cfg: ManagerBasedRlEnvCfg, scale: float) -> None:
  """Multiply pyramid stair step heights. The other columns stay as they are."""
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  sub_terrains = dict(generator.sub_terrains)
  for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stair = sub_terrains[name]
    low, high = stair.step_height_range
    sub_terrains[name] = replace(
      stair, step_height_range=(low * scale, high * scale)
    )
  generator.sub_terrains = sub_terrains


def _keep_every_third_terrain_row(cfg: ManagerBasedRlEnvCfg) -> None:
  """Drop to rows 0, 3, 6, and 9 of the 10-row rough grid.

  Difficulty is ``row / (num_rows - 1)``. Those four rows are the same
  difficulties as a 4-row grid, so the easiest and hardest patches stay
  and the map is shorter.
  """
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  stride = 3
  generator.num_rows = (generator.num_rows - 1) // stride + 1


def _apply_play_mode(cfg: ManagerBasedRlEnvCfg) -> None:
  """Infinite episode, no pushes, and the full training difficulty grid.

  A torso or head bump on the terrain does not reset the episode.
  """
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations.pop("torso_contact", None)
  cfg.curriculum = {}
  # Reset may land on any row (difficulty) and any terrain column.
  cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain,
    mode="reset",
    params={},
  )
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    generator = cfg.scene.terrain.terrain_generator
    # Keep the training layout: 10 difficulty rows, one column per terrain type.
    # Shrinking to 5x5 dropped the hard half of the curriculum.
    generator.curriculum = True
    generator.num_rows = 10
    cfg.scene.terrain.max_init_terrain_level = None
    generator.border_width = 10.0


def _switch_to_stairs_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the rough mix with a flat-plus-stairs curriculum grid."""
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = replace(_WALTER_FS_STAIRS_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 6


def _switch_to_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name not in ("terrain_scan", "body_height_scan")
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
