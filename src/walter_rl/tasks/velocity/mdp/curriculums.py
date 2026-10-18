"""WaLTER terrain curriculum helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_MIN_CMD_SPEED = 0.1


def terrain_levels_along_spawn(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Promote by travel along the spawn yaw, not by distance in any direction.

  Isotropic distance promotes a robot that yaws off the stairs and drives
  to the cell border. Progress here is world displacement projected onto the
  yaw latched at the start of the episode.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None

  along = _progress_along_spawn(env, env_ids, asset)
  cmd_speed = command[env_ids, 0].abs()
  has_cmd = cmd_speed > _MIN_CMD_SPEED
  promote_dist = terrain_generator.size[0] / 4
  move_up = has_cmd & (along > promote_dist)
  move_down = has_cmd & (along < cmd_speed * env.max_episode_length_s * 0.25)
  move_down = move_down & ~move_up
  if env.common_step_counter == 0 or not hasattr(env, "spawn_yaw_w"):
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)
  terrain.update_env_origins(env_ids, move_up, move_down)
  return _level_stats(env)


def _progress_along_spawn(env, env_ids: torch.Tensor, asset: Entity) -> torch.Tensor:
  """Meters traveled from the cell origin along the latched spawn yaw."""
  delta = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
  yaw = getattr(env, "spawn_yaw_w", None)
  valid = getattr(env, "_spawn_yaw_valid", None)
  if yaw is None or valid is None:
    return torch.zeros(env_ids.shape[0], device=env.device)
  heading = yaw[env_ids]
  along = torch.cos(heading) * delta[:, 0] + torch.sin(heading) * delta[:, 1]
  return torch.where(valid[env_ids], along, torch.zeros_like(along))


def _level_stats(env) -> dict[str, torch.Tensor]:
  """Mean terrain level, overall and per sub-terrain column."""
  terrain = env.scene.terrain
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }
  generator = terrain.cfg.terrain_generator
  if generator is None or terrain.terrain_origins is None:
    return result
  names = list(generator.sub_terrains.keys())
  if terrain.terrain_origins.shape[1] != len(names):
    return result
  for index, name in enumerate(names):
    mask = terrain.terrain_types == index
    if mask.any():
      result[name] = torch.mean(levels[mask])
  return result


def terrain_levels_hold_max(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Promote/demote by commanded-direction progress; hold at the hardest row.

  Uses displacement along the velocity command, not isotropic distance from
  spawn, so circling/tumbling in place cannot farm promotions. Clearing the
  top row keeps the env on the hardest difficulty instead of re-rolling.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None

  delta_xy = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
  cmd_xy = command[env_ids, :2]
  cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
  has_cmd = cmd_speed > _MIN_CMD_SPEED
  # Signed meters traveled along the commanded planar direction.
  progress = (delta_xy * cmd_xy).sum(dim=-1) / cmd_speed.clamp(min=1e-6)

  promote_dist = terrain_generator.size[0] / 4  # 2 m on 8 m tiles; size/2 was too hard early
  move_up = has_cmd & (progress > promote_dist)
  # Fail to cover a quarter of the commanded path length → demote (all columns).
  move_down = has_cmd & (progress < cmd_speed * env.max_episode_length_s * 0.25)
  move_down *= ~move_up

  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  assert terrain.env_origins is not None and terrain.terrain_origins is not None
  levels = terrain.terrain_levels
  levels[env_ids] = levels[env_ids] + move_up.long() - move_down.long()
  hard_row = terrain.max_terrain_level - 1
  levels[env_ids] = torch.clamp(levels[env_ids], min=0, max=hard_row)
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    levels[env_ids], terrain.terrain_types[env_ids]
  ]

  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels.float()),
    "max": torch.max(levels.float()),
  }
  sub_names = list(terrain_generator.sub_terrains.keys())
  num_cols = terrain.terrain_origins.shape[1]
  if num_cols == len(sub_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask].float())
  return result


_INVERTED_STAIRS = "pyramid_stairs_inv"
_HARD_INVERTED_STAIRS_FRACTION = 0.8
_PIN_MASK_ATTR = "_hard_inverted_stairs_mask"


def wrap_terrain_levels(func):
  """Return a curriculum that keeps most inverted stairs on the hardest row.

  The wrapped update still promotes and demotes. Afterwards, 80% of the
  environments on ``pyramid_stairs_inv`` are put back on the top row. The
  same environments stay pinned for the whole run.
  """

  def terrain_levels(env, env_ids, **params):
    result = func(env, env_ids, **params)
    _force_hard_inverted_stairs(env, env_ids)
    _refresh_terrain_level_stats(env, result)
    return result

  return terrain_levels


def _force_hard_inverted_stairs(env, env_ids) -> None:
  """Snap the pinned inverted-stair envs in this reset onto the top row."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return
  if terrain.terrain_origins is None or terrain.max_terrain_level < 1:
    return
  pinned = _pinned_this_reset(terrain, env_ids)
  if not bool(pinned.any()):
    return
  hard_row = terrain.max_terrain_level - 1
  terrain.terrain_levels[pinned] = hard_row
  terrain.env_origins[pinned] = terrain.terrain_origins[
    terrain.terrain_levels[pinned], terrain.terrain_types[pinned]
  ]


def _pinned_this_reset(terrain, env_ids) -> torch.Tensor:
  """Boolean mask of pinned inverted-stair envs included in ``env_ids``."""
  active = torch.zeros_like(_pin_mask(terrain))
  active[env_ids] = True
  return _pin_mask(terrain) & active


def _pin_mask(terrain) -> torch.Tensor:
  """Stable 80% of the inverted-stair column, chosen once."""
  stored = getattr(terrain, _PIN_MASK_ATTR, None)
  if stored is not None:
    return stored
  mask = torch.zeros(
    terrain.terrain_levels.shape[0],
    dtype=torch.bool,
    device=terrain.terrain_levels.device,
  )
  names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
  if _INVERTED_STAIRS in names:
    column = names.index(_INVERTED_STAIRS)
    ids = (terrain.terrain_types == column).nonzero(as_tuple=False).flatten()
    n_pin = int(round(_HARD_INVERTED_STAIRS_FRACTION * ids.numel()))
    if n_pin > 0:
      mask[ids[:n_pin]] = True
  setattr(terrain, _PIN_MASK_ATTR, mask)
  return mask


def _refresh_terrain_level_stats(env, result: dict[str, torch.Tensor]) -> None:
  """Rewrite logged level means after the inverted-stair pin."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return
  levels = terrain.terrain_levels.float()
  if "mean" in result:
    result["mean"] = torch.mean(levels)
  if "max" in result:
    result["max"] = torch.max(levels)
  generator = terrain.cfg.terrain_generator
  if generator is None:
    return
  names = list(generator.sub_terrains.keys())
  if terrain.terrain_origins.shape[1] != len(names):
    return
  for index, name in enumerate(names):
    if name not in result:
      continue
    mask = terrain.terrain_types == index
    if mask.any():
      result[name] = torch.mean(levels[mask])
