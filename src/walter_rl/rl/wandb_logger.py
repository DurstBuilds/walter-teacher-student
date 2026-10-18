"""Weights & Biases logger for WaLTER PPO runs."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Any

import wandb
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from torch.utils.tensorboard import SummaryWriter

WANDB_PROJECT = "walter-rl"


class WalterWandbLogWriter(WandbLogWriter):
  """Log scalars and checkpoints, and store configs W&B can serialize.

  The stock writer passes ``start_method="thread"`` into ``wandb.Settings``,
  which current W&B rejects. It also uploads the raw environment config,
  including reward callables that W&B cannot store.

  rsl-rl also logs reward and episode length against elapsed seconds unless
  the writer class is named ``WandbLogWriter``. Those rows jump W&B's step
  past the training iteration, and every later sample is dropped. This writer
  keeps the seconds series in TensorBoard only and uploads one row per
  iteration, plotted against ``iteration``.
  """

  def __init__(self, log_dir: str, project_name: str) -> None:
    """Open a TensorBoard file and a W&B run for this log directory."""
    SummaryWriter.__init__(self, log_dir, flush_secs=10)
    wandb.init(
      project=project_name,
      entity=os.environ.get("WANDB_USERNAME"),
      name=os.path.basename(log_dir),
      config={"log_dir": log_dir},
    )
    wandb.define_metric("iteration")
    self.logged_videos: set[str] = set()
    self._pending: dict[str, float] = {}
    self._pending_step: int | None = None
    self._axes: set[str] = set()

  def add_scalar(
    self,
    tag: str,
    scalar_value: float,
    global_step: int | None = None,
    walltime: float | None = None,
    new_style: bool = False,
  ) -> None:
    """Buffer one scalar. The iteration is uploaded when the step changes."""
    SummaryWriter.add_scalar(
      self,
      tag,
      scalar_value,
      global_step=global_step,
      walltime=walltime,
      new_style=new_style,
    )
    # rsl-rl uses a ``/time`` suffix for the elapsed-seconds axis. That step
    # is ahead of the iteration number, so it stays out of the W&B row.
    if tag.endswith("/time"):
      return
    step = 0 if global_step is None else int(global_step)
    if self._pending and step != self._pending_step:
      self._flush_scalars()
    self._pending_step = step
    self._pending[tag] = float(scalar_value)
    self._remember_axis(tag)

  def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
    """Upload a JSON-safe copy of the environment and training configs."""
    wandb.config.update(
      {
        "train_cfg": _jsonable(train_cfg),
        "env_cfg": _jsonable(env_cfg),
      }
    )

  def save_video(self, video: pathlib.Path, it: int) -> None:
    """Upload a video once, on the same iteration axis as the scalars."""
    self._flush_scalars()
    if video.name in self.logged_videos:
      return
    self._remember_axis("video")
    wandb.log(
      {"video": wandb.Video(str(video), format="mp4"), "iteration": it}
    )
    self.logged_videos.add(video.name)

  def stop(self) -> None:
    """Upload the last iteration and finish the W&B run."""
    self._flush_scalars()
    super().stop()

  def _remember_axis(self, tag: str) -> None:
    """Plot this metric against ``iteration``."""
    pattern = f"{tag.split('/', 1)[0]}/*" if "/" in tag else tag
    if pattern in self._axes:
      return
    self._axes.add(pattern)
    wandb.define_metric(pattern, step_metric="iteration")

  def _flush_scalars(self) -> None:
    """Upload the buffered iteration as one W&B row."""
    if not self._pending or self._pending_step is None:
      return
    payload: dict[str, float] = dict(self._pending)
    payload["iteration"] = self._pending_step
    wandb.log(payload)
    self._pending.clear()


def _jsonable(value: Any) -> Any:
  """Turn configs into values ``wandb.config`` can store."""
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return _jsonable(dataclasses.asdict(value))
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return str(value)
