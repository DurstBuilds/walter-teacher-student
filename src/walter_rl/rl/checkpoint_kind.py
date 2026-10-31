"""Tell state-machine and teacher checkpoints apart from their log path."""

from __future__ import annotations

from pathlib import Path

_TEACHER_RUN_MARKER = "walter_fs_teacher"


def is_teacher_checkpoint(path: Path | str) -> bool:
  """True when the checkpoint came from a ``walter_fs_teacher`` training run."""
  return _TEACHER_RUN_MARKER in Path(path).as_posix()
