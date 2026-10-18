"""Locations the panel reads and the processes it starts."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
  """Return the gen0WaLTER checkout that contains this package."""
  return Path(__file__).resolve().parents[3]
