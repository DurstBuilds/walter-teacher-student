"""One playback or training process at a time."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

_LOG_LIMIT = 200
_STOP_WAIT_S = 5
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_URL = re.compile(r"https?://[^\s<>\"]+")


class JobBusyError(Exception):
  """Raised when a second play or train request arrives."""


class JobManager:
  """Spawn, watch, and stop the single child process."""

  def __init__(self, cwd: Path) -> None:
    """Watch processes started in ``cwd``."""
    self._cwd = cwd
    self._lock = threading.Lock()
    self._process: subprocess.Popen[str] | None = None
    self._kind: str | None = None
    self._summary = ""
    self._state = "idle"
    self._lines: deque[str] = deque(maxlen=_LOG_LIMIT)

  def start(self, kind: str, command: list[str], summary: str) -> None:
    """Spawn one play or train process."""
    with self._lock:
      if self._state == "running":
        raise JobBusyError("Stop the current session before starting another.")
      process = _spawn(command, self._cwd)
      self._process = process
      self._kind = kind
      self._summary = summary
      self._state = "running"
      self._lines.clear()
    thread = threading.Thread(target=self._read_output, args=(process,), daemon=True)
    thread.start()

  def stop(self) -> None:
    """End the running process group."""
    with self._lock:
      process = self._process
      if process is None or self._state != "running":
        return
      self._state = "stopped"
    _terminate_group(process)

  def snapshot(self) -> dict:
    """Return the state the page polls."""
    with self._lock:
      lines = list(self._lines)
      state = self._state
      kind = self._kind
      summary = self._summary
      return_code = None if self._process is None else self._process.poll()
    viewer_url, wandb_url = _links(lines)
    if state == "running":
      return_code = None
    return {
      "state": state,
      "kind": kind,
      "summary": summary,
      "lines": lines,
      "viewer_url": viewer_url,
      "wandb_url": wandb_url,
      "return_code": return_code,
    }

  def _read_output(self, process: subprocess.Popen[str]) -> None:
    """Copy the child output into the log tail."""
    stdout = process.stdout
    if stdout is None:
      return
    for raw in stdout:
      line = _ANSI.sub("", raw).rstrip()
      with self._lock:
        self._lines.append(line)
    code = process.wait()
    with self._lock:
      if self._process is not process or self._state != "running":
        return
      self._state = "finished" if code == 0 else "failed"


def _spawn(command: list[str], cwd: Path) -> subprocess.Popen[str]:
  """Start a process in its own session so Stop can signal the group."""
  return subprocess.Popen(
    command,
    cwd=cwd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    text=True,
    encoding="utf-8",
    errors="replace",
  )


def _terminate_group(process: subprocess.Popen[str]) -> None:
  """Ask the process group to exit, then kill it if it stays up."""
  _signal_group(process, signal.SIGTERM)
  try:
    process.wait(timeout=_STOP_WAIT_S)
  except subprocess.TimeoutExpired:
    _signal_group(process, signal.SIGKILL)
    process.wait(timeout=_STOP_WAIT_S)


def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
  """Signal the child's process group. Ignore a process that already exited."""
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    return


def _links(lines: list[str]) -> tuple[str | None, str | None]:
  """Pick the viewer URL and the Weights & Biases URL out of the log."""
  viewer_url = None
  wandb_url = None
  for line in lines:
    for raw in _URL.findall(line):
      url = raw.rstrip(").,]>\"'")
      if "wandb.ai" in url:
        wandb_url = wandb_url or url
      elif viewer_url is None:
        viewer_url = url
  return viewer_url, wandb_url
