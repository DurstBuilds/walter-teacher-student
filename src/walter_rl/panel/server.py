"""Local browser panel for playback and training."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from walter_rl.panel.catalog import catalog, list_runs
from walter_rl.panel.commands import (
  PanelError,
  play_command,
  summary_text,
  train_command,
)
from walter_rl.panel.jobs import JobBusyError, JobManager
from walter_rl.panel.paths import repo_root

_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_MAX_BODY = 65_536

JOBS = JobManager(repo_root())


class _PanelServer(ThreadingHTTPServer):
  """Serve the panel and its API on one thread per request."""

  daemon_threads = True
  allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
  """Dispatch the panel page and the JSON API."""

  server_version = "WaLTERPanel/0.1"

  def do_GET(self) -> None:
    """Handle a GET route."""
    self._route("GET")

  def do_POST(self) -> None:
    """Handle a POST route."""
    self._route("POST")

  def log_message(self, fmt: str, *args: object) -> None:
    """Skip the job-status poll so the terminal stays readable."""
    if self.command == "GET" and urlparse(self.path).path == "/api/job":
      return
    super().log_message(fmt, *args)

  def _route(self, method: str) -> None:
    """Run one route, or return an error the page can show."""
    path = urlparse(self.path).path
    route = _ROUTES.get((method, path))
    if route is None:
      self.send_json(404, {"error": "Not found."})
      return
    try:
      route(self)
    except PanelError as exc:
      self.send_json(400, {"error": str(exc)})
    except JobBusyError as exc:
      self.send_json(409, {"error": str(exc)})

  def read_json(self) -> dict:
    """Return the request body as an object."""
    return _read_json(self)

  def send_json(self, status: int, payload: dict) -> None:
    """Write one JSON response."""
    body = json.dumps(payload).encode()
    _send_bytes(self, status, body, "application/json")


def serve(port: int = _DEFAULT_PORT) -> None:
  """Listen on localhost until the process is interrupted."""
  try:
    httpd = _PanelServer((_HOST, port), Handler)
  except OSError as exc:
    raise SystemExit(f"Could not listen on {_HOST}:{port}. {exc}") from exc
  print(f"WaLTER panel at http://{_HOST}:{port}", flush=True)
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    print("\nStopped.")
  finally:
    JOBS.stop()
    httpd.server_close()


def _favicon(handler: Handler) -> None:
  """Answer the browser's icon request with an empty response."""
  handler.send_response(204)
  handler.end_headers()


def _page(handler: Handler) -> None:
  """Serve the control panel."""
  page = Path(__file__).with_name("index.html").read_bytes()
  _send_bytes(handler, 200, page, "text/html; charset=utf-8")


def _catalog_route(handler: Handler) -> None:
  """Serve scenes, policies, and defaults."""
  handler.send_json(200, catalog())


def _checkpoints(handler: Handler) -> None:
  """Serve saved checkpoints grouped by run."""
  handler.send_json(200, {"runs": list_runs(repo_root())})


def _job(handler: Handler) -> None:
  """Serve the current session."""
  handler.send_json(200, JOBS.snapshot())


def _play(handler: Handler) -> None:
  """Start playback."""
  _start(handler, "play", play_command)


def _train(handler: Handler) -> None:
  """Start a new or continued training run."""
  _start(handler, "train", train_command)


def _stop(handler: Handler) -> None:
  """Stop the current session."""
  JOBS.stop()
  handler.send_json(200, JOBS.snapshot())


def _start(handler: Handler, kind: str, build) -> None:
  """Validate a request and spawn its process."""
  body = handler.read_json()
  command = build(repo_root(), body)
  JOBS.start(kind, command, summary_text(body))
  handler.send_json(200, JOBS.snapshot())


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
  """Parse a small JSON object from the request body."""
  raw = handler.rfile.read(_content_length(handler))
  if not raw:
    raise PanelError("Expected a JSON body.")
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise PanelError("Expected JSON.") from exc
  if not isinstance(payload, dict):
    raise PanelError("Expected a JSON object.")
  return payload


def _content_length(handler: BaseHTTPRequestHandler) -> int:
  """Return a body size the server is willing to read."""
  raw = handler.headers.get("Content-Length", "0")
  try:
    length = int(raw)
  except ValueError as exc:
    raise PanelError("Expected a JSON body.") from exc
  if length < 0 or length > _MAX_BODY:
    raise PanelError("Request is too large.")
  return length


def _send_bytes(
  handler: BaseHTTPRequestHandler,
  status: int,
  body: bytes,
  content_type: str,
) -> None:
  """Write a response with an explicit length."""
  handler.send_response(status)
  handler.send_header("Content-Type", content_type)
  handler.send_header("Content-Length", str(len(body)))
  handler.send_header("Cache-Control", "no-store")
  handler.end_headers()
  handler.wfile.write(body)


_ROUTES = {
  ("GET", "/"): _page,
  ("GET", "/favicon.ico"): _favicon,
  ("GET", "/api/catalog"): _catalog_route,
  ("GET", "/api/checkpoints"): _checkpoints,
  ("GET", "/api/job"): _job,
  ("POST", "/api/play"): _play,
  ("POST", "/api/train"): _train,
  ("POST", "/api/job/stop"): _stop,
}
