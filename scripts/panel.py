"""Open the WaLTER play and train panel in a browser.

Example:
  python scripts/panel.py
"""

from __future__ import annotations

import argparse

from walter_rl.panel.server import serve


def main() -> None:
  """Serve the panel on localhost."""
  args = _parse_args()
  serve(port=args.port)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()
  if args.port < 1 or args.port > 65535:
    parser.error("--port must be between 1 and 65535")
  return args


if __name__ == "__main__":
  main()
