"""
Local smoke: open Dolphin, menu into a match, report is_teams.

Uses local_paths (PhillipTeams settings + host ISO).
Opens a real window — close other Dolphin instances first.

  python -m slippi_ai.teams.check_teams_menu
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from absl import app, flags


ROOT = Path(r"C:\Users\langu\slippi-ai")

FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "Dolphin folder (optional).")
flags.DEFINE_string("iso", None, "Melee ISO (optional; .iso optional).")


def _ensure_path() -> None:
  if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(_argv) -> None:
  _ensure_path()
  from slippi_ai.teams.local_paths import resolve_paths

  path, iso = resolve_paths(dolphin=FLAGS.path, iso=FLAGS.iso)
  print("=== Teams CSS / is_teams smoke check ===")
  print(f"dolphin folder: {path}")
  print(f"iso: {iso}")
  print("Opening Dolphin (close other instances). Waiting for in-game...")
  print()

  import melee

  from slippi_ai import dolphin as dolphin_lib
  from slippi_ai.teams.env import make_four_ai_players

  players = make_four_ai_players(melee.Character.FOX)
  dol = dolphin_lib.Dolphin(
      path=str(path),
      iso=str(iso),
      players=players,
      stage=melee.Stage.FINAL_DESTINATION,
      headless=False,
      render=True,
      enable_teams=True,
      infinite_time=True,
      console_timeout=45.0,
  )

  t0 = time.time()
  gs = None
  err: str | None = None
  try:
    gs = dol.step()
  except Exception as e:  # noqa: BLE001
    err = f"{type(e).__name__}: {e}"
  finally:
    y_pulses = getattr(dol, "_teams_toggle_frames", None)
    try:
      dol.stop()
    except Exception:  # noqa: BLE001
      pass

  elapsed = time.time() - t0
  print(f"elapsed_s={elapsed:.1f}  y_toggle_pulses={y_pulses}")
  if err:
    print(f"error: {err}")
    print("TEAMS MENU: FAIL")
    raise SystemExit(1)

  assert gs is not None
  is_teams = bool(getattr(gs, "is_teams", False))
  n_players = len(getattr(gs, "players", {}) or {})
  frame = getattr(gs, "frame", None)
  print(f"in_game_frame={frame}  players={n_players}  is_teams={is_teams}")

  if is_teams:
    print("TEAMS MENU: ON")
    print("Y-toggle worked (gamestate.is_teams=True).")
    return

  print("TEAMS MENU: OFF (port fallback OK)")
  print(
      "is_teams stayed False - training can still use ports 1+2 vs 3+4. "
      "Not a blocker."
  )


if __name__ == "__main__":
  app.run(main)
