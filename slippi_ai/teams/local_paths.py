"""
Resolve Dolphin folder + Melee ISO on this Windows PC.

Prefers PhillipTeams settings; falls back to the host-confirmed ISO path.
Accepts a path with or without the ``.iso`` extension.
"""

from __future__ import annotations

import json
from pathlib import Path

# Host-confirmed (2026-08-25). Extension optional in conversation; file has .iso.
DEFAULT_ISO = Path(
    r"C:\External HD Copy\ThumbDrive\ISOs"
    r"\Super Smash Bros. Melee (USA) (v1.02).iso"
)

PHILLIP_TEAMS_SETTINGS = (
    Path.home()
    / "AppData"
    / "Roaming"
    / "PhillipTeams"
    / "phillip_teams_settings.json"
)


def normalize_iso_path(raw: str | Path | None) -> Path | None:
  """Return an existing .iso Path, or None."""
  if raw is None:
    return None
  text = str(raw).strip().strip('"')
  if not text:
    return None
  path = Path(text)
  if path.is_file():
    return path
  # User often omits .iso
  if path.suffix.lower() != ".iso":
    with_ext = Path(str(path) + ".iso")
    if with_ext.is_file():
      return with_ext
  return None


def load_phillip_teams_settings() -> dict:
  if not PHILLIP_TEAMS_SETTINGS.is_file():
    return {}
  try:
    data = json.loads(PHILLIP_TEAMS_SETTINGS.read_text(encoding="utf-8-sig"))
  except (OSError, json.JSONDecodeError):
    return {}
  return data if isinstance(data, dict) else {}


def resolve_dolphin_folder(explicit: str | Path | None = None) -> Path | None:
  if explicit:
    p = Path(explicit)
    if p.is_file():
      return p.parent
    if p.is_dir():
      return p
  cfg = load_phillip_teams_settings()
  exe = (cfg.get("dolphin_path") or "").strip()
  if exe and Path(exe).is_file():
    return Path(exe).parent
  # Common Slippi Launcher netplay install
  candidates = [
      Path.home()
      / "AppData"
      / "Roaming"
      / "Slippi Launcher"
      / "netplay",
      Path.home()
      / "AppData"
      / "Local"
      / "Slippi Launcher"
      / "netplay",
  ]
  for folder in candidates:
    if (folder / "Slippi Dolphin.exe").is_file():
      return folder
  return None


def resolve_iso(explicit: str | Path | None = None) -> Path | None:
  hit = normalize_iso_path(explicit)
  if hit is not None:
    return hit
  cfg = load_phillip_teams_settings()
  hit = normalize_iso_path(cfg.get("iso_path"))
  if hit is not None:
    return hit
  return normalize_iso_path(DEFAULT_ISO)


def resolve_paths(
    *,
    dolphin: str | Path | None = None,
    iso: str | Path | None = None,
) -> tuple[Path, Path]:
  """
  Raise SystemExit-friendly errors if missing.

  Returns (dolphin_folder, iso_path).
  """
  folder = resolve_dolphin_folder(dolphin)
  iso_path = resolve_iso(iso)
  if folder is None:
    raise FileNotFoundError(
        "Dolphin folder not found. Set PhillipTeams dolphin_path or pass --path."
    )
  if iso_path is None:
    raise FileNotFoundError(
        "Melee ISO not found. Expected:\n  "
        + str(DEFAULT_ISO)
        + "\n(or set PhillipTeams iso_path / pass --iso)."
    )
  return folder, iso_path


def describe_paths() -> str:
  try:
    folder, iso = resolve_paths()
    return f"dolphin={folder}\niso={iso}"
  except FileNotFoundError as e:
    return f"UNRESOLVED: {e}"
