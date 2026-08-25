"""
Step 3 demo — trainer plug-in on paper (no GPU / no Dolphin required).

  cd C:\\Users\\langu\\slippi-ai
  python -m slippi_ai.teams.step3_demo
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(r"C:\Users\langu\slippi-ai")


def _ensure_path() -> None:
  if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_local_dolphin_iso() -> dict:
  """Read PhillipTeams settings; report whether a live menu check can run."""
  settings = Path.home() / "AppData" / "Roaming" / "PhillipTeams" / (
      "phillip_teams_settings.json"
  )
  out = {
      "settings": str(settings),
      "dolphin_exe": "",
      "dolphin_folder": "",
      "iso": "",
      "ready": False,
  }
  if not settings.is_file():
    print("Dolphin/ISO: no PhillipTeams settings found")
    return out

  import json

  data = json.loads(settings.read_text(encoding="utf-8-sig"))
  exe = (data.get("dolphin_path") or "").strip()
  iso = (data.get("iso_path") or "").strip()
  out["dolphin_exe"] = exe
  out["iso"] = iso
  if exe:
    out["dolphin_folder"] = str(Path(exe).parent)
  exe_ok = bool(exe) and Path(exe).is_file()
  iso_ok = bool(iso) and Path(iso).is_file()
  out["ready"] = exe_ok and iso_ok
  print(
      f"Dolphin/ISO: exe={'OK' if exe_ok else 'MISSING'}  "
      f"iso={'OK' if iso_ok else 'MISSING'}  ready={out['ready']}"
  )
  if out["ready"]:
    print(f"  dolphin folder: {out['dolphin_folder']}")
    print(f"  iso: {iso}")
  return out


def main() -> int:
  _ensure_path()
  print("=== Teams IQ Step 3: trainer wiring (paper) ===")
  print()

  from slippi_ai.teams.trainer_bridge import describe_wiring, simulate_learner_inplace

  print(describe_wiring())
  print("--- one fake learner step ---")
  summary = simulate_learner_inplace(num_steps=16)
  plan = summary["plan"]
  print(
      f"plan: mode={plan.policy_mode} hours={plan.hours} "
      f"ports={plan.train_ports} skip_1v1_reward={plan.skip_stock_reward_recompute}"
  )
  print("timings:", summary["timings"])
  print("reward means:", summary["reward_means"])
  assert summary["states_are_1v1"]
  assert summary["ppo_ready_shape"]
  print("PPO-shaped trajectories: OK (1v1 states + Teams rewards)")
  print()

  paths = check_local_dolphin_iso()
  print()
  if paths["ready"]:
    print("STEP 3 OK - trainer hooks ready; Dolphin/ISO found on this PC.")
    print("Next local check:  python -m slippi_ai.teams.step4_demo")
    print("(opens Dolphin - live hybrid actor; see STEP4.md)")
  else:
    print("STEP 3 OK - trainer hooks ready; Dolphin/ISO not both found.")
    print("Skip live actor, or set paths in PhillipTeams settings.")
  print("GPU train still waits until you rent a box.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
