"""
Step 4 demo — live Dolphin hybrid actor (free polish, no GPU train).

Opens a Dolphin window, menus into a 4-Fox match, rolls out a short
trajectory with Teams IQ rewards + 1v1-projected states.

  cd C:\\Users\\langu\\slippi-ai
  python -m slippi_ai.teams.step4_demo
  python -m slippi_ai.teams.step4_demo --steps 32
  python -m slippi_ai.teams.step4_demo --iso "C:\\External HD Copy\\ThumbDrive\\ISOs\\Super Smash Bros. Melee (USA) (v1.02)"
"""

from __future__ import annotations

import sys
from pathlib import Path

from absl import app, flags


ROOT = Path(r"C:\Users\langu\slippi-ai")

FLAGS = flags.FLAGS
flags.DEFINE_integer("steps", 48, "Rollout length (frames of reward).")
flags.DEFINE_string("path", None, "Dolphin folder (optional; auto-detect).")
flags.DEFINE_string(
    "iso",
    None,
    "Melee ISO path (optional; .iso extension optional).",
)
flags.DEFINE_bool("fake", False, "Use FakeTeamsEnv instead of live Dolphin.")


def _ensure_path() -> None:
  if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(_argv) -> None:
  _ensure_path()
  print("=== Teams IQ Step 4: live Dolphin hybrid actor ===")
  print()

  from slippi_ai.teams.config import TeamsRLConfig
  from slippi_ai.teams.hybrid_worker import (
      build_hybrid_teams_actor,
      build_live_teams_actor,
  )
  from slippi_ai.teams.local_paths import describe_paths, resolve_paths
  from slippi_ai.teams.trainer_bridge import preserve_hybrid_rewards

  if FLAGS.fake:
    print("mode: FAKE (no Dolphin)")
    worker = build_hybrid_teams_actor(TeamsRLConfig())
  else:
    print("Resolved paths:")
    try:
      folder, iso = resolve_paths(dolphin=FLAGS.path, iso=FLAGS.iso)
      print(f"  dolphin: {folder}")
      print(f"  iso:     {iso}")
    except FileNotFoundError as e:
      print(e)
      print("Tip: pass --fake to dry-run without Dolphin.")
      raise SystemExit(2)
    print()
    print("Close other Dolphin windows. A match window will open briefly...")
    worker = build_live_teams_actor(
        TeamsRLConfig(),
        path=FLAGS.path,
        iso=FLAGS.iso,
        headless=False,
    )

  worker.start()
  try:
    trajs, timings = worker.rollout(FLAGS.steps)
  finally:
    worker.stop()

  kept = preserve_hybrid_rewards(trajs)
  print("timings:", timings)
  for port, traj in trajs.items():
    r = traj.rewards[:, 0]
    print(
        f"  port {port}: T={traj.states.p0.x.shape[0]} "
        f"reward_mean={float(r.mean()):.4f} "
        f"min={float(r.min()):.4f}"
    )
    assert hasattr(traj.states, "p0") and hasattr(traj.states, "p1")

  assert len(kept) == len(trajs)
  print()
  if FLAGS.fake:
    print("STEP 4 OK (fake) - hybrid shape still good.")
  else:
    print("STEP 4 OK (live) - Dolphin TeamsEnvironment fed the hybrid actor.")
    if not timings.get("is_teams"):
      print("Note: is_teams=False; using port teams 1+2 vs 3+4 (expected).")
  print("Still no overnight learn - that needs a rented GPU.")


if __name__ == "__main__":
  app.run(main)
