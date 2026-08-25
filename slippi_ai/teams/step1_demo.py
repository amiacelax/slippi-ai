"""
Step 1 demo — run on this Windows PC (no Dolphin / GPU).

Proves: FakeTeamsEnv → collector → Teams rewards inside Trajectory.

  cd C:\\Users\\langu\\slippi-ai
  python -m slippi_ai.teams.step1_demo
"""

from __future__ import annotations

import sys


def main() -> int:
    # Ensure local package
    if r"C:\Users\langu\slippi-ai" not in sys.path:
        sys.path.insert(0, r"C:\Users\langu\slippi-ai")

    from slippi_ai.teams.config import TeamsRLConfig
    from slippi_ai.teams.hybrid_worker import build_hybrid_teams_actor

    print("=== Teams IQ Step 1: plug collector into rollout worker ===")
    print("Using FakeTeamsEnv (no Dolphin). Teams rewards replace 1v1 rewards.")
    print()

    worker = build_hybrid_teams_actor(TeamsRLConfig(), num_envs=1)
    worker.start()
    trajectories, timings = worker.rollout(32)
    worker.stop()

    print("timings:", timings)
    for port, traj in trajectories.items():
        r = traj.rewards[:, 0]
        print(
            f"  port {port}: states T={traj.states.p0.x.shape[0]} "
            f"rewards mean={float(r.mean()):.4f} "
            f"min={float(r.min()):.4f} "
            f"(partner dies ~frame 20 -> big negative blip)"
        )
        # Show that states are 1v1-shaped (p0/p1 only)
        assert hasattr(traj.states, "p0") and hasattr(traj.states, "p1")
        assert not hasattr(traj.states, "ego")

    print()
    print("STEP 1 OK - worker returns Trajectory with Teams rewards + 1v1 states.")
    print("Next: python -m slippi_ai.teams.step2_demo  (see STEP2.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
