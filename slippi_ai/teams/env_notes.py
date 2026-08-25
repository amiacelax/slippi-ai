"""
Env status for 4-player Teams.

Implemented:
  - ``TeamsEnvironment`` (4 players, ego-centric TeamsGame obs)
  - ``dolphin.Dolphin(enable_teams=True)`` CSS Y-toggle (best-effort)
  - Port-convention fallback if ``is_teams`` never flips

Still needed for PPO:
  - jax.rl.run_lib agent_kwargs for ports 1–4
  - learner.update_rewards → teams.rollout.rewards_from_frames
  - Policy embed for partner + 2 opps (see observe.py)
"""

from __future__ import annotations

REQUIRED_PLAYERS = 4
DEFAULT_TEAM_A = (1, 2)
DEFAULT_TEAM_B = (3, 4)


def env_ready() -> bool:
    """True once TeamsEnvironment exists (Dolphin path still required at runtime)."""
    return True


def blocking_reasons() -> list[str]:
  return [
      "TeamsEnhancedEmbedModule compile needs JAX training env",
      "Live actor exists (step4) but PPO/GPU overnight not run",
      "Teams CSS toggle: is_teams often False on Slippi netplay (port fallback OK)",
  ]
