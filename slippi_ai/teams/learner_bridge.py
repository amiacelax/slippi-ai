"""
Monkeypatch / adapter notes for jax.rl.learner.

Stock path:
  update_rewards(traj, RewardConfig) -> compute_rewards(Game)

Teams hybrid path (Step 3 paper):
  HybridTeamsRolloutWorker already fills Trajectory.rewards with Teams IQ.
  Call trainer_bridge.preserve_hybrid_rewards so PPO does not wipe them
  with 1v1 update_rewards.

Full TeamsGame path (later):
  update_teams_rewards / learner_reward_hook on TeamsTrajectory.
"""

from __future__ import annotations

from slippi_ai.teams.reward import TeamsRewardConfig
from slippi_ai.teams.trajectory import TeamsTrajectory, update_teams_rewards


def learner_reward_hook(
    trajectory: TeamsTrajectory,
    reward_config: TeamsRewardConfig,
    *,
    role: str = "aggro",
) -> TeamsTrajectory:
  """Drop-in analogue of jax.rl.learner.update_rewards for Teams."""
  return update_teams_rewards(trajectory, reward_config, role=role)


INTEGRATION_CHECKLIST = [
    "DONE (paper): build_actor -> build_hybrid_teams_actor / trainer_bridge",
    "DONE (paper): preserve_hybrid_rewards skips 1v1 update_rewards wipe",
    "DONE (live): build_live_teams_actor -> TeamsEnvironment + Dolphin",
    "Trajectory.states type becomes TeamsGame when full embed lands",
    "Replace learner.update_rewards with learner_reward_hook (full Teams states)",
    "Policy.unroll still receives Game via teams_game_to_1v1 until embed lands",
    "Log partner deaths / 2v1 metrics in training dashboard",
]
