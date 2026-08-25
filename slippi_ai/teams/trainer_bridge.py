"""
Step 3 scaffolding — plug Teams into the jax.rl trainer *on paper*.

Stock jax.rl path (simplified):
  build_actor = lambda: evaluators.RolloutWorker(...)
  trajs, _ = actor.rollout(N)
  trajs = [update_rewards(t, RewardConfig) for t in trajs]  # 1v1 damage/death
  learner.ppo(trajs, ...)

Teams hybrid path (this module):
  build_actor = build_hybrid_teams_actor   # already returns Teams rewards
  trajs, _ = actor.rollout(N)
  trajs = preserve_hybrid_rewards(trajs)  # do NOT recompute with 1v1 RewardConfig
  # learner.ppo(...)  — same Trajectory shape; GPU box later

Full Teams embed path (later, after GPU):
  build_actor uses TeamsEnvironment + TeamsEnhancedEmbedModule
  update_rewards → learner_bridge.learner_reward_hook (TeamsGame states)
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from slippi_ai.evaluators import AbstractRolloutWorker, Trajectory
from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.hybrid_worker import build_hybrid_teams_actor
from slippi_ai.teams.learner_bridge import INTEGRATION_CHECKLIST, learner_reward_hook

Port = int


# Exact swap points relative to slippi_ai.jax.rl.run_lib / train_two_lib.
WIRING_MAP = (
    (
        "build_actor",
        "jax.rl.run_lib ~line 501 RolloutWorker(...)",
        "build_hybrid_teams_actor / build_live_teams_actor(TeamsRLConfig)",
    ),
    (
        "update_rewards",
        "jax.rl.learner.ppo -> update_rewards(t, RewardConfig)",
        "preserve_hybrid_rewards(trajs)  # keep Teams IQ rewards",
    ),
    (
        "ports",
        "PORT=1, ENEMY_PORT=2 (1v1)",
        "train_ports=(1, 2); opponents (3, 4) frozen teacher",
    ),
    (
        "dolphin players",
        "2 players in dolphin_kwargs",
        "TeamsRLConfig.dolphin_players() -> 4 AIs + enable_teams",
    ),
    (
        "embed (later)",
        "EnhancedEmbedModule Game(p0,p1)",
        "TeamsEnhancedEmbedModule + warm_start (Step 2; needs JAX)",
    ),
)


@dataclass(frozen=True)
class TeamsTrainerPlan:
  """Paper plan for one GPU training evening — no money spent yet."""

  policy_mode: str = "hybrid_1v1_teacher"
  hours: float = 3.0
  train_ports: tuple[int, ...] = (1, 2)
  opponent_mode: str = "frozen_teacher"
  skip_stock_reward_recompute: bool = True
  use_teams_embed: bool = False  # True only after JAX compile on GPU box


def default_plan(config: TeamsRLConfig | None = None) -> TeamsTrainerPlan:
  cfg = config or TeamsRLConfig()
  return TeamsTrainerPlan(
      policy_mode=cfg.policy_mode,
      hours=cfg.hours,
      train_ports=tuple(cfg.train_ports),
      opponent_mode=cfg.opponent_mode,
      skip_stock_reward_recompute=True,
      use_teams_embed=(cfg.policy_mode == "full_teams_embed"),
  )


def build_teams_actor(
    config: TeamsRLConfig | None = None,
    *,
    num_envs: int = 1,
    live: bool = False,
    path: str | None = None,
    iso: str | None = None,
) -> AbstractRolloutWorker:
  """Drop-in for jax.rl ``build_actor`` (fake by default; live Dolphin optional)."""
  if live:
    from slippi_ai.teams.hybrid_worker import build_live_teams_actor

    return build_live_teams_actor(
        config, num_envs=num_envs, path=path, iso=iso
    )
  return build_hybrid_teams_actor(config, num_envs=num_envs)


def preserve_hybrid_rewards(
    trajectories: tp.Mapping[Port, Trajectory] | list[Trajectory],
) -> list[Trajectory]:
  """
  Teams hybrid workers already put IQ rewards on Trajectory.rewards.

  Stock learner.ppo would overwrite them via 1v1 update_rewards — call this
  instead (or gate update_rewards when mode=hybrid_teams).
  """
  if isinstance(trajectories, dict):
    return [trajectories[p] for p in sorted(trajectories)]
  return list(trajectories)


def describe_wiring() -> str:
  lines = ["Teams <-> jax.rl wiring (paper):", ""]
  for name, stock, teams in WIRING_MAP:
    lines.append(f"[{name}]")
    lines.append(f"  stock:  {stock}")
    lines.append(f"  teams:  {teams}")
    lines.append("")
  lines.append("Integration checklist:")
  for item in INTEGRATION_CHECKLIST:
    lines.append(f"  - {item}")
  lines.append("")
  lines.append(f"learner_reward_hook: {learner_reward_hook.__name__}")
  return "\n".join(lines)


def simulate_learner_inplace(
    config: TeamsRLConfig | None = None,
    *,
    num_steps: int = 16,
) -> dict:
  """
  One fake trainer step: build_actor → rollout → preserve rewards.

  Does not call real PPO (needs JAX/GPU). Returns summary metrics.
  """
  plan = default_plan(config)
  actor = build_teams_actor(config)
  actor.start()
  traj_map, timings = actor.rollout(num_steps)
  actor.stop()
  trajs = preserve_hybrid_rewards(traj_map)

  means = {port: float(traj_map[port].rewards.mean()) for port in traj_map}
  return {
      "plan": plan,
      "timings": timings,
      "ports": list(traj_map),
      "reward_means": means,
      "num_trajectories": len(trajs),
      "states_are_1v1": all(hasattr(t.states, "p0") for t in trajs),
      "ppo_ready_shape": all(
          t.rewards.shape[0] == num_steps and t.states.p0.x.shape[0] == num_steps + 1
          for t in trajs
      ),
  }
