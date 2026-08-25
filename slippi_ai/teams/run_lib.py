"""
Teams RL run loop — scaffold.

Not a full jax.rl replacement yet. This module:
  1. Builds TeamsRLConfig / TeamsEnvironment kwargs
  2. Documents the hybrid training loop
  3. Exposes ``run_dry`` that validates config + imports without Dolphin/ISO

When ISO/Dolphin/GPU exist, ``run`` will:
  - spawn TeamsEnvironment(4 Fox)
  - for each train port: collect TeamsTrajectory
  - update_teams_rewards
  - policy step via compat_1v1 → medium-v2 (hybrid) OR future Teams embed
  - PPO update (wired later into jax.rl.learner)
"""

from __future__ import annotations

import logging
from typing import Optional

from slippi_ai.teams.config import PLACEHOLDER_DECISIONS, TeamsRLConfig
from slippi_ai.teams.env_notes import blocking_reasons
from slippi_ai.teams.layout import default_seat_ports
from slippi_ai.teams.observe import describe_migration
from slippi_ai.teams.trajectory import role_for_port


def build_dolphin_kwargs(
    config: TeamsRLConfig,
    *,
    path: Optional[str],
    iso: Optional[str],
    headless: bool = True,
) -> dict:
    return {
        "path": path,
        "iso": iso,
        "players": config.dolphin_players(),
        "stage": config.stage,
        "headless": headless,
        "enable_teams": True,
        "infinite_time": True,
    }


def describe_hybrid_loop(config: TeamsRLConfig) -> str:
    ports = default_seat_ports(1)
    lines = [
        "Hybrid Teams RL loop (PLACEHOLDER policy_mode=hybrid_1v1_teacher):",
        f"  ports: {ports.as_tuple()}",
        f"  train_ports: {config.train_ports}",
        f"  roles: "
        + ", ".join(f"{p}={role_for_port(p)}" for p in config.train_ports),
        f"  opponent_mode: {config.opponent_mode}",
        f"  focus enemy for 1v1 teacher: {config.focus_opponent}",
        "  each step:",
        "    1. TeamsEnvironment.step(controllers)",
        "    2. TeamsGame per ego seat",
        "    3. Game = teams_game_to_1v1(TeamsGame, focus)",
        "    4. medium-v2 policy.sample(Game)",
        "    5. append to TeamsTrajectory",
        "    6. update_teams_rewards(traj, reward_config_for_role)",
        "    7. PPO on advantages from Teams rewards (TODO wire learner)",
    ]
    return "\n".join(lines)


def run_dry(config: TeamsRLConfig | None = None) -> int:
    """Validate config without launching Dolphin. Returns 0 on success."""
    config = config or TeamsRLConfig()
    logging.info("Teams RL dry-run")
    logging.info(describe_hybrid_loop(config))
    logging.info("Migration:\n%s", describe_migration())
    logging.info("Placeholders:")
    for p in PLACEHOLDER_DECISIONS:
        logging.info("  [%s] assumed=%s — %s", p["id"], p["assumed"], p["why"])
    blockers = blocking_reasons()
    logging.info("Blockers for real GPU run:")
    for b in blockers:
        logging.info("  - %s", b)

    # Sanity: reward configs construct
    for role in ("aggro", "support"):
        cfg = config.reward_config_for_role(role)
        assert cfg.teammate_death_penalty == config.teammate_death_penalty
        assert cfg.role == role

    players = config.dolphin_players()
    assert set(players) == {1, 2, 3, 4}
    print("teams run_dry OK")
    print(describe_hybrid_loop(config))
    return 0


def run(
    config: TeamsRLConfig,
    *,
    path: Optional[str] = None,
    iso: Optional[str] = None,
) -> int:
    """
    Full run — PLACEHOLDER.

    Raises until Dolphin path/ISO provided and PPO wired.
    """
    if not path or not iso:
        print("Missing --path / --iso. Running dry-run instead.")
        return run_dry(config)

    blockers = blocking_reasons()
    print("Dolphin kwargs ready, but PPO still blocked:")
    for b in blockers:
        print(f"  - {b}")
    print("Next: wire TeamsTrajectory into jax.rl.learner.update_rewards path.")
    # Import env to prove it constructs once path/iso exist — don't auto-launch
    # overnight without user watching (can open windows / burn CPU).
    _ = build_dolphin_kwargs(config, path=path, iso=iso)
    return 2
