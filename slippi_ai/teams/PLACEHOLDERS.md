# -*- coding: utf-8 -*-
"""Assemble overnight PLACEHOLDER answers for host review."""

PLACEHOLDERS_FOR_HOST = """
# Teams IQ — placeholders I filled while you were away

Confirm or change these when you're back. Code already uses them.

## Locked (you answered)
- Roles: seat1 aggro / seat2 support
- Teammate death: 0.85
- First chars: Fox dittos
- Floaty-at-bay: Puff, Peach, Luigi, Sheik, ICs, Zelda, Doc, Mario, Pikachu, Mewtwo, Yoshi
- Yoshi edge: Falco/Marth dair spike at ≤40% or ≥90% only
- Push wrappers: yes

## Placeholders (foresight — change if wrong)

| ID | Assumed | Why |
|----|---------|-----|
| stage | Final Destination | simplest blastzones |
| opponent_mode | frozen_teacher on team B | stable while A learns |
| policy_mode | hybrid_1v1 (medium-v2 via compat) | works before 4p embed |
| focus | dynamic via focus.pick_focus | mirrors doubles_target |
| hours | 3 | evening GPU budget |
| ports | 1+2 vs 3+4 | standard |
| Teams CSS | pulse Y; fallback port teams | libmelee has no Teams helper |

## What got built this session
- teams/embed.py — numpy 4-player feature concat (debug size reference)
- teams/trajectory.py — TeamsTrajectory + update_teams_rewards
- teams/learner_bridge.py — drop-in for jax.rl.update_rewards
- teams/config.py + run_lib.py — dry-runnable RL config
- teams/focus.py — opp0/opp1 picker aligned with wrappers
- dolphin.enable_teams + TeamsEnvironment (earlier)

## Still incomplete (intentional)
- JAX EmbedConfig for TeamsGame
- RolloutWorker emitting TeamsTrajectory
- Actual GPU training loop
"""
