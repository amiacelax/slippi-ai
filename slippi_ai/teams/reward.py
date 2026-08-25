"""
Teams (2v2) reward — cooperative within team, competitive vs other team.

Unlike stock ``reward.compute_rewards`` (zero-sum p0 vs p1), this scores one
ego seat given partner + two opponents.

A-tier terms (always on by default):
  - own death / damage (1v1 baseline)
  - teammate death / damage (watch teammate)
  - prefer damaging / KOing the on-stage enemy over sharking (2v1 on stage)
  - role bias: aggro seat approaches enemies; support seat stays nearer partner

B/C terms are gated by CurriculumWeights so we can turn them up after A works.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np

from slippi_ai import reward as reward_1v1
from slippi_ai.teams.curriculum import (
    CurriculumWeights,
    FLOATY_OR_SLOW_CHARS,
    default_curriculum,
)
from slippi_ai.types import Player


@dataclasses.dataclass
class TeamsRewardConfig:
    damage_ratio: float = 0.01
    teammate_death_penalty: float = 0.85  # A: watch teammate (almost own death)
    teammate_damage_ratio: float = 0.006
    enemy_death_reward: float = 1.0
    enemy_damage_ratio: float = 0.01
    # Prefer converting the on-stage body (2v1) over both chasing offstage.
    offstage_enemy_ko_scale: float = 0.35
    onstage_enemy_ko_scale: float = 1.0
    stalling_penalty: float = 0.5
    stalling_threshold: float = reward_1v1.DEFAULT_STALLING_THRESHOLD
    approaching_factor: float = 0.02
    partner_proximity_support: float = 0.015  # support role: stay near mate
    space_penalty: float = 0.02  # B: don't sit on teammate
    floaty_onstage_bonus: float = 0.03  # B*: pressure floaty while on stage
    curriculum: CurriculumWeights = dataclasses.field(
        default_factory=default_curriculum
    )
    role: str = "aggro"  # "aggro" | "support" — A: team roles


def _deaths(player: Player) -> np.ndarray:
    return reward_1v1.process_deaths(player.action).astype(np.float32)


def _damages(player: Player) -> np.ndarray:
    return reward_1v1.process_damages(player.percent).astype(np.float32)


def _offstage_mask(player: Player, stage: np.ndarray, threshold: float) -> np.ndarray:
    # Use amount_offstage > small threshold as "offstage-ish".
    amt = reward_1v1.amount_offstage(player, stage)
    return (amt > 5.0)[1:].astype(np.float32)


def compute_teams_rewards(
    *,
    ego: Player,
    partner: Player,
    opp0: Player,
    opp1: Player,
    stage: np.ndarray,
    config: TeamsRewardConfig | None = None,
) -> np.ndarray:
    """
    Length (T-1) float32 rewards for one ego seat.

    Players are already ego-oriented time series (same convention as 1v1 Game.p0).
    """
    cfg = config or TeamsRewardConfig()
    cur = cfg.curriculum

    # --- Own survival (baseline) ---
    r = -(_deaths(ego) + cfg.damage_ratio * _damages(ego))

    # --- A: watch teammate ---
    if cur.watch_teammate > 0:
        r -= cur.watch_teammate * (
            cfg.teammate_death_penalty * _deaths(partner)
            + cfg.teammate_damage_ratio * _damages(partner)
        )

    # --- Enemy KOs / damage, with 2v1-on-stage bias ---
    for opp in (opp0, opp1):
        deaths = _deaths(opp)
        dmgs = _damages(opp)
        off = _offstage_mask(opp, stage, cfg.stalling_threshold)
        on = 1.0 - off
        scale = (
            cfg.onstage_enemy_ko_scale * on + cfg.offstage_enemy_ko_scale * off
        )
        if cur.prefer_2v1_on_stage > 0:
            scale = 1.0 + cur.prefer_2v1_on_stage * (scale - 1.0)
        r += cfg.enemy_death_reward * deaths * scale
        r += cfg.enemy_damage_ratio * dmgs * (
            0.7 + 0.3 * on
        )  # slightly prefer on-stage pressure

    # --- Stall offstage (own) ---
    stall = reward_1v1.is_stalling_offstage(
        ego, stage, cfg.stalling_threshold
    )[1:]
    r -= (cfg.stalling_penalty / 60.0) * stall.astype(np.float32)

    # --- Approach: aggro → enemies; support → partner (A: roles) ---
    if cur.team_roles > 0:
        if cfg.role == "support":
            r += (
                cur.team_roles
                * cfg.partner_proximity_support
                * reward_1v1.compute_approaching_factor(ego, partner)
            )
        else:
            # Approach nearer of the two enemies
            a0 = reward_1v1.compute_approaching_factor(ego, opp0)
            a1 = reward_1v1.compute_approaching_factor(ego, opp1)
            r += cur.team_roles * cfg.approaching_factor * np.maximum(a0, a1)

    # --- B: space around teammate (soft) ---
    if cur.space_around_teammate > 0:
        dx = ego.x[1:] - partner.x[1:]
        dy = ego.y[1:] - partner.y[1:]
        dist = np.sqrt(dx * dx + dy * dy)
        too_close = (dist < 12.0).astype(np.float32)
        r -= cur.space_around_teammate * cfg.space_penalty * too_close

    # --- B*: floaty / slow at bay (conditional) ---
    if cur.floaty_at_bay > 0:
        for opp in (opp0, opp1):
            char = opp.character
            # character may be scalar or array
            char0 = int(np.asarray(char).reshape(-1)[0])
            if char0 not in FLOATY_OR_SLOW_CHARS:
                continue
            off = _offstage_mask(opp, stage, cfg.stalling_threshold)
            # Reward approaching / damaging them while they are still on stage.
            approach = reward_1v1.compute_approaching_factor(ego, opp)
            r += (
                cur.floaty_at_bay
                * cfg.floaty_onstage_bonus
                * approach
                * (1.0 - off)
            )

    return r.astype(np.float32)


def teams_reward_summary(rewards: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "frames": int(rewards.shape[0]),
    }


# Live gamestate helper for PhillipTeams logging / future online shaping.
def live_stock_delta_reward(
    prev_stocks: Mapping[int, int],
    next_stocks: Mapping[int, int],
    *,
    ego: int,
    partner: int,
    enemies: tuple[int, int],
    config: TeamsRewardConfig | None = None,
) -> float:
    """Cheap stock-based signal for wrappers (not used by PPO until wired)."""
    cfg = config or TeamsRewardConfig()
    r = 0.0
    if next_stocks.get(ego, 0) < prev_stocks.get(ego, 0):
        r -= 1.0
    if next_stocks.get(partner, 0) < prev_stocks.get(partner, 0):
        r -= cfg.teammate_death_penalty
    for e in enemies:
        if next_stocks.get(e, 0) < prev_stocks.get(e, 0):
            r += cfg.enemy_death_reward
    return float(r)