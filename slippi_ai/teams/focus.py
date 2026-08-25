"""
Retarget helper: which enemy fills opp0 for the hybrid 1v1 teacher.

Mirrors PhillipTeams doubles_target scoring in a numpy/TeamsGame-friendly way
so training and live wrappers stay aligned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from slippi_ai.teams.curriculum import (
    FLOATY_OR_SLOW_CHARS,
    YOSHI_CHAR,
    yoshi_spike_edgeguard_ok,
)
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import Player


@dataclass
class FocusPick:
    focus: str  # "opp0" | "opp1"
    reason: str
    score_opp0: float
    score_opp1: float


def _xy(p: Player, t: int = -1) -> tuple[float, float]:
    return float(np.asarray(p.x).reshape(-1)[t]), float(np.asarray(p.y).reshape(-1)[t])


def _alive(p: Player, t: int = -1) -> bool:
    action = int(np.asarray(p.action).reshape(-1)[t])
    return action > 0xA


def _offstage(p: Player, t: int = -1, edge: float = 88.0) -> bool:
    x, y = _xy(p, t)
    return abs(x) > edge or y < -10.0


def _dist(a: Player, b: Player, t: int = -1) -> float:
    ax, ay = _xy(a, t)
    bx, by = _xy(b, t)
    return math.hypot(ax - bx, ay - by)


def score_focus_enemy(
    *,
    ego: Player,
    partner: Player,
    enemy: Player,
    other_on_stage: bool,
    role: str,
    t: int = -1,
) -> tuple[float, str]:
    reasons: list[str] = []
    score = 0.0
    if not _alive(enemy, t):
        return -1e9, "dead"

    off = _offstage(enemy, t)
    ego_char = int(np.asarray(ego.character).reshape(-1)[t])
    en_char = int(np.asarray(enemy.character).reshape(-1)[t])
    en_pct = float(np.asarray(enemy.percent).reshape(-1)[t])
    yoshi_spike = (
        off
        and en_char == YOSHI_CHAR
        and yoshi_spike_edgeguard_ok(ego_char=ego_char, yoshi_percent=en_pct)
    )

    if not off:
        score += 55.0
        reasons.append("2v1-on-stage")
    elif other_on_stage and not yoshi_spike:
        score -= 80.0
        reasons.append("skip-edge-for-2v1")

    d = _dist(ego, enemy, t)
    score += max(0.0, 45.0 - d * 0.55)

    if en_char in FLOATY_OR_SLOW_CHARS and not off:
        score += 28.0
        reasons.append("floaty-at-bay")

    if role == "support" and _alive(partner, t):
        score += max(0.0, 30.0 - _dist(partner, enemy, t) * 0.4)
        reasons.append("support-cover")

    if yoshi_spike:
        score += 95.0
        reasons.append("yoshi-spike-edge")
    elif off and other_on_stage:
        score -= 200.0
        reasons.append("stay-on-stage")

    return score, "+".join(reasons) if reasons else "closest"


def pick_focus(
    game: TeamsGame,
    *,
    role: str = "aggro",
    t: int = -1,
) -> FocusPick:
    """Choose opp0 vs opp1 as the 1v1 teacher focus for this frame."""
    other0_on = _alive(game.opp1, t) and not _offstage(game.opp1, t)
    other1_on = _alive(game.opp0, t) and not _offstage(game.opp0, t)
    s0, r0 = score_focus_enemy(
        ego=game.ego,
        partner=game.partner,
        enemy=game.opp0,
        other_on_stage=other0_on,
        role=role,
        t=t,
    )
    s1, r1 = score_focus_enemy(
        ego=game.ego,
        partner=game.partner,
        enemy=game.opp1,
        other_on_stage=other1_on,
        role=role,
        t=t,
    )
    if s1 > s0 * 1.18:
        return FocusPick("opp1", r1, s0, s1)
    return FocusPick("opp0", r0, s0, s1)
