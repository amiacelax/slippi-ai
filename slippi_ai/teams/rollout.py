"""
Rollout helpers: stack TeamsGame frames -> teams rewards.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from slippi_ai.teams.parse import stack_teams_games
from slippi_ai.teams.reward import (
    TeamsRewardConfig,
    compute_teams_rewards,
    teams_reward_summary,
)
from slippi_ai.teams.types_teams import TeamsGame


def rewards_from_frames(
    frames: Sequence[TeamsGame],
    *,
    config: TeamsRewardConfig | None = None,
) -> np.ndarray:
    """Length (T-1) rewards for one ego seat's frame history."""
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float32)
    stacked = stack_teams_games(frames)
    return compute_teams_rewards(
        ego=stacked.ego,
        partner=stacked.partner,
        opp0=stacked.opp0,
        opp1=stacked.opp1,
        stage=stacked.stage,
        config=config,
    )


def summarize_frames(
    frames: Sequence[TeamsGame],
    *,
    config: TeamsRewardConfig | None = None,
) -> dict[str, float]:
    r = rewards_from_frames(frames, config=config)
    if r.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "frames": 0}
    return teams_reward_summary(r)
