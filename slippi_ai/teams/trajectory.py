"""
Teams trajectory + reward bridge for (future) PPO.

Stock ``evaluators.Trajectory.states`` is ``Game``. For Teams we keep a parallel
``TeamsTrajectory`` whose rewards come from ``compute_teams_rewards``.

Hybrid path (PLACEHOLDER, runnable conceptually):
  - Policy samples from ``teams_game_to_1v1(states)``  → medium-v2
  - Rewards from full ``TeamsGame``                   → Teams IQ curriculum
"""

from __future__ import annotations

import dataclasses
import functools
import typing as tp

import numpy as np

from slippi_ai import utils
from slippi_ai.controller_heads import SampleOutputs, ControllerType as CT
from slippi_ai.teams.compat_1v1 import teams_game_to_1v1
from slippi_ai.teams.reward import TeamsRewardConfig, compute_teams_rewards
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import BoolArray, FloatArray, Game

RS = tp.TypeVar("RS")
Rank2 = tuple[int, int]


class TeamsTrajectory(tp.NamedTuple, tp.Generic[CT, RS]):
    """Parallel to evaluators.Trajectory but with TeamsGame states."""

    states: TeamsGame  # [T+1, B] nest
    name: np.ndarray  # [T+1, B]
    actions: SampleOutputs[CT]  # [T+1, B]
    rewards: FloatArray[Rank2]  # [T, B]
    is_resetting: BoolArray[Rank2]  # [T+1, B]
    initial_state: RS
    delayed_actions: list[SampleOutputs[CT]]

    @classmethod
    def batch(cls, trajectories: list[tp.Self]) -> tp.Self:
        return utils.map_nt(
            lambda axis, *ts: utils.concat_nest_nt(ts, axis),
            cls.batch_dims(),
            *trajectories,
        )

    @classmethod
    @functools.cache
    def batch_dims(cls) -> tp.Self:
        return cls(
            states=1,
            name=1,
            actions=1,
            rewards=1,
            is_resetting=1,
            initial_state=0,
            delayed_actions=0,
        )


def update_teams_rewards(
    trajectory: TeamsTrajectory,
    reward_config: TeamsRewardConfig | None = None,
    *,
    role: str | None = None,
) -> TeamsTrajectory:
    """Recompute rewards from TeamsGame states (T+1) → length T."""
    cfg = reward_config or TeamsRewardConfig()
    if role is not None and cfg.role != role:
        cfg = dataclasses.replace(cfg, role=role)

    states = trajectory.states
    rewards = compute_teams_rewards(
        ego=states.ego,
        partner=states.partner,
        opp0=states.opp0,
        opp1=states.opp1,
        stage=states.stage,
        config=cfg,
    )
    # Zero reward across episode boundaries (same as stock update_rewards).
    reset = np.asarray(trajectory.is_resetting)
    if reset.ndim >= 1 and reset.shape[0] == rewards.shape[0] + 1:
        # rewards may be [T] or [T, B]
        if rewards.ndim == 1:
            rewards = np.where(reset[1:], 0.0, rewards)
        else:
            rewards = np.where(reset[1:], 0.0, rewards)
    return trajectory._replace(rewards=rewards)


def teams_trajectory_as_1v1_states(
    trajectory: TeamsTrajectory,
    *,
    focus: str = "opp0",
) -> Game:
    """Project Teams states to Game for a 1v1 teacher/policy."""
    return teams_game_to_1v1(trajectory.states, focus=focus)


def role_for_port(port: int) -> str:
    """PLACEHOLDER seat→role map (host-confirmed: 1=aggro, 2=support)."""
    if int(port) in (2, 4):
        return "support"
    return "aggro"
