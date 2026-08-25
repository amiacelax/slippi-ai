"""
Offline fake rollout — no Dolphin.

Generates synthetic TeamsGame frames, picks focus, projects to 1v1, computes
Teams rewards. Proves the hybrid data path end-to-end without ISO/GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slippi_ai.teams.compat_1v1 import teams_game_to_1v1
from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.focus import pick_focus
from slippi_ai.teams.reward import compute_teams_rewards
from slippi_ai.teams.rollout import rewards_from_frames
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import (
    Buttons,
    Controller,
    FoDPlatforms,
    Nana,
    Player,
    Randall,
    Stick,
)
from slippi_db.parse_libmelee import _EMPTY_ITEMS


def _nana(T: int) -> Nana:
    z = lambda d: np.zeros(T, dtype=d)
    return Nana(
        exists=z(np.bool_),
        percent=z(np.uint16),
        facing=z(np.bool_),
        x=z(np.float32),
        y=z(np.float32),
        action=z(np.uint16),
        invulnerable=z(np.bool_),
        character=z(np.uint8),
        jumps_left=z(np.uint8),
        shield_strength=z(np.float32),
        on_ground=z(np.bool_),
    )


def _player(
    T: int,
    *,
    x: float,
    char: int = 2,
    action: int = 0xE,
) -> Player:
    zf = np.zeros(T, dtype=np.float32)
    stick = Stick(x=zf.copy(), y=zf.copy())
    buttons = Buttons(
        **{k: np.zeros(T, dtype=np.bool_) for k in Buttons._fields}
    )
    return Player(
        percent=np.zeros(T, dtype=np.uint16),
        facing=np.ones(T, dtype=np.bool_),
        x=np.full(T, x, dtype=np.float32),
        y=np.zeros(T, dtype=np.float32),
        action=np.full(T, action, dtype=np.uint16),
        invulnerable=np.zeros(T, dtype=np.bool_),
        character=np.full(T, char, dtype=np.uint8),
        jumps_left=np.full(T, 2, dtype=np.uint8),
        shield_strength=np.full(T, 60.0, dtype=np.float32),
        on_ground=np.ones(T, dtype=np.bool_),
        controller=Controller(
            main_stick=stick,
            c_stick=Stick(x=zf.copy(), y=zf.copy()),
            shoulder=zf.copy(),
            buttons=buttons,
        ),
        nana=_nana(T),
    )


@dataclass
class FakeRolloutResult:
    frames: int
    reward_mean: float
    focus_counts: dict[str, int]
    sample_1v1_opp_char: int


def fake_hybrid_rollout(
    *,
    T: int = 64,
    role: str = "aggro",
    config: TeamsRLConfig | None = None,
) -> FakeRolloutResult:
    """
    Simulate T frames: partner dies mid-way, puff on opp0, fox on opp1 far away.
    """
    config = config or TeamsRLConfig()
    partner = _player(T, x=-5)
    pa = np.full(T, 0xE, dtype=np.uint16)
    pa[T // 2 :] = 0
    partner = partner._replace(action=pa)

    # Move opp1 farther over time to encourage opp0 focus
    opp1_x = np.linspace(40, 130, T).astype(np.float32)
    opp1 = _player(T, x=40.0)
    opp1 = opp1._replace(x=opp1_x)

    tg = TeamsGame(
        ego=_player(T, x=-10),
        partner=partner,
        opp0=_player(T, x=12, char=15),  # puff
        opp1=opp1,
        stage=np.full(T, 32, dtype=np.uint8),
        randall=Randall(x=np.zeros(T, np.float32), y=np.zeros(T, np.float32)),
        fod_platforms=FoDPlatforms(
            left=np.zeros(T, np.float32), right=np.zeros(T, np.float32)
        ),
        items=_EMPTY_ITEMS,
    )

    focus_counts = {"opp0": 0, "opp1": 0}
    # Per-frame focus on last index of growing prefix would be heavy; sample
    # a few timesteps.
    for t in range(0, T, max(1, T // 8)):
        # Build single-frame view by slicing — use full nest pick on last dim
        pick = pick_focus(tg, role=role, t=t)
        focus_counts[pick.focus] += 1

    rewards = rewards_from_frames(
        # stack_teams_games expects sequence of Rank0 — fake_rollout uses T nest
        # directly via compute_teams_rewards
        [],
        config=config.reward_config_for_role(role),
    )
    # Direct path:
    rewards = compute_teams_rewards(
        ego=tg.ego,
        partner=tg.partner,
        opp0=tg.opp0,
        opp1=tg.opp1,
        stage=tg.stage,
        config=config.reward_config_for_role(role),
    )

    g = teams_game_to_1v1(tg, focus="opp0")
    return FakeRolloutResult(
        frames=T,
        reward_mean=float(np.mean(rewards)),
        focus_counts=focus_counts,
        sample_1v1_opp_char=int(g.p1.character[0]),
    )


def main() -> int:
    r = fake_hybrid_rollout()
    print("fake_hybrid_rollout", r)
    assert r.sample_1v1_opp_char == 15
    assert r.focus_counts["opp0"] >= r.focus_counts["opp1"]
    assert r.reward_mean < 0  # partner death mid-episode
    print("fake_rollout PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
