"""
Teams rollout worker scaffold.

Collects per-seat TeamsGame frames, applies Teams rewards, and (in hybrid mode)
feeds medium-v2 via teams_game_to_1v1 + focus.pick_focus.

Does not replace jax.rl RolloutWorker yet — this is the Teams-side collector
we will plug in once a Dolphin/ISO path is available on a training box.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

import numpy as np

from slippi_ai.teams.compat_1v1 import teams_game_to_1v1
from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.focus import pick_focus
from slippi_ai.teams.reward import TeamsRewardConfig
from slippi_ai.teams.rollout import rewards_from_frames
from slippi_ai.teams.trajectory import role_for_port
from slippi_ai.teams.types_teams import TeamsGame


@dataclass
class SeatBuffer:
    port: int
    role: str
    frames: list[TeamsGame] = field(default_factory=list)
    focuses: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()
        self.focuses.clear()


@dataclass
class TeamsRolloutResult:
    rewards: dict[int, np.ndarray]  # port -> [T-1]
    focus_hist: dict[int, dict[str, int]]
    num_steps: int


class TeamsRolloutCollector:
    """
    In-memory collector. Push one TeamsGame per train port each env step.
    Call ``finish`` to compute rewards.
    """

    def __init__(self, config: TeamsRLConfig | None = None):
        self.config = config or TeamsRLConfig()
        self.buffers: dict[int, SeatBuffer] = {
            p: SeatBuffer(port=p, role=role_for_port(p))
            for p in self.config.train_ports
        }

    def reset(self) -> None:
        for b in self.buffers.values():
            b.clear()

    def record(
        self,
        port: int,
        game: TeamsGame,
        *,
        focus: str | None = None,
    ) -> str:
        if port not in self.buffers:
            raise KeyError(f"port {port} not in train_ports")
        buf = self.buffers[port]
        if focus is None:
            focus = pick_focus(game, role=buf.role).focus
        buf.frames.append(game)
        buf.focuses.append(focus)
        return focus

    def hybrid_1v1_view(self, port: int, game: TeamsGame) -> object:
        """Game nest for medium-v2 policy."""
        focus = pick_focus(game, role=self.buffers[port].role).focus
        return teams_game_to_1v1(game, focus=focus), focus

    def finish(self) -> TeamsRolloutResult:
        rewards: dict[int, np.ndarray] = {}
        focus_hist: dict[int, dict[str, int]] = {}
        steps = 0
        for port, buf in self.buffers.items():
            cfg = self.config.reward_config_for_role(buf.role)
            r = np.zeros(0, dtype=np.float32)
            if len(buf.frames) >= 2:
                # Sequence of Rank0 frames
                try:
                    r = rewards_from_frames(buf.frames, config=cfg)
                except Exception:
                    r = np.zeros(0, dtype=np.float32)
            elif len(buf.frames) == 1:
                g = buf.frames[0]
                # Already time-stacked nest [T, ...]
                tlen = int(np.asarray(g.ego.x).shape[0]) if np.ndim(g.ego.x) >= 1 else 1
                if tlen >= 2:
                    from slippi_ai.teams.reward import compute_teams_rewards

                    r = compute_teams_rewards(
                        ego=g.ego,
                        partner=g.partner,
                        opp0=g.opp0,
                        opp1=g.opp1,
                        stage=g.stage,
                        config=cfg,
                    )
            rewards[port] = r
            steps = max(steps, int(r.shape[0]) if r.size else 0)
            hist: dict[str, int] = {"opp0": 0, "opp1": 0}
            for f in buf.focuses:
                hist[f] = hist.get(f, 0) + 1
            focus_hist[port] = hist
        return TeamsRolloutResult(
            rewards=rewards, focus_hist=focus_hist, num_steps=steps
        )


def describe_integration() -> str:
    return (
        "Wire-up checklist:\n"
        "1. TeamsEnvironment.step → TeamsGame per train port\n"
        "2. collector.record(port, game)\n"
        "3. game_1v1, focus = collector.hybrid_1v1_view(port, game)\n"
        "4. agent.step(game_1v1)  # medium-v2 until Teams embed ships\n"
        "5. result = collector.finish() → Teams rewards into PPO advantages\n"
        "6. Later: swap step 3–4 for Teams embed policy"
    )
