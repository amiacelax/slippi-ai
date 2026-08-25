"""
Step 1: Hybrid Teams rollout worker.

Stock RL workers compute 1v1 rewards from Game(p0, p1).
This worker:
  - steps a Teams env (FakeTeamsEnv by default; LiveTeamsEnv with Dolphin)
  - records TeamsGame frames
  - builds Trajectory with **Teams IQ rewards**
  - stores **1v1-projected Game** states so a medium-v2 policy can still train

Run:  python -m slippi_ai.teams.step1_demo
Live: python -m slippi_ai.teams.step4_demo
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np

from slippi_ai import utils
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.data import NAME_DTYPE
from slippi_ai.evaluators import AbstractRolloutWorker, Timings, Trajectory
from slippi_ai.teams.compat_1v1 import teams_game_to_1v1
from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.live_env import TeamsStep
from slippi_ai.teams.parse import stack_teams_games
from slippi_ai.teams.reward import compute_teams_rewards
from slippi_ai.teams.rollout_worker import TeamsRolloutCollector
from slippi_ai.teams.trajectory import role_for_port
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import (
    Buttons,
    Controller,
    FoDPlatforms,
    Game,
    Nana,
    Player,
    Randall,
    Stick,
)
from slippi_db.parse_libmelee import _EMPTY_ITEMS

Port = int


def _nana0() -> Nana:
  return Nana(
      exists=np.bool_(False),
      percent=np.uint16(0),
      facing=np.bool_(False),
      x=np.float32(0),
      y=np.float32(0),
      action=np.uint16(0),
      invulnerable=np.bool_(False),
      character=np.uint8(0),
      jumps_left=np.uint8(0),
      shield_strength=np.float32(0),
      on_ground=np.bool_(False),
  )


def make_rank0_player(
    *,
    x: float = 0.0,
    char: int = 2,
    action: int = 0xE,
    percent: int = 0,
) -> Player:
  zf = np.float32(0.0)
  stick = Stick(x=zf, y=zf)
  buttons = Buttons(
      A=np.bool_(False),
      B=np.bool_(False),
      X=np.bool_(False),
      Y=np.bool_(False),
      Z=np.bool_(False),
      L=np.bool_(False),
      R=np.bool_(False),
      D_UP=np.bool_(False),
  )
  return Player(
      percent=np.uint16(percent),
      facing=np.bool_(True),
      x=np.float32(x),
      y=np.float32(0.0),
      action=np.uint16(action),
      invulnerable=np.bool_(False),
      character=np.uint8(char),
      jumps_left=np.uint8(2),
      shield_strength=np.float32(60.0),
      on_ground=np.bool_(True),
      controller=Controller(
          main_stick=stick,
          c_stick=Stick(x=zf, y=zf),
          shoulder=zf,
          buttons=buttons,
      ),
      nana=_nana0(),
  )


def make_rank0_teams_game(
    *,
    frame: int = 0,
    partner_dead: bool = False,
) -> TeamsGame:
  partner_action = 0 if partner_dead else 0xE
  return TeamsGame(
      ego=make_rank0_player(x=-10.0),
      partner=make_rank0_player(x=-5.0, action=partner_action),
      opp0=make_rank0_player(x=12.0, char=15),
      opp1=make_rank0_player(x=40.0 + frame * 0.5),
      stage=np.uint8(32),
      randall=Randall(x=np.float32(0), y=np.float32(0)),
      fod_platforms=FoDPlatforms(left=np.float32(0), right=np.float32(0)),
      items=_EMPTY_ITEMS,
  )


def _dummy_sample_outputs() -> SampleOutputs:
  """Controller-shaped zeros so batch_nest_nt works."""
  ctrl = make_rank0_player().controller
  return SampleOutputs(controller_state=ctrl, logits=ctrl)


# Back-compat alias
FakeTeamsStep = TeamsStep


class FakeTeamsEnv:
  """No Dolphin / ISO. Good enough to prove Step 1 on Windows."""

  def __init__(self, ports: tuple[int, ...] = (1, 2)):
    self.ports = ports
    self._frame = 0
    self._started = False
    self.last_is_teams = False

  def stop(self) -> None:
    pass

  def reset(self) -> TeamsStep:
    self._frame = 0
    self._started = True
    return self._make(needs_reset=True)

  def step(self, _controllers: dict | None = None) -> TeamsStep:
    if not self._started:
      return self.reset()
    self._frame += 1
    return self._make(needs_reset=False)

  def _make(self, needs_reset: bool) -> TeamsStep:
    partner_dead = self._frame >= 20
    return TeamsStep(
        gamestates={
            p: make_rank0_teams_game(
                frame=self._frame, partner_dead=partner_dead
            )
            for p in self.ports
        },
        needs_reset=needs_reset,
        is_teams=False,
    )


class HybridTeamsRolloutWorker(AbstractRolloutWorker):
  """
  Drop-in style worker for Teams hybrid training.

  Trajectory.states  = 1v1 Game (focus-projected)  -> medium-v2 compatible
  Trajectory.rewards = Teams IQ rewards             -> teammate death, 2v1, ...
  """

  def __init__(
      self,
      *,
      config: TeamsRLConfig | None = None,
      num_envs: int = 1,
      env: tp.Any | None = None,
      mode_label: str = "hybrid_teams_fake",
  ):
    self.config = config or TeamsRLConfig()
    self._num_envs = num_envs
    self._train_ports = tuple(self.config.train_ports)
    self._env = env or FakeTeamsEnv(ports=self._train_ports)
    self._mode_label = mode_label
    self._collector = TeamsRolloutCollector(self.config)
    self._peek: TeamsStep | None = None

  def start(self) -> None:
    self._peek = self._env.reset()

  def stop(self) -> None:
    self._env.stop()

  def reset_env(self) -> None:
    self._peek = self._env.reset()
    self._collector.reset()

  def update_variables(self, updates: tp.Mapping[Port, tp.Any]) -> None:
    del updates  # real agents on training box later

  def rollout(
      self, num_steps: int, verbose: bool = False
  ) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    del verbose
    if self._peek is None:
      self.start()
    assert self._peek is not None

    self._collector.reset()
    games_1v1: dict[Port, list[Game]] = {p: [] for p in self._train_ports}
    teams_frames: dict[Port, list[TeamsGame]] = {
        p: [] for p in self._train_ports
    }
    sample_outputs: dict[Port, list[SampleOutputs]] = {
        p: [] for p in self._train_ports
    }
    is_resetting: list[bool] = []
    saw_teams = False

    # num_steps + 1 states (same overlap convention as stock RolloutWorker)
    for i in range(num_steps + 1):
      step = self._peek
      is_resetting.append(bool(step.needs_reset))
      saw_teams = saw_teams or bool(getattr(step, "is_teams", False))
      for port, tg in step.gamestates.items():
        if port not in self._train_ports:
          continue
        focus = self._collector.record(port, tg)
        games_1v1[port].append(teams_game_to_1v1(tg, focus=focus))
        teams_frames[port].append(tg)
        sample_outputs[port].append(_dummy_sample_outputs())
      if i < num_steps:
        self._peek = self._env.step({})

    reset_arr = np.asarray(is_resetting, dtype=np.bool_)
    trajectories: dict[Port, Trajectory] = {}

    for port in self._train_ports:
      stacked = stack_teams_games(teams_frames[port])
      cfg = self.config.reward_config_for_role(role_for_port(port))
      rewards = compute_teams_rewards(
          ego=stacked.ego,
          partner=stacked.partner,
          opp0=stacked.opp0,
          opp1=stacked.opp1,
          stage=stacked.stage,
          config=cfg,
      )
      assert rewards.shape[0] == num_steps, (
          f"port {port}: got {rewards.shape[0]} rewards, want {num_steps}"
      )
      rewards = np.where(reset_arr[1:], 0.0, rewards)

      states = utils.batch_nest_nt(games_1v1[port])
      actions = utils.batch_nest_nt(sample_outputs[port])

      def add_b(x: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.asarray(x), axis=1)

      trajectories[port] = Trajectory(
          states=utils.map_nt(add_b, states),
          name=np.zeros([num_steps + 1, self._num_envs], dtype=NAME_DTYPE),
          actions=utils.map_nt(add_b, actions),
          rewards=rewards[:, None].astype(np.float32),
          is_resetting=reset_arr[:, None],
          initial_state=None,
          delayed_actions=[],
      )

    timings: Timings = {
        "mode": self._mode_label,
        "num_steps": num_steps,
        "ports": list(self._train_ports),
        "is_teams": saw_teams
        or bool(getattr(self._env, "last_is_teams", False)),
    }
    return trajectories, timings


def build_hybrid_teams_actor(
    config: TeamsRLConfig | None = None,
    *,
    num_envs: int = 1,
) -> HybridTeamsRolloutWorker:
  """Factory matching jax.rl run_lib's build_actor() pattern (fake env)."""
  return HybridTeamsRolloutWorker(
      config=config,
      num_envs=num_envs,
      mode_label="hybrid_teams_fake",
  )


def build_live_teams_actor(
    config: TeamsRLConfig | None = None,
    *,
    num_envs: int = 1,
    path: str | None = None,
    iso: str | None = None,
    headless: bool = False,
) -> HybridTeamsRolloutWorker:
  """Same hybrid Trajectory shape, but steps a real Dolphin TeamsEnvironment."""
  from slippi_ai.teams.live_env import build_live_teams_env

  if num_envs != 1:
    raise ValueError("Live Teams actor currently supports num_envs=1 only")
  env = build_live_teams_env(
      config, path=path, iso=iso, headless=headless
  )
  return HybridTeamsRolloutWorker(
      config=config,
      num_envs=1,
      env=env,
      mode_label="hybrid_teams_live",
  )
