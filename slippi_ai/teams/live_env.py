"""
Adapter: TeamsEnvironment → same reset/step shape as FakeTeamsEnv.

Live Dolphin opens a window. Use only when ISO + Dolphin are configured.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.env import TeamsEnvironment, TeamsEnvOutput
from slippi_ai.teams.local_paths import resolve_paths
from slippi_ai.teams.run_lib import build_dolphin_kwargs
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import Controller

Port = int


@dataclass
class TeamsStep:
  """Shared env step for fake + live hybrid workers."""

  gamestates: dict[int, TeamsGame]
  needs_reset: bool
  is_teams: bool = False


class LiveTeamsEnv:
  """
  Thin wrapper so HybridTeamsRolloutWorker can swap FakeTeamsEnv → Dolphin.

  Neutral controllers are sent for every AI port each frame (blocking input).
  """

  def __init__(
      self,
      *,
      config: TeamsRLConfig | None = None,
      path: str | None = None,
      iso: str | None = None,
      headless: bool = False,
      train_ports: tuple[int, ...] | None = None,
  ):
    self.config = config or TeamsRLConfig()
    self._train_ports = tuple(train_ports or self.config.train_ports)
    folder, iso_path = resolve_paths(dolphin=path, iso=iso)
    kwargs = build_dolphin_kwargs(
        self.config,
        path=str(folder),
        iso=str(iso_path),
        headless=headless,
    )
    # Netplay Slippi: keep a visible window; headless needs mainline/EXI.
    if not headless:
      kwargs["headless"] = False
      kwargs["render"] = True
    self._env = TeamsEnvironment(
        kwargs,
        train_ports=self._train_ports,
    )
    self._all_ports = tuple(sorted(kwargs["players"]))
    self._neutral = _neutral_controller()
    self._started = False
    self.last_is_teams = False

  @property
  def ports(self) -> tuple[int, ...]:
    return self._train_ports

  def stop(self) -> None:
    self._env.stop()

  def reset(self) -> TeamsStep:
    out = self._env.current_state()
    self._started = True
    return self._to_step(out)

  def step(
      self, controllers: dict[Port, Controller] | None = None
  ) -> TeamsStep:
    if not self._started:
      return self.reset()
    merged = {p: self._neutral for p in self._all_ports}
    if controllers:
      merged.update(controllers)
    out = self._env.step(merged)
    return self._to_step(out)

  def _to_step(self, out: TeamsEnvOutput) -> TeamsStep:
    self.last_is_teams = bool(out.is_teams)
    games = {int(p): g for p, g in out.gamestates.items()}
    # Ensure train ports always present (parser should provide them).
    missing = [p for p in self._train_ports if p not in games]
    if missing:
      raise RuntimeError(
          f"LiveTeamsEnv missing train-port observations: {missing} "
          f"(have {sorted(games)})"
      )
    return TeamsStep(
        gamestates=games,
        needs_reset=bool(out.needs_reset),
        is_teams=bool(out.is_teams),
    )


def _neutral_controller() -> Controller:
  import numpy as np

  from slippi_ai.types import Buttons, Nana, Player, Stick

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
  return Controller(
      main_stick=stick,
      c_stick=Stick(x=zf, y=zf),
      shoulder=zf,
      buttons=buttons,
  )


def build_live_teams_env(
    config: TeamsRLConfig | None = None,
    *,
    path: str | None = None,
    iso: str | None = None,
    headless: bool = False,
) -> LiveTeamsEnv:
  return LiveTeamsEnv(config=config, path=path, iso=iso, headless=headless)
