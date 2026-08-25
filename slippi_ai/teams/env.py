"""
Parallel RL env for 2v2 Teams (4 controllers).

Stock ``slippi_ai.envs.Environment`` still requires exactly 2 players.
This class is separate so 1v1 training stays untouched.
"""

from __future__ import annotations

import typing as tp
from typing import Mapping, Optional

from melee import GameState

from slippi_ai import dolphin, utils
from slippi_ai.controller_lib import send_controller
from slippi_ai.teams.layout import TeamsPorts, default_seat_ports
from slippi_ai.teams.menu import preferred_costume_for_port
from slippi_ai.teams.parse import TeamsParser, resolve_ports_for_ego
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import Controller

Port = int
Controllers = Mapping[Port, Controller]


def is_initial_frame(gamestate: GameState) -> bool:
    return gamestate.frame == -123


class TeamsEnvOutput(tp.NamedTuple):
    """Per-ego TeamsGame observations keyed by controlled port."""

    gamestates: Mapping[int, TeamsGame]
    needs_reset: bool
    is_teams: bool
    raw: Optional[GameState] = None


class TeamsEnvironment:
    """
    Wrap dolphin with 4 players.

    Expected seating (override with ``seat_layout``):
      Team A ports 1+2, Team B ports 3+4.
    Observations are ego-centric TeamsGame for each AI on team A by default
    (both teams if ``observe_all_ai``).
    """

    def __init__(
        self,
        dolphin_kwargs: dict,
        *,
        train_ports: tuple[int, ...] = (1, 2),
        seat_layout: TeamsPorts | None = None,
        observe_all_ai: bool = False,
        assign_costumes: bool = True,
    ):
        players: dict[Port, dolphin.Player] = dolphin_kwargs["players"]
        if len(players) != 4:
            raise ValueError(
                f"TeamsEnvironment requires exactly 4 players, got {len(players)}"
            )

        # Optional costume hint for team colors before is_teams is reliable.
        if assign_costumes:
            for port, player in players.items():
                if isinstance(player, dolphin.MenuingPlayer) and player.costume is None:
                    player.costume = preferred_costume_for_port(int(port))

        actual = dict(dolphin_kwargs)
        actual.setdefault("enable_teams", True)
        self._dolphin = dolphin.Dolphin(**actual)
        self._players = players
        self._train_ports = tuple(int(p) for p in train_ports)
        self._seat_layout = seat_layout or default_seat_ports(1)
        self._observe_all_ai = observe_all_ai
        self._prev_state: Optional[GameState] = None
        self._parsers: dict[int, TeamsParser] = {}

    def stop(self):
        self._dolphin.stop()

    def _ego_ports(self) -> list[int]:
        out = []
        for port, player in self._players.items():
            if not isinstance(player, dolphin.AI):
                continue
            if self._observe_all_ai or port in self._train_ports:
                out.append(int(port))
        return out

    def _ensure_parsers(self, gamestate: GameState) -> None:
        for ego in self._ego_ports():
            ports = resolve_ports_for_ego(
                gamestate, ego, fallback_ports=self._seat_layout
            )
            self._parsers[ego] = TeamsParser(ports, allow_missing_players=True)

    def current_state(self) -> TeamsEnvOutput:
        if self._prev_state is None:
            self._prev_state = self._step_with_menu()

        needs_reset = is_initial_frame(self._prev_state)
        if needs_reset or not self._parsers:
            self._ensure_parsers(self._prev_state)

        games = {
            ego: self._parsers[ego].get_teams_game(self._prev_state)
            for ego in self._ego_ports()
            if ego in self._parsers
        }
        return TeamsEnvOutput(
            gamestates=games,
            needs_reset=needs_reset,
            is_teams=bool(getattr(self._prev_state, "is_teams", False)),
            raw=self._prev_state,
        )

    def _step_with_menu(self) -> GameState:
        # Dolphin.step runs CSS/stage menuing; enable_teams pulses Y on CSS.
        return self._dolphin.step()

    def step(self, controllers: Controllers) -> TeamsEnvOutput:
        for port, controller in controllers.items():
            send_controller(self._dolphin.controllers[int(port)], controller)
        self._prev_state = self._step_with_menu()
        return self.current_state()


def make_four_ai_players(
    character: "melee.Character | None" = None,
) -> dict[int, dolphin.AI]:
    """Convenience: 4 AI seats, Fox default, costumes by team."""
    import melee

    char = character or melee.Character.FOX
    return {
        1: dolphin.AI(character=char, costume=0),
        2: dolphin.AI(character=char, costume=0),
        3: dolphin.AI(character=char, costume=1),
        4: dolphin.AI(character=char, costume=1),
    }
