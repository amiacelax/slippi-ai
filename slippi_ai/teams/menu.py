"""
Teams CSS helpers — enable Melee Teams and color-code seats.

libmelee's MenuHelper has no Teams path. We layer a small state machine on top:

1. Use stock menu_helper_simple to reach Versus CSS.
2. On CSS, pulse Y until ``gamestate.is_teams`` (Melee toggles Teams with Y).
3. Costumes: seat layout uses index as costume so partners share a palette feel;
   real team_id comes from the game once Teams is on.

If Y-toggle fails on some Dolphin builds, training can still run with
port-convention teams (1+2 vs 3+4) via ``team_map_from_gamestate`` fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import melee
from melee.controller import Controller
from melee.gamestate import GameState


@dataclass
class TeamsMenuAssist:
    """Per-console assist; one instance shared across ports is fine."""

    frames_trying_teams: int = 0
    teams_confirmed: bool = False
    max_toggle_attempts: int = 180  # ~3s at 60fps

    def maybe_enable_teams(
        self,
        gamestate: GameState,
        controller: Controller,
        *,
        is_leader: bool,
    ) -> None:
        """
        Call once per frame on CSS for the leader controller.
        Non-leaders should not spam Y.
        """
        if gamestate.menu_state not in (
            melee.Menu.CHARACTER_SELECT,
            melee.Menu.SLIPPI_ONLINE_CSS,
        ):
            return

        if getattr(gamestate, "is_teams", False):
            self.teams_confirmed = True
            return

        if not is_leader:
            return

        if self.frames_trying_teams >= self.max_toggle_attempts:
            return

        # Pulse Y every other frame to toggle Teams on CSS.
        if self.frames_trying_teams % 2 == 0:
            controller.press_button(melee.Button.BUTTON_Y)
        else:
            controller.release_button(melee.Button.BUTTON_Y)
        self.frames_trying_teams += 1

    def reset(self) -> None:
        self.frames_trying_teams = 0
        self.teams_confirmed = False


def preferred_costume_for_port(port: int) -> int:
    """Rough team coloring: ports 1–2 costume 0, ports 3–4 costume 1."""
    if port in (1, 2):
        return 0
    if port in (3, 4):
        return 1
    return (port - 1) % 4
