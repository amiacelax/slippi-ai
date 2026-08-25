"""
4-port parser: libmelee GameState -> ego-centric TeamsGame.

Reuses stock ``get_player`` / item helpers from ``parse_libmelee``.
Does **not** touch the 1v1 Arrow ``Game`` schema.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

import melee

from slippi_ai import utils
from slippi_ai.teams.layout import TeamsPorts, default_seat_ports, resolve_teams_ports
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import FoDPlatforms, InvalidGameError, Randall, reify_tuple_type, Item, Items
from slippi_db import parsing_utils
from slippi_db.parse_libmelee import (
    _EMPTY_ITEMS,
    get_item,
    get_player,
)


def team_map_from_gamestate(
    gamestate: melee.GameState,
    *,
    fallback_ports: TeamsPorts | None = None,
) -> dict[int, int]:
    """
    Prefer live ``team_id`` when ``is_teams``; else port convention 1+2 vs 3+4.
    """
    ports = sorted(int(p) for p in gamestate.players.keys())
    if getattr(gamestate, "is_teams", False):
        team_of: dict[int, int] = {}
        for p in ports:
            pl = gamestate.players[p]
            team_of[p] = int(getattr(pl, "team_id", 0) or 0)
        # If everyone somehow shares one team id, fall back.
        if len(set(team_of.values())) >= 2:
            return team_of

    fb = fallback_ports or default_seat_ports(1)
    # Map by absolute ports from the default seating template.
    return {
        fb.ego: 0,
        fb.partner: 0,
        fb.opp0: 1,
        fb.opp1: 1,
    }


def resolve_ports_for_ego(
    gamestate: melee.GameState,
    ego: int,
    *,
    fallback_ports: TeamsPorts | None = None,
) -> TeamsPorts:
    team_of = team_map_from_gamestate(gamestate, fallback_ports=fallback_ports)
    if ego not in team_of and fallback_ports is not None:
        # Ego might be seat2 — rebuild from fixed layout.
        return (
            default_seat_ports(1)
            if ego == fallback_ports.ego
            else default_seat_ports(2)
            if ego == fallback_ports.partner
            else resolve_teams_ports(ego=ego, team_of=team_of)
        )
    return resolve_teams_ports(ego=ego, team_of=team_of, ports=sorted(team_of.keys()))


class TeamsParser:
    """Parses one ego-centric TeamsGame frame from a 4-player GameState."""

    def __init__(
        self,
        ports: TeamsPorts,
        *,
        allow_missing_players: bool = False,
    ):
        self.ports = ports
        self.allow_missing_players = allow_missing_players
        self.item_assigner = parsing_utils.ItemAssigner()

    def _player_or_empty(self, gamestate: melee.GameState, port: int):
        pl = gamestate.players.get(port) or gamestate.players.get(int(port))
        if pl is not None:
            return get_player(pl)
        if not self.allow_missing_players:
            raise InvalidGameError(f"missing player on port {port}")
        # Zeroed player nest
        return utils.map_nt(lambda t: t(0), reify_tuple_type(type(get_player(next(iter(gamestate.players.values()))))))

    def get_teams_game(self, gamestate: melee.GameState) -> TeamsGame:
        ports_this_frame = sorted(int(p) for p in gamestate.players.keys())
        if len(ports_this_frame) < 2:
            raise InvalidGameError(f"need at least 2 players, got {ports_this_frame}")

        needed = set(self.ports.as_tuple())
        missing = needed - set(ports_this_frame)
        if missing and not self.allow_missing_players:
            # Mid-stockout: allow temporarily if at least ego exists
            if self.ports.ego not in ports_this_frame:
                raise InvalidGameError(f"ego port {self.ports.ego} missing; have {ports_this_frame}")

        if gamestate.stage is melee.Stage.YOSHIS_STORY:
            randall_y, randall_x_left, randall_x_right = melee.randall_position(gamestate.frame)
            randall_x = (randall_x_left + randall_x_right) / 2
        else:
            randall_y = randall_x = 0.0

        if gamestate.fod_platforms:
            fod_platforms = FoDPlatforms(
                left=gamestate.fod_platforms.left,
                right=gamestate.fod_platforms.right,
            )
        else:
            fod_platforms = FoDPlatforms(np.float32(0), np.float32(0))

        slots = self.item_assigner.assign(
            [item.spawn_id for item in gamestate.projectiles]
        )
        items_dict = {}
        for slot, item in zip(slots, gamestate.projectiles):
            items_dict[f"item_{slot}"] = get_item(item)
        items = _EMPTY_ITEMS._replace(**items_dict)

        def slot_player(port: int):
            pl = gamestate.players.get(port) or gamestate.players.get(int(port))
            if pl is None:
                # Dead / despawned — synthesize from any existing player then zero stocks feel
                any_pl = next(iter(gamestate.players.values()))
                base = get_player(any_pl)
                # Mark as dead-ish action 0
                return base._replace(action=np.uint16(0), percent=np.uint16(0))
            return get_player(pl)

        return TeamsGame(
            ego=slot_player(self.ports.ego),
            partner=slot_player(self.ports.partner),
            opp0=slot_player(self.ports.opp0),
            opp1=slot_player(self.ports.opp1),
            stage=np.uint8(gamestate.stage.value),
            randall=Randall(x=np.float32(randall_x), y=np.float32(randall_y)),
            fod_platforms=fod_platforms,
            items=items,
        )


def stack_teams_games(frames: Sequence[TeamsGame]) -> TeamsGame:
    """Stack Rank0 frames into a length-T TeamsGame nest."""
    if not frames:
        raise ValueError("no frames to stack")
    return utils.batch_nest_nt(list(frames))


def ports_from_libmelee_players(
    players: Mapping[int, object],
    *,
    ego: int,
) -> TeamsPorts:
    team_of: dict[int, int] = {}
    for port, pl in players.items():
        tid = getattr(pl, "team_id", None)
        if tid is None:
            continue
        team_of[int(port)] = int(tid)
    return resolve_teams_ports(ego=ego, team_of=team_of)


def assert_four_unique(ports: Sequence[int]) -> None:
    uniq = set(int(p) for p in ports)
    if len(uniq) != 4:
        raise ValueError(f"expected 4 unique ports, got {ports!r}")
