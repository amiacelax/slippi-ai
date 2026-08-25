"""
Proposed ``TeamsGame`` nest — parallel to stock ``types.Game``.

Not registered in Arrow / GAME_TYPE yet. Used by reward tests and as the
target schema for the next coding pass.
"""

from __future__ import annotations

from typing import Generic, NamedTuple

from slippi_ai.types import FoDPlatforms, Items, Player, Randall, S, UInt8Array


class TeamsGame(NamedTuple, Generic[S]):
    """Ego-centric 2v2 frame nest."""

    ego: Player[S]
    partner: Player[S]
    opp0: Player[S]
    opp1: Player[S]
    stage: UInt8Array[S]
    randall: Randall[S]
    fod_platforms: FoDPlatforms[S]
    items: Items[S]


def teams_game_from_players(
    *,
    ego: Player,
    partner: Player,
    opp0: Player,
    opp1: Player,
    stage: UInt8Array,
    randall: Randall,
    fod_platforms: FoDPlatforms,
    items: Items,
) -> TeamsGame:
    return TeamsGame(
        ego=ego,
        partner=partner,
        opp0=opp0,
        opp1=opp1,
        stage=stage,
        randall=randall,
        fod_platforms=fod_platforms,
        items=items,
    )
