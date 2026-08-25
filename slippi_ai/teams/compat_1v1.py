"""
Bridge: run stock 1v1 medium-v2 inside a 4-player Teams env.

The policy still only "sees" ego vs one opponent (opp0). Full TeamsGame is
kept for rewards / logging. Targeting wrappers (PhillipTeams) or
``pick_focus_opponent`` choose which enemy fills opp0 each frame.
"""

from __future__ import annotations

from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import Game


def teams_game_to_1v1(game: TeamsGame, *, focus: str = "opp0") -> Game:
    """
    Project TeamsGame -> stock Game(p0, p1) for medium-v2.

    ``focus`` is ``opp0`` or ``opp1`` — which enemy the 1v1 brain fights.
    """
    if focus == "opp0":
        opp = game.opp0
    elif focus == "opp1":
        opp = game.opp1
    else:
        raise ValueError(f"focus must be opp0 or opp1, got {focus!r}")

    return Game(
        p0=game.ego,
        p1=opp,
        stage=game.stage,
        randall=game.randall,
        fod_platforms=game.fod_platforms,
        items=game.items,
    )


def describe_teacher_warmup() -> str:
    return (
        "Warm-start plan:\n"
        "1. TeamsEnvironment produces TeamsGame per seat\n"
        "2. teams_game_to_1v1(focus=opp0) feeds medium-v2\n"
        "3. rewards_from_frames uses full TeamsGame (partner death, 2v1, …)\n"
        "4. Later: replace step 2 with a real 4-player embed/network"
    )
