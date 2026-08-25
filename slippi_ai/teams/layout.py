"""Port / seat mapping for 2v2 Teams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TeamsPorts:
    """Ego-centric port assignment for one controlled seat."""

    ego: int
    partner: int
    opp0: int
    opp1: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.ego, self.partner, self.opp0, self.opp1)

    def enemies(self) -> tuple[int, int]:
        return (self.opp0, self.opp1)


def resolve_teams_ports(
    *,
    ego: int,
    team_of: Mapping[int, int],
    ports: Sequence[int] | None = None,
) -> TeamsPorts:
    """
    Build ego/partner/opp0/opp1 from team_id map.

    ``team_of`` maps port -> team id (e.g. 0 or 1). Missing ports are ignored.
    Opponent order is sorted by port for stability (wrappers may re-rank live).
    """
    if ports is None:
        ports = tuple(sorted(team_of.keys()))
    else:
        ports = tuple(int(p) for p in ports)

    if ego not in team_of:
        raise ValueError(f"ego port {ego} missing from team_of")

    my_team = team_of[ego]
    partners = [p for p in ports if p != ego and team_of.get(p) == my_team]
    enemies = sorted(p for p in ports if team_of.get(p) is not None and team_of[p] != my_team)

    if len(partners) < 1:
        raise ValueError(f"no partner found for ego={ego} team={my_team}")
    if len(enemies) < 2:
        # Pad with a dummy only if we somehow have one enemy (stockouts mid-game).
        while len(enemies) < 2:
            enemies.append(enemies[-1] if enemies else ego)

    return TeamsPorts(
        ego=int(ego),
        partner=int(partners[0]),
        opp0=int(enemies[0]),
        opp1=int(enemies[1]),
    )


def default_seat_ports(seat: int = 1) -> TeamsPorts:
    """
    Assumed training seating (host recommendation for overnight work):

      Team A (trained / red): ports 1 + 2
      Team B (opponents):     ports 3 + 4

    Seat 1 = port 1 (aggro bias), seat 2 = port 2 (support bias).
    """
    if seat <= 1:
        return TeamsPorts(ego=1, partner=2, opp0=3, opp1=4)
    return TeamsPorts(ego=2, partner=1, opp0=3, opp1=4)


def all_seat_layouts(
    team_a: Iterable[int] = (1, 2),
    team_b: Iterable[int] = (3, 4),
) -> list[TeamsPorts]:
    """Layouts for every controlled port on team A (training batch)."""
    a = list(team_a)
    b = sorted(team_b)
    out: list[TeamsPorts] = []
    for ego in a:
        partner = next(p for p in a if p != ego)
        out.append(TeamsPorts(ego=ego, partner=partner, opp0=b[0], opp1=b[1]))
    return out
