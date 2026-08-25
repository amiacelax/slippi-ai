"""
Warm-start a Teams policy from a 1v1 medium-v2-style checkpoint.

Mapping (PLACEHOLDER, host-ok):
  p0 (ego in 1v1)  -> ego
  p1 (opponent)    -> opp0
  partner          <- copy of ego  (or zeros — see PARTNER_INIT)
  opp1             <- copy of opp0

This module is JAX-free: it describes path replacements on flattened param
name lists so we can unit-test the mapping without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# How to init partner / second opponent when teacher is 1v1-only.
PARTNER_INIT = "copy_ego"  # or "zeros"
OPP1_INIT = "copy_opp0"  # or "zeros" | "copy_ego"


@dataclass(frozen=True)
class WarmStartPlan:
    partner_init: str = PARTNER_INIT
    opp1_init: str = OPP1_INIT
    # Substring markers in flattened param paths (Flax/nnx vary; treat as hints).
    teacher_p0: str = "p0"
    teacher_p1: str = "p1"
    teams_ego: str = "ego"
    teams_partner: str = "partner"
    teams_opp0: str = "opp0"
    teams_opp1: str = "opp1"


DEFAULT_PLAN = WarmStartPlan()


def map_param_name(name: str, plan: WarmStartPlan = DEFAULT_PLAN) -> list[str]:
    """
    Given a teacher param path containing p0/p1, return Teams target path(s).

    Example: '.../game/p0/percent/...' -> ['.../teams_game/ego/percent/...']
    """
    if f"/{plan.teacher_p0}/" in name or name.endswith(f"/{plan.teacher_p0}"):
        base = name.replace(f"/{plan.teacher_p0}/", f"/{plan.teams_ego}/")
        base = base.replace(f".{plan.teacher_p0}.", f".{plan.teams_ego}.")
        out = [base]
        if plan.partner_init == "copy_ego":
            out.append(
                base.replace(f"/{plan.teams_ego}/", f"/{plan.teams_partner}/").replace(
                    f".{plan.teams_ego}.", f".{plan.teams_partner}."
                )
            )
        return out

    if f"/{plan.teacher_p1}/" in name or name.endswith(f"/{plan.teacher_p1}"):
        base = name.replace(f"/{plan.teacher_p1}/", f"/{plan.teams_opp0}/")
        base = base.replace(f".{plan.teacher_p1}.", f".{plan.teams_opp0}.")
        out = [base]
        if plan.opp1_init == "copy_opp0":
            out.append(
                base.replace(f"/{plan.teams_opp0}/", f"/{plan.teams_opp1}/").replace(
                    f".{plan.teams_opp0}.", f".{plan.teams_opp1}."
                )
            )
        return out

    # Shared trunks (stage, controller head, etc.) keep the same name.
    return [name]


def build_assignment_table(
    teacher_names: Iterable[str],
    plan: WarmStartPlan = DEFAULT_PLAN,
) -> dict[str, str]:
    """
    teacher_param_name -> teams_param_name (one teacher leaf can fan out).

    Returns a flat dict of *destination* -> *source* for loading.
    """
    dest_to_src: dict[str, str] = {}
    for src in teacher_names:
        for dest in map_param_name(src, plan):
            dest_to_src[dest] = src
    return dest_to_src


def describe() -> str:
    return (
        f"Warm-start: p0->ego (+partner via {PARTNER_INIT}), "
        f"p1->opp0 (+opp1 via {OPP1_INIT}). "
        "Unmapped Teams leaves stay at init. Shared heads copy 1:1."
    )
