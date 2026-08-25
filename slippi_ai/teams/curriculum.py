"""
A–F curriculum weights for Teams IQ.

Source: host doubles tier list (Aug 2026) + floaty-at-bay as low B*.
F-tier is human/meta — weight 0 for the neural net.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CurriculumWeights:
    """How strongly each doubles idea should influence reward / wrappers."""

    # A — teach first
    team_roles: float = 1.0
    watch_teammate: float = 1.0
    prefer_2v1_on_stage: float = 1.0

    # B
    pass_to_teammate: float = 0.7
    space_around_teammate: float = 0.7
    alternate_pressure: float = 0.7
    # Conditional — only when a slow/floaty is in the match
    floaty_at_bay: float = 0.55

    # C
    sandwich: float = 0.45
    check_percent_before_throw: float = 0.45
    smart_edgeguards: float = 0.45

    # D / E — later
    stall_no_hero: float = 0.2
    edgeguard_jobs: float = 0.2
    let_hitstun_breathe: float = 0.15
    backup_before_pass: float = 0.15

    # F — off for the model
    consistent_teammate: float = 0.0
    communication: float = 0.0

    notes: dict[str, str] = field(default_factory=dict)


def default_curriculum() -> CurriculumWeights:
    return CurriculumWeights(
        notes={
            "floaty_at_bay": (
                "low B*; Puff/Peach/Luigi/Sheik/ICs/Zelda/Doc/Mario/Pikachu/"
                "Mewtwo/Yoshi when present. Yoshi: edgeguard with Falco/Marth "
                "dair spike at low % or >=90% only."
            ),
            "source": "japaneselanguagementor.com/doubles/ Player 1 ranking Aug 2026",
            "trainer_gate": "4-player env + partner in observation before GPU fine-tune",
            "host_answers": "roles=aggro/support, death=0.85, fox dittos, wrappers=yes",
        }
    )


# Host-confirmed B* list (updated Aug 25 2026).
# Values verified against libmelee on this machine.
FLOATY_OR_SLOW_CHARS = frozenset(
    {
        15,  # JIGGLYPUFF
        9,   # PEACH
        17,  # LUIGI
        7,   # SHEIK
        10,  # POPO (Ice Climbers)
        11,  # NANA
        19,  # ZELDA
        21,  # DOC (Dr. Mario)
        0,   # MARIO
        12,  # PIKACHU
        16,  # MEWTWO
        14,  # YOSHI
    }
)

FLOATY_OR_SLOW_NAMES = frozenset(
    {
        "JIGGLYPUFF",
        "PEACH",
        "LUIGI",
        "SHEIK",
        "POPO",
        "NANA",
        "ZELDA",
        "DOC",
        "MARIO",
        "PIKACHU",
        "MEWTWO",
        "YOSHI",
    }
)

# Yoshi edgeguard: rare; Falco/Marth dair spike at low % or when DJ armor soft (>=90).
YOSHI_CHAR = 14
YOSHI_SPIKE_LOW_PCT = 40.0
YOSHI_SPIKE_HIGH_PCT = 90.0
SPIKE_EDGE_CHARS = frozenset({22, 18})  # FALCO, MARTH


def yoshi_spike_edgeguard_ok(*, ego_char: int, yoshi_percent: float) -> bool:
    """True when Falco/Marth should chase Yoshi offstage for a spike."""
    if int(ego_char) not in SPIKE_EDGE_CHARS:
        return False
    pct = float(yoshi_percent)
    return pct <= YOSHI_SPIKE_LOW_PCT or pct >= YOSHI_SPIKE_HIGH_PCT
