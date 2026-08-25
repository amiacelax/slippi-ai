"""
Teams RL config — placeholders filled with host answers + foresight defaults.

When you disagree, edit CONFIRMED/PLACEHOLDERS; run_lib reads this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import melee

from slippi_ai.teams.curriculum import CurriculumWeights, default_curriculum
from slippi_ai.teams.reward import TeamsRewardConfig


@dataclass
class TeamsRLConfig:
    # --- Host-confirmed -------------------------------------------------
    teammate_death_penalty: float = 0.85
    seat1_role: str = "aggro"
    seat2_role: str = "support"
    character: melee.Character = melee.Character.FOX
    floaty_chars: tuple[str, ...] = (
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
    )
    push_wrappers: bool = True

    # --- Foresight placeholders (override when back) --------------------
    # PLACEHOLDER: FD first; BF/DL later for platforms.
    stage: melee.Stage = melee.Stage.FINAL_DESTINATION
    team_a_ports: tuple[int, int] = (1, 2)
    team_b_ports: tuple[int, int] = (3, 4)
    # PLACEHOLDER: train both seats on team A; team B is frozen teacher copies.
    train_ports: tuple[int, ...] = (1, 2)
    opponent_mode: str = "frozen_teacher"  # or "co_train" later
    # PLACEHOLDER: hybrid = medium-v2 via compat_1v1 until 4p embed exists.
    policy_mode: str = "hybrid_1v1_teacher"
    focus_opponent: str = "opp0"  # which enemy fills p1 for hybrid
    hours: float = 3.0
    # PLACEHOLDER reward extras
    damage_ratio: float = 0.01
    stalling_penalty: float = 0.5
    approaching_factor: float = 0.02

    curriculum: CurriculumWeights = field(default_factory=default_curriculum)

    def reward_config_for_role(self, role: str) -> TeamsRewardConfig:
        return TeamsRewardConfig(
            damage_ratio=self.damage_ratio,
            teammate_death_penalty=self.teammate_death_penalty,
            stalling_penalty=self.stalling_penalty,
            approaching_factor=self.approaching_factor,
            curriculum=self.curriculum,
            role=role,
        )

    def dolphin_players(self) -> dict:
        from slippi_ai import dolphin

        char = self.character
        return {
            self.team_a_ports[0]: dolphin.AI(character=char, costume=0),
            self.team_a_ports[1]: dolphin.AI(character=char, costume=0),
            self.team_b_ports[0]: dolphin.AI(character=char, costume=1),
            self.team_b_ports[1]: dolphin.AI(character=char, costume=1),
        }


# Human-readable list of every PLACEHOLDER for the morning review.
PLACEHOLDER_DECISIONS = [
    {
        "id": "p_stage",
        "assumed": "FINAL_DESTINATION",
        "why": "Simplest blastzones for first Teams RL",
        "alts": ["BATTLEFIELD", "DREAMLAND", "YOSHIS_STORY"],
    },
    {
        "id": "p_opponent_mode",
        "assumed": "frozen_teacher",
        "why": "Stable opponent while team A learns; co-train later",
        "alts": ["co_train", "self_play_all_four"],
    },
    {
        "id": "p_policy_mode",
        "assumed": "hybrid_1v1_teacher",
        "why": "medium-v2 works today; full 4p embed is next real model change",
        "alts": ["full_teams_embed"],
    },
    {
        "id": "p_focus",
        "assumed": "opp0",
        "why": "Primary enemy slot; wrappers/retarget can swap who is opp0",
        "alts": ["dynamic_via_doubles_target"],
    },
    {
        "id": "p_hours",
        "assumed": 3.0,
        "why": "Matches Fox fine-tune evening budget",
        "alts": [1.0, 6.0, 12.0],
    },
]
