"""
Host answers locked (Aug 24 2026) + remaining assumptions.

Code uses CONFIRMED where filled in; ASSUMED only for unasked defaults.
"""

# --- CONFIRMED by host -------------------------------------------------

CONFIRMED = {
    "roles": {
        "seat_1_port_1": "aggro",
        "seat_2_port_2": "support",
        "roles_can_swap_when_partner_dying": True,
    },
    "teammate_death_penalty": 0.85,
    "first_gpu_chars": "Fox/Fox vs Fox/Fox",
    "floaty_at_bay_chars": [
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
    ],
    "yoshi_spike_edge": {
        "chars": ["FALCO", "MARTH"],
        "low_pct": 40,
        "high_pct": 90,
        "note": "dair spike only; otherwise treat Yoshi as keep-at-bay on stage",
    },
    "push_wrappers_before_gpu": True,
}

# --- Still assumed (not asked / low stakes) ----------------------------

ASSUMED = {
    "training_ports": {
        "team_a": [1, 2],
        "team_b": [3, 4],
    },
    "prefer_2v1_on_stage": True,
    "floaty_at_bay_tier": "low_B_star",
    "do_not_break_1v1_medium_v2": True,
    "gpu_training_tonight": False,
    "first_gpu_experiment": "4p_teams_from_medium_v2_teacher_fox_ditto_teams",
    "stage_for_first_rl": "FINAL_DESTINATION",
    # merge confirmed floaty list into assumed for older imports
    "floaty_chars": CONFIRMED["floaty_at_bay_chars"],
    "roles": CONFIRMED["roles"],
    "teammate_death_penalty": CONFIRMED["teammate_death_penalty"],
}

# --- All questions answered --------------------------------------------

QUESTIONS = []
