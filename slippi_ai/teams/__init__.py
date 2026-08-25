"""
Teams (2v2) extensions for slippi-ai.

This package is intentionally **parallel** to the stock 1v1 Game(p0, p1) path.
Nothing here changes Arrow schemas or breaks medium-v2 loading.

Layout (ego-centric):
  ego     = controlled agent
  partner = teammate
  opp0    = primary opponent (usually on-stage focus)
  opp1    = secondary opponent

Curriculum priorities come from PhillipTeams docs/DOUBLES_CURRICULUM.md
(A = teach first).
"""

from slippi_ai.teams.curriculum import CurriculumWeights, default_curriculum
from slippi_ai.teams.layout import TeamsPorts, resolve_teams_ports
from slippi_ai.teams.reward import TeamsRewardConfig, compute_teams_rewards
from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.teams.parse import TeamsParser, stack_teams_games
from slippi_ai.teams.rollout import rewards_from_frames
from slippi_ai.teams.config import TeamsRLConfig
from slippi_ai.teams.focus import pick_focus

__all__ = [
    "CurriculumWeights",
    "TeamsGame",
    "TeamsParser",
    "TeamsPorts",
    "TeamsRLConfig",
    "TeamsRewardConfig",
    "compute_teams_rewards",
    "default_curriculum",
    "pick_focus",
    "resolve_teams_ports",
    "rewards_from_frames",
    "stack_teams_games",
]
