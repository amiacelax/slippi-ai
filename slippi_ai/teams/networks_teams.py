"""
Notes + pure helpers for Teams EnhancedEmbedModule.

Real nnx Module: ``teams.jax_networks.TeamsEnhancedEmbedModule`` (needs JAX).
This file stays JAX-free for selftest / Windows play venv.
"""

from __future__ import annotations

from slippi_ai.teams.embed import teams_embed_size
from slippi_ai.teams.observe import describe_migration


TEAMS_CONCAT_ORDER = ("ego", "partner", "opp0", "opp1", "stage", "controller")


def expected_feature_width_numpy() -> int:
  """Matches teams.embed.embed_teams_game_numpy (no controller/name yet)."""
  return teams_embed_size()


def describe() -> str:
  return (
      describe_migration()
      + "\n\nConcat order: "
      + " | ".join(TEAMS_CONCAT_ORDER)
      + f"\nNumpy feature width (players+stage): {expected_feature_width_numpy()}"
      + "\nJAX module: slippi_ai.teams.jax_networks.TeamsEnhancedEmbedModule"
  )
