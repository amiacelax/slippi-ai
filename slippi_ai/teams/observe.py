"""
Future observation layout for Teams (documentation + shape helpers).

Stock medium-v2 embeds Game(p0, p1) only. Teams needs four player slots:

  embed = concat(ego, partner, opp0, opp1)   # or ego+partner vs opp0+opp1 towers

Until the network is rebuilt, PhillipTeams fakes partner awareness via
``doubles_target`` (which opponent the 1v1 brain sees).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeamsObserveSpec:
    """Target feature layout for a Teams policy (see jax_networks)."""

    slots: tuple[str, ...] = ("ego", "partner", "opp0", "opp1")
    include_partner_controller: bool = False
    include_enemy_controllers: bool = False
    # Compatibility: pad missing players with zeros so a 1v1 teacher can warm-start.
    pad_missing_to_four: bool = True

    @property
    def num_players(self) -> int:
        return len(self.slots)


DEFAULT_OBSERVE_SPEC = TeamsObserveSpec()


def describe_migration() -> str:
  return (
      "DONE: TeamsGame NamedTuple + TeamsParser(ports=ego/partner/opp0/opp1)\n"
      "DONE: TeamsEnvironment (4 controllers) + reward/rollout helpers\n"
      "DONE: compat_1v1.teams_game_to_1v1 for medium-v2 teacher bridge\n"
      "DONE: EmbedConfig.make_teams_game_embedding() in jax/embed.py\n"
      "DONE: warm_start param map p0→ego/partner, p1→opp0/opp1\n"
      "DONE: TeamsRolloutCollector hybrid path\n"
      "DONE: TeamsEnhancedEmbedModule (teams/jax_networks.py)\n"
      "DONE: trainer_bridge (build_actor + preserve_hybrid_rewards)\n"
      "DONE: LiveTeamsEnv + build_live_teams_actor (Step 4)\n"
      "NEXT: GPU run with TeamsRLConfig / real PPO\n"
  )
