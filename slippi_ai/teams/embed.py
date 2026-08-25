"""
Numpy feature embedding for TeamsGame (no JAX required).

This is a **scaffold** for the future JAX EmbedConfig. It proves the 4-player
layout and gives a stable feature width for tests. The real network will reuse
stock player embed pieces from ``slippi_ai.jax.embed``.

PLACEHOLDER: scales / one-hots match medium-v2-ish conventions but are not
bit-identical to the trained embed. Do not train against this numpy path —
use it for reward/debug and as a size reference for the JAX port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slippi_ai.teams.types_teams import TeamsGame
from slippi_ai.types import Player

# Keep in sync with jax embed OneHotEmbedding sizes where possible.
ACTION_SIZE = 0x195  # melee action enum upper bound used elsewhere
CHAR_SIZE = 32
STAGE_SIZE = 64


@dataclass(frozen=True)
class TeamsEmbedConfig:
    xy_scale: float = 0.05
    shield_scale: float = 0.01
    include_controller: bool = False  # PLACEHOLDER: partner controller off
    mask_dead: bool = True


def _is_dead_action(action: np.ndarray) -> np.ndarray:
    return action <= 0xA


def embed_player_numpy(
    player: Player,
    *,
    cfg: TeamsEmbedConfig,
) -> np.ndarray:
    """
    Embed one Player nest → [..., D] float32.

    Time/batch dims are preserved from the input arrays.
    """
    x = player.x.astype(np.float32) * cfg.xy_scale
    y = player.y.astype(np.float32) * cfg.xy_scale
    percent = player.percent.astype(np.float32) / 100.0
    facing = player.facing.astype(np.float32)
    on_ground = player.on_ground.astype(np.float32)
    invuln = player.invulnerable.astype(np.float32)
    jumps = player.jumps_left.astype(np.float32) / 2.0
    shield = player.shield_strength.astype(np.float32) * cfg.shield_scale

    # One-hot style soft embeds (dense, not sparse) for scaffolding size.
    action = player.action.astype(np.int32)
    char = player.character.astype(np.int32)
    action_oh = np.eye(ACTION_SIZE, dtype=np.float32)[np.clip(action, 0, ACTION_SIZE - 1)]
    char_oh = np.eye(CHAR_SIZE, dtype=np.float32)[np.clip(char, 0, CHAR_SIZE - 1)]

    parts = [
        x[..., None],
        y[..., None],
        percent[..., None],
        facing[..., None],
        on_ground[..., None],
        invuln[..., None],
        jumps[..., None],
        shield[..., None],
        action_oh,
        char_oh,
    ]
    out = np.concatenate(parts, axis=-1)

    if cfg.mask_dead:
        dead = _is_dead_action(player.action)
        out = np.where(dead[..., None], 0.0, out)

    return out.astype(np.float32)


def player_embed_size(cfg: TeamsEmbedConfig | None = None) -> int:
    cfg = cfg or TeamsEmbedConfig()
    # 8 scalars + action one-hot + char one-hot
    return 8 + ACTION_SIZE + CHAR_SIZE


def embed_teams_game_numpy(
    game: TeamsGame,
    *,
    cfg: TeamsEmbedConfig | None = None,
) -> np.ndarray:
    """
    Concatenate ego|partner|opp0|opp1|stage → [..., D].

    Order matches TeamsObserveSpec.slots.
    """
    cfg = cfg or TeamsEmbedConfig()
    players = [
        embed_player_numpy(game.ego, cfg=cfg),
        embed_player_numpy(game.partner, cfg=cfg),
        embed_player_numpy(game.opp0, cfg=cfg),
        embed_player_numpy(game.opp1, cfg=cfg),
    ]
    stage = game.stage.astype(np.int32)
    stage_oh = np.eye(STAGE_SIZE, dtype=np.float32)[
        np.clip(stage, 0, STAGE_SIZE - 1)
    ]
    return np.concatenate([*players, stage_oh], axis=-1)


def teams_embed_size(cfg: TeamsEmbedConfig | None = None) -> int:
    cfg = cfg or TeamsEmbedConfig()
    return 4 * player_embed_size(cfg) + STAGE_SIZE


def describe_jax_port() -> str:
    return (
        "JAX port plan (PLACEHOLDER until implemented):\n"
        "1. Copy PlayerConfig.make_embedding() four times as ego/partner/opp0/opp1\n"
        "2. Build StructEmbedding over TeamsGame fields\n"
        "3. EnhancedEmbedModule: concat four _embed_player calls\n"
        "4. Warm-start: map medium-v2 p0→ego, p1→opp0; copy opp0→opp1; "
        "copy ego→partner or random init partner\n"
        "5. Until then: use compat_1v1 + this numpy embed only for debugging"
    )
