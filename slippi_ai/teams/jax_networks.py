"""
Teams EnhancedEmbedModule — 4-player concat (ego|partner|opp0|opp1).

Requires JAX + flax (training box). Importing this module on a play-only
Windows venv will fail; use ``step2_demo`` which skips gracefully.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai.data import StateAction, Action
from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax.networks import (
    Array,
    ControllerRNN,
    EmbedModule,
    EnhancedEmbedModule,
    MultiEmbed,
)
from slippi_ai.jax import jax_utils
from slippi_ai.types import Controller, Nana, Player, S
from slippi_ai.teams.types_teams import TeamsGame

PlayerOrNana = tp.TypeVar("PlayerOrNana", Player, Nana)


class TeamsEmbeddings(tp.NamedTuple, tp.Generic[Action]):
  """Same as Embeddings, but game nest is TeamsGame (4 players)."""

  config: embed_lib.EmbedConfig[Action]
  num_names: int
  embed_game: embed_lib.Embedding[TeamsGame, TeamsGame]
  embed_action: embed_lib.Embedding[Controller, Action]
  embed_state_action: embed_lib.Embedding[
      StateAction[tp.Any, Controller], StateAction[tp.Any, Action]
  ]

  @classmethod
  def init(
      cls,
      config: embed_lib.EmbedConfig[Action],
      num_names: int,
      embed_action: tp.Optional[embed_lib.Embedding[Controller, Action]] = None,
  ) -> tp.Self:
    embed_game = config.make_teams_game_embedding()
    embed_action = (
        config.make_controller_embedding()
        if embed_action is None
        else embed_action
    )
    embed_state_action = embed_lib.get_teams_state_action_embedding(
        embed_game=embed_game,
        embed_action=embed_action,
        num_names=num_names,
    )
    return cls(
        config=config,
        num_names=num_names,
        embed_game=embed_game,
        embed_action=embed_action,
        embed_state_action=embed_state_action,
    )


class TeamsEnhancedEmbedModule(nnx.Module, EmbedModule[Action]):
  """
  Drop-in parallel to EnhancedEmbedModule for TeamsGame.

  Concat order: ego | partner | opp0 | opp1 | stage | randall | fod | items |
  name | controller.
  """

  @classmethod
  def default_config(cls) -> dict[str, tp.Any]:
    return EnhancedEmbedModule.default_config()

  def __init__(
      self,
      rngs: nnx.Rngs,
      embeddings: TeamsEmbeddings[Action],
      hidden_size: int,
      item_mlp_layers: int,
      rnn_cell: str = "lstm",
      use_self_nana: bool = True,
      use_controller_rnn: bool = False,
      use_learned_char: bool = True,
      use_learned_action: bool = True,
      use_char_action_joint: bool = True,
      use_item_sum: bool = True,
      use_items: bool = True,
      hybrid_embed: bool = False,
  ):
    self._use_self_nana = use_self_nana
    self._use_learned_char = use_learned_char
    self._hybrid_embed = hybrid_embed
    self._use_learned_action = use_learned_action
    self._use_char_action_joint = use_char_action_joint
    self._use_item_sum = use_item_sum
    self._use_items = use_items

    self._embed_game = embeddings.embed_game
    self._embed_controller = embeddings.embed_action
    self._embed_state_action = embeddings.embed_state_action

    self._item_embedding = embeddings.config.make_item_embedding()
    self._item_mlp = jax_utils.MLP(
        rngs=rngs,
        input_size=self._item_embedding.size,
        features=[hidden_size] * item_mlp_layers,
    )

    self._use_controller_rnn = use_controller_rnn
    if use_controller_rnn:
      self._controller_rnn = ControllerRNN(
          rngs=rngs,
          embed_controller=self._embed_controller,
          hidden_size=hidden_size,
          rnn_cell=rnn_cell,
      )

    embed_char = embed_lib.embed_char
    self._embed_char = nnx.Embed(
        num_embeddings=embed_char.size,
        features=hidden_size,
        rngs=rngs,
    )
    self._embed_action = nnx.Embed(
        num_embeddings=embed_lib.embed_action.size,
        features=hidden_size,
        rngs=rngs,
    )
    self._embed_char_action = MultiEmbed(
        sizes=(embed_char.size, embed_lib.embed_action.size),
        features=hidden_size,
        rngs=rngs,
        embedding_init=nnx.initializers.zeros,
    )

    output_shape = jax_utils.eval_shape_method(self.__call__, self.dummy(()))
    assert output_shape.ndim == 1
    self._output_size = output_shape.shape[0]

  @property
  def output_size(self) -> int:
    return self._output_size

  def dummy(self, shape: S) -> StateAction[S, Action]:
    return self._embed_state_action.dummy(shape)

  def encode(
      self, state_action: StateAction[S, Controller]
  ) -> StateAction[S, Action]:
    return self._embed_state_action.from_state(state_action)

  def encode_game(self, game: TeamsGame[S]) -> TeamsGame[S]:
    return self._embed_game.from_state(game)

  def _embed_player_or_nana(
      self, raw: PlayerOrNana, default: PlayerOrNana
  ) -> Array:
    if self._use_learned_action:
      action = self._embed_action(raw.action)
      if self._use_char_action_joint:
        action = action + self._embed_char_action(raw.character, raw.action)
      if self._hybrid_embed:
        action = jnp.concatenate([action, default.action], axis=-1)
    else:
      action = default.action

    if self._use_learned_char:
      char = self._embed_char(raw.character)
      if self._hybrid_embed:
        char = jnp.concatenate([char, default.character], axis=-1)
    else:
      char = default.character

    parts = [
        default.percent,
        default.facing,
        default.x,
        default.y,
        action,
        default.invulnerable,
        char,
        default.jumps_left,
        default.shield_strength,
        default.on_ground,
    ]
    if isinstance(default, Nana):
      parts.append(default.exists)
    return jnp.concatenate(parts, axis=-1)

  def _embed_player(
      self, raw: Player, default: Player, with_nana: bool
  ) -> Array:
    parts = [self._embed_player_or_nana(raw, default)]
    if with_nana:
      parts.append(self._embed_player_or_nana(raw.nana, default.nana))
    return jnp.concatenate(parts, axis=-1)

  def __call__(self, state_action: StateAction[tp.Any, Action]) -> Array:
    raw_game = state_action.state
    default_state_action_embed = self._embed_state_action.map(
        lambda e, v: e(v), state_action
    )
    default_game = default_state_action_embed.state

    parts = [
        self._embed_player(
            raw_game.ego, default_game.ego, with_nana=self._use_self_nana
        ),
        self._embed_player(
            raw_game.partner, default_game.partner, with_nana=True
        ),
        self._embed_player(raw_game.opp0, default_game.opp0, with_nana=True),
        self._embed_player(raw_game.opp1, default_game.opp1, with_nana=True),
        default_game.stage,
        *default_game.randall,
        *default_game.fod_platforms,
    ]

    if self._use_items:
      stacked_items = jax.tree.map(
          lambda *args: jnp.stack(args, axis=-1), *raw_game.items
      )
      item_embed = self._item_embedding(stacked_items)
      assert item_embed.shape[-2:] == (
          len(raw_game.items),
          self._item_embedding.size,
      )
      if self._use_item_sum:
        item_embed = self._item_mlp(item_embed)
        item_embed = jnp.where(stacked_items.exists[..., None], item_embed, 0)
        items_embed = jnp.sum(item_embed, axis=-2)
      else:
        items_embed = item_embed.reshape(*item_embed.shape[:-2], -1)
      parts.append(items_embed)

    parts.append(tp.cast(Array, default_state_action_embed.name))

    if self._use_controller_rnn:
      parts.append(self._controller_rnn(state_action.action))
    else:
      parts.append(self._embed_controller(state_action.action))

    return jnp.concatenate(parts, axis=-1)


def build_teams_embed_module(
    rngs: nnx.Rngs,
    embed_config: embed_lib.EmbedConfig[Action],
    num_names: int,
    enhanced_config: tp.Optional[dict] = None,
    embed_action: tp.Optional[embed_lib.Embedding[Controller, Action]] = None,
) -> TeamsEnhancedEmbedModule:
  """Factory: TeamsEmbeddings + TeamsEnhancedEmbedModule."""
  embeddings = TeamsEmbeddings.init(
      config=embed_config,
      num_names=num_names,
      embed_action=embed_action,
  )
  cfg = enhanced_config or TeamsEnhancedEmbedModule.default_config()
  return TeamsEnhancedEmbedModule(
      rngs=rngs,
      embeddings=embeddings,
      **cfg,
  )


def jax_available() -> bool:
  return True  # import succeeded if this module loaded
