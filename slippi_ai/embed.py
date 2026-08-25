"""Compat shim: older pickles import ``slippi_ai.embed``.

That module moved to ``slippi_ai.tf.embed``. Re-export so unpickle works.
"""

from slippi_ai.tf.embed import *  # noqa: F403
from slippi_ai.tf import embed as _embed

# Explicit common names some pickles expect at module level.
ItemsType = _embed.ItemsType
EmbedConfig = _embed.EmbedConfig
PlayerConfig = _embed.PlayerConfig
ControllerConfig = _embed.ControllerConfig
ItemsConfig = _embed.ItemsConfig
