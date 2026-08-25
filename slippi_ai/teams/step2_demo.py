"""
Step 2 demo — run on this Windows PC (JAX optional).

Proves: Teams embed wiring + warm-start map are ready.
Full nnx compile only runs if JAX/flax are installed (training box).

  cd C:\\Users\\langu\\slippi-ai
  python -m slippi_ai.teams.step2_demo
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\langu\slippi-ai")


def _ensure_path() -> None:
  if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_source_wiring() -> None:
  """JAX-free: prove the Teams module + embed helpers exist in source."""
  embed_py = (ROOT / "slippi_ai" / "jax" / "embed.py").read_text(encoding="utf-8")
  assert "def make_teams_game_embedding" in embed_py
  assert "def make_teams_state_action_embedding" in embed_py
  assert "def get_teams_state_action_embedding" in embed_py

  mod_py = (ROOT / "slippi_ai" / "teams" / "jax_networks.py").read_text(
      encoding="utf-8"
  )
  tree = ast.parse(mod_py)
  names = {
      n.name
      for n in tree.body
      if isinstance(n, (ast.ClassDef, ast.FunctionDef))
  }
  assert "TeamsEnhancedEmbedModule" in names
  assert "TeamsEmbeddings" in names
  assert "build_teams_embed_module" in names
  assert "raw_game.ego" in mod_py
  assert "raw_game.partner" in mod_py
  assert "raw_game.opp0" in mod_py
  assert "raw_game.opp1" in mod_py
  print("source wiring OK (TeamsEnhancedEmbedModule + make_teams_*)")


def check_warm_start_and_order() -> None:
  from slippi_ai.teams.networks_teams import TEAMS_CONCAT_ORDER, describe
  from slippi_ai.teams.warm_start import build_assignment_table, map_param_name

  assert TEAMS_CONCAT_ORDER[:4] == ("ego", "partner", "opp0", "opp1")
  mapped = map_param_name("policy/embed/game/p0/percent/scale")
  assert any("ego" in m for m in mapped)
  assert any("partner" in m for m in mapped)
  table = build_assignment_table(
      [
          "policy/embed/game/p0/percent/scale",
          "policy/embed/game/p1/x/scale",
      ]
  )
  assert any("ego" in d for d in table)
  assert any("opp0" in d for d in table)
  print("warm_start + concat order OK")
  print(" ", describe().splitlines()[0])


def try_jax_compile() -> str:
  """Optional: construct module and measure output width."""
  try:
    import jax  # noqa: F401
    from flax import nnx
  except ImportError as e:
    print(f"JAX compile: SKIPPED ({e})")
    print("  (Expected on Windows play venv — full compile on GPU box later.)")
    return "skipped"

  from slippi_ai.jax import embed as embed_lib
  from slippi_ai.teams.jax_networks import (
      TeamsEnhancedEmbedModule,
      build_teams_embed_module,
  )

  rngs = nnx.Rngs(0)
  cfg = embed_lib.EmbedConfig()
  enhanced = TeamsEnhancedEmbedModule.default_config()
  enhanced["use_items"] = False
  mod = build_teams_embed_module(
      rngs=rngs,
      embed_config=cfg,
      num_names=1,
      enhanced_config=enhanced,
  )
  dummy = mod.dummy(())
  out = mod(dummy)
  print(f"JAX compile: OK  output_size={mod.output_size}  out.shape={out.shape}")
  return "ok"


def main() -> int:
  _ensure_path()
  print("=== Teams IQ Step 2: Teams neural embed wiring ===")
  print()

  check_source_wiring()
  check_warm_start_and_order()
  status = try_jax_compile()

  print()
  if status == "ok":
    print("STEP 2 OK - Teams embed constructs under JAX.")
  else:
    print(
        "STEP 2 OK - Teams embed code + warm-start ready "
        "(JAX compile deferred to GPU box)."
    )
  print("Next (step 3): python -m slippi_ai.teams.step3_demo  (see STEP3.md)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
