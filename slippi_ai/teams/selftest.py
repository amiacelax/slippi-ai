"""Smoke tests for teams layout, parse, reward, rollout (no Dolphin / GPU)."""

from __future__ import annotations

import sys
import types

import numpy as np

from slippi_ai.teams.curriculum import FLOATY_OR_SLOW_CHARS, default_curriculum
from slippi_ai.teams.decisions import ASSUMED, CONFIRMED, QUESTIONS
from slippi_ai.teams.layout import all_seat_layouts, default_seat_ports, resolve_teams_ports
from slippi_ai.teams.parse import (
    TeamsParser,
    resolve_ports_for_ego,
    stack_teams_games,
    team_map_from_gamestate,
)
from slippi_ai.teams.reward import TeamsRewardConfig, compute_teams_rewards
from slippi_ai.teams.rollout import rewards_from_frames, summarize_frames
from slippi_ai.types import (
    Buttons,
    Controller,
    Nana,
    Player,
    Stick,
)


def _empty_nana(T: int) -> Nana:
    z_u8 = np.zeros(T, dtype=np.uint8)
    z_u16 = np.zeros(T, dtype=np.uint16)
    z_f = np.zeros(T, dtype=np.float32)
    z_b = np.zeros(T, dtype=np.bool_)
    return Nana(
        exists=z_b,
        percent=z_u16,
        facing=z_b,
        x=z_f,
        y=z_f,
        action=z_u16,
        invulnerable=z_b,
        character=z_u8,
        jumps_left=z_u8,
        shield_strength=z_f,
        on_ground=z_b,
    )


def _fake_player(T: int, *, x: float = 0.0, action: int = 0xE, char: int = 2) -> Player:
    z_u8 = np.zeros(T, dtype=np.uint8)
    z_u16 = np.full(T, action, dtype=np.uint16)
    z_f = np.zeros(T, dtype=np.float32)
    z_b = np.ones(T, dtype=np.bool_)
    stick = Stick(x=z_f.copy(), y=z_f.copy())
    buttons = Buttons(
        A=np.zeros(T, dtype=np.bool_),
        B=np.zeros(T, dtype=np.bool_),
        X=np.zeros(T, dtype=np.bool_),
        Y=np.zeros(T, dtype=np.bool_),
        Z=np.zeros(T, dtype=np.bool_),
        L=np.zeros(T, dtype=np.bool_),
        R=np.zeros(T, dtype=np.bool_),
        D_UP=np.zeros(T, dtype=np.bool_),
    )
    ctrl = Controller(
        main_stick=stick,
        c_stick=Stick(x=z_f.copy(), y=z_f.copy()),
        shoulder=z_f.copy(),
        buttons=buttons,
    )
    return Player(
        percent=np.zeros(T, dtype=np.uint16),
        facing=z_b.copy(),
        x=np.full(T, x, dtype=np.float32),
        y=np.zeros(T, dtype=np.float32),
        action=z_u16,
        invulnerable=np.zeros(T, dtype=np.bool_),
        character=np.full(T, char, dtype=np.uint8),
        jumps_left=np.full(T, 2, dtype=np.uint8),
        shield_strength=np.full(T, 60.0, dtype=np.float32),
        on_ground=z_b.copy(),
        controller=ctrl,
        nana=_empty_nana(T),
    )


def test_layout() -> None:
    ports = default_seat_ports(1)
    assert ports.as_tuple() == (1, 2, 3, 4)
    ports2 = default_seat_ports(2)
    assert ports2.ego == 2 and ports2.partner == 1
    resolved = resolve_teams_ports(ego=1, team_of={1: 0, 2: 0, 3: 1, 4: 1})
    assert resolved.partner == 2
    assert set(resolved.enemies()) == {3, 4}
    assert len(all_seat_layouts()) == 2
    print("layout OK")


def test_reward_teammate_death() -> None:
    T = 8
    ego = _fake_player(T, x=-10)
    partner = _fake_player(T, x=-5)
    partner_action = np.full(T, 0xE, dtype=np.uint16)
    partner_action[4:] = 0x0
    partner = partner._replace(action=partner_action)
    opp0 = _fake_player(T, x=20)
    opp1 = _fake_player(T, x=30)
    stage = np.full(T, 32, dtype=np.uint8)

    cfg = TeamsRewardConfig(role="aggro")
    r = compute_teams_rewards(
        ego=ego, partner=partner, opp0=opp0, opp1=opp1, stage=stage, config=cfg
    )
    assert r.shape == (T - 1,)
    assert r[3] < -0.5, f"expected teammate death penalty, got {r[3]}"
    print("reward teammate-death OK", float(r[3]))


def test_curriculum() -> None:
    c = default_curriculum()
    assert c.watch_teammate == 1.0
    assert c.communication == 0.0
    assert c.floaty_at_bay == 0.55
    assert FLOATY_OR_SLOW_CHARS >= frozenset({15, 9, 17, 7, 10, 11, 19, 21, 0, 12, 16, 14})
    assert "YOSHI" in CONFIRMED["floaty_at_bay_chars"]
    assert CONFIRMED["yoshi_spike_edge"]["high_pct"] == 90
    assert set(CONFIRMED["floaty_at_bay_chars"]) >= {
        "JIGGLYPUFF",
        "PEACH",
        "LUIGI",
        "SHEIK",
        "POPO",
        "ZELDA",
        "DOC",
        "MARIO",
        "PIKACHU",
        "MEWTWO",
        "YOSHI",
    }
    assert QUESTIONS == []
    print("curriculum OK")


def _fake_libmelee_player(
    *,
    team_id: int,
    char: int = 2,
    x: float = 0.0,
    action: int = 14,
):
    import melee

    pos = types.SimpleNamespace(x=x, y=0.0)
    cs = melee.ControllerState()
    return types.SimpleNamespace(
        percent=0,
        facing=True,
        position=pos,
        action=types.SimpleNamespace(value=action),
        character=types.SimpleNamespace(value=char),
        jumps_left=2,
        shield_strength=60.0,
        on_ground=True,
        invulnerable=False,
        nana=None,
        controller_state=cs,
        team_id=team_id,
        off_stage=False,
        stock=4,
    )


def test_parse_and_stack() -> None:
    import melee

    players = {
        1: _fake_libmelee_player(team_id=0, x=-20),
        2: _fake_libmelee_player(team_id=0, x=-10),
        3: _fake_libmelee_player(team_id=1, x=10, char=15),  # puff
        4: _fake_libmelee_player(team_id=1, x=20),
    }
    gs = types.SimpleNamespace(
        players=players,
        stage=melee.Stage.FINAL_DESTINATION,
        frame=0,
        fod_platforms=None,
        projectiles=[],
        is_teams=True,
    )
    team_of = team_map_from_gamestate(gs)  # type: ignore[arg-type]
    assert team_of[1] == 0 and team_of[3] == 1

    ports = resolve_ports_for_ego(gs, 1)  # type: ignore[arg-type]
    assert ports.as_tuple() == (1, 2, 3, 4)

    parser = TeamsParser(ports)
    frames = []
    for i in range(5):
        gs.frame = i
        # Kill partner on last frames
        if i >= 3:
            players[2].action = types.SimpleNamespace(value=0)
        frames.append(parser.get_teams_game(gs))  # type: ignore[arg-type]

    stacked = stack_teams_games(frames)
    assert stacked.ego.x.shape == (5,)
    r = rewards_from_frames(frames, config=TeamsRewardConfig(role="aggro"))
    assert r.shape == (4,)
    summary = summarize_frames(frames)
    assert summary["frames"] == 4
    print("parse+stack+rollout OK", summary)
    print("  puff char in opp0:", int(frames[0].opp0.character))


def test_env_import() -> None:
    from slippi_ai.teams.env import TeamsEnvironment, make_four_ai_players
    from slippi_ai.dolphin import Dolphin

    # Construction without dolphin path should fail later; just import + player map.
    players = make_four_ai_players()
    assert set(players) == {1, 2, 3, 4}
    assert players[1].costume == 0 and players[3].costume == 1
    # Confirm Dolphin accepts enable_teams kwarg
    import inspect

    sig = inspect.signature(Dolphin.__init__)
    assert "enable_teams" in sig.parameters
    print("env import OK")


def test_compat_1v1() -> None:
    from slippi_ai.teams.compat_1v1 import teams_game_to_1v1
    from slippi_ai.teams.types_teams import TeamsGame
    from slippi_ai.types import FoDPlatforms, Randall
    from slippi_db.parse_libmelee import _EMPTY_ITEMS

    T = 3
    ego = _fake_player(T, x=-10)
    partner = _fake_player(T, x=-5)
    opp0 = _fake_player(T, x=10, char=15)
    opp1 = _fake_player(T, x=20)
    tg = TeamsGame(
        ego=ego,
        partner=partner,
        opp0=opp0,
        opp1=opp1,
        stage=np.full(T, 32, dtype=np.uint8),
        randall=Randall(x=np.zeros(T, np.float32), y=np.zeros(T, np.float32)),
        fod_platforms=FoDPlatforms(
            left=np.zeros(T, np.float32), right=np.zeros(T, np.float32)
        ),
        items=_EMPTY_ITEMS,
    )
    g = teams_game_to_1v1(tg, focus="opp0")
    assert g.p0.x[0] == -10
    assert int(g.p1.character[0]) == 15
    g2 = teams_game_to_1v1(tg, focus="opp1")
    assert int(g2.p1.character[0]) == 2
    print("compat_1v1 OK")
    return tg


def test_embed_and_focus(tg=None) -> None:
    from slippi_ai.teams.embed import (
        embed_teams_game_numpy,
        teams_embed_size,
        player_embed_size,
    )
    from slippi_ai.teams.focus import pick_focus
    from slippi_ai.teams.types_teams import TeamsGame
    from slippi_ai.types import FoDPlatforms, Randall
    from slippi_db.parse_libmelee import _EMPTY_ITEMS

    if tg is None:
        T = 4
        tg = TeamsGame(
            ego=_fake_player(T, x=-10),
            partner=_fake_player(T, x=-5),
            opp0=_fake_player(T, x=10, char=15),  # puff on stage
            opp1=_fake_player(T, x=120),  # far / off-ish
            stage=np.full(T, 32, dtype=np.uint8),
            randall=Randall(x=np.zeros(T, np.float32), y=np.zeros(T, np.float32)),
            fod_platforms=FoDPlatforms(
                left=np.zeros(T, np.float32), right=np.zeros(T, np.float32)
            ),
            items=_EMPTY_ITEMS,
        )

    feat = embed_teams_game_numpy(tg)
    assert feat.shape[-1] == teams_embed_size()
    assert feat.shape[0] == tg.ego.x.shape[0]
    assert player_embed_size() * 4 + 64 == teams_embed_size()

    pick = pick_focus(tg, role="aggro")
    assert pick.focus in ("opp0", "opp1")
    # Puff on stage at x=10 should beat far enemy
    assert pick.focus == "opp0", pick
    print("embed+focus OK", feat.shape, pick.focus, pick.reason)


def test_run_dry() -> None:
    from slippi_ai.teams.run_lib import run_dry
    from slippi_ai.teams.config import TeamsRLConfig, PLACEHOLDER_DECISIONS

    assert run_dry(TeamsRLConfig()) == 0
    assert len(PLACEHOLDER_DECISIONS) >= 4
    print("run_dry OK")


def test_update_teams_rewards_traj() -> None:
    from slippi_ai.teams.trajectory import (
        TeamsTrajectory,
        update_teams_rewards,
        role_for_port,
    )
    from slippi_ai.teams.types_teams import TeamsGame
    from slippi_ai.types import FoDPlatforms, Randall
    from slippi_db.parse_libmelee import _EMPTY_ITEMS

    T = 6
    partner = _fake_player(T, x=-5)
    partner_action = np.full(T, 0xE, dtype=np.uint16)
    partner_action[3:] = 0
    partner = partner._replace(action=partner_action)
    tg = TeamsGame(
        ego=_fake_player(T, x=-10),
        partner=partner,
        opp0=_fake_player(T, x=10),
        opp1=_fake_player(T, x=20),
        stage=np.full(T, 32, dtype=np.uint8),
        randall=Randall(x=np.zeros(T, np.float32), y=np.zeros(T, np.float32)),
        fod_platforms=FoDPlatforms(
            left=np.zeros(T, np.float32), right=np.zeros(T, np.float32)
        ),
        items=_EMPTY_ITEMS,
    )
    # Minimal dummy actions — SampleOutputs structure varies; use object stub
    dummy_actions = types.SimpleNamespace()  # type: ignore
    traj = TeamsTrajectory(
        states=tg,
        name=np.zeros(T, dtype=np.int32),
        actions=dummy_actions,  # type: ignore[arg-type]
        rewards=np.zeros(T - 1, dtype=np.float32),
        is_resetting=np.zeros(T, dtype=np.bool_),
        initial_state=None,
        delayed_actions=[],
    )
    out = update_teams_rewards(traj, role="aggro")
    assert out.rewards.shape == (T - 1,)
    assert out.rewards[2] < -0.5
    assert role_for_port(1) == "aggro"
    assert role_for_port(2) == "support"
    print("update_teams_rewards OK", float(out.rewards[2]))


def test_fake_rollout() -> None:
    from slippi_ai.teams.fake_rollout import fake_hybrid_rollout

    r = fake_hybrid_rollout(T=32)
    assert r.sample_1v1_opp_char == 15
    assert r.focus_counts["opp0"] >= r.focus_counts["opp1"]
    print("fake_rollout OK", r)


def test_warm_start_and_collector() -> None:
    from slippi_ai.teams.warm_start import build_assignment_table, map_param_name, describe
    from slippi_ai.teams.rollout_worker import TeamsRolloutCollector
    from slippi_ai.teams.types_teams import TeamsGame
    from slippi_ai.types import FoDPlatforms, Randall
    from slippi_db.parse_libmelee import _EMPTY_ITEMS

    mapped = map_param_name("policy/embed/game/p0/percent/scale")
    assert any("ego" in m for m in mapped)
    assert any("partner" in m for m in mapped)
    mapped1 = map_param_name("policy/embed/game/p1/x/scale")
    assert any("opp0" in m for m in mapped1)
    table = build_assignment_table(
        [
            "policy/embed/game/p0/percent/scale",
            "policy/embed/game/p1/x/scale",
            "policy/head/kernel",
        ]
    )
    assert "policy/head/kernel" in table.values() or any(
        v.endswith("kernel") for v in table.values()
    )
    print("warm_start OK", describe())

    # Time-stacked single nest as one "frame sequence" via fake Rank0 list
    T = 8
    partner = _fake_player(T, x=-5)
    pa = np.full(T, 0xE, dtype=np.uint16)
    pa[4:] = 0
    partner = partner._replace(action=pa)
    tg = TeamsGame(
        ego=_fake_player(T, x=-10),
        partner=partner,
        opp0=_fake_player(T, x=10, char=15),
        opp1=_fake_player(T, x=20),
        stage=np.full(T, 32, dtype=np.uint8),
        randall=Randall(x=np.zeros(T, np.float32), y=np.zeros(T, np.float32)),
        fod_platforms=FoDPlatforms(
            left=np.zeros(T, np.float32), right=np.zeros(T, np.float32)
        ),
        items=_EMPTY_ITEMS,
    )
    col = TeamsRolloutCollector()
    # Push the stacked game once — finish uses compute path fallback
    col.record(1, tg, focus="opp0")
    result = col.finish()
    assert 1 in result.rewards
    assert result.rewards[1].shape[0] == T - 1
    assert result.focus_hist[1]["opp0"] == 1
    print("collector OK", float(result.rewards[1].mean()))


def test_step1_hybrid_worker() -> None:
  from slippi_ai.teams.hybrid_worker import build_hybrid_teams_actor

  w = build_hybrid_teams_actor()
  w.start()
  trajs, timings = w.rollout(16)
  w.stop()
  assert timings["mode"] == "hybrid_teams_fake"
  assert set(trajs) == {1, 2}
  assert trajs[1].rewards.shape == (16, 1)
  assert hasattr(trajs[1].states, "p0")
  print("step1 hybrid worker OK", float(trajs[1].rewards.mean()))


def test_step2_embed_wiring() -> None:
  """JAX-free: Teams embed source + warm-start; optional JAX compile."""
  from slippi_ai.teams.networks_teams import TEAMS_CONCAT_ORDER
  from slippi_ai.teams.step2_demo import check_source_wiring, check_warm_start_and_order

  assert TEAMS_CONCAT_ORDER[0] == "ego"
  check_source_wiring()
  check_warm_start_and_order()
  try:
    import jax  # noqa: F401
    from flax import nnx
    from slippi_ai.jax import embed as embed_lib
    from slippi_ai.teams.jax_networks import (
        TeamsEnhancedEmbedModule,
        build_teams_embed_module,
    )

    enhanced = TeamsEnhancedEmbedModule.default_config()
    enhanced["use_items"] = False
    mod = build_teams_embed_module(
        rngs=nnx.Rngs(0),
        embed_config=embed_lib.EmbedConfig(),
        num_names=1,
        enhanced_config=enhanced,
    )
    out = mod(mod.dummy(()))
    assert out.ndim == 1
    print("step2 JAX embed OK", int(mod.output_size))
  except ImportError:
    print("step2 JAX embed SKIPPED (no jax) — wiring OK")


def test_step3_trainer_bridge() -> None:
  from slippi_ai.teams.trainer_bridge import (
      describe_wiring,
      preserve_hybrid_rewards,
      simulate_learner_inplace,
  )
  from slippi_ai.teams.hybrid_worker import build_hybrid_teams_actor
  from slippi_ai.teams.local_paths import normalize_iso_path, DEFAULT_ISO

  text = describe_wiring()
  assert "build_actor" in text
  assert "preserve_hybrid_rewards" in text
  summary = simulate_learner_inplace(num_steps=8)
  assert summary["ppo_ready_shape"]
  assert summary["num_trajectories"] == 2

  w = build_hybrid_teams_actor()
  w.start()
  traj_map, _ = w.rollout(4)
  w.stop()
  trajs = preserve_hybrid_rewards(traj_map)
  assert len(trajs) == 2
  assert trajs[0].rewards.shape[0] == 4

  # Host ISO path (with or without .iso)
  bare = str(DEFAULT_ISO).removesuffix(".iso")
  hit = normalize_iso_path(bare)
  if hit is not None:
    assert hit.suffix.lower() == ".iso"
  print("step3 trainer_bridge OK", summary["reward_means"])


def test_live_env_import_and_paths() -> None:
  from slippi_ai.teams.local_paths import describe_paths, resolve_paths
  from slippi_ai.teams.live_env import LiveTeamsEnv, TeamsStep
  from slippi_ai.teams.hybrid_worker import build_live_teams_actor

  assert TeamsStep is not None
  assert LiveTeamsEnv is not None
  assert callable(build_live_teams_actor)
  print("paths:\n", describe_paths())
  folder, iso = resolve_paths()
  assert folder.is_dir()
  assert iso.is_file()
  print("live_env import + paths OK", iso.name)


def main() -> int:
  test_layout()
  test_reward_teammate_death()
  test_curriculum()
  test_parse_and_stack()
  test_env_import()
  tg = test_compat_1v1()
  test_embed_and_focus(tg)
  test_update_teams_rewards_traj()
  test_run_dry()
  test_fake_rollout()
  test_warm_start_and_collector()
  test_step1_hybrid_worker()
  test_step2_embed_wiring()
  test_step3_trainer_bridge()
  test_live_env_import_and_paths()
  print("ASSUMED keys:", sorted(ASSUMED.keys()))
  print("teams selftest PASSED")
  return 0


if __name__ == "__main__":
    sys.exit(main())
