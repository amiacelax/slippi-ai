# Teams IQ (2v2) — overnight scaffold

Parallel package under `slippi_ai/teams/`. Does **not** modify the stock
`Game(p0, p1)` schema or break `medium-v2`.

## What's here

| Module | Role |
|--------|------|
| `layout.py` | ego / partner / opp0 / opp1 port mapping |
| `curriculum.py` | A–F weights from your doubles tier list |
| `reward.py` | Teams reward: teammate death, 2v1-on-stage, roles, floaty* |
| `parse.py` | **TeamsParser** — 4-port GameState → TeamsGame |
| `rollout.py` | stack frames → `compute_teams_rewards` |
| `env.py` | **TeamsEnvironment** — 4 AI Dolphin wrapper |
| `menu.py` | CSS Teams toggle assist (Y) |
| `types_teams.py` | `TeamsGame` nest |
| `compat_1v1.py` | TeamsGame → stock Game for medium-v2 |
| `embed.py` | numpy 4-player embed (JAX port size reference) |
| `focus.py` | dynamic opp0/opp1 picker |
| `trajectory.py` / `learner_bridge.py` | TeamsTrajectory + PPO reward hook |
| `config.py` / `run_lib.py` | hybrid RL config + dry-run |
| `fake_rollout.py` | end-to-end hybrid path without Dolphin |
| `warm_start.py` | medium-v2 p0/p1 → ego/partner/opp0/opp1 param map |
| `rollout_worker.py` | TeamsRolloutCollector hybrid loop |
| `networks_teams.py` | Concat order notes (JAX-free) |
| `jax_networks.py` | `TeamsEnhancedEmbedModule` (needs JAX) |
| `jax/embed.py` | `make_teams_game_embedding()` (needs JAX to import) |
| `STEP1.md` / `step1_demo.py` | Hybrid rewards worker smoke |
| `STEP2.md` / `step2_demo.py` | Teams embed wiring smoke |
| `STEP3.md` / `step3_demo.py` | Trainer plug-in (paper) |
| `STEP4.md` / `step4_demo.py` | Live Dolphin hybrid actor |
| `live_env.py` / `local_paths.py` | TeamsEnvironment adapter + ISO/Dolphin resolve |
| `trainer_bridge.py` | build_actor + preserve Teams rewards |
| `check_teams_menu.py` | Local Dolphin is_teams smoke |
| `observe.py` | embed migration notes |
| `decisions.py` | host-confirmed answers |
| `selftest.py` | no-GPU smoke tests |

Dolphin: optional `enable_teams=True` pulses Y on CSS during menuing.

## What's NOT done yet

1. Rent GPU and run overnight PPO (JAX train box)
2. Optional: force Teams CSS on builds where `is_teams` stays false

## How to test locally (no GPU)

```bash
cd C:\Users\langu\slippi-ai
python -m slippi_ai.teams.selftest
```

## Curriculum source

`PhillipTeams/docs/DOUBLES_CURRICULUM.md` + site rankings.
Host answers locked: aggro/support seats, death=0.85, Fox dittos,
floaty list = Puff/Peach/Luigi/Sheik.
