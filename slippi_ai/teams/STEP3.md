# Step 3 — what you do (plain English)



No rented GPU yet. This step only **wires the trainer on paper** and

(optionally) checks Teams mode on your local Dolphin.



## Part A — trainer scaffolding (safe, no window)



```bat

cd C:\Users\langu\slippi-ai

C:\Users\langu\phillip-teams\.venv-ai\Scripts\python.exe -m slippi_ai.teams.step3_demo

```



You want `STEP 3 OK`.



That proves:

- `build_actor` → hybrid Teams worker (same shape as jax.rl)

- Teams rewards are **kept** (stock 1v1 reward recompute would wipe them)

- Fake rollout looks like something PPO could eat later



## Part B — Dolphin Teams menu check (opens a window)



Only if Part A said Dolphin/ISO are ready (they should be — PhillipTeams

already has your paths).



**Close other Dolphin windows first**, then:



```bat

cd C:\Users\langu\slippi-ai

C:\Users\langu\phillip-teams\.venv-ai\Scripts\python.exe -m slippi_ai.teams.check_teams_menu

```



A Dolphin window will open for up to ~90 seconds, try to toggle Teams with

**Y** on character select, then quit.



Look for one of:

- `TEAMS MENU: ON` — great, `is_teams` flipped

- `TEAMS MENU: OFF (port fallback OK)` — still fine; we train with ports 1+2 vs 3+4



## What you do NOT do yet



- Rent a GPU / overnight train



## After both parts



Tell me the two result lines (`STEP 3 OK` + `TEAMS MENU: …`).

Note from agent run on this PC: Dolphin/ISO found; menu check got
`TEAMS MENU: OFF (port fallback OK)` — Y pulsed but `is_teams` stayed
false. Training still uses ports 1+2 vs 3+4. Not a blocker.


