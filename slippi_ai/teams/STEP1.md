# Step 1 — what you do (plain English)

You do **not** need Linux for this step.

## What Step 1 is

Hook the Teams reward collector into a rollout worker so training trajectories
use **doubles rewards** (teammate death, 2v1, …) while the brain still sees a
normal 1v1 Game (so medium-v2 still works).

## What you run (one command)

Open a terminal:

```bat
cd C:\Users\langu\slippi-ai
C:\Users\langu\phillip-teams\.venv-ai\Scripts\python.exe -m slippi_ai.teams.step1_demo
```

You should see `STEP 1 OK` and reward stats for ports 1 and 2.

## What that proved

- Fake 4-player match (no Dolphin window)
- Worker built the same kind of `Trajectory` the RL trainer expects
- Rewards came from Teams IQ rules, not stock 1v1

## What you do NOT do yet

- Rent a GPU
- Install Linux
- Change PhillipTeams for this step

## After Step 1 works

Tell me and we do **Step 2** (see `STEP2.md` — Teams neural embed wiring).
