# Step 4 — live Dolphin actor (plain English)

No rented GPU. This swaps the **fake** match for a **real** Dolphin window
so the trainer-shaped worker reads live 4-player frames.

## ISO (your path)

Either of these work (`.iso` optional):

`C:\External HD Copy\ThumbDrive\ISOs\Super Smash Bros. Melee (USA) (v1.02).iso`

## What you run

**Close other Dolphin windows first**, then:

```bat
cd C:\Users\langu\slippi-ai
C:\Users\langu\phillip-teams\.venv-ai\Scripts\python.exe -m slippi_ai.teams.step4_demo
```

Optional:

```bat
... -m slippi_ai.teams.step4_demo --steps 32
... -m slippi_ai.teams.step4_demo --fake
```

You want `STEP 4 OK (live)`.

## What that proves

- Real Dolphin + your ISO
- `TeamsEnvironment` (4 Fox) → hybrid actor
- Same Trajectory shape as Step 1/3 (1v1 states + Teams rewards)

`is_teams=False` is OK — we still seat ports 1+2 vs 3+4.

## What you do NOT do yet

- Rent GPU / overnight train
