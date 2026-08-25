# Step 2 — what you do (plain English)



You do **not** need a rented GPU for this step on Windows.



## What Step 2 is



Wire the **Teams neural embed**: the brain sees four players

(`ego | partner | opp0 | opp1`) instead of only `p0 | p1`, and we keep the

warm-start map from medium-v2 (`p0→ego/partner`, `p1→opp0/opp1`).



## What you run (one command)



Open a terminal:



```bat

cd C:\Users\langu\slippi-ai

C:\Users\langu\phillip-teams\.venv-ai\Scripts\python.exe -m slippi_ai.teams.step2_demo

```



You should see `STEP 2 OK`.



If it prints `JAX compile: SKIPPED`, that is **fine** on this PC — play venv

has no JAX. The embed **code** is still in place for the later GPU box.



## What that proved



- Teams embed factory + concat order exist

- Warm-start path map still matches medium-v2 → Teams

- Optional: if JAX is installed, a dummy forward pass builds and prints size



## What you do NOT do yet



- Rent a GPU / install Linux

- Train overnight

- Change PhillipTeams for this step



## After Step 2 works



Tell me and we do **Step 3** (see `STEP3.md` — trainer wiring + optional Dolphin check).


