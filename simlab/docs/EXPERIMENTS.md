# Experiments

This file describes planned experiments. None of the experiments below have
been run yet; they are scheduled for after the `learner-v0` harness is
approved.

## Planned experiment: obs_only vs obs_drives

Compare two observation encodings for the learned GRU policy on the Balanced 5
and Hard 5 benchmarks.

- Conditions:
  - `obs_only` — 16 rays x (distance / 20.0 + one-hot of the 4 ray types) plus
    energy/100, health/100, speed/max_speed -> 83 dims.
  - `obs_drives` — the same 83 dims plus the 8 drive values (simple DriveCore
    with FixedDecay retention) -> 91 dims.
- Training seeds: 3 training seeds each condition, drawn from `100000` upward
  (never overlapping dev seeds 0-19 or test seeds 1000-1019).
- Training horizon: 300000 environment steps per training seed (the smoke-test
  setting), Balanced scenario, 5 agents, episodes truncated at 2000 steps with
  automatic reset to a new seed.

### Exact commands

Run each of the six training jobs (3 seeds x 2 conditions) as a background
process writing to `logs/`:

```powershell
cd C:\AI\SimulationCode
.\.venv\Scripts\python.exe -m simlab.cli train `
  --obs obs_only --total-steps 300000 --workers 4 --seed 100000 `
  --out results/train/obs_only_100000 --eval-every 20
.\.venv\Scripts\python.exe -m simlab.cli train `
  --obs obs_only --total-steps 300000 --workers 4 --seed 100001 `
  --out results/train/obs_only_100001 --eval-every 20
.\.venv\Scripts\python.exe -m simlab.cli train `
  --obs obs_only --total-steps 300000 --workers 4 --seed 100002 `
  --out results/train/obs_only_100002 --eval-every 20
# ...same three with --obs obs_drives -> results/train/obs_drives_1000xx
```

Then evaluate the final checkpoints on the held-out test seeds (1000-1019):

```powershell
.\.venv\Scripts\python.exe -m simlab.cli eval `
  --policy learned_gru --checkpoint results/train/obs_only_100000/final.pt `
  --scenario balanced --agents 5 --seeds 20 --seed-offset 1000 `
  --cap 10000 --workers 4 --out results/eval/obs_only_100000_balanced5.json --compare
# repeat for each condition/seed and for --scenario hard
```

### Estimated wall time

The smoke test (one job, obs_only, 300000 env steps, 4 workers) ran at about
440 env steps/s aggregate on this machine, so one training job takes roughly
11-12 minutes. With 6 training jobs this experiment is estimated at:

- ~66-72 min if run sequentially;
- ~22-24 min if run two at a time (one per obs condition, so each of the two
  cores is kept busy by one worker set);
- ~11-12 min if all six run in parallel (not recommended - six 4-worker jobs
  oversubscribe the CPU and each slows to roughly a sixth of its solo speed).

The 6 training jobs dominate; the 12 eval jobs (each 20 seeds, cap 10000,
4 workers, ~2-3 min each) add ~25-35 min. Total plan: ~45-60 min with two
training jobs at a time plus the eval jobs.

### Evaluation protocol

For each trained checkpoint report on the test seeds (1000-1019), Balanced 5
and Hard 5:

- cap percent with a 95% bootstrap confidence interval over seeds;
- median survival (steps);
- death-cause split (starvation / hazard / truncation);
- food left (mean).

`eval --compare` prints the verdict against random, heuristic_noflee and
heuristic using the dev/test-seed baselines in `docs/baselines.json`.

#### Pairwise comparison between conditions

For each condition, summarise the per-training-seed cap CI. A condition is
considered better than the other only if its cap CI does not overlap the
other condition's cap CI across the same benchmark (Balanced 5 or Hard 5),
mirroring the rule for comparing a learned policy against the heuristic.

#### Rule for claiming improvement

A learned policy counts as better than the heuristic only if its cap CI lower
bound is above the heuristic's cap CI upper bound on the same scenario and
seed set (dev or test). Otherwise it is classified NOT DISTINGUISHABLE.
