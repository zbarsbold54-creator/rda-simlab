# Diagnostics

## Nondeterministic multi-worker PPO training (fixed, det-v0)

### Symptom
Same config + same seed produced different weights when training was run twice
in separate processes at `workers > 1`. `workers=1` was fully reproducible.

### Localization (CPU, obs_only, Balanced 5, seed 7, 2 iterations, two runs each)

```text
workers=1   max_abs_diff = 0.000e+00
workers=2   max_abs_diff = 8.131e-04   (before fix)
workers=4   max_abs_diff = 6.796e-03   (before fix)
```

`workers=1` deterministic, `workers>1` not => the cause is in multi-worker
coordination, not in a single worker's rollout or in the PPO update itself.

### Cause
`PPOTrainer._collect` read rollout results from the shared `mp.Queue` with
`results = [self._result_q.get() for _ in self._procs]` and then concatenated
segments in that **arrival order**. `mp.Queue.get()` returns whichever worker
finished first, so the segment concatenation order (and therefore the
advantage array ordering and minibatch chunking) depended on nondeterministic
worker completion timing. Each result carries its `rank`, but it was ignored.

Per-worker rollout streams were already deterministic (world and action-sampling
RNGs are pure functions of worker rank / episode index), so sorting the results
by worker id fully restores cross-process reproducibility.

### Fix
- `_collect`: sort collected results by `rank` before concatenating (fixed
  worker-id order).
- `_rollout_worker`: re-seed the per-iteration action-sampling RNG from
  `_rollout_seed(training_seed, worker_id, iteration)` (a pure function),
  delivered via the broadcast job. Previously the stream was seeded only from
  `TRAIN_SEED_BASE + rank * 1000` and advanced implicitly across iterations.

No algorithm change. Training numbers for a given seed change after this fix
(expected; the RNG streams are reassigned). Checkpoint loading and eval are
unaffected (eval of `tests/fixtures/learner_v0_final.pt` unchanged: cap 5.0%,
median 3705).

### Verification
```text
workers=1   max_abs_diff = 0.000e+00   (after fix)
workers=2   max_abs_diff = 0.000e+00   (after fix)
workers=4   max_abs_diff = 0.000e+00   (after fix)
87 passed  (full suite)  ruff clean
legacy eval: cap_pct=5.0% median_survival=3705
```

## Training-seed plumbing (fixed, det-v1)

`_build_net` (ppo.py) hardcoded `"seed": 0` in the policy config, so the
training seed never reached `torch.manual_seed` and two training seeds
(100000 vs 100001) produced identical initial weights (max_abs_diff 0.0).
Changed to pass `cfg.seed`. Audit of every consumer of the training seed:

- policy init: feeds `torch.manual_seed` via `TorchPolicy._build_from_cfg`
  (now correct after det-v1);
- per-worker rollout action RNG: `_rollout_seed(cfg.seed, rank, iteration)`,
  a pure function (correct);
- minibatch shuffling: global torch RNG (`torch.randperm`) deterministically
  seeded by the same manual_seed (correct);
- world/episode seeds: `_WorkerEnv.seed = TRAIN_SEED_BASE + rank*1000`
  (100000+), incremented per episode, so training world seeds are always
  >= 100000 and never overlap dev (0-19) or test (1000-1019);
- eval: `_quick_eval` uses dev seeds 0-4 only for EVAL, never training.
- left unchanged: `build_policy_cfg(arch)` (the `--arch` path) uses the arch
  selection's own seed, not `cfg.seed`; not on the default GRU obs_only path.

Verified: two training seeds -> different initial weights; same seed -> bit
identical initial weights across separate processes. Gate 0(b) re-ran bit
identical with workers=4 after the fix. Legacy eval unchanged (cap 5.0%,
median 3705).

## Horizon diagnostic (diag-v0)

### Question
The default GRU obs_only trainer dies of starvation around step ~3700 on the
10000-step cap. Does the effective advantage horizon (rollout length and/or
gamma) explain it? Three arms, 300k env steps each, obs_only, workers=4,
seeds 100000 and 100001, commit 29cfe22:

| arm | change | effective horizon (1/(1-gamma)) |
|-----|--------|--------------------------------|
| R256 | baseline (rollout 256, gamma 0.995) | ~200 |
| R1024 | rollout 1024 | ~200 (longer contiguous truncation) |
| G999 | gamma 0.999 | ~1000 |

### Final eval (dev seeds 0-19, cap 10000)

```text
arm      seed    Balanced cap%  Balanced median  Hard cap%  Hard median
R256    100000   0.0           2701            0.0      2699
R256    100001   1.0           1502            0.0       599
R1024   100000  28.0           8051           11.0      5110
R1024   100001  25.0           4477           18.0      3605
G999    100000  11.0           3665           11.0      3611
G999    100001   5.0           5092            8.0      2720
```

### Decision rule (pre-declared)
An arm "helps" iff BOTH its seeds beat BOTH R256 seeds on cap% in Balanced AND
Hard. Seed noise dominates iff the two R256 seeds disagree more than the arm
differences.

- R1024: both seeds (28/25 balanced, 11/18 hard) beat both R256 seeds
  (0/1 balanced, 0/0 hard). => HELPS.
- G999: both seeds (11/5 balanced, 11/8 hard) beat both R256 seeds. => HELPS.
- R256's two seeds differ by 1.0pp (balanced); arm differences are 5-28pp,
  so seed noise does not dominate.

### Interpretation
Both longer-horizon arms help; R1024 is clearly strongest. Note R1024 runs far
fewer updates (75 vs 294) for the same env steps (4x longer rollouts), so the
gain is not "more training" but a longer contiguous advantage-truncation /
credit-assignment window over which the starvation-death signal can be
attributed to earlier food-seeking actions. G999 (longer discount) helps but
less than longer rollouts. The default horizon (~200) is too short to credit
sustained food-seeking, which is why the baseline starves on both seeds.
