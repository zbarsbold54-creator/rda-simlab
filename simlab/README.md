# SimLab

A deterministic 2D survival world with a GUI, built to test drive-based and retention-based architectures against plain baselines. Python 3.11, numpy core, PySide6 and pyqtgraph for the GUI.

## What it does

- **World:** numpy core, deterministic, with snapshots and replay. Agents need food and avoid hazards.
- **Presets:** Balanced (default: hazard detection radius 16, food supply ratio 2.0), Solo (1 agent, 1 hazard), Hard (12 hazards).
- **Plugins:** slots for `Policy`, `DriveCore` and `RetentionRule`. Architectures are registered in a torch-free `ArchSpec` registry and built lazily. Drop a file in `plugins/architectures/` to add one.
- **Training and evaluation:** PPO on CPU with 4 workers (about 1,000 env steps/s), and a frozen evaluation harness (`eval --compare`) that checks config and policy hashes against `docs/baselines.json` and prints `STALE` or `NOT COMPARABLE` when they differ.
- **Cockpit:** Experiment, Queue, Live, Curves, Results and Gates tabs; an architecture dock; an active-policy banner; a per-agent panel (energy, food, death cause); checkpoint loading.

## Presets

21 architecture presets. Parameter counts:

| Preset | Params |
|---|---|
| `gru_legacy` (the first learner) | 9,798 |
| `gru` | 70,118 |
| `transformer` | 73,606 |
| `rda_gru_*`, `rda_const` | 70,398 |
| `rda_tf_*` | 69,986 |

Retention rules: fixed, constant, surprise-gated (`surprise`), `surprise_rand`, `per_drive`, `per_drive_tied`, plus additive `_norm` variants. The gate maps surprise to the retention value `lam = 0.9 + 0.0984375 * clamp(m/4, 0, 1)`, with `m(t) = 0.99 * m(t-1) + s`.

## Seed protocol

| Set | Seeds | Use |
|---|---|---|
| Dev | 0-19 | tune |
| Test | 1000-1019 | report |
| Training | 100000 and up | train and init |

Evaluation: 5 agents, cap 10,000 ticks, frozen policy, `batched=False`. Report cap% with a 95% interval.

## Results

Reference policies, dev seeds, Balanced / Hard cap%: random 0 / not recorded; first PPO GRU (obs_only, 300k steps) 5 / 11; heuristic about 80 / 46; oracle_avoid (a world-state reference ceiling, not a baseline) about 98 / 96.

Test seeds 1000-1019, mean cap% per arm (all `rda_gru_surprise`, about 300k steps). Source: `logs/overnight/testeval_summary.csv`.

| Arm | n | Hard | Balanced |
|---|---|---|---|
| A | 3 | 2.7 | 5.7 |
| C | 6 | 5.7 | 14.0 |
| D2 | 6 | 12.8 | 26.5 |
| D5 | 6 | 16.5 | 30.7 |
| D | 12 | 25.3 | 38.2 |

- Fresh-init GRU across 12 init seeds: Hard mean 27.3% (range 0-55). Arm D's paired gain over its own step-0 init is about +8.8 points, SD about 30, t about 1.0, so not significant.
- D seed 100000 is the best of 12: Hard 93% [87, 97], Balanced 100%. The D median Hard cap is 5.5% (D-family median 3.5% across D/D2/D5, from `testeval_summary.csv`).
- Outcomes on Hard (D 100000 / heuristic / fresh-init / do_nothing): reached cap 93 / 57 / 36 / 0; hazard death 7 / 43 / 60 / 100; starvation 0 / 0 / 4 / 0.
- PPO horizon diagnostic (obs_only, 300k steps): rollout 1024 reached 28% / 11% and 25% / 18% (Balanced / Hard, seeds 100000 and 100001) against 0-1% for rollout 256; gamma 0.999 reached 11 / 11 and 5 / 8.
- Gate saturation: the retention gate sits near its bounds from the start and moves toward the lower bound with training (rda_gru_surprise near-min 0.2% to 30.9%, near-max 99.8% to 64.8%).
- Performance: the PPO update is slower on CUDA than CPU for this GRU (59.8 s vs 12.3 s for 20,480 steps), so training runs on CPU with a single-thread update.

Not yet run: a plain GRU or transformer trained for the same budget as the RDA arms, and the `obs_only` vs `obs_drives` vs constant grid.

## Integrity notes

- Earlier hand-typed result tables disagreed with the raw files; `tools/build_testeval_summary.py` regenerates the summary from `logs/overnight/testeval_*.json`.
- A GUI-vs-harness discrepancy came from a 40,960-step checkpoint being loaded instead of the 303,104-step one; the batched-action mode was also fixed (`gui-v1.6`).
- The `food_eaten` metric counted only agents that died; fixed in `63a9719`.
- `do_nothing` is a registered baseline (about 0-1% cap), which ruled out the worry that D survives by idling.
- `tools/decay_probe.py` measures gate behavior, not survival, so its numbers are not comparable to cap%.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[learn]"        # CPU torch, imported lazily
python -m simlab.cli eval --help
python -m simlab.cli train --help
```

Documentation: `docs/BASELINES.md`, `docs/LEARNER_API.md`, `docs/EXTENDING.md`, `docs/EXPERIMENTS.md`, `PROGRESS.md`.

## License

MIT.
