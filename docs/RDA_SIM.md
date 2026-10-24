# RDA for SimLab: Design Note

This note specifies how SimLab becomes ready for **RDA (Recurrent Drive /
Affective) research in the GUI**. It ports the *logic* of
`reference/rda_nanochat.py` (read-only; never imported) into the SimLab
learner, without changing world dynamics, scenarios, `baselines.json`, or the
heuristic/oracle baselines.

Reference contracts: `docs/ARCHITECTURES.md` (arch-v0), `docs/LEARNER_API.md`.
GUI version marker: `gui-arch-v0.x` (the Architecture panel lists specs from the
`ArchSpec` registry; new specs appear automatically with generated forms).

---

## 1. Two-sided split (who owns what)

SimLab keeps a strict split. The **world side** produces *instantaneous*,
deterministic, numpy drive signals, once per step, using only information up to
time `t`. The **learner side** (torch) turns that signal into a persistent,
policy-usable state.

```
world (numpy)                     learner (torch)
-------------------------          ----------------------------------------
DriveCore.step(obs)  -> d_t  ->   Retention.step(d_t, ...) -> E_t, lam_t
                                            | (FiLM / concat) v
                                   obs -> encoder -> features -> policy/value
```

Reused sim-side products:
- `simlab/plugins/drive_cores.py` `RDA8DriveCore` (EMA over raw drive) stays as
  the legacy world-side drive core; it is unchanged. The new world-side core is
  `InstantDriveCore`, which returns the *raw* `d_t` (no EMA).
- The learner-side Retention is a **new** torch module that replaces the legacy
  `NotImplementedError` `RDABackbone` stub.

### 1.1 Torch-free core/CLI/GUI

`simlab.core`, `simlab.cli`, and `simlab.gui` never import torch. All new
learner code lives under `simlab/training/` (torch import allowed only there)
and is reached through lazy `ArchSpec` loaders.

---

## 2. Sim-side instantaneous drives (`drives/definitions.py`)

The eight drive definitions live in **one file**, `simlab/drives/definitions.py`.
Each definition is an entry with:

- `name` — canonical id (fixed order, `DRIVE_NAMES`);
- `label` — a human-readable display label (currently **PLACEHOLDER**; the user
  will edit these);
- `docstring` — semantic description of the drive;
- `compute(obs, last_rays) -> float` — a deterministic numpy function of the
  current observation only (plus the previous raw ray vector for derived
  quantities like `learning_progress`), all information up to `t`.

Order (fixed, matches `LEARNER_API.md` and the `obs_drives` encoding trailing
8 dims):

| # | Drive | Phase |
|---|-------|-------|
| 1 | novelty | PLACEHOLDER |
| 2 | uncertainty | PLACEHOLDER |
| 3 | threat | PLACEHOLDER |
| 4 | integrity | PLACEHOLDER |
| 5 | agency | PLACEHOLDER |
| 6 | learning_progress | PLACEHOLDER |
| 7 | arousal | PLACEHOLDER |
| 8 | homeostatic_deviation | PLACEHOLDER |

`InstantDriveCore` calls `raw_drive_vector(obs, last_rays)` and returns the
`(n_live, 8)` array unchanged. `DRIVE_NAMES` is re-exported so the inspector and
`obs_drives` encoding keep the same order. The existing
`drive_cores._raw_drive` is refactored to delegate to the definitions so there
is exactly one source of truth.

### 2.1 Where the inspector shows drives

`InspectorWidget` renders one bar per `frame.drive_names` / `detail.drives`.
The instantaneous drive names flow from `InstantDriveCore.names` (`DRIVE_NAMES`),
so the eight definitions appear in the inspector when `instant` is selected as
the drive core.

---

## 3. Learner-side Retention module (`simlab/training/retention.py`)

The Retention holds the persistent state `E_t` (the learner's memory). It is
carried in the policy state exactly like the GRU hidden state:

- `initial_state` returns zeros for `E`, the trace `m`, the previous observation
  embedding and previous `d` (per variant);
- `done_mask` reset (mask = 0) zeroes the carried state, so `E`/`m` reset at
  episode boundaries;
- the PPO update recomputes each rollout chunk starting from the stored initial
  state (zeros at chunk start, same as the GRU path) and carries `E` through the
  chunk.

### 3.1 Core recurrence (all variants share one code path)

```
E_t = lam_t * E_{t-1} + (1 - lam_t) * write_t
```

- `write_t` is the *event* the learner commits at time `t`. Default: the
  drive signal `d_t` (bounded in [0,1]) optionally squashed; port the reference
  `tanh(event_term + pe)` convex form and keep the result in [-1,1].
- `E_t` stays within `[-1, 1]` by construction (convex combination), matching
  the reference's bounded state and the `|E|` rail shown in the inspector.

### 3.2 Variants (one code path, config-selected)

| Variant | Meaning |
|---------|---------|
| `fixed` | `lam_t` is a scalar or per-drive constant `decay` (baseline 0.9). |
| `constant` | `E_t` is a learned constant vector (control; independent of observations). |
| `surprise` (G1) | shared gate: `lam_t = f(m_{t-1})`, see 3.3. |
| `surprise_rand` (control) | `lam_t` sampled i.i.d. from the empirical marginal histogram of the G1 gate (see 3.4). |
| `per_drive` (G3) | per-drive traces `m_{t,i}` and `lam_{t,i}`, each driven by its own prediction error of `d_t` from `d_{t-1}`. |
| `per_drive_tied` | G3 but all gates tied to the shared G1 gate. |

All variants share the same observation encoder and policy/value heads (see
§4). Gate parameters start **fixed** (no trainable gate weights). Trainable gate
parameters are exposed only behind a `trainable_gate` config flag, and any such
parameters are kept out of the scalar optimizer group (i.e. they are ordinary
network parameters, never the `log_std_turn` / `log_std_thrust` scalars the PPO
noise uses).

### 3.3 Surprise gate G1

- **Auxiliary head** (independent of `E`, small, own auxiliary loss, weight
  configurable): predicts the observation embedding `o_t` from `o_{t-1}`.
  `s_t = normalized error` = (prediction error normalized by a running mean/std,
  stop-gradient on the gate path).
- **Trace**: `m_t = rho * m_{t-1} + s_t` (`rho` configurable).
- **Gate**: `lam_t` is a fixed, monotone-increasing function of `m_{t-1}`:
  baseline `0.9`, rising toward a configurable max (default `0.9984375`).
- **Past-only rule** (see §6): at the surprising step `t`, the gate still uses
  `m_{t-1}` (which does not yet contain the current surprise), so the event is
  written strongly; from `t+1` onward the trace includes the surprise and the
  event is protected.

### 3.4 `surprise_rand` control (empirical marginal)

- A running histogram of the G1 gate's `lam` values is maintained **during
  training only** (inference freezes it).
- `surprise_rand` samples `lam_t` i.i.d. from that histogram.
- Independence guarantee: gate values depend only on the RNG seed and the frozen
  histogram, never on the data. This is exactly what the independence test
  checks (`same seed -> same gate values for two different observation
  streams`). We **never** permute gate values within a time window — permuting
  leaks future information and is forbidden.

### 3.5 Per-drive gates (G3)

- A small per-dimension predictor reconstructs `d_t` from `d_{t-1}`; each drive
  `i` has its own error, trace `m_{t,i}`, and `lam_{t,i}`.
- `per_drive_tied` ties every gate to the shared G1 gate (identical to G1's
  `lam_t` broadcast to all drives).

---

## 4. Fusion and the 8 presets

### 4.1 Fusion

- `film` — port the reference's per-position causal FiLM: a `film_scale` /
  `film_shift` linear map from `E_t` (zero-initialized) plus the `_unit_rms`
  normalization and saturation guard so the gain stays bounded and the
  modulation is an identity at init.
- `concat` — concatenate `E_t` with the encoded observation features before the
  heads.

### 4.2 Presets

All eight presets are `backbone` `ArchSpec`s; they appear in the GUI's backbone
dropdown and generate forms from their config dataclasses. They share the same
observation encoder and policy/value heads; `match_params` picks a hidden size so
parameter counts are within 5% of each other.

| Preset | Encoder | Retention | Fusion |
|--------|---------|-----------|--------|
| `gru` | GRU | — (existing) | concat/film |
| `transformer` | causal Transformer (K) | — (existing) | concat/film |
| `rda` | GRU | fixed decay | concat/film |
| `rda_const` | GRU | constant E | concat/film |
| `rda_g1` | GRU | surprise (G1) | concat/film |
| `rda_g1_rand` | GRU | surprise_rand | concat/film |
| `rda_g3` | GRU | per_drive | concat/film |
| `rda_g3_tied` | GRU | per_drive_tied | concat/film |

**Transformer preset.** Causal over the last `K` observation embeddings (`K`
configurable, default 32). The update uses a **vectorized** causal-attention
forward (no per-timestep Python loop in the batched path); rollouts use an
incremental forward with a ring buffer. The vectorized batched forward is
verified to equal the incremental step-by-step forward to `1e-5`. Rollout
length stays a config value (diag-v0 favored 1024).

### 4.3 Introspection (inspector)

Each RDA variant exposes an `introspect()` dict for the inspector / tests:
`d_t`, `E_t`, `lam_t` (per drive), `m_t`, `s_t`, FiLM `gamma`/`beta`, and
`|E|` versus the rail (1.0).

---

## 5. Integration points

- `backbones.py`: `RDABackbone` replaces the stub; `BACKBONES` maps the eight
  preset names to the implementation.
- `arch_spec.py`: `add_spec` registers one config dataclass per preset (torch-free).
- `torch_policy.py`: for any `rda*` backbone, `obs_mode` is forced to
  `obs_drives` so `d_t` is always present in the trailing 8 dims of `x`, for
  both the training loop and `act()`.
- `ppo.py`: unchanged trainer loop — `frame_step(Xb[:, si], h, None)` reads
  `d_t` from `x`; `E`/`m` are part of the carried state and are reset by
  `done_mask`. The vectorized batched `forward` exists for the transformer and
  is exercised by the equivalence test.
- `drives/definitions.py`: single source of truth for the 8 drives; used by
  `InstantDriveCore` and by the inspector.

---

## 6. Past-only rule (exact accounting)

At step `t`:

| Quantity | Uses info up to | Computed in |
|----------|-----------------|-------------|
| `d_t` (sim side) | `t` | `InstantDriveCore.step` (world) |
| `write_t` (event) | `t` (from `d_t`) | Retention.step |
| `s_t` (surprise) | `t` (predicts `o_t` from `o_{t-1}`) | aux head, then running-normalized |
| `m_t` (trace) | `t` (from `s_t`) | Retention.step |
| `lam_t` (gate) | `t-1` (uses `m_{t-1}`) | Retention.step |
| `E_t` (state) | `t` (uses `lam_t`, `write_t`) | Retention.step |
| FiLM gains | `t` (from `E_t`) | fusion |

The **gate uses only information up to `t-1`**; the **write may use `d_t`**. A
perturbation of the observation after step `t` therefore changes nothing at `t`
(both `d_t` and `E_t` and `lam_t` for the action logits at `t`).

---

## 7. Test plan (CPU)

- Causality: perturb observations after `t`; `d_t`, `E_t`, `lam_t`, `m_t`, and
  action logits at `t` unchanged (all variants).
- Gate timing: injected surprise at `t` leaves `lam_t` unchanged and raises
  `lam_{t+1}`.
- `surprise_rand` independence: same RNG seed -> same gate values for two
  different observation streams.
- `constant`: `E_t` does not depend on observations.
- Transformer: vectorized update == incremental forward (`max abs diff < 1e-5`),
  incl. `done_mask` resets.
- Smoke: all presets x 2 PPO iterations, workers=4, run twice, weights
  bit-identical; param counts printed and within 5%.
- Full suite, ruff, torch-free import tests, `eval --compare` shows no STALE.

---

## 8. Known placeholders / intentionally skipped

- Drive descriptions are **PLACEHOLDER** for the user to fill in.
- The `surprise` aux head uses a small MLP on the observation embedding; the
  reference's bigram NLL table is language-specific and not ported.
- Trainable gate parameters are implemented but off by default (`trainable_gate`
  flag); the scalar optimizer-group exclusion is enforced by never putting them
  on `log_std_*`.

---

## 9. Research cockpit (GUI)

The cockpit (`simlab/gui/cockpit.py`) is a dock in the main window with six
tabs. It is designed so you can run, watch, and compare retention experiments
without touching code.

### 9.1 Experiment builder (tab 1)

- Pick preset(s) from the registry (arms), training seeds, total steps, rollout
  length, and workers; save as TOML under `runs/`. The arm table shows each
  arm's param count (computed lazily on the GUI thread, torch off-thread).
- **Protocol guards** (`simlab/research/experiment.py`):
  - training seeds `>= 100000` only;
  - workers pinned to 4 by default, with a visible warning if changed;
  - dev seeds `0-19`; test seeds `1000-1019` behind a confirm dialog
    ("final report only");
  - warning if fewer than 3 seeds per arm.

### 9.2 Run queue (tab 2)

`simlab/research/queue.py` launches the grid as a single resumable detached
queue: one serving process plus an exclusive lock file (with staleness recovery),
per-run status, pause-after-current-run, resume after crash/restart, and Stop.
Each run writes a results JSON with seed, init checksum, worker count, and the
git commit.

### 9.3 Watch live (tab 3)

A "Follow latest checkpoint" mode polls for the newest checkpoint (interval
configurable) and plays it in the world view while training continues in its
subprocess. The inspector shows `d_t`, `E_t` (8 drive curves), `lambda_t`, and
FiLM gains for RDA presets; speed/pause/step controls are the existing ones.

### 9.4 Training curves (tab 4)

Live plots from the run logs (`learning.csv` / quick-eval comment lines): return,
entropy, KL, value loss, and checkpoint cap%/median, with arms overlaid.

### 9.5 Results table (tab 5)

`simlab/research/results.py` aggregates per arm per seed: cap%, median,
starve/hazard/trunc for Balanced and Hard on dev seeds, mean +/- spread across
seeds, and differences vs a chosen control arm (rda_const for RDA arms). It
prints only numbers and per-seed values (no verdict words) and exports to CSV.

### 9.6 Gate diagnostics (tab 6)

`simlab/research/gates.py` + `simlab/research/replay.py` plot, live or from a
loaded checkpoint: `lam_t` (per drive), `m_t`, `s_t`, `E_t`, and `|E|` against
the saturation rail; the fraction of steps where `lam` is at its maximum, a
`lam` histogram, and the correlation of `lam` with logged salient events
(hazard contact, food eaten, damage). The control arm's `lam` distribution is
overlaid when a control is selected, and a warning fires if `|E|` stays within
0.02 of the rail for more than N steps.
