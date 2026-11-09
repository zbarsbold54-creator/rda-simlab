# Learner API

This document specifies the exact interface a learned policy or drive core must
implement to plug into SimLab's full pipeline. It is the contract for the
future emotion-modulated learner. Read this together with
`simlab/plugins/example_learned_policy.py`, which is a working numpy-only
template.

## Full pipeline

Every simulation step runs, in order, for the live agents:

1. Sense: `World._observation_batch()` produces one `Observation` per live agent.
2. Drive core: `drive_core.step(live_observations, {"rng": world._rng})`
   returns a `(n_live, n_drives)` float array; each row becomes that agent's
   `drive_state`.
3. Policy: `policy.act(obs, drive_state)` (or `select_action` for world-state
   policies) returns per-agent actions.
4. `World.step` applies the actions and advances physics.

This sequence is shared by the headless `Runner` and the GUI worker
(`simlab/core/pipeline.py`), so the two paths cannot drift apart.

## Observation layout

`Observation` is a frozen dataclass (`simlab/core/types.py`). A policy that
declares `needs_observations = True` receives the following fields:

| Field | Shape | Range | Meaning |
|-------|-------|-------|---------|
| `rays` | tuple of 16 floats | 0.0–20.0 | Distance along each ray to the nearest detected object; clipped at 20.0 |
| `ray_types` | tuple of 16 ints | 1/2/3/4 | Object type at that ray's hit |
| `proprioception` | tuple of 3 floats | 0.0–1.0 | `(energy/max_energy, health/100, speed/speed_factor)` |
| `agent_position` | tuple of 2 floats | world units | Absolute (x, y) position |
| `agent_energy` | float | 0.0–max_energy | Raw energy |
| `step_count` | int | 0+ | Global step index |

### Ray geometry

- 16 rays at equal angular offsets over `linspace(-135°, +135°)`, with the
  centre ray forced to 0° (forward). Field of view is 270°, forward-centred.
- `ray_types` codes:
  - `1` food patch
  - `2` hazard
  - `3` nothing detected (open space; the default fill value)
  - `4` another agent (including dead agents)
- A ray with type `3` still carries a distance (the clipped 20.0 or the
  nearest open hit); prefer filtering by `ray_types` before reading `rays`.

### Proprioception

`proprioception = (energy / max_energy, health / 100.0, speed / speed_factor)`.
All three entries are in [0.0, 1.0]. `speed_factor` is the world constant
(8.0 by default), so a speed of 0.0 means stopped and 1.0 means full speed.

## Learner observation encoding

For learned policies the raw `Observation` is encoded into a fixed-size float
vector (numpy, in `simlab/training/encoding.py`). The learner never sees
absolute positions or the step count. Per ray we keep the normalized distance
and a one-hot of the ray-type channel, then append the three normalized
proprioceptive values. Option B appends the 8 drive values.

| Option | Dimensions | Layout |
|--------|-----------|--------|
| `obs_only` | 83 | 16 x (distance/20.0 + 4 one-hot) + (energy/max, health/100, speed/max_speed) |
| `obs_drives` | 91 | 83 above + 8 drive values |

Index ranges (0-based):

| Indices | Content |
|---------|---------|
| 0 | ray 0 distance / RAY_RANGE (20.0) |
| 1–4 | ray 0 ray-type one-hot (type 1 food, 2 hazard, 3 open, 4 agent) |
| 5 | ray 1 distance |
| 6–9 | ray 1 one-hot |
| ... | repeat for rays 2–15 |
| 80–82 | proprioception: energy/max_energy, health/100, speed/speed_factor |
| 83–90 | (obs_drives only) drive values in the order below |

`obs_dim(with_drives)` gives the vector length.

## Action

`Action` is a frozen dataclass (`simlab/core/types.py`):

| Field | Range | Meaning |
|-------|-------|---------|
| `turn` | -1.0–1.0 | Steering multiplier; actual turn = `turn * max_turn_rate * dt` |
| `thrust` | 0.0–1.0 | Throttle; speed = `thrust * speed_factor` |
| `eat` | bool | Issue an eat request this step |
| `action_id` | str | Cosmetic label carried by the action (e.g. `"heuristic_eat"`) |

Values are clipped on construction (`turn` to ±1, `thrust` to [0, 1]).

## Drive state

`drive_state` is given to the policy as a tuple. With the built-in
`BasicDriveCore` it has 8 entries in this order, each in [0.0, 1.0]:

`(novelty, uncertainty, threat, integrity, agency, learning_progress, arousal, homeostatic_deviation)`

If no drive core is registered, `drive_state` is the empty tuple `()`. A
learned policy must tolerate both lengths (padding/truncating as needed);
`ExampleLearnedPolicy._features` shows one way.

## Policy interface

```python
class Policy(Protocol):
    needs_observations: bool = True

    def act(self, obs, drive_state) -> Action: ...

    # optional:
    def reset(self, n_agents: int, rng) -> None: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
```

- `needs_observations` defaults to `True`. Set it to `False` only if the
  policy never reads `obs`/`drive_state`. Fast paths (GUI Max, bench) may skip
  per-step sensing, drive updates and snapshots only when no active component
  needs observations.
- `act` is called once per live agent per step, in stable iteration order,
  without an agent id. Index recurrent hidden state by a stable per-agent
  identity, not by call order, if the live set can change (this is the one
  simplification in the template).
- World-state reference policies (e.g. `oracle_*`) instead implement
  `select_action(state, world_state, [], step)`, receive `needs_observations =
  False` and are clearly marked as probes, not baselines.
- `reset(n_agents, rng)` is called at episode start (and internally when the
  live-agent count changes) for any stateful policy.
- `get_state()`/`set_state()` are the snapshot hooks. `get_state` must return
  a JSON-friendly structure (numpy arrays are canonicalized); `set_state`
  restores it exactly so scrub-back replay reproduces identical actions.

## Drive core interface

```python
class DriveCore(Protocol):
    names: tuple[str, ...]
    needs_observations: bool = True

    def reset(self, n_agents: int, rng) -> None: ...
    def step(self, obs_batch, info) -> np.ndarray: ...
    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
```

- `step` returns a `(n_live, len(names))` float32 array; row order matches the
  live-agent order passed in `obs_batch`.
- `info` carries at least `{"rng": world._rng}`.
- Retention (`RetentionRule`) runs after raw drive values are computed:
  `E_new = clamp(decay * E_prev + (1 - decay) * drives, 0, 1)` via
  `FixedDecay` (default `lambdas = 0.9`). A recurrent drive core must be
  stepped every step; the fast path only skips it when nothing needs
  observations.

## Episode lifecycle

- An episode runs `run_step` until either `terminated` (all agents dead) or
  `truncated` (step count reaches the cap).
- Death is classified after the run: `hazard` if cumulative `hazard_loss`
  exceeds cumulative `starve_loss`, else `starvation`; an agent alive at the
  cap is `truncation`.
- `action_log` records the actions issued each step and is exported by
  `export_action_log()`.

## Snapshot / restore

`Runner.snapshot()` returns a `Snapshot` capturing the world, RNG state, and
(via `get_state`) the policy and drive-core internal state. `Runner.restore()`
puts them back, so replaying from a snapshot with `set_state` reproduces
identical actions (the scrub-back test asserts this).

## Throughput (measured)

Measured on the dev machine, world 50x50, default config, step cap 1000,
numpy-only. The single-process full pipeline runs (sensing, drive core,
policy, every step) at about 3280 steps/s with 1 agent and 1200 steps/s with
5 agents. World-only stepping (no sensing/drives/policy) is roughly 5520
steps/s at 1 agent and 3130 steps/s at 5 agents.

Multiprocessing eval wall time is dominated by per-seed process spawn and
collection overhead. Setting `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`
inside worker processes removes BLAS thread oversubscription and improves
per-worker throughput substantially. Measured aggregate (summed over workers)
for a 20-seed, cap-10000, Balanced-5 eval:

| Workers | Aggregate steps/s (no threads pin) | Aggregate steps/s (threads=1) |
|---------|-----------------------------------|-------------------------------|
| 1 | 973 | 1343 |
| 2 | 1980 | 2500 |
| 4 | 3725 | 4255 |
| 8 | 5385 | 6670 |

The 4-worker threads=1 run is ~3.2x the single-process wall time. Numbers are
machine-dependent.

## torch policy

Torch is allowed inside a plugin/learned policy and in the training module. It
must never be imported by `simlab.core`, and `simlab.cli` must not load torch
unless the `train` command runs. Learned policies import torch lazily (inside
`__init__`/methods), so importing `simlab.core` or `simlab.cli` never loads
torch (verified by a test). Evaluation runs each seed in its own process, so a
torch policy is isolated and does not leak into the core. Training runs on CPU
only (`torch.set_num_threads(1)` per rollout worker; no CUDA).

## The learned GRU plugin

`simlab/plugins/learned_gru.py` is the reference learned policy. It is a
registered Policy with:

- a GRU cell (hidden 64) plus actor/critic heads, parameters shared across
  agents;
- `needs_observations = True`; it reads only `obs` and `drive_state` through
  the Policy interface;
- per-agent hidden state:
  - `reset(n_agents, rng)` zeroes/randomizes hidden state;
  - `get_state()` returns a JSON-friendly copy of the hidden array;
  - `set_state(state)` restores it exactly, so scrub-back replay reproduces
    identical actions;
- weights loaded from a checkpoint via `--policy learned_gru --checkpoint
  PATH`; the checkpoint file hash is folded into the policy hash so stale
  comparisons are caught.

Deterministic evaluation uses the action means (no sampling), so a given
checkpoint produces identical outputs across runs.

## Training

`python -m simlab.cli train --obs {obs_only,obs_drives} --total-steps N
--workers W --seed S --out DIR --eval-every K` runs recurrent PPO (GAE,
clip, Adam with linear LR decay, entropy bonus, sequence minibatches of length
32 with episode masking). Rollout workers each own a World and a CPU copy of
the policy. Training seeds start at 100000 so they never overlap dev seeds
(0–19) or test seeds (1000–1019). Checkpoints, a `learning.csv` curve, and a
`config.json` are written to `--out`.

## Commands

Evaluate on dev seeds (0–19) and compare against baselines:

```powershell
python -m simlab.cli eval --policy NAME --scenario balanced --agents 5 `
  --seeds 20 --cap 10000 --workers 4 --compare
```

Evaluate on held-out test seeds (1000–1019):

```powershell
python -m simlab.cli eval --policy NAME --scenario balanced --agents 5 `
  --seeds 20 --cap 10000 --workers 4 --compare --seed-offset 1000
```

Scenarios: `balanced`, `solo`, `hard`. Workers default to 1; use `--workers`
for multiprocessing (one process per seed). See `simlab/cli.py` for the full
flag list (`--allow-stale`, `--drive`, `--out`, ...).
