# Extending SimLab

Drop a Python module into `simlab/plugins/`; the GUI imports plugin modules and their `@register` decorators automatically. Names become selectable in the Scenario dock where supported. Implement the interface from `simlab/interfaces/` and keep plugin imports lightweight.

## Add a drive

A drive core implements `reset(n_agents, rng)` and `step(obs_batch, info)`, returning an `(agents, drives)` NumPy array in `[0, 1]`. Decorate and save this module under `simlab/plugins/`:

```python
import numpy as np
from simlab.core.registry import register


@register("drive", "curiosity")
class CuriosityDrive:
    names = ("novelty",)

    def reset(self, n_agents, rng):
        self.rng = rng

    def step(self, obs_batch, info):
        return np.asarray(
            [[min(len(obs.rays) / 32, 1)] for obs in obs_batch], dtype=np.float32
        )
```

Select `curiosity` in the Scenario dock. Replace the example formula with one value per agent for every name.

## Add a policy

A policy implements `act(observation, drive_state)` and returns an `Action`:

```python
from simlab.core.registry import register
from simlab.core.types import Action


@register("policy", "coast")
class CoastPolicy:
    def act(self, obs, drive_state=None):
        return Action(turn=0.0, thrust=0.2, eat=False)
```

The name `coast` appears in the Scenario dock. Policies may instead implement `select_action(agent_state, world_state, available_actions, step)` for a world-level controller.

## Add a learned policy (template)

For a policy with recurrent internal state that must be stepped every step, implement `needs_observations = True`, `reset(n_agents, rng)`, `get_state()` and `set_state()`. `reset` resizes/zeroes hidden state; `get_state`/`set_state` let snapshots capture and restore it so scrub-back replay reproduces identical actions. See `simlab/plugins/example_learned_policy.py` for the full template (a numpy-only GRU-like cell with fixed random weights; it must not import torch). Key points:

- Declare `needs_observations = True` so fast paths never skip sensing/drives for your policy.
- Build input features only from `obs` (rays + proprioception) and the `drive_state` tuple passed into `act`.
- Keep hidden state in a NumPy array; `get_state()` returns a copy, `set_state()` restores it.
- `act` is called once per live agent per step without an agent id — index hidden state by a stable per-agent identity if your policy depends on which agent is acting.
- Register as usual with `@register("policy", "my_policy")`; the Scenario dock and the eval CLI then discover it.

## Plug in your own architecture

Learned policies are built by composing a world-side **DriveCore** (numpy) with
a learner-side **Backbone** (torch) behind the `TorchPolicy` adapter. Backbones
register in the existing `backbone` slot; drive cores in the existing `drive`
slot. Build a policy from a config dict:

```python
from simlab.training.torch_policy import make_policy

cfg = {
    "backbone": "transformer",   # registered in the existing `backbone` slot
    "drive_core": "rda8",        # world-facing numpy drive core
    "fusion": "concat",          # "concat" | "film"
    "obs_mode": "obs_drives",
    "kwargs": {"context": 16, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "match_params": 9798,        # optional: match a param count within 5%
    "seed": 0,
}
policy = make_policy(cfg)        # -> TorchPolicy implementing the Policy port
policy.save_checkpoint("logs/my_arch/final.pt")  # stores the arch config
```

Checkpoints store the architecture config; loading one rebuilds the backbone from
the stored config. Built-in backbones: `gru` (legacy GRU), `transformer` (causal
ring buffer), and the RDA presets `rda`, `rda_const`, `rda_g1`,
`rda_g1_rand`, `rda_g3`, `rda_g3_tied` (see `docs/RDA_SIM.md`). Built-in drive
cores: `none`, `constant`, `instant`, `rda8`. Eval the resulting policy with
`simlab.cli eval --policy learned_torch --checkpoint logs/my_arch/final.pt`; the
eval table prints `param_count` per config.

## Add a retention rule

Retention rules live in the learner side (`simlab/training/retention.py`) and
implement the shared recurrence `E_t = lam_t * E_{t-1} + (1 - lam_t) * d_t`.
A new variant is a class with `step`/`initial_state` methods plus a `config`
dataclass; register it in the `RETENTION` registry in `retention.py` and add a
preset mapping in `backbones.py` (`BACKBONES`) so it appears in the GUI's
backbone dropdown with a generated form. See `docs/RDA_SIM.md` §3 for the
surprise (G1), control (`surprise_rand`), and per-drive (G3) variants, their
gate-timing rules, and the past-only rule that must be preserved.

The legacy plugin registration path is retained for world-side rules:

```python
from simlab.core.registry import register


@register("retention", "fast_decay")
class FastDecay:
    def step(self, previous, drives, context):
        return 0.5 * previous + 0.5 * drives
```

## Add a drive definition

The eight world-side drives are defined in `simlab/drives/definitions.py` — one
file, one entry per drive, each with a `name`, `label`, `docstring`, and a pure
numpy `compute(obs, last_rays)` returning a value in `[0, 1]`. Add a new drive to
`DRIVE_NAMES` (append at the end; the order is the fixed `obs_drives` encoding)
and provide its `compute`. `InstantDriveCore` picks it up automatically, and the
inspector renders it. Drive labels are current **PLACEHOLDER** strings you can
edit in place without touching any other file.

## Add an architecture and see it in the GUI

Every backbone and drive core is described by an `ArchSpec`
(`simlab.training.arch_spec`), a torch-free metadata record. A spec has:

- `name` — unique id used in configs, hashes and checkpoints.
- `kind` — `"backbone"` or `"drive_core"`.
- `description` — human-readable text shown in the panel.
- `config_type` — a `@dataclass` whose typed fields drive the auto-generated form.
  Field `metadata` is read for `min`, `max`, `choices` and `help`.
- `loader` — a callable that returns the actual class; it is invoked only when a
  policy is built (so torch is never imported at GUI startup).
- `source` — where the spec came from (built-in or a plugin file).

The GUI panel has no knowledge of any specific architecture: it renders dropdowns
from every registered spec and builds a form from the selected spec's fields.
Validation (ranges, choices, normalization) also runs from the spec alone, so a
new architecture appears in the GUI with no code changes to the panel.

To add your own architecture, drop a file into the plugin folder (default
`plugins/architectures`, override with the `SIMLAB_ARCH_DIR` environment
variable) and press **Rescan**. Import errors and invalid specs are collected and
shown in the "Plugin errors" list; they never crash the GUI. The **New
architecture template** button writes a documented example file into that folder.

Example plugin file (a backbone spec):

```python
from dataclasses import dataclass, field

from simlab.training.arch_spec import register_arch


@dataclass
class MyConfig:
    width: int = field(default=64, metadata={"min": 8, "max": 512, "help": "hidden width"})


@register_arch("backbone", "mybb", description="My backbone", config=MyConfig)
class MyBackbone:
    def __init__(self, **kwargs):
        self.width = int(kwargs.get("width", 64))

    # implement the Backbone interface (initial_state, forward, param_count,
    # config_dict); the TorchPolicy adapter wraps it for the sim Policy port.
```

Select `mybb` in the panel, set the fields, press **Build** (loads torch lazily,
shows param count and architecture hash), or **Save experiment** to write a TOML
selection under `runs/` that **Load experiment** restores exactly. Checkpoints
store the architecture config, so loading one rebuilds the backbone from the
stored config.
