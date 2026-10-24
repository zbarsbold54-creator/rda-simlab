# Extending SimLab

This file explains how to extend the two research-facing extension points for
RDA: drive definitions and retention rules.

## Add a drive definition

Drive signals `d_t` are the instantaneous, deterministic, numpy side of the
RDA interface. They are defined once in
[simlab/drives/definitions.py](simlab/drives/definitions.py).

1. Write a pure function `_my_drive(obs, last_rays) -> float` returning a value
   clipped to `[0, 1]`. `obs` exposes `rays`, `ray_types`, and
   `proprioception (energy, health, speed)`. `last_rays` is the previous raw ray
   vector (used to compute change signals). Use only information available at
   step `t` so the drive stays causal.
2. Add a `DriveDef` entry to the `_DRIVES` tuple with `name`, `label` (a
   PLACEHOLDER string you will edit), `docstring`, and `compute`.
3. The learner-side Retention reads `d_t` from the trailing 8 dims of the
   `obs_drives` encoding, so `N_DRIVES` and the order of `_DRIVES` must match
   the `DRIVE_NAMES` tuple. Keep `N_DRIVES` in sync everywhere.
4. The drive core (`simlab/plugins/drive_cores.py`) calls
   `raw_drive_vector`, so no further wiring is needed for the GUI or inspector.

There is no test specific to individual drive values; the causality tests in
`tests/test_rda.py` assert that future observations do not change `d_t`, `E_t`,
`lam_t`, `m_t`, or action logits at time `t`.

## Add a retention rule

Retention rules live in [simlab/training/retention.py](simlab/training/retention.py)
as modes of the `Retention` module. All variants share one state-update path:

`E_t = lam_t * E_{t-1} + (1 - lam_t) * d_t`

To add a new rule:

1. Add a mode string to the `Retention.__init__` dispatch. The mode determines:
   - the state layout (`state_dim` and the `_split`/`_assemble` pair),
   - whether the gate uses a shared G1 surprise trace, per-drive traces, or a
     fixed/learned value,
   - how the auxiliary loss is computed (if any).
2. Follow the past-only rule: the gate `lam_t` may use only information up to
   `t-1` (the trace `m_{t-1}`). The write at `t` may use `d_t`. Document where
   each quantity is computed in `docs/RDA_SIM.md`.
3. If the rule introduces an auxiliary head, register its parameters normally
   (they are included in `TorchPolicy.parameters()`). Keep trainable gate
   parameters out of the scalar optimizer group unless `trainable_gates` is
   enabled.
4. Register the variant as an ArchSpec in
   [simlab/training/arch_spec.py](simlab/training/arch_spec.py) by adding it
   to `_RDA_VARIANTS` (aliases) or to `_RETENTION_DESC` (combos); the factory
   loop then exposes it in the GUI dropdowns with a generated form, and
   `build_policy_cfg` forces `obs_drives` for any `rda*` name.
5. The backbone factory (`simlab/training/backbones.py`) maps the preset name
   to the retention mode. Add the mapping there and confirm `config_dict`
   round-trips the variant name so checkpoints restore the correct rule.
6. Add a causality test in `tests/test_rda.py` for the new variant.

All RDA variants share the obs encoder and heads, and the `match_params` helper
picks a hidden / `d_model` size so parameter counts land within 5% of the target
(the GUI and tests size-match to the transformer reference). The **auxiliary
loss is uniform across every retention variant** (`pred_o` and `pred_d` heads,
both always included with the same weight) — only the gate differs between
variants. `test_aux_loss_parity_across_variants` verifies equality when the
aux-head weights are shared.
