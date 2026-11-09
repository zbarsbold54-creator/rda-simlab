# SimLab Learner Architectures

This note defines how the SimLab learner becomes architecture-agnostic. It is
the design contract for the `arch-v0` change. The World is untouched; this is
plumbing only.

## Current state (inspected)

- Policy plugin port (`simlab/interfaces/policy.py`): `act(obs, drive_state) ->
  Action`, optional `reset(n_agents, rng)`, `get_state()`, `set_state()`,
  `needs_observations`.
- DriveCore port (`simlab/interfaces/drive_core.py`): `names`,
  `needs_observations`, `reset(n_agents, rng)`,
  `step(obs_batch, info) -> np.ndarray` `(n_live, n_drives)`. The built-in
  `BasicDriveCore` additionally exposes `get_state()/set_state()` so snapshots
  can capture/restore its recurrence.
- GRU policy is built in `simlab/plugins/learned_gru.py` (a `_Net` with a
  `gru_ih/gru_hh` cell, `policy_head/value_head`, and per-drive
  `log_std_turn/log_std_thrust`). PPO trainer `simlab/training/ppo.py`
  constructs it directly (`_build_net`) and reaches into `_net`, `_hidden`,
  `log_std_*`. Hidden state resets on episode end by rebuilding the world and
  calling `policy.reset(n_agents, rng)` in `_WorkerEnv._reset_episode`; within
  a rollout, `_ppo_update` zeroes the hidden state with the episode
  `done`/`mask` at each sequence step.
- `obs_only` (83 dims) vs `obs_drives` (91 dims) is encoded in
  `simlab/training/encoding.py` (`obs_dim`, `encode_observation`), selecting
  whether the 8 drive values are appended.
- Config hash: `resolved_config_hash` over `SimLabConfig.to_dict()`. Policy
  hash: `policy_source_hash` hashes the policy class's source files plus, when
  a checkpoint path is given, the checkpoint file bytes. Checkpoints
  (`LearnedGRU.save_checkpoint`) store `{in_dim, hidden, with_drives,
  state_dict}`.

## Proposed slot model

### a) Sim-side DriveCore (numpy, deterministic)

Implementations `none`, `constant`, `rda8` registered in the existing `drive`
slot. `rda8` keeps an EMA state

    E[t] = decay * E[t-1] + (1 - decay) * drive[t]

with `decay` configurable as a scalar or per-drive length-8 vector. `drive[t]`
depends only on information up to step `t`.

Conflict note: the idealized `step(obs, events, state) -> (drive_vector[8],
new_state)` signature conflicts with the wired pipeline, which calls
`drive_core.step(obs_batch, info) -> np.ndarray` and restores recurrence through
`get_state()/set_state()`. Smallest change (adopted): keep the existing
DriveCore port, and express `rda8`'s recurrence as internal `_previous`
(EMA) / `_last_rays` state exposed through `get_state()/set_state()`, exactly
like `BasicDriveCore`. Snapshots capture drive state via `get_state()` beside
world snapshots (the existing mechanism); the World / snapshot schema is not
modified.

### b) Learner-side Backbone (torch)

`simlab/training/backbones.py` defines the contract:

- `initial_state(batch) -> Tensor` of shape `(batch, state_dim)`;
- `forward(obs_seq, drive_seq, state, done_mask) -> (logits, value, new_state)`
  with `obs_seq (B,T,obs_dim)`, `drive_seq (B,T,n_drives)` or `None`,
  `state (B,state_dim)`, `done_mask (B,T)`; returns `logits (B,T,3)`,
  `value (B,T)`, `new_state (B,state_dim)`;
- `param_count() -> int`, `config_dict() -> dict`.

Fusion `concat` (append `drive_seq` to the encoded obs) or `film`
(per-channel affine modulation) is applied to `drive_seq`. Backbones are
registered in the existing registry under the `backbone` type: `gru`,
`transformer`, `rda` (stub). The registry itself is not replaced — no new
top-level registry.

### c) TorchPolicy adapter

`simlab/training/torch_policy.py` implements the sim Policy port by wrapping any
Backbone. It exposes the low-level surfaces the trainer already uses
(`sample_actions_batch`, `evaluate_batch`, `frame_step`, `net_state_dict`,
`param_count`, `hidden_size`, `log_std_turn/thrust`). Trainer and eval depend
only on this adapter.

### d) `make_policy(cfg)`

Keys: `backbone`, `drive_core`, `fusion`, `kwargs`, plus `seed`,
`obs_mode`/`with_drives`, optional `match_params` (target parameter count) and
optional `checkpoint`. Architecture config is stored in checkpoints (and folded
into the policy hash through the checkpoint bytes); loading a checkpoint
rebuilds from its stored config. Legacy `learner-v0` checkpoints (no `arch`
key) load as the `gru` backbone with obs mode inferred from `in_dim` (83 ->
obs_only, 91 -> obs_drives).

Backbones: `gru` reproduces the existing GRU cell exactly (no behavior
change); `transformer` is a small causal Transformer over the last `K` embedded
obs steps (`K` is a named config value), state = ring buffer, reset on
`done_mask`; `rda` is a stub raising `NotImplementedError` (user supplies it
later through the slot). The observation encoder and output heads are shared by
all backbones.
