# Figures

No figures have been generated.

Figures belong here only once they are produced by a script in
`diagnostics/` and the underlying run is recorded in
`../research/EXPERIMENT_LOG.md`. A figure whose data cannot be regenerated
does not belong in this repository.

## Planned

Listed as intentions, not commitments. Each names the script that would
produce it, so that the plan is executable rather than aspirational.

- **State trajectories** — per-dimension `|E|` over steps, A vs B'.
  Source: `diagnostics/probe_pe_film.py`.
- **The event ratchet** — `event_proj` weight norm and event magnitude over
  steps, with the frozen-base control overlaid, before the fix.
  Source: `diagnostics/probe_event_growth.py`.
- **The FiLM self-excitation** — FiLM gain over steps for C1, C2 and C3,
  showing that C3 grows with a flat PE and C2 does not grow at all. This is
  the decisive figure for the mechanism claim.
  Source: `diagnostics/probe_pe_film.py`.
- **Gradient dormancy** — per-group gradient norms at steps 0 and 1,
  showing the state pathway at exactly zero then nonzero.
  Source: `diagnostics/pe_gradient_onset.py`.
- **Causality** — the perturbation test, showing state positions `< k`
  bit-identical under a change to `targets[:, k]`.
  Source: `diagnostics/test_pe_film_invariants.py`.

Keep a `.gitkeep` here so the directory exists in the repository.
