# RDA: Regulatory Drive Architecture (nanochat fork)

A fork of nanochat that adds a persistent state of eight drives and feeds it to the transformer. The code is in `nanochat/rda.py` and is switched on with `--rda`.

## The mechanism

- **State:** eight drives (novelty, uncertainty, threat, integrity, agency, learning progress, arousal, homeostatic deviation), updated as `E[t] = decay * E[t-1] + (1 - decay) * drive[t]`, with `drive = tanh(event_effect + pe_effect)`.
- **Use:** the state is turned into a per-position FiLM modulation (scale and shift) of the transformer. It is causal: position `p` only sees the state built from tokens before it.
- **Not yet used:** the state-prediction-error `dpe` is computed but never consumed.
- **Horizon:** decay 0.95 gives a horizon of about 20 positions. The state is within-sequence inertia, not memory across documents. Batches are independent packed samples, so persistence across batches would need a dataloader change.

## Arms

| Arm | What it is | Flag |
|---|---|---|
| A | Baseline nanochat | none |
| B' | Drive-state FiLM (leak-fixed) | `--rda` |
| C | Constant-state FiLM control (a learned vector replaces the drive state) | `--rda --rda-const-state` |
| D | B' with eight per-drive decays, horizons about 10 to 1280 tokens | `--rda-decays` |
| D_const | Constant-state control for D | |
| M, M_const | Gated-write state (closed-gate bias -5 to -9) and its control | `--rda-gated-write` (on `mech2-20261015`, not on `master`) |

M and M_const are on `mech2-20261015` (not on `master`). The gated-write run failed with a non-finite state at about steps 12-19 (see `paper/claims.md`); it has not produced a valid result.

## Results

FineWeb-Edu final validation BPB, depth 8, sequence length 1024, device batch 2, 600 iterations, seeds 42/43/44.

| Arm | s42 | s43 | s44 | Mean | ms/step | Peak MiB |
|---|---|---|---|---|---|---|
| A | 1.300680 | 1.287114 | 1.293590 | 1.293795 | 1273 | 3707 |
| B' | 1.277996 | 1.275198 | 1.275858 | 1.276351 | about 1990 | 4592 |
| C | 1.277748 | 1.274950 | 1.274079 | 1.275592 | about 1727 | 5173 |

- B' minus A: -0.0227 / -0.0119 / -0.0177, mean -0.0174.
- C minus B': -0.00025 / -0.00025 / -0.00178, mean -0.00076. The gain comes from the FiLM path, not from the state content.
- Eval-time ablation on trained checkpoints: B' s43 shuffled rows +0.00084, time-reversed +0.00052; s44 +0.00226 and +0.00144. State content is worth about 0.001-0.002 BPB. A constant replacement costs +0.03 to +0.06, which mostly measures distribution shift. C is unchanged by every variant.
- Multi-timescale decay (seed 43): D 1.275386 vs D_const 1.274947. D is no better than its control.
- Long language pilot (width 512, window pattern SSSL, 4,578 iterations): A 1.055337, B' 1.055067, C 1.053788. No gain from RDA. This is a different setup from the 600-iteration runs.

### Synthetic delayed-recall benchmark

Key-value events with salient and ordinary classes, filler, controlled delays and queries; segment-masked attention (`--segment-len 128`) so the state is the only cross-segment channel; 16-way scoring, chance 0.0625; answer weight 20.

- Manipulation check passed: arm A without masking recalls at 0.995-1.000 at delays 192-832, with a no-context control of 0.057-0.067.
- D and D_const learn the within-segment task (about 1.0) but sit at chance across segments. Paired difference near 0, verdict `WITHIN_NULL_CI`.
- Seeds 102-104 of the D/D_const grid did not run as of the last record.

## Integrity notes

- A label leak (`film_state[p]` depended on `x[p+1]`) was found and fixed in `969e067` with a regression test and a pre-flight gate. The first B' canary gain of 0.0749 BPB is withdrawn.
- The 0.0137 BPB spread of the baseline is used as a threshold only as a proposal (`research/Control Arm.md`).
- The VRAM-spill explanation for the arm C slowdown was withdrawn. At device batch 2, C is about 13% faster than B'. At device batch 4 the 8 GB GPU is nearly full and timings are noisy.
- Final loss depends on device batch size, so arms must be compared at the same one.
- Scale is toy: depth 8, three seeds, one machine.

## Reproduce

Windows 11, Python 3.11, RTX 5060 Ti 8 GB.

```powershell
$env:NANOCHAT_DATASET = 'fineweb-edu'      # the default climbmix is not on disk
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_ab_v3_dbs2.ps1
```

- Launch driver scripts as a fresh child process as above; running them in-process broke their `nvidia-smi` query.
- Checkout `c19de86` to reproduce the 3-arm result. Results are in `results/ab_v3_dbs2/`.
- The ablation harness is `diagnostics/ablate_state.py`; the leak test is `diagnostics/test_no_label_leak.py`.

## Layout

| Path | Contents |
|---|---|
| `nanochat/rda.py` | The mechanism |
| `scripts/base_train.py`, `nanochat/loss_eval.py` | Training and evaluation with RDA flags |
| `nanochat/recall_data.py` | Recall benchmark generator |
| `diagnostics/` | Probes, leak test, ablation, profiling |
| `research/`, `paper/` | Notes, specs and the paper draft |

## License

MIT, inherited from nanochat (Andrej Karpathy).
