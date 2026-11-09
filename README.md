# RDA + SimLab

Research on the **Regulatory Drive Architecture (RDA)**: a persistent internal state of eight drives, tested in two places, a small transformer language model and a 2D survival simulation.

The question is functional, not philosophical: does a persistent regulatory state change how a model learns, what it retains and how it behaves, and which drives matter? Nothing here claims subjective emotion.

## What is in the repo

| Folder | What it is | How it is tested |
|---|---|---|
| `rda/` | Fork of [nanochat](https://github.com/karpathy/nanochat) with an 8-drive state that conditions the transformer through per-position causal FiLM | FineWeb-Edu validation BPB, plus a synthetic delayed-recall benchmark |
| `simlab/` | Deterministic 2D survival world with a PySide6 GUI, plugin slots for policies, drive cores and retention rules, PPO training and a frozen evaluation harness | Survival: share of agents reaching the 10,000-tick cap (cap%) |
| `results/` | Curated summaries and raw result files from both parts | |

The two parts are separate code bases with separate Python environments. SimLab does not import `rda/`. Its RDA presets (`rda_gru_*`, `rda_tf_*`) are an independent implementation of the retention ideas (surprise-gated and per-drive retention) on a GRU or small transformer core. See `rda/README.md` and `simlab/README.md`.

## The eight drives

Novelty, uncertainty, threat, integrity, agency, learning progress, arousal, homeostatic deviation. The state follows `E[t] = lambda * E[t-1] + (1 - lambda) * d[t]`, where `d` is built from events and prediction error. The retention `lambda` is what the experiments vary: fixed, constant, surprise-gated, or per-drive.

## Combined results

Every comparison is against a constant-state control, because most of the early gains turned out to come from the extra FiLM path, not from the state content.

| Question | Result | Where |
|---|---|---|
| Does drive-state FiLM beat the baseline on language loss? | Yes: B' minus A is -0.0174 BPB on average (3 seeds, 600 iterations, toy scale) | `rda/` |
| Is that gain due to state content? | No: constant-state FiLM (C) is within -0.00076 BPB of B'; shuffling the state at eval time costs only 0.001-0.002 BPB | `rda/` |
| Does it hold in a long run? | No gain: val BPB A 1.055337, B' 1.055067, C 1.053788 (4,578 iterations, width 512) | `rda/` |
| Do multi-timescale decays help? | No: D 1.275386 vs its constant control 1.274947 (1 seed) | `rda/` |
| Can the state carry information across segments? | No: D and D_const are at chance on cross-segment recall while the benchmark itself is valid (manipulation check passed on arm A) | `rda/` |
| Does RDA retention beat a plain GRU in the simulation? | Not shown: arm D reaches 25.3% mean cap% on Hard, below the fresh-init mean of 27.3%; paired gain over init about +8.8 points, t about 1.0 | `simlab/` |
| How hard is the simulation? | Hard, dev seeds: heuristic about 46%, oracle_avoid about 96% | `simlab/` |

So far, neither part has shown that the drive state's content helps. Two things are still untested: the gated-write mechanism (M and M_const, on `mech2-20261015`, failed with a non-finite state) and a size-matched plain GRU or transformer trained for the same budget in SimLab.

## Integrity notes

- A one-token label leak in `rda.py` was found and fixed (commit `969e067`). The first B' gain of 0.0749 BPB was withdrawn.
- The 0.0137 BPB threshold is a proposal from the control-arm note, not a calibrated or pre-registered threshold.
- The earlier explanation of the arm C slowdown (VRAM spill) was withdrawn in the repo's own notes.
- In SimLab, `logs/overnight/testeval_summary.csv` is the source of truth for per-arm results; earlier hand-typed tables were wrong in places.
- Results are toy scale: depth 8, a few seeds, one machine.

## Setup

Windows 11, Python 3.11, RTX 5060 Ti (8 GB), 32 GB RAM. Use one virtual environment per part.

- `rda/`: see `rda/README.md` (GPU torch, set `NANOCHAT_DATASET=fineweb-edu`).
- `simlab/`: see `simlab/README.md` (CPU torch is enough; training is faster on CPU than on this GPU for the GRU).

## License

MIT, inherited from nanochat (Andrej Karpathy).
