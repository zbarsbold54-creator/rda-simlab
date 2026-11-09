# Combined results index

Curated results from both parts. Raw logs and checkpoints are not copied here; each entry links to the source file in `rda/` or `simlab/`.

## RDA (nanochat fork)

- `rda/results/ab_v3_dbs2/SUMMARY.md` — 3-arm FineWeb-Edu BPB table (A / B' / C, 600 iterations, seeds 42/43/44).
- `rda/results/ab_v3_dbs2/logs/ab_v3_dbs2_results.json` — machine-readable BPB for the 3-arm run.
- `rda/results/multiscale_decay/SUMMARY.md` — D vs D_const (seed 43, no gain).
- `rda/results/multiscale_decay/results.json` — machine-readable multiscale decay results.
- `rda/results/recall_benchmark/main/main_d_dconst.json` — cross-segment recall (at chance, `WITHIN_NULL_CI`).
- `rda/results/lm_pilot/val_bpb.csv` — long language pilot val BPB (A 1.055337, B' 1.055067, C 1.053788).

## SimLab

- `simlab/docs/baselines.json` — frozen evaluation baselines (config + policy hashes).
- `simlab/logs/overnight/testeval_summary.csv` — per-arm cap% source of truth (94 rows, test seeds 1000-1019).
- `simlab/logs/overnight/testeval_*.json` — raw per-run eval JSONs (94 files).

## Figma charts

- `results/figma/` — SimLab Results charts (see `results/figma/README.md`). PNG export pending.
