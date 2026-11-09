# Evidence archive — evidence-20261015

Branch: evidence-20261015 (from paper-draft-20261015, commit a5ce58c)
Created: 2026-10-15
Total input bytes: ~3.05 MiB (under 50 MB cap; no truncation needed)
No checkpoints copied. No GPU work performed.

## Source files copied (read-only; originals untouched)

| Path | Size | SHA-256 (first 16 chars) | Source |
|---|---|---|---|
| lm_pilot/A/val_bpb.csv | 307 | (see p1a_hashes.txt) | pilot arm A |
| lm_pilot/Bprime/val_bpb.csv | 307 | (see p1a_hashes.txt) | pilot arm B' |
| lm_pilot/C/val_bpb.csv | 307 | (see p1a_hashes.txt) | pilot arm C |
| lm_pilot/state.json | 1692 | — | pilot driver state |
| lm_pilot/driver.log | 7865 | — | pilot driver (interruption + resume) |
| lm_pilot/A/train.log | 807723 | — | pilot A (first 200 + last 200 lines kept; full >5 MB) |
| lm_pilot/Bprime/train.log | 853558 | — | pilot B' (same) |
| lm_pilot/C/train.log | 927275 | — | pilot C (same) |
| recall-A-nomask-s43-w20/eval_result.txt | 10729 | — | positive control |
| results/recall_benchmark/main/main_d_dconst.json | 101738 | 845AA5C... | recall grid (current) |
| results/recall_benchmark/main/main_d_dconst.json.bak-20261014-172553 | 51364 | — | earlier backup |
| results/recall_benchmark/main/main_d_dconst.json.bak-20261014-180342 | 76478 | — | earlier backup |
| results/recall_benchmark/main/main_d_dconst.json.pre-promote-20261014-175926 | 51364 | — | pre-promote |
| results/recall_benchmark/main/main_d_dconst.json.pre-promote-20261014-191304 | 76478 | — | pre-promote |
| results/recall_benchmark/main/main_d_dconst.json.tmp.promoted-20261014-175939 | 76478 | — | promoted temp |
| results/recall_benchmark/main/main_d_dconst.json.tmp.promoted-20261014-191304 | 101738 | 845AA5C... | promoted temp (matches current) |
| STATUS_followup.md | 181 | — | gated follow-up status |
| evidence/windows_events_20261015.txt | 3626 | — | Part 1c event export |
| evidence/gated_cpu_tests_20261015.txt | 2478 | — | Part 1d test output (7 passed) |
| evidence/p1a_identity.py | — | — | seed-101 identity script |
| evidence/p1a_out.txt | — | — | seed-101 identity output |
| evidence/p1b_config.py | — | — | config comparison script |
| evidence/p1b_out.txt | — | — | config comparison output |

## Patches (mech2-20261015; UNREVIEWED — never applied to paper branch)

- an_fix.patch (566 B) — label/nan fix; not applied
- followup_driver_hardening.patch (831 B) — driver hardening; not applied
- gated_write_commit.patch (38515 B) — gated mechanism; not applied
- mech_diff.patch (10934 B) — mechanism diff; not applied

## Part 1 outputs included

- p1a_identity.py + p1a_out.txt (seed-101 identity, pooled recomputation, backup comparison, checkpoint hashes)
- p1b_config.py + p1b_out.txt (600-iter vs pilot config comparison; differences marked)
- windows_events_20261015.txt (16 System events; 1074/109 restart near interruption; 6005/6006 after resume)
- gated_cpu_tests_20261015.txt (pytest 7 passed, 1 warning; assertion `max|E_next|=nan` reproduced)

## Part 3 paper changes (on paper-draft-20261015, commit a5ce58c)

- DRAFT.md: decay bound corrected; "pre-registered" replaced; spill withdrawn; pilot separate; interruption cited; gated only from files; seed-101 kept with [PENDING]; positive control cited.
- claims.md: 337 numeric + 25 qualitative rows; 0 unmatched numbers.
- OPEN_ITEMS.md: placeholders, unverified items, conflicts (repository wins).
- RDA_research_paper_print.docx: body rebuilt from DRAFT.md; title block kept.

## Not included (deliberate)

- No checkpoint .pt files (would exceed cap; hashes in p1a_out.txt instead).
- No code edits to nanochat/ or diagnostics/.
- No GPU work; CUDA_VISIBLE_DEVICES='' only inside the Part 1d command.
