# Open Items

Status of `DRAFT.md` (paper-draft-20261015). Every number in the draft is listed in [claims.md](claims.md) with its file and key or line. The checker found no unmatched numbers. Where the request and the repository disagree, the repository wins and the draft follows it.

## Placeholders and author decisions

- **[AUTHOR TO CONFIRM] minimum effect of interest.** `docs/recall_spec.md` line 52 records 0.02. The draft leaves the slot open (Section 3.4) and does not use the value to claim anything.
- **Citations.** Five `[CITE: ...]` markers in Section 2 (nanochat; FiLM; Muon and AdamW; recurrent memory in language models; computational affect and drive representations). None was filled from memory.
- **Title block of the .docx.** The author names, email, and keywords were carried over from the earlier print version. Confirm them before circulation.
- **Threshold wording.** The draft calls 0.0137 a "pre-set yardstick" and says the records describe it as a proposal. Confirm which wording the authors want.

## [UNVERIFIED] items

1. **Recall driver stop after eval-D_const-101.** `results/recall_benchmark/main/main_d_dconst.txt` ends at line 3533 after `SUBPROCESS_END ... exit_code=0`, with no completion record. The JSON `completed` flag is false. Cause unknown.
2. **Arm-C interruption in the long pilot.** The driver log shows the last C checkpoint at step 3000 (line 57), then a stale-lock takeover of PID 29624 at 06:01:46 (line 58). The brief attributes the stop to a Windows servicing restart. No restart record is in the retained files. Operating-system event logs were not consulted and would be needed to verify this.
3. **Why only seeds 43 and 101 ran.** `run_recall_main_d_dconst.ps1` lists seeds 43, 101, 102, 103, 104. No record explains the omission.
4. **Non-finite state in the gated-write run.** The brief reports a numerical failure at about steps 12-19. The only status record, `STATUS_followup.md`, lists MECH_GATED as failed with "Smoke directory unexpectedly has checkpoint 20" (line 5). No non-finite record was found in `research/`, `logs/`, `lm_pilot_smoke/`, or the `mech2-20261015` commit messages.
5. **CPU tests on `mech2-20261015`.** Test files exist (`tests/test_gated_label_leak.py`, `tests/test_gated_optimizer.py`, `tests/test_mechanism_gated.py`), but no pass record was found. The tests were not run in this pass, which would need a separate worktree and CPU time.
6. **Recall-grid code and the leak fix.** Ancestry is verified for `5566d90`, `1cf28c1`, and `5bb8933`. The recall-grid code commit is not recorded in the results JSON, so its ancestry relative to `969e067` is unknown.
7. **"Data position approximate after resume."** `run_lm_pilot.ps1` (lines 207 and 282-283) sets this note on resume and clears it after the next checkpoint. The records do not explain it. The draft relies on the metadata identity at step 3000 (`pq_idx 3, rg_idx 7, epoch 1` in all three arms) and on the resume code path (`scripts/base_train.py` line 450). No post-resume position was checked directly.

## Conflicts between the brief and the repository

| # | Brief says | Repository says | Handling in the draft |
|---|---|---|---|
| 1 | Positive-control result to be reported; earlier open items said it was not found | `recall-A-nomask-s43-w20/eval_result.txt` exists at the repository root. It is untracked, lives outside `results/recall_benchmark/`, and has no research record | Reported (Section 5.6). The untracked status is an open item: commit it or record it |
| 2 | "Pre-registered" 0.0137 threshold | `research/Control Arm.md` line 134: "a proposal, not a calibrated value". `research/EXPERIMENT_LOG.md` lines 30-32: cannot be distinguished from seed noise with three seeds | Called a pre-set yardstick; the records' description is stated |
| 3 | Arm-C slowdown was a VRAM spill at device batch 4 | `research/EXPERIMENT_LOG.md` E18(b) withdraws the VRAM explanation. Shared memory was 14.5 MiB (no spill). E18 does not state its batch size. The preceding E17 A/B flags use device batch 4 (line 2173) | Follows the record: explanation withdrawn, cause open. The batch-4 link is stated as an inference |
| 4 | Decay bound at delay 832 about 2.3e-4 | Grid maximum is 4.2540e-4 at decay 0.9984375. 2.3116e-4 is the value at decay 0.996875 | Draft reports 4.25e-4 as the grid maximum and labels 2.3e-4 as the 0.996875 value |
| 5 | Long pilot continues the 600-iteration setup | Pilot used width 512 (aspect 64) and window pattern SSSL; the comparison used width 768 and window L. The pilot passed no head-dim or window flags | Configuration difference shown in Table 1; pilot results not attributed to FiLM or state content |
| 6 | Arm C stopped by a Windows servicing restart | Only a stale-lock takeover is logged | See unverified item 2 |
| 7 | Mechanism variant: non-finite state at about steps 12-19 | `STATUS_followup.md`: "Smoke directory unexpectedly has checkpoint 20" | See unverified item 4 |
| 8 | Seed-101 D and D_const results both came from the promoted artifact | Promoted artifact and current JSON are byte-identical (same SHA-256), but the two seed-101 runs have identical pooled CROSS values (accuracy 0.062663, gap +0.000163) while their per-delay values differ | Reported as observed. Not explained; check whether one pooled block was copied (see item 9) |
| 9 | Pooled CROSS values for D-101 and D_const-101 | Identical pooled values. Per-delay values differ at several delays (for example delay 448 ORDINARY: 0.062500 vs 0.060547) | Open. Recompute pooled values from per-delay counts for both runs before any contrast is used |
| 10 | Stray value: B'-seed-42 post-fix BPB | `research/EXPERIMENT_LOG.md` line 601 and `research/BPB.md` line 69 show B'-s42 at 1.137101 (post-fix). The withdrawn value is 1.226305 | Not used in the draft. The two numbers come from different configurations and should not be placed side by side without a note |
| 11 | Pilot evidence labelled "single seed" | Correct. The aspect and window differences are also a configuration change | Stated in Sections 4.2, 5.2, and 9 |

## Other open items

- **Claim extraction script.** `claims.md` was produced by a parser kept outside the repository. Commit the parser, or record its logic, if the ledger must be regenerable.
- **Unanalysed diagnostic.** `ROW_VALUE_COLLISION` reports 516 of 8,192 rows (6.30%) with equal SALIENT and ORDINARY values. The rate is close to the 1/16 expected from independent 16-way draws (computed). Its effect on scoring has not been analysed.
- **Planning notes out of date.** `METHODS.md`, `EXPERIMENTS.md`, and `LIMITATIONS.md` still describe some runs as not run (for example E14 "NOT RUN, BLOCKED" and the A/B comparison). They were not edited, so the draft does not rely on them.
- **Descriptive look before amendment.** `docs/recall_spec.md` lines 104-112 record that seed-43 recall rows were viewed before the power amendment. The draft states this.
- **Word-level claims.** The draft contains no instance of "significant" or related forms. Confirm this if the text is edited.
- **.docx.** `paper/RDA_research_paper_print.docx` was previously untracked. Its body was regenerated from `DRAFT.md`, and its title block was kept. Figures and tables should be checked visually in Word.
- **Working tree.** Unrelated uncommitted changes exist in `research/`, `results/`, and root-level files. None was staged. The branch `paper-draft-20261015` was created from `lm-pilot-20261014` HEAD and carries only the `paper/` changes.
- **No GPU work** was done. No results folder, JSON, checkpoint, or code file was modified.
