"""Evaluate the synthetic delayed-recall task with paired bootstrap intervals."""

import argparse
import json
import os
import time

os.environ["NANOCHAT_DTYPE"] = "float32"

import numpy as np
import torch

from diagnostics.recall_models import load_model, parse_decays
from nanochat.recall_data import (
    DELAY_GRID,
    FILLER_START,
    N_VALUES,
    VALUE_START,
    build_batch,
    build_eval_grid,
)


def bootstrap_interval(values, rng, repetitions=5000):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def paired_bootstrap_pvalue(differences, rng, repetitions=5000):
    """One-sided paired-bootstrap p-value: P(resample mean <= 0)."""
    differences = np.asarray(differences, dtype=np.float64)
    indices = rng.integers(
        0, len(differences), size=(repetitions, len(differences))
    )
    samples = differences[indices].mean(axis=1)
    return float((samples <= 0.0).mean())


def holm_correction(pvalues, alpha=0.05):
    """Holm-Bonferroni correction; returns adjusted p-values and flags."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted, [bool(adjusted[i] < alpha) for i in range(m)]


@torch.inference_mode()
def evaluate(checkpoint_dir, step, seed, n_rows, batch_size, device,
             decay_override=None, check_cross_leak=True,
             eval_seed_base=1_000_000_000):
    started_at = time.perf_counter()
    model, _, _ = load_model(
        checkpoint_dir, step, device, seed, segment_len=128,
        decay_override=decay_override,
    )
    rows = build_eval_grid(n_rows=n_rows, seed_base=eval_seed_base)
    class_names = ("SALIENT", "ORDINARY")
    class_index = {name: index for index, name in enumerate(class_names)}
    correct = np.zeros((n_rows, 2), dtype=np.bool_)
    no_context_correct = np.zeros((n_rows, 2), dtype=np.bool_)
    full_vocab_correct = np.zeros((n_rows, 2), dtype=np.bool_)
    buckets = np.zeros(n_rows, dtype=np.int64)
    predictions = np.zeros((n_rows, 2), dtype=np.int64)
    no_context_predictions = np.zeros((n_rows, 2), dtype=np.int64)
    true_values = np.zeros((n_rows, 2), dtype=np.int64)
    row_value_equal = np.zeros(n_rows, dtype=np.bool_)
    for start in range(0, n_rows, batch_size):
        subset = rows[start:start + batch_size]
        inputs, targets, _, metadata = build_batch(subset, device=device)
        logits = model(inputs)
        no_context_inputs = inputs.clone()
        for row_offset, row_meta in enumerate(metadata):
            for event in row_meta["events"]:
                event_key = event["event_key_position"]
                query_key = event["query_position"]
                neutral_positions = (event_key, event_key + 1, query_key)
                for position in neutral_positions:
                    no_context_inputs[row_offset, position] = FILLER_START
                if query_key + 1 < no_context_inputs.size(1):
                    no_context_inputs[row_offset, query_key + 1] = FILLER_START
        no_context_logits = model(no_context_inputs)
        for row_offset, row_meta in enumerate(metadata):
            row_index = start + row_offset
            row_events = row_meta["events"]
            row_value_equal[row_index] = (
                row_events[0]["value"] == row_events[1]["value"]
            )
            for event in row_events:
                class_id = class_index[event["class"]]
                query_position = event["query_position"]
                # The task is 16-way answer retrieval, so score within the
                # value vocabulary; full-vocabulary argmax is separate.
                value_logits = logits[
                    row_offset, query_position,
                    VALUE_START:VALUE_START + N_VALUES,
                ]
                predicted = VALUE_START + int(value_logits.argmax().item())
                no_context_value_logits = no_context_logits[
                    row_offset, query_position,
                    VALUE_START:VALUE_START + N_VALUES,
                ]
                no_context_predicted = VALUE_START + int(
                    no_context_value_logits.argmax().item()
                )
                full_vocab_predicted = int(
                    logits[row_offset, query_position].argmax().item()
                )
                expected = int(targets[row_offset, query_position].item())
                predictions[row_index, class_id] = predicted
                no_context_predictions[row_index, class_id] = (
                    no_context_predicted
                )
                true_values[row_index, class_id] = expected
                correct[row_index, class_id] = predicted == expected
                no_context_correct[row_index, class_id] = (
                    no_context_predicted == expected
                )
                full_vocab_correct[row_index, class_id] = (
                    full_vocab_predicted == expected
                )
                buckets[row_index] = event["delay_bucket"]
        del logits
        del no_context_logits
        del no_context_inputs
    del model
    torch.cuda.empty_cache()

    rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed + 7919)
    permutation_matches = np.zeros((n_rows, 2), dtype=np.bool_)
    no_context_permutation_matches = np.zeros((n_rows, 2), dtype=np.bool_)
    other_event_matches = np.zeros((n_rows, 2), dtype=np.bool_)
    for bucket_index in range(len(DELAY_GRID)):
        bucket_rows = np.flatnonzero(buckets == bucket_index)
        paired_rows = permutation_rng.permutation(bucket_rows)
        while np.any(paired_rows == bucket_rows):
            paired_rows = permutation_rng.permutation(bucket_rows)
        permutation_matches[bucket_rows] = (
            predictions[bucket_rows] == true_values[paired_rows]
        )
        no_context_permutation_matches[bucket_rows] = (
            no_context_predictions[bucket_rows] == true_values[paired_rows]
        )
        other_event_matches[bucket_rows, 0] = (
            predictions[bucket_rows, 0] == true_values[bucket_rows, 1]
        )
        other_event_matches[bucket_rows, 1] = (
            predictions[bucket_rows, 1] == true_values[bucket_rows, 0]
        )
    print(
        f"ROW_VALUE_COLLISION rows={n_rows} equal={int(row_value_equal.sum())} "
        f"percent={100.0 * row_value_equal.mean():.6f}",
        flush=True,
    )
    all_results = {}
    cell_records = []
    for bucket_index, delay in enumerate(DELAY_GRID):
        bucket_mask = buckets == bucket_index
        class_accuracy = {}
        for class_id, class_name in enumerate(class_names):
            values = correct[bucket_mask, class_id].astype(np.float64)
            interval = bootstrap_interval(values, rng)
            accuracy = float(values.mean())
            class_accuracy[class_name] = accuracy
            no_context_values = no_context_correct[
                bucket_mask, class_id
            ].astype(np.float64)
            no_context_interval = bootstrap_interval(no_context_values, rng)
            no_context_accuracy = float(no_context_values.mean())
            no_context_null_values = no_context_permutation_matches[
                bucket_mask, class_id
            ].astype(np.float64)
            no_context_null_interval = bootstrap_interval(
                no_context_null_values, rng
            )
            no_context_null = float(no_context_null_values.mean())
            no_context_flag = (
                "ABOVE_NULL_CI"
                if no_context_accuracy > no_context_null_interval[1]
                else "WITHIN_NULL_CI"
            )
            no_context_diff = (
                no_context_values - no_context_null_values
            )
            no_context_pvalue = paired_bootstrap_pvalue(no_context_diff, rng)
            cell_records.append({
                "delay": int(delay),
                "class": class_name,
                "accuracy": no_context_accuracy,
                "null": no_context_null,
                "pvalue": no_context_pvalue,
                "flag": no_context_flag,
                "diff": no_context_diff,
            })
            print(
                f"NO_CONTEXT delay={delay} class={class_name} "
                f"accuracy={no_context_accuracy:.6f} "
                f"ci95=[{no_context_interval[0]:.6f},"
                f"{no_context_interval[1]:.6f}] "
                f"permutation_null={no_context_null:.6f} "
                f"permutation_ci95=[{no_context_null_interval[0]:.6f},"
                f"{no_context_null_interval[1]:.6f}] "
                f"pvalue={no_context_pvalue:.6f} "
                f"flag={no_context_flag}",
                flush=True,
            )
            permutation_values = permutation_matches[
                bucket_mask, class_id
            ].astype(np.float64)
            permutation_interval = bootstrap_interval(
                permutation_values, rng
            )
            permutation_accuracy = float(permutation_values.mean())
            full_values = full_vocab_correct[bucket_mask, class_id].astype(np.float64)
            full_accuracy = float(full_values.mean())
            print(
                f"ACCURACY delay={delay} class={class_name} n={len(values)} "
                f"accuracy={accuracy:.6f} ci95=[{interval[0]:.6f},{interval[1]:.6f}] "
                f"permutation_null={permutation_accuracy:.6f} "
                f"permutation_ci95=[{permutation_interval[0]:.6f},"
                f"{permutation_interval[1]:.6f}] "
                f"full_vocab_accuracy={full_accuracy:.6f}",
                flush=True,
            )
            if check_cross_leak and delay >= 192 and interval[0] > 0.09:
                raise SystemExit(
                    f"LEAK: CROSS delay {delay} {class_name} "
                    f"CI lower bound exceeds 0.09"
                )
        paired = (
            correct[bucket_mask, 0].astype(np.float64)
            - correct[bucket_mask, 1].astype(np.float64)
        )
        paired_interval = bootstrap_interval(paired, rng)
        print(
            f"PAIRED delay={delay} salient_minus_ordinary={paired.mean():.6f} "
            f"ci95=[{paired_interval[0]:.6f},{paired_interval[1]:.6f}]",
            flush=True,
        )
        all_results[str(delay)] = {
            "SALIENT": class_accuracy["SALIENT"],
            "ORDINARY": class_accuracy["ORDINARY"],
            "SALIENT_no_context": float(
                no_context_correct[bucket_mask, 0].astype(np.float64).mean()
            ),
            "ORDINARY_no_context": float(
                no_context_correct[bucket_mask, 1].astype(np.float64).mean()
            ),
            "SALIENT_permutation_null": float(
                permutation_matches[bucket_mask, 0].astype(np.float64).mean()
            ),
            "ORDINARY_permutation_null": float(
                permutation_matches[bucket_mask, 1].astype(np.float64).mean()
            ),
            "SALIENT_full_vocab": float(
                full_vocab_correct[bucket_mask, 0].astype(np.float64).mean()
            ),
            "ORDINARY_full_vocab": float(
                full_vocab_correct[bucket_mask, 1].astype(np.float64).mean()
            ),
            "paired_difference": float(paired.mean()),
        }
    holm_pvalues = np.array(
        [record["pvalue"] for record in cell_records], dtype=np.float64
    )
    holm_adjusted, holm_flags = holm_correction(holm_pvalues)
    for record, adjusted, significant in zip(
        cell_records, holm_adjusted, holm_flags
    ):
        print(
            f"HOLM delay={record['delay']} class={record['class']} "
            f"pvalue={record['pvalue']:.6f} "
            f"holm_pvalue={adjusted:.6f} "
            f"significant={'YES' if significant else 'NO'}",
            flush=True,
        )
    pooled_no_context = no_context_correct.astype(np.float64).mean(axis=1)
    pooled_no_context_null = no_context_permutation_matches.astype(
        np.float64
    ).mean(axis=1)
    pooled_diff = pooled_no_context - pooled_no_context_null
    pooled_interval = bootstrap_interval(pooled_no_context, rng)
    pooled_null_interval = bootstrap_interval(pooled_no_context_null, rng)
    pooled_pvalue = paired_bootstrap_pvalue(pooled_diff, rng)
    pooled_flag = (
        "ABOVE_NULL_CI"
        if pooled_interval[0] > pooled_null_interval[1]
        else "WITHIN_NULL_CI"
    )
    print(
        f"NO_CONTEXT_POOLED accuracy={pooled_no_context.mean():.6f} "
        f"ci95=[{pooled_interval[0]:.6f},{pooled_interval[1]:.6f}] "
        f"permutation_null={pooled_no_context_null.mean():.6f} "
        f"permutation_ci95=[{pooled_null_interval[0]:.6f},"
        f"{pooled_null_interval[1]:.6f}] "
        f"pvalue={pooled_pvalue:.6f} flag={pooled_flag}",
        flush=True,
    )
    cross_rows = buckets >= 2
    cross_accuracy_by_row = correct[cross_rows].astype(np.float64).mean(axis=1)
    cross_permutation_by_row = permutation_matches[cross_rows].astype(
        np.float64
    ).mean(axis=1)
    cross_accuracy_ci = bootstrap_interval(cross_accuracy_by_row, rng)
    cross_permutation_ci = bootstrap_interval(cross_permutation_by_row, rng)
    print(
        f"CROSS_POOLED accuracy={cross_accuracy_by_row.mean():.6f} "
        f"ci95=[{cross_accuracy_ci[0]:.6f},{cross_accuracy_ci[1]:.6f}] "
        f"permutation_null={cross_permutation_by_row.mean():.6f} "
        f"permutation_ci95=[{cross_permutation_ci[0]:.6f},"
        f"{cross_permutation_ci[1]:.6f}]",
        flush=True,
    )
    for class_id, class_name in enumerate(class_names):
        print(
            f"CROSS_MATCH class={class_name} "
            f"same_row_other_event="
            f"{other_event_matches[cross_rows, class_id].mean():.6f} "
            f"permuted_other_row="
            f"{permutation_matches[cross_rows, class_id].mean():.6f}",
            flush=True,
        )
    print(
        "CROSS_MATCH pooled "
        f"same_row_other_event={other_event_matches[cross_rows].mean():.6f} "
        f"permuted_other_row={permutation_matches[cross_rows].mean():.6f}",
        flush=True,
    )
    print(f"EVAL_WALL_SECONDS={time.perf_counter() - started_at:.3f}", flush=True)
    print("RESULT_JSON=" + json.dumps(all_results), flush=True)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, default=600)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--n-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--eval-seed-base", type=int,
                        default=1_000_000_000,
                        help="seed base for the eval stream")
    parser.add_argument("--rda-decays", default=None)
    parser.add_argument("--allow-cross-above-0.09",
                        dest="allow_cross_above_0_09",
                        action="store_true",
                        help="do not apply the untrained-A cross-delay leak stop")
    args = parser.parse_args()
    if args.n_rows <= 0 or args.n_rows % len(DELAY_GRID):
        parser.error(f"--n-rows must be a positive multiple of {len(DELAY_GRID)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for recall evaluation")
    decays = parse_decays(args.rda_decays) if args.rda_decays else None
    evaluate(
        args.checkpoint_dir, args.step, args.seed, args.n_rows,
        args.device_batch_size, torch.device("cuda"), decay_override=decays,
        check_cross_leak=not args.allow_cross_above_0_09,
        eval_seed_base=args.eval_seed_base,
    )


if __name__ == "__main__":
    main()
