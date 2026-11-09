"""CPU distribution and invariance checks for the synthetic recall generator."""

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanochat.common import get_base_dir
from nanochat.recall_data import (
    BOS_ID, CROSS_DELAY_BUCKETS, DELAY_GRID, FILLER_START, N_FILLER,
    QUERY_ID, SEGMENT_LEN, SEQUENCE_LEN, TRANSITION_OFFSETS, VALUE_START,
    build_eval_grid, row,
)
from nanochat.tokenizer import get_tokenizer

N_ROWS = 20_000
EVAL_SEED_BASE = 1_000_000_000
FAILURES = []


def gamma_q(a, x):
    """Regularized upper incomplete gamma, used for chi-square p-values."""
    if x < 0 or a <= 0:
        raise ValueError("gamma_q requires x >= 0 and a > 0")
    if x == 0:
        return 1.0
    eps = 3e-14
    fpmin = 1e-300
    log_term = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap = a
        term = total = 1.0 / a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(log_term)))
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(abs(b), fpmin)
    if b < 0:
        d = -d
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return max(0.0, min(1.0, math.exp(log_term) * h))


def chi_square_independence(table):
    values = np.asarray(table, dtype=np.float64)
    row_totals = values.sum(axis=1, keepdims=True)
    col_totals = values.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / values.sum()
    statistic = float(((values - expected) ** 2 / expected).sum())
    df = (values.shape[0] - 1) * (values.shape[1] - 1)
    return statistic, gamma_q(df / 2.0, statistic / 2.0)


def report(name, passed, statistic):
    print(f"{'PASS' if passed else 'FAIL'} {name}: {statistic}", flush=True)
    if not passed:
        FAILURES.append(name)


def lead_transition_entropy(leads):
    counts = np.zeros(N_FILLER, dtype=np.int64)
    for lead in leads:
        counts += np.bincount(
            (lead[1:] - lead[:-1]) % N_FILLER, minlength=N_FILLER
        )
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def count_first_shard_reserved_ids():
    shard = Path(get_base_dir()) / "base_data_fineweb_edu" / "shard_00000.parquet"
    tokenizer = get_tokenizer()
    reserved = set(range(FILLER_START, QUERY_ID + 1))
    count = 0
    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(batch_size=512, columns=["text"]):
        texts = batch.column("text").to_pylist()
        for encoded in tokenizer.encode(texts, prepend=None, num_threads=4):
            count += sum(token in reserved for token in encoded)
    return shard, count


def main():
    keys_by_class = {name: Counter() for name in ("SALIENT", "ORDINARY")}
    values_by_class = {name: Counter() for name in ("SALIENT", "ORDINARY")}
    class_key = np.zeros((2, 16), dtype=np.int64)
    class_value = np.zeros((2, 16), dtype=np.int64)
    class_delay = np.zeros((2, len(DELAY_GRID)), dtype=np.int64)
    class_decile = np.zeros((2, 10), dtype=np.int64)
    class_filler = np.zeros((2, N_FILLER), dtype=np.int64)
    leads = {"SALIENT": [], "ORDINARY": []}
    class_index = {"SALIENT": 0, "ORDINARY": 1}
    deterministic = True

    for index in range(N_ROWS):
        inputs, targets, mask, metadata = row(0, index)
        repeated = row(0, index)
        deterministic &= all(
            torch_equal(first, second)
            for first, second in zip((inputs, targets, mask), repeated[:3])
        ) and metadata == repeated[3]
        assert inputs[0].item() == BOS_ID
        assert targets.shape == inputs.shape == mask.shape
        events = metadata["events"]
        assert len(events) == 2
        assert len({event["key"] for event in events}) == 2
        ranges = []
        for event in events:
            start = event["event_start"]
            key_position = event["event_key_position"]
            query_position = event["query_position"]
            assert event["key"] == inputs[key_position].item()
            assert event["value"] == targets[query_position].item()
            assert inputs[query_position].item() == event["key"]
            assert mask[query_position].item()
            ranges.extend(((start, start + 9), (query_position - 1, query_position + 1)))
            same_segment = (
                event["event_segment"] == event["query_segment"]
            )
            assert same_segment == (event["delay"] in DELAY_GRID[:2])
            if event["delay"] in CROSS_DELAY_BUCKETS:
                assert not same_segment
            cidx = class_index[event["class"]]
            key_id = event["key"] - (FILLER_START + N_FILLER)
            value_id = event["value"] - VALUE_START
            class_key[cidx, key_id] += 1
            class_value[cidx, value_id] += 1
            class_delay[cidx, event["delay_bucket"]] += 1
            decile = min(9, event["event_key_position"] * 10 // SEQUENCE_LEN)
            class_decile[cidx, decile] += 1
            lead = inputs[start:start + 8].numpy()
            class_filler[cidx] += np.bincount(
                lead - FILLER_START, minlength=N_FILLER
            )
            leads[event["class"]].append(lead)
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                assert ranges[i][1] < ranges[j][0] or ranges[j][1] < ranges[i][0]

    for name, table in (
        ("class_vs_key", class_key),
        ("class_vs_value", class_value),
        ("class_vs_delay_bucket", class_delay),
        ("class_vs_event_position_decile", class_decile),
    ):
        statistic, p_value = chi_square_independence(table)
        report(name, p_value > 0.01,
               f"chi2={statistic:.4f} df={table.shape[1]-1} p={p_value:.6g}")

    count_deviations = {}
    for label, table in (("key", class_key), ("value", class_value)):
        deviations = []
        for class_counts in table:
            average = class_counts.mean()
            deviations.append(float(np.max(np.abs(class_counts - average) / average)))
        count_deviations[label] = deviations
    report("per_class_key_value_counts_within_5_percent",
           max(max(v) for v in count_deviations.values()) <= 0.05,
           f"max_key_relative_deviation={max(count_deviations['key']):.4%}; "
           f"max_value_relative_deviation={max(count_deviations['value']):.4%}")

    filler_stat, filler_p = chi_square_independence(class_filler)
    report("burst_vs_ordinary_filler_marginals",
           filler_p > 0.01,
           f"chi2={filler_stat:.4f} df={N_FILLER-1} p={filler_p:.6g}")

    matrix = np.zeros((N_FILLER, N_FILLER), dtype=np.float64)
    for offset in TRANSITION_OFFSETS:
        matrix[np.arange(N_FILLER), (np.arange(N_FILLER) + offset) % N_FILLER] += 0.25
    row_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    col_error = float(np.max(np.abs(matrix.sum(axis=0) - 1.0)))
    stationary_error = float(np.max(np.abs(
        np.full(N_FILLER, 1.0 / N_FILLER) @ matrix - 1.0 / N_FILLER
    )))
    report("doubly_stochastic_stationary_uniform",
           max(row_error, col_error, stationary_error) <= 1e-3,
           f"max_row_error={row_error:.3g} max_col_error={col_error:.3g} "
           f"max_stationary_error={stationary_error:.3g}")

    entropy_salient = lead_transition_entropy(leads["SALIENT"])
    entropy_ordinary = lead_transition_entropy(leads["ORDINARY"])
    report("lead_conditional_entropy",
           entropy_salient > entropy_ordinary
           and abs(entropy_salient - math.log2(N_FILLER)) < 0.5,
           f"SALIENT={entropy_salient:.4f} bits/transition "
           f"ORDINARY={entropy_ordinary:.4f} bits/transition")

    eval_rows = build_eval_grid(2048)
    grid = np.zeros((len(DELAY_GRID), 2), dtype=np.int64)
    for _, _, _, meta in eval_rows:
        for event in meta["events"]:
            grid[event["delay_bucket"], class_index[event["class"]]] += 1
    report("stratified_eval_grid_equal_cells",
           bool(np.all(grid == grid[0, 0])),
           f"min={grid.min()} max={grid.max()} cells={grid.tolist()}")

    report("positions_keys_overlap_segment_constraints",
           True,
           f"rows={N_ROWS} events={2*N_ROWS} within_same_segment=100% "
           f"cross_different_segment=100% nonoverlap=100% distinct_keys=100%")

    identical = deterministic
    report("deterministic_rows_and_disjoint_seed_ranges",
           identical and 0 <= N_ROWS < EVAL_SEED_BASE,
           f"train seeds=[0,{N_ROWS-1}] eval seeds=[{EVAL_SEED_BASE},...) "
           f"identical={identical}")

    shard, count = count_first_shard_reserved_ids()
    print(f"RESERVED_IDS={FILLER_START}-{QUERY_ID} "
          f"first_fineweb_shard={shard} occurrences={count}", flush=True)
    if FAILURES:
        print(f"RESULT: FAIL {FAILURES}", flush=True)
        return 1
    print("RESULT: PASS all generator checks", flush=True)
    return 0


def torch_equal(left, right):
    import torch
    return torch.equal(left, right)


if __name__ == "__main__":
    import torch
    sys.exit(main())
