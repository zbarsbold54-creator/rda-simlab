"""Deterministic synthetic delayed-recall rows for the RDA benchmark."""

from functools import lru_cache

import numpy as np
import torch

SEQUENCE_LEN = 1024
SEGMENT_LEN = 128
N_FILLER = 512
N_KEYS = 16
N_VALUES = 16
FILLER_START = 32200
KEY_START = FILLER_START + N_FILLER
VALUE_START = KEY_START + N_KEYS
QUERY_ID = VALUE_START + N_VALUES
BOS_ID = 32759
DELAY_GRID = (16, 48, 192, 320, 448, 576, 704, 832)
CROSS_DELAY_BUCKETS = (2, 3, 4, 5, 6, 7)
TRANSITION_OFFSETS = (1, 13, 47, 173)
KEY_SALT = 211
VALUE_SALT = 419


def _rng(seed, index, salt=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index), salt]))


@lru_cache(maxsize=None)
def _valid_event_starts(delay, segment_len):
    candidates = []
    for start in range(1, SEQUENCE_LEN - 9):
        event_key = start + 8
        query_key = event_key + int(delay)
        if query_key > SEQUENCE_LEN - 1 or query_key - 1 < 1:
            continue
        same_segment = event_key // segment_len == query_key // segment_len
        if (delay in DELAY_GRID[:2]) != same_segment:
            continue
        if any(start <= pos <= start + 9
               for pos in (query_key - 1, query_key, query_key + 1)):
            continue
        candidates.append(start)
    return np.asarray(candidates, dtype=np.int32)


def _overlap(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def _placements_do_not_overlap(start_a, delay_a, start_b, delay_b):
    blocks = []
    for start, delay in ((start_a, delay_a), (start_b, delay_b)):
        key_pos = start + 8
        query_key = key_pos + delay
        blocks.extend(((start, start + 9), (query_key - 1, query_key + 1)))
    return all(
        not _overlap(*blocks[i], *blocks[j])
        for i in range(4) for j in range(i + 1, 4)
    )


def _sample_placements(rng, delays, segment_len):
    candidates = [_valid_event_starts(delay, segment_len) for delay in delays]
    if any(len(values) == 0 for values in candidates):
        raise ValueError(f"No feasible event positions for delays {delays}")
    for _ in range(10000):
        first = int(rng.choice(candidates[0]))
        feasible_second = np.asarray([
            int(value) for value in candidates[1]
            if _placements_do_not_overlap(first, delays[0], int(value), delays[1])
        ], dtype=np.int32)
        if len(feasible_second):
            return first, int(rng.choice(feasible_second))
    raise RuntimeError(f"Could not place non-overlapping events for delays {delays}")


def row(seed, index, *, delay_bucket=None, segment_len=SEGMENT_LEN, bos_id=BOS_ID):
    """Build one row as (inputs, next-token targets, answer mask, metadata).

    Inputs and targets are one sequence window long. Each answer mask position
    selects the value token predicted from the corresponding query-key input.
    """
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    if delay_bucket is None:
        rng = _rng(seed, index)
        delays = [DELAY_GRID[int(rng.integers(len(DELAY_GRID)))] for _ in range(2)]
    else:
        if not 0 <= int(delay_bucket) < len(DELAY_GRID):
            raise ValueError("delay_bucket must index DELAY_GRID")
        delays = [DELAY_GRID[int(delay_bucket)]] * 2
        rng = _rng(seed, index, salt=100 + int(delay_bucket))

    starts = _sample_placements(rng, delays, segment_len)
    class_order = rng.permutation(2)
    class_by_slot = ["ORDINARY", "ORDINARY"]
    class_by_slot[int(class_order[0])] = "SALIENT"
    key_symbols = _rng(seed, index, KEY_SALT).choice(
        N_KEYS, size=2, replace=False
    )
    value_symbols = _rng(seed, index, VALUE_SALT).integers(
        N_VALUES, size=2
    )

    filler_offsets = np.asarray(TRANSITION_OFFSETS, dtype=np.int32)
    chain_choices = rng.integers(len(filler_offsets), size=SEQUENCE_LEN)
    filler = (int(rng.integers(N_FILLER))
              + np.cumsum(np.r_[0, filler_offsets[chain_choices[:-1]]])) % N_FILLER
    tokens = np.empty(SEQUENCE_LEN + 1, dtype=np.int64)
    tokens[0] = int(bos_id)
    tokens[1:] = FILLER_START + filler
    events = []

    for slot, (start, delay) in enumerate(zip(starts, delays)):
        event_key = start + 8
        query_key = event_key + delay
        event_class = class_by_slot[slot]
        if event_class == "SALIENT":
            tokens[start:start + 8] = (
                FILLER_START + rng.integers(N_FILLER, size=8)
            )
        key_id = KEY_START + int(key_symbols[slot])
        value_id = VALUE_START + int(value_symbols[slot])
        tokens[event_key] = key_id
        tokens[event_key + 1] = value_id
        tokens[query_key - 1] = QUERY_ID
        tokens[query_key] = key_id
        tokens[query_key + 1] = value_id
        events.append({
            "class": event_class,
            "event_start": int(start),
            "event_key_position": int(event_key),
            "query_position": int(query_key),
            "delay": int(delay),
            "delay_bucket": int(DELAY_GRID.index(delay)),
            "key": int(key_id),
            "value": int(value_id),
            "event_segment": int(event_key // segment_len),
            "query_segment": int(query_key // segment_len),
        })

    inputs = torch.from_numpy(tokens[:SEQUENCE_LEN].copy()).long()
    targets = torch.from_numpy(tokens[1:SEQUENCE_LEN + 1].copy()).long()
    loss_mask = torch.zeros(SEQUENCE_LEN, dtype=torch.bool)
    for event in events:
        loss_mask[event["query_position"]] = True
    return inputs, targets, loss_mask, {
        "seed": int(seed),
        "index": int(index),
        "events": events,
    }


def build_eval_grid(n_rows=2048, seed_base=1_000_000_000,
                    segment_len=SEGMENT_LEN, bos_id=BOS_ID):
    """Build delay/class-balanced eval rows with randomized value assignment."""
    rows_per_bucket = n_rows // len(DELAY_GRID)
    if (n_rows <= 0 or n_rows % len(DELAY_GRID)
            or rows_per_bucket % N_VALUES):
        raise ValueError(
            "n_rows must give a multiple of 16 rows per delay bucket"
        )
    rows = []
    for index in range(n_rows):
        bucket = index // rows_per_bucket
        rows.append(row(seed_base, index, delay_bucket=bucket,
                        segment_len=segment_len, bos_id=bos_id))

    for bucket in range(len(DELAY_GRID)):
        first = bucket * rows_per_bucket
        bucket_rows = rows[first:first + rows_per_bucket]
        value_rng = _rng(seed_base, bucket, VALUE_SALT)
        for class_name in ("SALIENT", "ORDINARY"):
            events = [
                (item, event)
                for item in bucket_rows
                for event in item[3]["events"]
                if event["class"] == class_name
            ]
            values = np.tile(
                np.arange(N_VALUES, dtype=np.int64),
                len(events) // N_VALUES,
            )
            value_rng.shuffle(values)
            for (item, event), value_symbol in zip(events, values):
                value_id = VALUE_START + int(value_symbol)
                event_key = event["event_key_position"]
                query_key = event["query_position"]
                item[0][event_key + 1] = value_id
                item[1][event_key] = value_id
                if query_key + 1 < item[0].numel():
                    item[0][query_key + 1] = value_id
                item[1][query_key] = value_id
                event["value"] = value_id
    return rows


def build_batch(rows, device=None):
    """Stack generated rows into input/target/loss-mask tensors and metadata."""
    inputs = torch.stack([item[0] for item in rows])
    targets = torch.stack([item[1] for item in rows])
    loss_mask = torch.stack([item[2] for item in rows])
    if device is not None:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)
    return inputs, targets, loss_mask, [item[3] for item in rows]
