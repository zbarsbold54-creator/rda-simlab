"""Measure paired salient/ordinary RDA state elevation on recall eval rows."""

import argparse
import os
import time

os.environ["NANOCHAT_DTYPE"] = "float32"

import numpy as np
import torch

from diagnostics.recall_models import load_model, parse_decays
from nanochat.recall_data import build_batch, build_eval_grid


def bootstrap_ci(values, seed=43017, repetitions=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[indices].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975])


def paired_cohens_d(differences):
    differences = np.asarray(differences, dtype=np.float64)
    standard_deviation = differences.std(ddof=1)
    if standard_deviation == 0:
        if differences.mean() == 0:
            return 0.0
        return float(np.copysign(np.inf, differences.mean()))
    return float(differences.mean() / standard_deviation)


@torch.inference_mode()
def evaluate(checkpoint_dir, step, seed, n_rows, batch_size, decays):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the RDA manipulation check")
    started = time.perf_counter()
    device = torch.device("cuda")
    model, _, _ = load_model(
        checkpoint_dir,
        step,
        device,
        seed,
        segment_len=128,
        decay_override=decays,
    )
    class_index = {"SALIENT": 0, "ORDINARY": 1}
    key_states = np.zeros((n_rows, 2, 8), dtype=np.float64)
    offsets = (-16, -8, 0, 8, 16, 32, 64)
    window_sums = np.zeros((len(offsets), 2), dtype=np.float64)
    window_counts = np.zeros((len(offsets), 2), dtype=np.int64)
    rows = build_eval_grid(n_rows=n_rows)
    original_modulate = model.state.modulate

    def capture_state(hidden, state):
        captured["state"] = state.detach().float()
        return original_modulate(hidden, state)

    model.state.modulate = capture_state
    for start in range(0, n_rows, batch_size):
        subset = rows[start:start + batch_size]
        inputs, targets, _, metadata = build_batch(subset, device=device)
        captured = {}
        model(inputs, targets)
        state = captured["state"]
        for row_offset, row_meta in enumerate(metadata):
            row_index = start + row_offset
            for event in row_meta["events"]:
                cls = class_index[event["class"]]
                key_position = event["event_key_position"]
                key_states[row_index, cls] = (
                    state[row_offset, key_position].cpu().numpy()
                )
                for offset_index, offset in enumerate(offsets):
                    position = key_position + offset
                    if 0 <= position < state.size(1):
                        window_sums[offset_index, cls] += float(
                            state[row_offset, position].norm().item()
                        )
                        window_counts[offset_index, cls] += 1
        del state, inputs, targets, captured
    del model
    torch.cuda.empty_cache()

    norm_by_class = np.linalg.norm(key_states, axis=2)
    paired_norm_diff = norm_by_class[:, 0] - norm_by_class[:, 1]
    d_value = paired_cohens_d(paired_norm_diff)
    bootstrap_ds = np.empty(5000, dtype=np.float64)
    rng = np.random.default_rng(43018)
    for index in range(len(bootstrap_ds)):
        sample = paired_norm_diff[
            rng.integers(0, n_rows, size=n_rows)
        ]
        bootstrap_ds[index] = paired_cohens_d(sample)
    d_ci = np.quantile(bootstrap_ds, [0.025, 0.975])
    print(
        f"MANIPULATION paired_cohens_d={d_value:.6f} "
        f"ci95=[{d_ci[0]:.6f},{d_ci[1]:.6f}] "
        f"mean_norm_salient={norm_by_class[:, 0].mean():.6f} "
        f"mean_norm_ordinary={norm_by_class[:, 1].mean():.6f} "
        f"verdict={'PASS' if d_value >= 0.5 and d_ci[0] > 0 else 'FAIL'}",
        flush=True,
    )
    differences = key_states[:, 0, :] - key_states[:, 1, :]
    for drive in range(8):
        print(
            f"DRIVE {drive} paired_cohens_d="
            f"{paired_cohens_d(differences[:, drive]):.6f} "
            f"mean_difference={differences[:, drive].mean():.6f}",
            flush=True,
        )
    for offset_index, offset in enumerate(offsets):
        means = window_sums[offset_index] / window_counts[offset_index]
        print(
            f"WINDOW offset={offset} n_salient={window_counts[offset_index, 0]} "
            f"mean_norm_salient={means[0]:.6f} "
            f"n_ordinary={window_counts[offset_index, 1]} "
            f"mean_norm_ordinary={means[1]:.6f}",
            flush=True,
        )
    print(f"MANIP_CHECK_WALL_SECONDS={time.perf_counter() - started:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, default=1200)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--n-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rda-decays", default=None)
    args = parser.parse_args()
    if args.n_rows <= 0 or args.n_rows % 8:
        parser.error("--n-rows must be a positive multiple of 8")
    decays = parse_decays(args.rda_decays) if args.rda_decays else None
    evaluate(
        args.checkpoint_dir,
        args.step,
        args.seed,
        args.n_rows,
        args.device_batch_size,
        decays,
    )


if __name__ == "__main__":
    main()
