"""Run the pre-registered paired-state manipulation check and constant-state harness."""

import argparse
import os
import time

os.environ["NANOCHAT_DTYPE"] = "float32"

import numpy as np
import torch

from diagnostics.recall_models import load_model
from nanochat.recall_data import build_batch, build_eval_grid


N_ROWS = 2048
TRAIN_SEED = 43
EVAL_SEED_BASE = 1_000_000_000
SEGMENT_LEN = 128
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261011


def paired_cohens_d(differences):
    differences = np.asarray(differences, dtype=np.float64)
    standard_deviation = differences.std(ddof=1)
    if standard_deviation == 0:
        if differences.mean() == 0:
            return 0.0
        return float(np.copysign(np.inf, differences.mean()))
    return float(differences.mean() / standard_deviation)


def bootstrap_d_ci(differences):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPS):
        sample = differences[rng.integers(0, len(differences), size=len(differences))]
        estimates[index] = paired_cohens_d(sample)
    return np.quantile(estimates, [0.025, 0.975])


@torch.inference_mode()
def measure_checkpoint(checkpoint_dir, step, batch_size, expected_const_state):
    device = torch.device("cuda")
    model, meta, _ = load_model(
        checkpoint_dir,
        step,
        device,
        TRAIN_SEED,
        segment_len=SEGMENT_LEN,
    )
    has_const_state = model.const_state_vec is not None
    if has_const_state != expected_const_state:
        raise RuntimeError(
            f"checkpoint constant-state status mismatch for {checkpoint_dir}"
        )
    rows = build_eval_grid(
        n_rows=N_ROWS,
        seed_base=EVAL_SEED_BASE,
        segment_len=SEGMENT_LEN,
    )
    key_states = np.full((N_ROWS, 2, 8), np.nan, dtype=np.float64)
    event_delays = np.full((N_ROWS, 2), -1, dtype=np.int64)
    class_index = {"SALIENT": 0, "ORDINARY": 1}
    original_modulate = model.state.modulate

    def capture_state(hidden, state):
        captured["state"] = state.detach().float().clone()
        return original_modulate(hidden, state)

    model.state.modulate = capture_state
    for start in range(0, N_ROWS, batch_size):
        subset = rows[start:start + batch_size]
        inputs, targets, _, metadata = build_batch(subset, device=device)
        captured = {}
        model(inputs, targets)
        state = captured.get("state")
        if state is None or state.ndim != 3 or state.shape[-1] != 8:
            raise RuntimeError("RDA state capture did not produce [batch, time, 8]")
        for row_offset, row_meta in enumerate(metadata):
            row_index = start + row_offset
            if len(row_meta["events"]) != 2:
                raise RuntimeError(f"row {row_index} does not contain two events")
            for event in row_meta["events"]:
                cls = class_index[event["class"]]
                key_position = event["event_key_position"]
                key_states[row_index, cls] = (
                    state[row_offset, key_position].cpu().numpy()
                )
                event_delays[row_index, cls] = event["delay"]
        del state, inputs, targets, captured
    del model
    torch.cuda.empty_cache()

    if not np.isfinite(key_states).all() or (event_delays < 0).any():
        raise RuntimeError("evaluation grid did not yield a complete paired measurement")

    norms = np.linalg.norm(key_states, axis=2)
    norm_differences = norms[:, 0] - norms[:, 1]
    drive_differences = key_states[:, 0, :] - key_states[:, 1, :]
    return {
        "meta": meta,
        "has_const_state": has_const_state,
        "key_states": key_states,
        "norms": norms,
        "norm_differences": norm_differences,
        "drive_differences": drive_differences,
        "event_delays": event_delays,
    }


def report_rows(label, result):
    for row_index in range(N_ROWS):
        drive_values = ",".join(
            f"{value:.8g}" for value in result["drive_differences"][row_index]
        )
        print(
            f"ROW arm={label} index={row_index} "
            f"delay_salient={result['event_delays'][row_index, 0]} "
            f"delay_ordinary={result['event_delays'][row_index, 1]} "
            f"norm_salient={result['norms'][row_index, 0]:.8g} "
            f"norm_ordinary={result['norms'][row_index, 1]:.8g} "
            f"norm_difference={result['norm_differences'][row_index]:.8g} "
            f"drive_differences={drive_values}",
            flush=True,
        )


def report_arm(label, result):
    d_value = paired_cohens_d(result["norm_differences"])
    ci = bootstrap_d_ci(result["norm_differences"])
    print(
        f"MANIPULATION arm={label} paired_cohens_d={d_value:.6f} "
        f"ci95=[{ci[0]:.6f},{ci[1]:.6f}] "
        f"mean_norm_salient={result['norms'][:, 0].mean():.6f} "
        f"mean_norm_ordinary={result['norms'][:, 1].mean():.6f}",
        flush=True,
    )
    for drive in range(8):
        differences = result["drive_differences"][:, drive]
        print(
            f"DRIVE arm={label} index={drive} "
            f"paired_cohens_d={paired_cohens_d(differences):.6f} "
            f"mean_difference={differences.mean():.6f}",
            flush=True,
        )
    return d_value, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-checkpoint-dir", required=True)
    parser.add_argument("--dconst-checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, default=1200)
    parser.add_argument("--device-batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the RDA manipulation check")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(
        f"SETTINGS n_rows={N_ROWS} training_seed={TRAIN_SEED} "
        f"eval_seed_base={EVAL_SEED_BASE} segment_len={SEGMENT_LEN} "
        f"bootstrap_reps={BOOTSTRAP_REPS} bootstrap_seed={BOOTSTRAP_SEED}",
        flush=True,
    )

    d_result = measure_checkpoint(
        args.d_checkpoint_dir, args.step, args.device_batch_size,
        expected_const_state=False,
    )
    dconst_result = measure_checkpoint(
        args.dconst_checkpoint_dir, args.step, args.device_batch_size,
        expected_const_state=True,
    )
    dconst_max_abs = float(np.max(np.abs(dconst_result["norm_differences"])))
    print(f"HARNESS_DCONST_MAX_ABS_NORM_DIFFERENCE={dconst_max_abs:.12g}", flush=True)
    if dconst_max_abs > 1e-6:
        print("HARNESS=FAIL; D verdict withheld", flush=True)
        print(
            f"PEAK_CUDA_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()} "
            f"PEAK_CUDA_RESERVED_BYTES={torch.cuda.max_memory_reserved()}",
            flush=True,
        )
        print(f"WALL_SECONDS={time.perf_counter() - started:.3f}", flush=True)
        raise SystemExit(2)
    print("HARNESS=PASS", flush=True)

    report_rows("D_const", dconst_result)
    report_arm("D_const", dconst_result)
    report_rows("D", d_result)
    d_value, ci = report_arm("D", d_result)
    passed = d_value >= 0.5 and ci[0] > 0.0
    print(
        f"VERDICT_D={'PASS' if passed else 'FAIL'} "
        f"criterion=d>=0.5_and_ci95_excludes_0",
        flush=True,
    )
    print(
        f"PEAK_CUDA_ALLOCATED_BYTES={torch.cuda.max_memory_allocated()} "
        f"PEAK_CUDA_RESERVED_BYTES={torch.cuda.max_memory_reserved()}",
        flush=True,
    )
    print(f"WALL_SECONDS={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
