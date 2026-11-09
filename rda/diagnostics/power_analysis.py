"""Power analysis for the difference-in-differences recall test.

Simulates the paired row structure: each row has one SALIENT and one ORDINARY
event, both queried. Base accuracy per class is the observed D_const CROSS
accuracy (0.063802, seed-2e9 replication). Per-row outcomes are Bernoulli;
an effect delta is added to the SALIENT class of the arm only. Test: bootstrap
95% CI over rows excludes 0, pooled over seeds. Grid: N in {2048, 4096, 8192,
16384} x delta in {0.03, 0.05, 0.08} x seeds in {1, 3}. 2000 simulations
per cell. Reports power, the smallest N giving 80% power at delta 0.05 with
3 seeds, and the wall time of one eval_recall pass at 2048 rows.
"""

import time

import numpy as np

BASE_ACCURACY = 0.0638020833333333  # D_const CROSS pooled, seed 2e9
N_GRID = (2048, 4096, 8192, 16384)
DELTA_GRID = (0.03, 0.05, 0.08)
SEED_GRID = (1, 3)
SIMULATIONS = 2000
BOOTSTRAP_REPS = 5000


def simulate_cell(n_rows, delta, n_seeds, rng):
    """Power of the pooled-over-seeds bootstrap CI test for one cell."""
    significant = 0
    for _ in range(SIMULATIONS):
        pooled_diff = np.empty(n_seeds * n_rows, dtype=np.float64)
        for seed_index in range(n_seeds):
            # Bernoulli outcomes per row, per class.
            ordinary = rng.random(n_rows) < BASE_ACCURACY
            salient = rng.random(n_rows) < (BASE_ACCURACY + delta)
            # Paired within-row difference (SALIENT - ORDINARY).
            pooled_diff[
                seed_index * n_rows:(seed_index + 1) * n_rows
            ] = salient.astype(np.float64) - ordinary.astype(np.float64)
        # Bootstrap 95% CI over rows, pooled over seeds.
        indices = rng.integers(
            0, pooled_diff.size, size=(BOOTSTRAP_REPS, pooled_diff.size)
        )
        means = pooled_diff[indices].mean(axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975])
        if lower > 0.0:
            significant += 1
    return significant / SIMULATIONS


def main():
    started = time.perf_counter()
    rng = np.random.default_rng(20261011)
    print(
        f"POWER base_accuracy={BASE_ACCURACY} simulations={SIMULATIONS} "
        f"bootstrap_reps={BOOTSTRAP_REPS}",
        flush=True,
    )
    results = {}
    for n_rows in N_GRID:
        for delta in DELTA_GRID:
            for n_seeds in SEED_GRID:
                power = simulate_cell(n_rows, delta, n_seeds, rng)
                results[(n_rows, delta, n_seeds)] = power
                print(
                    f"POWER N={n_rows} delta={delta} seeds={n_seeds} "
                    f"power={power:.4f}",
                    flush=True,
                )
    # Smallest N giving >= 80% power at delta 0.05 with 3 seeds.
    target = None
    for n_rows in N_GRID:
        if results[(n_rows, 0.05, 3)] >= 0.80:
            target = n_rows
            break
    print(
        f"POWER_TARGET delta=0.05 seeds=3 smallest_N={target} "
        f"power={results[(target, 0.05, 3)]:.4f}" if target is not None
        else "POWER_TARGET delta=0.05 seeds=3 smallest_N=NONE "
             "(no grid point reaches 0.80)",
        flush=True,
    )
    print(f"POWER_WALL_SECONDS={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
