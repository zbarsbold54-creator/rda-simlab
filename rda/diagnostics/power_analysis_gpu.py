"""CUDA power analysis matching diagnostics.power_analysis."""

import argparse
import time

import torch

BASE_ACCURACY = 0.0638020833333333
SIMULATIONS = 2000
BOOTSTRAP_REPS = 5000
DEFAULT_ROWS = "2048,0.03,1;2048,0.03,3;2048,0.05,1"
DEFAULT_OUTPUT = (
    "results/recall_benchmark/calib/power_analysis_gpu.txt"
)
RNG_SEED = 20261011
SIMULATION_CHUNK = 32


def parse_rows(value):
    rows = []
    for item in value.split(";"):
        n_rows, delta, n_seeds = item.split(",")
        rows.append((int(n_rows), float(delta), int(n_seeds)))
    return rows


def simulate_cell(n_rows, delta, n_seeds, generator):
    row_count = n_rows * n_seeds
    significant = 0
    started = time.perf_counter()
    remaining = SIMULATIONS
    while remaining:
        batch_size = min(SIMULATION_CHUNK, remaining)
        salient = (
            torch.rand(
                (batch_size, row_count),
                device="cuda",
                generator=generator,
            )
            < BASE_ACCURACY + delta
        )
        ordinary = (
            torch.rand(
                (batch_size, row_count),
                device="cuda",
                generator=generator,
            )
            < BASE_ACCURACY
        )
        positive_count = (salient & ~ordinary).sum(dim=1).float()
        negative_count = (~salient & ordinary).sum(dim=1).float()
        p_positive = positive_count / row_count
        p_negative = negative_count / row_count
        bootstrap_shape = (BOOTSTRAP_REPS, batch_size)
        total_draws = torch.full(
            bootstrap_shape, row_count, device="cuda", dtype=torch.float32
        )
        plus_probs = p_positive.unsqueeze(0).expand(bootstrap_shape)
        bootstrap_positive = torch.binomial(
            total_draws, plus_probs, generator=generator
        )
        remaining_draws = total_draws - bootstrap_positive
        denominator = (1.0 - p_positive).clamp_min(1e-12)
        conditional_negative = (p_negative / denominator).clamp(0.0, 1.0)
        negative_probs = conditional_negative.unsqueeze(0).expand(
            bootstrap_shape
        )
        bootstrap_negative = torch.binomial(
            remaining_draws, negative_probs, generator=generator
        )
        bootstrap_means = (
            bootstrap_positive - bootstrap_negative
        ) / row_count
        lower = torch.quantile(bootstrap_means, 0.025, dim=0)
        significant += int((lower > 0.0).sum().item())
        remaining -= batch_size
    torch.cuda.synchronize()
    return significant / SIMULATIONS, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=DEFAULT_ROWS,
                        help="semicolon-separated N,delta,seeds triples")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    print(
        f"POWER_GPU base_accuracy={BASE_ACCURACY} simulations={SIMULATIONS} "
        f"bootstrap_reps={BOOTSTRAP_REPS} seed={args.seed}",
        flush=True,
    )
    with open(args.output, "a", encoding="utf-8") as output:
        output.write(
            f"POWER_GPU base_accuracy={BASE_ACCURACY} "
            f"simulations={SIMULATIONS} bootstrap_reps={BOOTSTRAP_REPS} "
            f"seed={args.seed}\n"
        )
        output.flush()
        for n_rows, delta, n_seeds in parse_rows(args.rows):
            power, elapsed = simulate_cell(
                n_rows, delta, n_seeds, generator
            )
            line = (
                f"POWER N={n_rows} delta={delta} seeds={n_seeds} "
                f"power={power:.4f}"
            )
            print(line, flush=True)
            print(
                f"ROW_WALL_SECONDS N={n_rows} delta={delta} "
                f"seeds={n_seeds} seconds={elapsed:.3f}",
                flush=True,
            )
            output.write(line + "\n")
            output.flush()


if __name__ == "__main__":
    main()
