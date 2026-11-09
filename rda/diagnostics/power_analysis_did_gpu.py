"""CUDA power analysis for the planned contrast: D vs D_const
difference in the SALIENT-minus-ORDINARY gap, with training-seed
variance included.

Model: for each arm (D, D_const), K independent training seeds. Each
seed's true SALIENT-minus-ORDINARY gap = arm gap + N(0, sigma_seed^2).
Each seed's eval gap = that value + paired-row sampling noise with N
paired rows (pooled CROSS, base accuracy 0.0638). Statistic: mean over
seeds of (gap_D - gap_Dconst) paired by seed index.

Tests:
 (a) one-sided exact sign-flip test over seeds (all K sign assignments);
 (b) threshold rule: estimated difference exceeds the largest seed
     spread of the gap within either arm;
 (c) one-sided label-permutation test across all 2K per-seed eval
     gaps (exhaustive: 20 arrangements for K=3, 252 for K=5).

Grid: K in {3, 5}, N in {1536, 4096, 8192}, true difference-in-
differences in {0.0, 0.01, 0.02, 0.03, 0.05}, sigma_seed in
{0.005, 0.01, 0.02}. Bootstrap is not used.
"""

import argparse
import itertools
import math
import time
from pathlib import Path

import torch

BASE_ACCURACY = 0.0638020833333333  # D_const CROSS pooled
SIMULATIONS = 2000
K_GRID = (3, 5)
N_GRID = (1536, 4096, 8192)
DELTA_GRID = (0.0, 0.01, 0.02, 0.03, 0.05)
SIGMA_GRID = (0.005, 0.01, 0.02)
DEFAULT_OUTPUT = (
    "results/recall_benchmark/calib/power_analysis_did_gpu_v5.txt"
)
RNG_SEED = 20261011
SIMULATION_CHUNK = 2000
SELFCHECK_SIMULATIONS = 20000
SELFCHECK_CHUNK = 2000


def sign_flip_p(per_seed_diff):
    """Exact one-sided sign-flip p-values; accepts (K,) or (batch, K)."""
    was_1d = per_seed_diff.ndim == 1
    differences = per_seed_diff.unsqueeze(0) if was_1d else per_seed_diff
    differences = differences.to(torch.float64)
    k_seeds = differences.shape[1]
    signs = torch.tensor(
        [
            [1.0 if (assignment >> bit) & 1 else -1.0
             for bit in range(k_seeds)]
            for assignment in range(2 ** k_seeds)
        ],
        device=differences.device,
        dtype=torch.float64,
    )
    null_sums = differences @ signs.T
    observed_sums = differences.sum(dim=1, keepdim=True)
    p_values = (null_sums >= observed_sums).to(torch.float32).mean(dim=1)
    return p_values[0] if was_1d else p_values


def permutation_p(eval_gap_d, eval_gap_dconst):
    """Exact one-sided label-permutation p-values; accepts batched or 1D gaps."""
    was_1d = eval_gap_d.ndim == 1
    gaps_d = eval_gap_d.unsqueeze(0) if was_1d else eval_gap_d
    gaps_c = eval_gap_dconst.unsqueeze(0) if was_1d else eval_gap_dconst
    gaps_d = gaps_d.to(torch.float64)
    gaps_c = gaps_c.to(torch.float64)
    k_seeds = gaps_d.shape[1]
    contrasts = []
    for selected in itertools.combinations(range(2 * k_seeds), k_seeds):
        weights = [-1.0] * (2 * k_seeds)
        for index in selected:
            weights[index] = 1.0
        contrasts.append(weights)
    contrast_matrix = torch.tensor(
        contrasts, device=gaps_d.device, dtype=torch.float64
    )
    combined = torch.cat((gaps_d, gaps_c), dim=1)
    perm_sums = combined @ contrast_matrix.T
    observed_sums = gaps_d.sum(dim=1) - gaps_c.sum(dim=1)
    p_values = (perm_sums >= observed_sums.unsqueeze(1)).to(
        torch.float32
    ).mean(dim=1)
    return p_values[0] if was_1d else p_values


def generate_eval_gaps(k_seeds, n_rows, delta, sigma_seed, batch_size, generator):
    """Generate arm gaps with independent training-seed effects and eval noise."""
    seed_noise_d = torch.randn(
        (batch_size, k_seeds), device="cuda", generator=generator
    ) * sigma_seed
    seed_noise_c = torch.randn(
        (batch_size, k_seeds), device="cuda", generator=generator
    ) * sigma_seed
    true_gap_d = delta + seed_noise_d
    true_gap_c = seed_noise_c
    totals = torch.full(
        (batch_size, k_seeds), n_rows, device="cuda", dtype=torch.float32
    )
    ordinary_probs = torch.full_like(totals, BASE_ACCURACY)
    salient_probs_d = (BASE_ACCURACY + true_gap_d).clamp(0.0, 1.0)
    salient_probs_c = (BASE_ACCURACY + true_gap_c).clamp(0.0, 1.0)
    salient_d = torch.binomial(totals, salient_probs_d, generator=generator)
    ordinary_d = torch.binomial(totals, ordinary_probs, generator=generator)
    salient_c = torch.binomial(totals, salient_probs_c, generator=generator)
    ordinary_c = torch.binomial(totals, ordinary_probs, generator=generator)
    return (
        (salient_d - ordinary_d) / n_rows,
        (salient_c - ordinary_c) / n_rows,
    )


def run_self_check(seed):
    """Validate exact-test size on Gaussian and simulated data before logging."""
    print("=== ISOLATION A: iid Gaussian gaps, 20,000 simulations ===", flush=True)
    gaussian_generator = torch.Generator(device="cuda")
    gaussian_generator.manual_seed(seed)
    gaussian_results = {}
    for k_seeds in K_GRID:
        sign_rejections = 0
        permutation_rejections = 0
        remaining = SELFCHECK_SIMULATIONS
        while remaining:
            batch_size = min(SELFCHECK_CHUNK, remaining)
            gaps_d = torch.randn(
                (batch_size, k_seeds),
                device="cuda",
                generator=gaussian_generator,
            )
            gaps_c = torch.randn(
                (batch_size, k_seeds),
                device="cuda",
                generator=gaussian_generator,
            )
            sign_rejections += int(
                (sign_flip_p(gaps_d - gaps_c) <= 0.05).sum().item()
            )
            permutation_rejections += int(
                (permutation_p(gaps_d, gaps_c) <= 0.05).sum().item()
            )
            remaining -= batch_size
        sf_size = sign_rejections / SELFCHECK_SIMULATIONS
        perm_size = permutation_rejections / SELFCHECK_SIMULATIONS
        gaussian_results[k_seeds] = (sf_size, perm_size)
        print(
            f"ISOLATION_A K={k_seeds} sign_flip={sf_size:.4f} "
            f"permutation={perm_size:.4f}",
            flush=True,
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    generated_results = {}
    all_positive_fractions = {}
    k5_diffs = []
    k5_tied_sets = 0
    for k_seeds in K_GRID:
        sign_rejections = 0
        permutation_rejections = 0
        differences_all = []
        positive_counts = torch.zeros(k_seeds, device="cuda")
        all_positive_count = 0
        tie_set_count = 0
        remaining = SELFCHECK_SIMULATIONS
        while remaining:
            batch_size = min(SELFCHECK_CHUNK, remaining)
            gaps_d, gaps_c = generate_eval_gaps(
                k_seeds, 4096, 0.0, 0.01, batch_size, generator
            )
            differences = gaps_d - gaps_c
            sf_p = sign_flip_p(differences)
            perm_p = permutation_p(gaps_d, gaps_c)
            sign_rejections += int((sf_p <= 0.05).sum().item())
            permutation_rejections += int((perm_p <= 0.05).sum().item())
            positive_counts += (differences > 0).sum(dim=0)
            all_positive_count += int((differences > 0).all(dim=1).sum().item())
            tie_set_count += int((differences == 0).any(dim=1).sum().item())
            if k_seeds == 5:
                differences_all.append(differences)
            remaining -= batch_size
        sf_size = sign_rejections / SELFCHECK_SIMULATIONS
        perm_size = permutation_rejections / SELFCHECK_SIMULATIONS
        generated_results[k_seeds] = (sf_size, perm_size)
        all_positive_fractions[k_seeds] = (
            all_positive_count / SELFCHECK_SIMULATIONS
        )
        fractions_positive = (
            positive_counts / SELFCHECK_SIMULATIONS
        ).cpu().tolist()
        print(
            f"ISOLATION_B K={k_seeds} sign_flip={sf_size:.4f} "
            f"permutation={perm_size:.4f} "
            f"positive_by_seed={fractions_positive} "
            f"p_all_positive={all_positive_count / SELFCHECK_SIMULATIONS:.4f} "
            f"sets_with_zero_diff={tie_set_count}",
            flush=True,
        )
        if k_seeds == 5:
            k5_diffs = torch.cat(differences_all, dim=0)
            k5_tied_sets = tie_set_count
            mean_diff = k5_diffs.mean().item()
            positive_fractions = torch.tensor(fractions_positive)
            assert abs(mean_diff) < 0.0005
            assert torch.max(torch.abs(positive_fractions - 0.5)).item() < 0.01
            print(
                f"ISOLATION_B_K5_DIFF mean={mean_diff:.8f} "
                f"sd={k5_diffs.std(unbiased=True).item():.8f} "
                f"zero_diff_sets={k5_tied_sets} "
                "tie_rule=exact-zero gaps retained; permutation/sign-flip "
                "comparisons use >= including statistic ties",
                flush=True,
            )

    # Gaussian isolation targets diagnose the test functions themselves.
    assert abs(gaussian_results[5][0] - 0.031) <= 0.005
    assert abs(gaussian_results[5][1] - 0.048) <= 0.006
    assert abs(gaussian_results[3][1] - 0.050) <= 0.006
    assert gaussian_results[3][0] == 0.0
    # The generated eval gaps are discrete; ties are counted by >= and may
    # make exact tests conservative. Gate on requested calibration targets.
    assert abs(generated_results[5][0] - 0.031) <= 0.005
    assert abs(generated_results[5][1] - 0.048) <= 0.006
    assert abs(generated_results[3][1] - 0.050) <= 0.006
    assert generated_results[3][0] == 0.0
    assert abs(all_positive_fractions[5] - 0.031) <= 0.005
    return gaussian_results, generated_results


def run_unit_tests():
    print("=== UNIT TESTS ===")
    for k, label in [(3, "K=3"), (5, "K=5")]:
        all_pos = torch.ones(k, device="cuda")
        p = sign_flip_p(all_pos).item()
        expected = 1.0 / (2 ** k)
        assert abs(p - expected) < 1e-6, f"{label} all-positive sign-flip failed"
        print(f"{label} all-pos: p={p:.6f} expected={expected:.6f} PASS")
        all_neg = -torch.ones(k, device="cuda")
        p_neg = sign_flip_p(all_neg).item()
        assert abs(p_neg - 1.0) < 1e-6, f"{label} all-negative sign-flip failed"
        print(f"{label} all-neg: p={p_neg:.6f} expected=1.000000 PASS")
    mixed = torch.tensor([1.0, 1.0, -1.0], device="cuda")
    p_mix = sign_flip_p(mixed).item()
    assert abs(p_mix - 0.5) < 1e-6, "K=3 mixed sign-flip failed"
    print(f"K=3 mixed (+,+,-): p={p_mix:.6f} expected=0.500000 PASS")
    d3 = torch.ones(3, device="cuda")
    c3 = torch.zeros(3, device="cuda")
    p3 = permutation_p(d3, c3).item()
    assert abs(p3 - 0.05) < 1e-6, "K=3 permutation failed"
    print(f"K=3 perm (D>const): p={p3:.6f} expected=0.050000 PASS")
    d5 = torch.ones(5, device="cuda")
    c5 = torch.zeros(5, device="cuda")
    p5 = permutation_p(d5, c5).item()
    assert abs(p5 - 1 / 252) < 1e-6, "K=5 permutation failed"
    print(f"K=5 perm (D>const): p={p5:.6f} expected={1/252:.6f} PASS")
    print("=== END UNIT TESTS ===")


def simulate_cell(k_seeds, n_rows, delta, sigma_seed, generator):
    """Power of both tests for one cell, vectorized over simulations."""
    significant_sign = 0
    significant_threshold = 0
    significant_perm = 0
    remaining = SIMULATIONS
    while remaining:
        batch_size = min(SIMULATION_CHUNK, remaining)
        eval_gap_d, eval_gap_dconst = generate_eval_gaps(
            k_seeds, n_rows, delta, sigma_seed, batch_size, generator
        )
        per_seed_diff = eval_gap_d - eval_gap_dconst
        diff = per_seed_diff.mean(dim=1)
        p_sign = sign_flip_p(per_seed_diff)
        significant_sign += int((p_sign <= 0.05).sum().item())

        spread_d = eval_gap_d.max(dim=1).values - eval_gap_d.min(dim=1).values
        spread_dconst = (
            eval_gap_dconst.max(dim=1).values
            - eval_gap_dconst.min(dim=1).values
        )
        largest_spread = torch.maximum(spread_d, spread_dconst)
        significant_threshold += int((diff > largest_spread).sum().item())
        p_perm = permutation_p(eval_gap_d, eval_gap_dconst)
        significant_perm += int((p_perm <= 0.05).sum().item())
        remaining -= batch_size
    torch.cuda.synchronize()
    return (
        significant_sign / SIMULATIONS,
        significant_threshold / SIMULATIONS,
        significant_perm / SIMULATIONS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--k-values", nargs="+", type=int, default=K_GRID)
    parser.add_argument("--n-values", nargs="+", type=int, default=N_GRID)
    args = parser.parse_args()
    if any(k_seeds not in K_GRID for k_seeds in args.k_values):
        parser.error(f"--k-values must be selected from {K_GRID}")
    if any(n_rows <= 0 for n_rows in args.n_values):
        parser.error("--n-values must contain positive integers")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if Path(args.output).exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    run_unit_tests()
    gaussian_results, generated_results = run_self_check(args.seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    header = (
        f"POWER_DID_GPU base_accuracy={BASE_ACCURACY} "
        f"simulations={SIMULATIONS} bootstrap_reps=0 "
        f"seed={args.seed}"
    )
    print(header, flush=True)
    with open(args.output, "x", encoding="utf-8") as output:
        output.write(header + "\n")
        output.write(
            "SELF_CHECK_PASS "
            f"gaussian_K3_sf={gaussian_results[3][0]:.4f} "
            f"gaussian_K3_perm={gaussian_results[3][1]:.4f} "
            f"gaussian_K5_sf={gaussian_results[5][0]:.4f} "
            f"gaussian_K5_perm={gaussian_results[5][1]:.4f} "
            f"generated_K3_sf={generated_results[3][0]:.4f} "
            f"generated_K3_perm={generated_results[3][1]:.4f} "
            f"generated_K5_sf={generated_results[5][0]:.4f} "
            f"generated_K5_perm={generated_results[5][1]:.4f}\n"
        )
        output.flush()
        # Minimum achievable p-value of the exact sign-flip test.
        for k_seeds in args.k_values:
            min_p = 1.0 / (2 ** k_seeds)
            line = f"MIN_P K={k_seeds} min_p={min_p:.6f} min_p_perm={1.0/math.comb(2*k_seeds, k_seeds):.6f}"
            print(line, flush=True)
            output.write(line + "\n")
        output.flush()
        for k_seeds, n_rows, delta, sigma_seed in itertools.product(
            args.k_values, args.n_values, DELTA_GRID, SIGMA_GRID
        ):
            started = time.perf_counter()
            power_sign, power_threshold, power_perm = simulate_cell(
                k_seeds, n_rows, delta, sigma_seed, generator
            )
            elapsed = time.perf_counter() - started
            line = (
                f"POWER K={k_seeds} N={n_rows} delta={delta} "
                f"sigma_seed={sigma_seed} "
                f"power_sign_flip={power_sign:.4f} "
                f"power_threshold={power_threshold:.4f} "
                f"power_perm={power_perm:.4f}"
            )
            print(line, flush=True)
            print(
                f"ROW_WALL_SECONDS K={k_seeds} N={n_rows} "
                f"delta={delta} sigma_seed={sigma_seed} "
                f"seconds={elapsed:.3f}",
                flush=True,
            )
            output.write(line + "\n")
            output.flush()


if __name__ == "__main__":
    main()
