"""Build a hashed bigram surprise table from training shards.

This script reads training text shards, counts bigram occurrences using a fixed
hash table (≈2^20 buckets), and outputs:
  - A PyTorch .pt file with the bucket counts (non-trainable buffer)
  - Config and hash metadata
  - Fixed mean mu and std sigma of the surprise values s over the sample

Usage:
    python diagnostics/build_surprise_table.py \
        --shards data/train/shards/*.txt \
        --out tests/golden_retention_config.pt \
        --seed 42 \
        --n-tokens 1000000

The output .pt file contains:
    'buckets': LongTensor ([N_buckets]) - hashed bigram counts
    'config': dict with hyperparameters
    'hash_config': dict with hash function details
    'mu': float - mean of s over the sample
    'sigma': float - std of s over the sample
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

N_BUCKETS = 2 ** 20  # ≈ 1M buckets for bigram hashing
UNIGRAM_FALLBACK = True  # use unigram counts for unseen tokens


def bigram_hash(token_id_prev, token_id_curr):
    """Deterministic hash of a bigram into [0, N_BUCKETS).

    Uses splitmix64-inspired mixing so that adjacent tokens map
    well and the hash is fully deterministic.
    """
    # Combine two u32s into a u64, then apply splitmix64
    combined = (token_id_prev * 2654435761 + token_id_curr) & 0xFFFFFFFFFFFFFFFF
    # Splitmix64: z = (z + 0x9e3779b97f4a7c15) & mask, then loop
    z = combined + 0x9e3779b97f4a7c15
    z = (z | 1) & 0xFFFFFFFFFFFFFFFF  # ensure odd
    z = z - ((z >> 1) & 0xFFFFFFFFFFFFFFFF)
    z = z ^ (z >> 32)
    z = z * 0xbf58476d1ce4e5b9
    z = z ^ (z >> 27)
    z = z * 0x94d049bb133111eb
    z = z ^ (z >> 31)
    return z % N_BUCKETS


def read_shard_tokens(shard_path):
    """Read token IDs from a shard file, trying multiple encodings."""
    content = None
    for enc in ["utf-16", "utf-16-le", "utf-16-be", "utf-8"]:
        try:
            with open(shard_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeError:
            continue

    if content is None:
        print(f"  WARNING: could not decode {shard_path} with any encoding, skipping")
        return []

    # Parse lines - handle both newline styles
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    tokens = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                tokens.append(int(line))
            except ValueError:
                # Skip non-integer lines (raw text handling)
                pass
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Build a hashed bigram surprise table from training shards"
    )
    parser.add_argument(
        "--shards",
        type=str,
        nargs="+",
        required=True,
        help="Paths to training text shards (one token id per line, or raw text)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .pt file path (e.g. tests/golden_retention_config.pt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=1000000,
        help="Number of tokens to sample for statistics (default: 1M)",
    )
    parser.add_argument(
        "--max-surprise",
        type=float,
        default=20.0,
        help="Cap surprise values at this many nats (default: 20.0)",
    )
    parser.add_argument(
        "--bucket-cap",
        type=int,
        default=None,
        help="Hard cap on bucket counts (prevents overflow; default: no cap)",
    )
    args = parser.parse_args()

    shard_paths = [Path(p) for p in args.shards]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Read and sample tokens from shards
    # -------------------------------------------------------------------------
    print(f"Reading shards: {shard_paths}", flush=True)

    all_tokens = []
    for sp in shard_paths:
        tokens_from_shard = read_shard_tokens(sp)
        print(f"  {sp}: {len(tokens_from_shard)} tokens", flush=True)
        all_tokens.extend(tokens_from_shard)

    total_tokens = len(all_tokens)
    print(f"Total tokens read: {total_tokens}", flush=True)

    # Sample tokens if we have more than n-tokens
    if total_tokens > args.n_tokens:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        indices = torch.randperm(total_tokens, generator=g).tolist()
        sampled_indices = indices[: args.n_tokens]
        tokens = [all_tokens[i] for i in sorted(sampled_indices)]
        # Re-shuffle for sequential processing (bigram order matters)
        import random
        random.seed(args.seed)
        random.shuffle(tokens)
    else:
        tokens = all_tokens

    print(f"Using {len(tokens)} sampled tokens", flush=True)

    # -------------------------------------------------------------------------
    # 2. Build bigram count table (hashed)
    # -------------------------------------------------------------------------
    bucket_counts = torch.zeros(N_BUCKETS, dtype=torch.int64)

    # Also track unigram counts for fallback
    unigram_counts = torch.zeros(256 + 1, dtype=torch.int64)  # +1 for "unknown"

    prev_token = None
    seen_pairs = 0

    for i, tok in enumerate(tokens):
        if prev_token is not None:
            h = bigram_hash(prev_token, tok)
            bucket_counts[h] += 1
            if args.bucket_cap is not None:
                bucket_counts[h] = min(bucket_counts[h], args.bucket_cap)
            seen_pairs += 1
        # Track unigram
        unigram_counts[tok] += 1
        prev_token = tok

    print(
        f"Built bigram table: {seen_pairs} pairs into {N_BUCKETS} buckets",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 3. Compute surprise s_t = -log P(x_t | x_{t-1})
    # -------------------------------------------------------------------------
    print("Computing surprise values...", flush=True)

    surprises = []
    prev_token = None

    for tok in tokens:
        if prev_token is not None:
            h = bigram_hash(prev_token, tok)
            count = bucket_counts[h].item()

            # Compute probability: P(x_t | x_{t-1}) = count / unigram(x_{t-1})
            unigram_prev = unigram_counts[prev_token].item()
            if unigram_prev > 0:
                prob = count / unigram_prev
            else:
                # Unseen pair: use uniform fallback over vocabulary range
                # We'll estimate vocab size from max token id + 1
                vocab_size = unigram_counts.shape[0] - 1  # exclude unknown
                if vocab_size > 0:
                    prob = 1.0 / vocab_size
                else:
                    prob = 1.0 / N_BUCKETS  # absolute fallback

            # Cap probability to avoid -log(0) and extreme values
            prob = max(prob, 1e-10)
            prob = min(prob, 1.0 - 1e-10)

            s = -math.log(prob)
            # Cap surprise at max_surprise
            s = min(s, args.max_surprise)

            surprises.append(s)
        else:
            # First token has no preceding bigram; assign neutral surprise
            surprises.append(0.0)

        prev_token = tok

    surprises = torch.tensor(surprises, dtype=torch.float64)
    print(
        f"Computed {len(surprises)} surprise values in [0, {args.max_surprise}]",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 4. Compute mean mu and std sigma of surprises
    # -------------------------------------------------------------------------
    mu = surprises.mean().item()
    sigma = surprises.std(unbiased=False).item()  # population std
    print(f"Surprise statistics: mu={mu:.6f}, sigma={sigma:.6f}", flush=True)

    # -------------------------------------------------------------------------
    # 5. Save output
    # -------------------------------------------------------------------------
    output = {
        "buckets": bucket_counts,
        "unigram_counts": unigram_counts,  # per-token unigram counts for fallback
        "config": {
            "n_buckets": N_BUCKETS,
            "max_surprise": args.max_surprise,
            "bucket_cap": args.bucket_cap,
            "unigram_fallback": UNIGRAM_FALLBACK,
            "n_tokens_sampled": len(tokens),
            "hash_function": "splitmix64-inspired bigram hash",
        },
        "hash_config": {
            "function": "splitmix64-inspired",
            "input_width": 2 * 32,  # two u32 token ids combined into u64
            "key": "bigram_hash",
        },
        "mu": mu,
        "sigma": sigma,
    }

    torch.save(output, str(out_path))
    print(f"Saved surprise table to {out_path}", flush=True)
    print(f"  buckets shape: {bucket_counts.shape}", flush=True)
    print(f"  mu={mu:.6f}, sigma={sigma:.6f}", flush=True)


if __name__ == "__main__":
    main()